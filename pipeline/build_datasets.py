"""
Hybrid Music Recommender - dataset builder

Builds the small and big modelling datasets from real open data, measures the
AcousticBrainz join rate, and produces descriptive statistics.

Data sources (openly available; MLHD+ is non-commercial use only,
AcousticBrainz is CC0):
  - MLHD+          : ListenBrainz's MusicBrainz-cleaned listening history.
                     Interactions, keyed on recording MBID, with timestamps.
                     https://musicbrainz.org/doc/MLHD+
  - AcousticBrainz : Essentia audio analysis, flat feature dumps (rhythm /
                     tonal / lowlevel CSVs), keyed on recording MBID.
                     https://acousticbrainz.org/download

Everything joins on the MusicBrainz recording ID, so no fuzzy string matching.

Inputs (staged from the raw dumps beforehand; staging code not included):
    ab_items.parquet           one row per recording: mbid plus the AcousticBrainz
                               features (bpm, danceability, onset_rate, key_key,
                               key_scale, tuning_frequency, average_loudness,
                               dynamic_complexity, mfcc_zero_mean)
    mlhd_interactions.parquet  one row per listen: user_id, recording_mbid,
                               listened_at (unix seconds)

Run order:
    python build_datasets.py joinrate      # coverage measurement
    python build_datasets.py build_a       # pairs + train-only profiles
    python build_datasets.py build_b       # positives, negatives, datasets
    python build_datasets.py stats         # descriptive statistics
"""

import sys
from pyspark.sql import SparkSession, functions as F, Window

SEED = 42            # fixed so every split and sample is reproducible
K = 4                # negatives sampled per positive
POS_TARGET = 400_000
POOL = 200_000
LO, HI = 1104537600, 1420070399     # 2005-01-01 .. 2014-12-31 UTC


def session(name):
    s = (SparkSession.builder.appName(name).master("local[2]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "200")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    return s


# ---------------------------------------------------------------- step 1
def joinrate():
    """The one measurement capable of invalidating the design.

    AcousticBrainz stopped collecting in 2022, so any recording released since
    has no audio features. Coverage is reported two ways, because they answer
    different questions: coverage of unique recordings says how much of the
    catalogue can be described, coverage of listens says how much of what
    people actually play can be described. The second is what the model sees.
    """
    spark = session("joinrate")
    items = spark.read.parquet("ab_items.parquet").select("mbid").distinct()
    m = spark.read.parquet("mlhd_interactions.parquet")

    n_listens = m.count()
    recs = m.groupBy("recording_mbid").agg(F.count("*").alias("listens"))
    n_recs = recs.count()

    j = recs.join(items, recs.recording_mbid == items.mbid, "inner")
    rec_cov = j.count()
    listen_cov = j.agg(F.sum("listens")).collect()[0][0]

    print("distinct recordings      ", n_recs)
    print("covered by AcousticBrainz", rec_cov, f"({100*rec_cov/n_recs:.2f}%)")
    print("listens                  ", n_listens)
    print("covered by AcousticBrainz", listen_cov, f"({100*listen_cov/n_listens:.2f}%)")
    spark.stop()


# ---------------------------------------------------------------- step 2
def build_a():
    """Interaction table plus the user and item profiles.

    Every aggregate here is computed on the TRAINING slice only. als_score and
    these profiles are learned features, and computing them over the full data
    is the standard way a recommender posts excellent offline numbers that
    collapse on new data.
    """
    spark = session("build_a")
    m = (spark.read.parquet("mlhd_interactions.parquet")
         .filter((F.col("listened_at") >= LO) & (F.col("listened_at") <= HI)))
    items = spark.read.parquet("ab_items.parquet")
    m = m.join(items.select("mbid"), m.recording_mbid == items.mbid, "inner").drop("mbid")

    (m.groupBy("user_id", "recording_mbid")
       .agg(F.count("*").alias("play_count"),
            F.min("listened_at").alias("first_listen"),
            F.max("listened_at").alias("last_listen"))
       .write.mode("overwrite").parquet("pairs.parquet"))

    pairs = spark.read.parquet("pairs.parquet")

    # Split on TIME, not at random. The listens carry timestamps, so a model is
    # never allowed to see a listener's future. This is also what makes the
    # cold-start definition real rather than simulated.
    split_ts = int(pairs.approxQuantile("last_listen", [0.8], 0.001)[0])
    open("SPLIT_TS.txt", "w").write(str(split_ts))
    train = pairs.filter(F.col("last_listen") <= split_ts)

    (train.groupBy("user_id")
          .agg(F.sum("play_count").alias("listen_count_total"),
               F.countDistinct("recording_mbid").alias("distinct_recordings"))
          .write.mode("overwrite").parquet("uprof.parquet"))
    (train.groupBy("recording_mbid")
          .agg(F.sum("play_count").alias("recording_listen_total"),
               F.countDistinct("user_id").alias("distinct_listeners"),
               F.min("first_listen").alias("item_first_listen"))
          .write.mode("overwrite").parquet("iprof.parquet"))
    print("split timestamp", split_ts, "pairs", pairs.count())
    spark.stop()


# ---------------------------------------------------------------- step 3
def build_b():
    """Positives, negatives and the final modelling table.

    MLHD+ records only what people played, so there are no explicit negatives
    and a classifier cannot be trained on it as it stands. Negatives are
    sampled POPULARITY-WEIGHTED rather than uniformly, so the model is forced
    to learn taste instead of the much easier trick of telling famous
    recordings from obscure ones. The seed is fixed and the same negative set
    is reused by every model, so the comparison measures models rather than
    sampling luck.
    """
    spark = session("build_b")
    split_ts = int(open("SPLIT_TS.txt").read().strip())
    pairs = spark.read.parquet("pairs.parquet")
    uprof = spark.read.parquet("uprof.parquet")
    iprof = spark.read.parquet("iprof.parquet")
    items = spark.read.parquet("ab_items.parquet")

    n_pairs = pairs.count()
    pos = (pairs.sample(False, POS_TARGET / n_pairs * 1.15, SEED).limit(POS_TARGET)
           .withColumn("played", F.lit(1))
           .withColumn("split", F.when(F.col("last_listen") <= split_ts, "train")
                                 .otherwise("test")))
    pos.write.mode("overwrite").parquet("pos.parquet")
    pos = spark.read.parquet("pos.parquet")

    pool = (iprof.select("recording_mbid", "recording_listen_total")
            .withColumn("w", F.rand(SEED) / F.log1p(F.col("recording_listen_total")))
            .orderBy("w").limit(POOL)
            .withColumn("idx", F.row_number().over(Window.orderBy("w")) - 1)
            .select("idx", F.col("recording_mbid").alias("neg_mbid")))
    pool.write.mode("overwrite").parquet("pool.parquet")
    pool = spark.read.parquet("pool.parquet")

    cand = (pos.select("user_id", "split")
            .withColumn("idx", F.explode(F.array(
                *[F.floor(F.rand(SEED + i) * POOL).cast("int") for i in range(K)])))
            .join(F.broadcast(pool), "idx").drop("idx")
            .withColumnRenamed("neg_mbid", "recording_mbid")
            .dropDuplicates(["user_id", "recording_mbid"]))

    # Anti-join against every observed pair, so a sampled "negative" is never
    # something the listener actually played.
    neg = (cand.join(pairs.select("user_id", "recording_mbid"),
                     ["user_id", "recording_mbid"], "left_anti")
           .withColumn("played", F.lit(0)).withColumn("play_count", F.lit(0))
           .withColumn("first_listen", F.lit(None).cast("long"))
           .withColumn("last_listen", F.lit(None).cast("long")))
    neg.write.mode("overwrite").parquet("neg.parquet")
    neg = spark.read.parquet("neg.parquet")

    cols = ["user_id", "recording_mbid", "played", "play_count",
            "first_listen", "last_listen", "split"]
    model = (pos.select(*cols).unionByName(neg.select(*cols))
             .join(items, F.col("recording_mbid") == items.mbid, "left").drop("mbid")
             .join(uprof, "user_id", "left")
             .join(iprof, "recording_mbid", "left")
             .withColumn("log1p_play_count", F.log1p(F.col("play_count")))
             .withColumn("days_since_first_listen",
                         F.when(F.col("item_first_listen").isNotNull(),
                                ((F.lit(split_ts) - F.col("item_first_listen")) / 86400)
                                .cast("int")))
             .drop("item_first_listen")
             .fillna({"listen_count_total": 0, "distinct_recordings": 0,
                      "recording_listen_total": 0, "distinct_listeners": 0}))

    model.write.mode("overwrite").parquet("big_dataset.parquet")
    big = spark.read.parquet("big_dataset.parquet")
    big.sample(False, 0.05, SEED).write.mode("overwrite").parquet("small_dataset.parquet")
    print("big", big.count(), "small", spark.read.parquet("small_dataset.parquet").count())
    spark.stop()


# ---------------------------------------------------------------- step 4
def stats():
    """Descriptive statistics for the small dataset."""
    spark = session("stats")
    sm = spark.read.parquet("small_dataset.parquet")
    sm.cache()
    print("rows", sm.count(), "columns", len(sm.columns))
    sm.printSchema()

    numeric = ["played", "play_count", "log1p_play_count", "bpm", "danceability",
               "onset_rate", "tuning_frequency", "average_loudness",
               "dynamic_complexity", "mfcc_zero_mean", "listen_count_total",
               "distinct_recordings", "recording_listen_total",
               "distinct_listeners", "days_since_first_listen"]
    sm.select(numeric).summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show(truncate=False)

    sm.groupBy("played").count().orderBy("played").show()
    sm.groupBy("split").count().orderBy("split").show()
    sm.groupBy("key_scale").count().orderBy(F.desc("count")).show(10, False)
    sm.groupBy("key_key").count().orderBy(F.desc("count")).show(12, False)
    sm.select([F.sum(F.col(c).isNull().cast("int")).alias(c)
               for c in sm.columns]).show(vertical=True, truncate=False)
    spark.stop()


if __name__ == "__main__":
    {"joinrate": joinrate, "build_a": build_a,
     "build_b": build_b, "stats": stats}[sys.argv[1]]()
