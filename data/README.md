# Data

`small_dataset.parquet` is a 5% random sample (seed 42) of the full modelling table. The full `big_dataset.parquet` (1,978,602 rows, 16 part files, about 69 MB) is not committed. Rebuild it with `pipeline/build_datasets.py`.

| File | Rows | Columns |
|---|---|---|
| `small_dataset.parquet` | 98,987 | 22 |
| `big_dataset.parquet/` (not included) | 1,978,602 | 22 |

Class balance in both: about 20% `played = 1`, 80% `played = 0`.
The `split` column is `train` before 6 Oct 2012 and `test` after.

## Loading

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("recommender").getOrCreate()

small = spark.read.parquet("data/small_dataset.parquet")
train = small.filter("split = 'train'")
test  = small.filter("split = 'test'")
```

For the big dataset, point Spark at the folder, not a part file. Spark reads all 16 files as one table.

## Columns

| Group | Columns |
|---|---|
| Keys | `user_id`, `recording_mbid` |
| Target | `played` (0/1), `play_count`, `log1p_play_count` |
| Audio | `bpm`, `danceability`, `onset_rate`, `key_key`, `key_scale`, `tuning_frequency`, `average_loudness`, `dynamic_complexity`, `mfcc_zero_mean` |
| Listener profile | `listen_count_total`, `distinct_recordings` |
| Song profile | `recording_listen_total`, `distinct_listeners`, `days_since_first_listen` |
| Bookkeeping | `first_listen`, `last_listen`, `split` |

Listener and song profiles are computed from the training period only.

## Quirks

- `danceability` is **not** on a 0 to 1 scale. It's raw Essentia output (mean 1.15, max 14.07). Clip or scale it before modelling.
- `average_loudness` **is** on a 0 to 1 scale. It isn't decibels, despite the name.
- `bpm` of 0.0 means the beat tracker failed, not that the track has no tempo. Treat it as missing.
- `user_id` values are MLHD+'s anonymized listener IDs.

## License

Derived from [MLHD+](https://musicbrainz.org/doc/MLHD+), which MetaBrainz lists as **non-commercial use only**, and [AcousticBrainz](https://acousticbrainz.org/download), which is CC0. Use this data for non-commercial research and education only. The repo's MIT license covers the code, not this data.
