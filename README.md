# Hybrid Music Recommender (PySpark)

A music recommender that combines **collaborative filtering** (what similar listeners played) with **content-based filtering** (what the audio sounds like), built on real listening data and run on Google Cloud Dataproc.

Collaborative filtering is strong once a song has a listening history and useless before that. A track released yesterday has no plays, so CF can't score it. Audio features exist from day one. This project measures how much each approach contributes, and whether a learned hybrid beats both.

Course project for CIS 415 (Big Data & AI in Business) by Vrishabh Gupta.

## Data

Two open datasets, joined on the MusicBrainz recording ID (exact key match, no fuzzy string matching):

| Source | What it contributes | License |
|---|---|---|
| [MLHD+](https://musicbrainz.org/doc/MLHD+) (MetaBrainz) | Timestamped listening histories, cleaned to MusicBrainz IDs | Non-commercial use only |
| [AcousticBrainz](https://acousticbrainz.org/download) | Audio analysis per recording: tempo, danceability, key, loudness, timbre | CC0 |

| Dataset | Rows | In this repo |
|---|---|---|
| `small_dataset.parquet` | 98,987 (5% random sample) | Yes, in `data/` |
| `big_dataset.parquet` | 1,978,602 | No, rebuild it with `pipeline/build_datasets.py` |

About 20% of rows are plays (`played = 1`). MLHD+ only records what people played, so the non-plays are sampled negatives: 4 per play, weighted toward popular songs so the model has to learn taste rather than just telling famous tracks from obscure ones. Train and test are split by **time** (6 Oct 2012), so no model ever sees a listener's future.

Column list and data quirks: [`data/README.md`](data/README.md).

## Approach

Seven models, all trained on the same rows with the same class weights:

| # | Model | Uses |
|---|---|---|
| 1 | Popularity baseline | How many people played the song |
| 2 | Collaborative filtering | Implicit-feedback ALS on play counts |
| 3 | Content-based | Cosine similarity between a song's audio profile and the listener's average taste |
| 4 | Score blend | `alpha * CF + (1 - alpha) * content`, alpha tuned on training data |
| 5 to 7 | Hybrid: Logistic Regression, Random Forest, Gradient-Boosted Trees | Audio, popularity, listener history, and both engines' scores as features |

### Leakage controls

Recommender benchmarks are easy to inflate by accident. Four things keep these numbers honest:

1. **Fair cold-start test.** Every sampled negative existed before the split, so a brand-new song would always be a positive. Each new song a listener played is paired with four other new songs they didn't play, so popularity is zero on both sides.
2. **Leave-one-out popularity.** A training row's own plays are removed from its song's popularity counts.
3. **Out-of-fold engine scores.** ALS and content scores for training rows come from models fit on the other half of the data, so no row is scored by a model that saw it.
4. **Train-only statistics.** Every median, percentile and scaling constant comes from training rows only.

## Results (small dataset)

Warm test = songs that existed before the split. New songs = songs first heard after the split.

| Model | F1 | ROC-AUC | PR-AUC | New songs ROC-AUC |
|---|---|---|---|---|
| Popularity | 0.512 | 0.805 | 0.571 | 0.500 |
| Collaborative filtering (ALS) | 0.190 | 0.548 | 0.430 | 0.500 |
| Content-based | 0.315 | 0.547 | 0.219 | 0.534 |
| Blend (90% CF, 10% content) | 0.318 | 0.587 | 0.299 | 0.537 |
| Logistic Regression | 0.537 | 0.823 | 0.583 | 0.520 |
| Random Forest | 0.544 | 0.816 | 0.566 | 0.503 |
| **Gradient-Boosted Trees** | 0.537 | **0.824** | **0.588** | 0.520 |

What stands out:

- **The hybrids win, but not by much.** Gradient-Boosted Trees beat popularity on PR-AUC (0.588 vs 0.571). A small win is expected here; a huge one would point to leakage.
- **Popularity does most of the work.** The number of listeners a song has accounts for about 55% of the GBT's feature importance.
- **Content is the only thing that helps with new songs.** Popularity and CF are stuck at 0.500 (a coin flip) on songs nobody had heard yet. Content and the blend are the only models above 0.53.
- **CF is starved on 5% of the data.** Most listeners in the sample have too few plays for ALS to learn from. Scaling to the full dataset is the main test of whether that changes.

**Status:** scaling to the full 1.98M-row dataset on Dataproc is in progress. Big-data results will be added here.

## Run it

The notebook detects where it's running (Colab, Dataproc or local). Switching from small to big data means changing one line: `data_file_name` in the config cell.

**Locally** (needs Java 17+):

```bash
pip install -r requirements.txt
jupyter notebook notebooks/hybrid_recommender.ipynb
```

In the config cell, set `LOCAL_DATA_DIR = "../data/"`, then run all cells. Expect 15 to 20 minutes on the small data, most of it training the Gradient-Boosted Trees.

**Google Colab or Dataproc:** see [`docs/running_on_gcp.md`](docs/running_on_gcp.md).

## Repo layout

```
notebooks/hybrid_recommender.ipynb   the full pipeline: features, both engines, 7 models, evaluation
pipeline/build_datasets.py           builds the big and small datasets from the staged MLHD+ and AcousticBrainz data
data/small_dataset.parquet           98,987-row sample, ready to run
data/README.md                       columns, data quirks, license
data/descriptive_statistics.txt      summary statistics for the small dataset
docs/running_on_gcp.md               Cloud Storage, Colab and Dataproc setup
```

## License

Code is MIT licensed (see `LICENSE`). The data in `data/` is **not** covered by that license: it is derived from MLHD+ (non-commercial use only) and AcousticBrainz (CC0), and is shared for non-commercial research and education.

## Acknowledgments

Listening data from MLHD+ by [MetaBrainz](https://metabrainz.org), built on the Music Listening Histories Dataset by Vigliensoni and Fujinaga (ISMIR 2017). Audio features from AcousticBrainz, a MetaBrainz and Music Technology Group (UPF) project.
