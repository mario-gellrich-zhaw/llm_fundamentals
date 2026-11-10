# Data

All notebooks use a small subset of the [Yelp Open Dataset](https://yelp.com/dataset):
the McDonald's and IKEA locations and all reviews of the IKEA locations.

**The data files are not in this repository.** The Yelp terms do not allow publishing
the data. Download `llm_fundamentals_data.zip` from Moodle and put it into this folder;
the first cells of `01_Explore_Yelp_Data` unpack it here. The steps are in the main
[README](../README.md#data).
Git ignores the files, so they are never pushed to GitHub.

## Files

| File | Rows | Content |
|---|---|---|
| `businesses_mcdonalds_ikea.csv` | 716 | One row per location: 708 McDonald's, 8 IKEA (6 stores and 2 store restaurants) |
| `ikea_reviews.csv` | 1,743 | All reviews of the 8 IKEA locations, 2007–2022 |
| `fewshot_examples.json` | 3 | Labelled IKEA reviews used as few-shot examples in `03_Classify_Reviews` (labels by us) |
| `TERMS_OF_USE.txt` | – | Short note on the terms of use, included in the Moodle zip |
| `processed/reviews_sample.csv` | 500 | Anonymized random sample, written by `01_Explore_Yelp_Data`, read by `03_Classify_Reviews` and `04_Mini_RAG` (not in the zip) |
| `processed/silver_labels.csv` | up to 100 | Written by `03_Classify_Reviews`, read by `05_Local_Model` (not in the zip) |
| `raw/` | – | Place for the full Yelp files, only needed to rebuild the subset |

All of these are ignored by git; only this README and the empty folders `raw/` and
`processed/` are in the repository.

### `businesses_mcdonalds_ikea.csv`

| Column | Meaning |
|---|---|
| `chain` | `McDonald's` or `IKEA` |
| `business_id` | Yelp ID of the location |
| `name` | Name on Yelp, e.g. `IKEA` or `IKEA Restaurant` |
| `city`, `state` | Location (11 metro areas in the US and Canada) |
| `stars` | Average rating of the location, rounded to 0.5 |
| `review_count` | Number of reviews according to the business file |

### `ikea_reviews.csv`

| Column | Meaning |
|---|---|
| `review_id` | Yelp ID of the review |
| `business_id` | Yelp ID of the location (links to the business file) |
| `store`, `city`, `state` | Location of the review |
| `stars` | Rating of the review, 1–5 |
| `date` | Date and time of the review (UTC) |
| `text` | Full review text, original spelling |

4 review texts appear more than once (9 rows); `01_Explore_Yelp_Data` keeps each text
once when it draws the sample.

### `fewshot_examples.json`

A list of 3 examples, each with `text` (review), `yelp_stars`, `sentiment` (our label),
`aspects` (our labels per aspect) and `note`. `03_Classify_Reviews` uses `text` and
`sentiment`.

### `processed/reviews_sample.csv`

A random sample of 500 unique reviews (fixed seed, without the three few-shot examples).
The texts are anonymized; Yelp IDs are replaced by short IDs.

| Column | Meaning |
|---|---|
| `id` | Short ID of the review in the sample, `R001` to `R500` |
| `store`, `city` | Location of the review |
| `stars` | Rating of the review, 1–5 |
| `date` | Date and time of the review (UTC) |
| `text` | Review text after local anonymization: names of people, e-mail addresses and phone numbers are replaced by `<REDACTED>` |

### `processed/silver_labels.csv`

100 reviews from the sample, 20 per star rating (fewer if the model gave no valid label).

| Column | Meaning |
|---|---|
| `id` | Short ID of the review in the sample |
| `stars` | Rating of the review, 1–5 |
| `text` | Anonymized review text |
| `silver` | Silver label from `gpt-6-luna` with high reasoning effort: `positive`, `negative` or `mixed` |

## How the subset was selected

- McDonald's: business name exactly `McDonald's` or `McDonalds` (case-insensitive).
- IKEA: business name starts with `IKEA`.
- Reviews: all reviews in the review file whose `business_id` is one of the IKEA locations.

The `review_count` in the business file (1,688 for IKEA) is not the same as the number of
rows in the review file (1,743).

To recompute the numbers shown in the lecture from the files in `data/`:

```bash
python scripts/reproduce_slide_numbers.py
```

For the lecturer only: to rebuild the subset from the full data set (Parquet copy on
Hugging Face: `yashraizad/yelp-open-dataset-business`, `yashraizad/yelp-open-dataset-reviews`),
save the two files in `data/raw/` and run:

```bash
pip install pyarrow
python scripts/prepare_data.py --business data/raw/business.parquet --reviews data/raw/review.parquet
```

## Licence

The Yelp Open Dataset is licensed for non-commercial, academic use, and the data may not
be displayed or distributed publicly (Yelp Dataset Terms of Use, section 4A). Use the data
only in this course, don't share it, and delete it after the semester. Review texts can
contain names of people: don't search for or contact the authors.

## Before external API calls

Keep the original data on your computer. The last part of notebook 01 draws the sample,
anonymizes it locally with Presidio and spaCy, shows texts before and after, and saves
`processed/reviews_sample.csv`. Notebooks 03–05 read only this sample. Notebook 02 counts
the tokens of all reviews locally and sends none of them.

Automatic anonymization can miss personal details, so the files in `processed/` stay
private too. The API helpers anonymize every text once more before sending it. See the
section on anonymization in the main README.

Silver labels are model-generated references, so reported accuracy against them is
model agreement. Reference failures are excluded and counted; prediction failures count
as errors. These labels cannot establish independent human-level accuracy.
