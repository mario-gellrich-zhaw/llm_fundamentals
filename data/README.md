# Data

All notebooks use a small subset of the [Yelp Open Dataset](https://yelp.com/dataset):
the McDonald's and IKEA locations and all reviews of the IKEA locations.

**The data files are not in this repository.** The Yelp terms do not allow publishing
the data. Download `llm_fundamentals_data.zip` from Moodle and unpack it in the top
folder of the repository; the steps are in the main [README](../README.md#data).
Git ignores the files, so they are never pushed to GitHub.

## Files

| File | Rows | Content |
|---|---|---|
| `businesses_mcdonalds_ikea.csv` | 716 | One row per location: 708 McDonald's, 8 IKEA (6 stores and 2 store restaurants) |
| `ikea_reviews.csv` | 1,743 | All reviews of the 8 IKEA locations, 2007–2022 |
| `fewshot_examples.json` | 3 | Labelled IKEA reviews used as few-shot examples in `03_Classify_Reviews` (labels by us) |
| `raw/` | – | Place for the full Yelp files if you want to rebuild the subset (ignored by git) |
| `processed/` | – | Files the notebooks write, e.g. `silver_labels.csv` (ignored by git) |

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

## How the subset was selected

- McDonald's: business name exactly `McDonald's` or `McDonalds` (case-insensitive).
- IKEA: business name starts with `IKEA`.
- Reviews: all reviews in the review file whose `business_id` is one of the IKEA locations.

The `review_count` in the business file (1,688 for IKEA) is not the same as the number of
rows in the review file (1,743).

To rebuild the subset from the full data set (Parquet copy on Hugging Face:
`yashraizad/yelp-open-dataset-business`, `yashraizad/yelp-open-dataset-reviews`):

```bash
pip install pyarrow
python scripts/prepare_data.py --business data/raw/business.parquet --reviews data/raw/review.parquet
python scripts/reproduce_slide_numbers.py
```

## Licence

The Yelp Open Dataset is licensed for non-commercial, academic use, and the data may not
be displayed or distributed publicly (Yelp Dataset Terms of Use, section 4A). Use the data
only in this course, don't share it, and delete it after the semester. Review texts can
contain names of people: don't search for or contact the authors.
