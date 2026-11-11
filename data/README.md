# Data

All notebooks use a small subset of the [Yelp Open Dataset](https://yelp.com/dataset):
the McDonald's and IKEA locations and all reviews of both chains.

**The data files are not in this repository.** The Yelp terms do not allow publishing
the data. Download `llm_fundamentals_data.zip` from Moodle and put it into this folder;
the first cells of `01_Explore_Yelp_Data` unpack it here. The steps are in the main
[README](../README.md#data).
Git ignores the files, so they are never pushed to GitHub.

## Files

| File | Rows | Content |
|---|---|---|
| `businesses_mcdonalds_ikea.csv` | 718 | One row per location: 710 McDonald's, 8 IKEA (6 stores and 2 store restaurants) |
| `ikea_reviews.csv` | 1,743 | All reviews of the 8 IKEA locations, 2007–2022 |
| `mcdonalds_reviews.csv` | 18,280 | All reviews of the 710 McDonald's locations, 2006–2022 |
| `fewshot_examples.json` | 3 | Labelled IKEA reviews used as few-shot examples in `03_Classify_Reviews` (labels by us) |
| `silver_reference.csv` | 40 | Prepared sample IDs and silver labels for notebook 03, included in the Moodle zip |
| `TERMS_OF_USE.txt` | – | Short note on the terms of use, included in the Moodle zip |
| `processed/reviews_sample.csv` | 500 | Anonymized sample of 250 reviews per chain, written by `01_Explore_Yelp_Data`, read by `03_Classify_Reviews` and `04_Extract_Information` (not in the zip) |
| `processed/silver_labels.csv` | 40 | Written by `03_Classify_Reviews`, read by `05_Local_Model` (not in the zip) |
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

### `ikea_reviews.csv` and `mcdonalds_reviews.csv`

| Column | Meaning |
|---|---|
| `review_id` | Yelp ID of the review |
| `business_id` | Yelp ID of the location (links to the business file) |
| `store`, `city`, `state` | Location of the review |
| `stars` | Rating of the review, 1–5 |
| `date` | Date and time of the review (UTC) |
| `text` | Full review text, original spelling |

The original CSVs retain all repeated texts: IKEA has 1,738 distinct texts and McDonald's has 18,245. `01_Explore_Yelp_Data` keeps each text once when it draws the teaching sample; it does not modify these source CSVs.

### `fewshot_examples.json`

A list of 3 examples, each with `text` (review), `yelp_stars`, `sentiment` (our label),
`aspects` (our labels per aspect) and `note`. `03_Classify_Reviews` uses `text` and
`sentiment`.

### `processed/reviews_sample.csv`

A random sample of 500 unique reviews: 250 per chain (fixed seed, without the three few-shot examples). Equal chain sizes support teaching comparisons; they do not preserve the original dataset proportions.
The texts are anonymized; Yelp IDs are replaced by short IDs.

| Column | Meaning |
|---|---|
| `id` | Short ID of the review in the sample, `R001` to `R500` |
| `chain` | `IKEA` or `McDonald's` |
| `store`, `city` | Location of the review |
| `stars` | Rating of the review, 1–5 |
| `date` | Date and time of the review (UTC) |
| `text` | Review text after local anonymization: names of people, e-mail addresses and phone numbers are replaced by placeholders such as `<PERSON>` |

### `silver_reference.csv`

Prepared once from existing `gpt-6-luna` reference labels with high reasoning effort.
Contains only `id` (the sample ID from notebook 01) and `silver` (the sentiment label).
The 40 cases include 20 reviews per chain, with four per star rating and chain.
IDs refer to the fixed sample produced by notebook 01; keep its sampling unchanged.
Notebook 03 loads these labels without generating them again. Keep this file in the
private course data package, together with the other Yelp-derived data.

### `processed/silver_labels.csv`

The same 40 reviews, with their locally anonymized texts and prepared labels.
Notebook 03 joins them by sample ID and saves this file for notebook 05.

| Column | Meaning |
|---|---|
| `id` | Short ID of the review in the sample |
| `chain` | `IKEA` or `McDonald's` |
| `stars` | Rating of the review, 1–5 |
| `text` | Anonymized review text |
| `silver` | Silver label from `gpt-6-luna` with high reasoning effort: `positive`, `negative` or `mixed` |

## How the subset was selected

- McDonald's: business name fully matches `^McDonald['’]?s(?:['’]s)?$` (case-insensitive), including the four apostrophe variants in the dataset.
- IKEA: business name starts with `IKEA`.
- Reviews: all reviews in the review file whose `business_id` is one of the selected locations. No additional filters, deduplication or text changes.

The business-file `review_count` sums (IKEA: 1,688; McDonald's: 17,427) differ from the review-file row counts (1,743 and 18,280). This is expected. The lecture uses review-file counts.

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

The script validates the source row counts and all expected subset statistics in
`subset_validation.json`. It reports differences without changing the selection.
It never downloads source files.

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
private too. Notebooks 03–05 send only these anonymized texts to the API. See the
section on anonymization in the main README.

Silver labels are model-generated references, so reported accuracy against them is
model agreement. Reference failures are excluded and counted; prediction failures count
as errors. These labels cannot establish independent human-level accuracy.
