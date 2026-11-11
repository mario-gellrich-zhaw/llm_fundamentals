(
    (
        "Extract the unchanged IKEA and "
        "McDonald's subsets from local Yelp "
        "Parquet "
        "files.\n"
        "\n"
        "Lecturer only: no downloads or model "
        "calls. Keep the output private; "
        "students\n"
        "obtain the subset from Moodle. Requires "
        "pandas and "
        "pyarrow.\n"
        "\n"
        "    python scripts/prepare_data.py "
        "--business data/raw/business.parquet "
        "\\\n"
        "                                   "
        "--reviews data/raw/review.parquet\n"
        "\n"
        "Business names are matched exactly as "
        "specified in the course extraction "
        "task.\n"
        "Every matching review row is retained, "
        "including duplicate texts. Reviews "
        "are\n"
        "sorted by date without changing their "
        "text. Expected counts are checks, "
        "never\n"
        "extra selection criteria. Differences "
        "are reported in subset_validation.json.\n"
    )
)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MCDONALDS_PATTERN = r"^McDonald['’]?s(?:['’]s)?$"
BUSINESS_COLUMNS = [
    "chain",
    "business_id",
    "name",
    "city",
    "state",
    "stars",
    "review_count",
]
REVIEW_INPUT_COLUMNS = ["review_id", "business_id", "stars", "date", "text"]
REVIEW_OUTPUT_COLUMNS = [
    "review_id",
    "business_id",
    "store",
    "city",
    "state",
    "stars",
    "date",
    "text",
]
EXPECTED_INPUT = {"business_rows": 150_346, "review_rows": 6_990_280}
EXPECTED = {
    "IKEA": {
        "business_records": 8,
        "states_provinces": 5,
        "state_codes": ["AB", "FL", "IN", "MO", "PA"],
        "review_rows": 1_743,
        "distinct_review_texts": 1_738,
        "mean_review_stars": 3.26,
        "reviews_by_stars_1_to_5": [397, 155, 243, 496, 452],
        "earliest_review_date": "2007-02-23",
        "latest_review_date": "2022-01-06",
        "business_review_count_sum": 1_688,
    },
    "McDonald's": {
        "business_records": 710,
        "states_provinces": 14,
        "review_rows": 18_280,
        "distinct_review_texts": 18_245,
        "mean_review_stars": 1.78,
        "reviews_by_stars_1_to_5": [12_353, 1_865, 1_317, 1_144, 1_601],
        "earliest_review_date": "2006-07-19",
        "latest_review_date": "2022-01-19",
        "business_review_count_sum": 17_427,
    },
}


def chain_stats(businesses: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    """Compute aggregate checks without modifying either source table."""
    dates = pd.to_datetime(reviews["date"])
    return {
        "business_records": len(businesses),
        "states_provinces": businesses["state"].nunique(),
        "state_codes": sorted(businesses["state"].dropna().unique().tolist()),
        "review_rows": len(reviews),
        "distinct_review_texts": reviews["text"].nunique(),
        "mean_review_stars": (
            round(float(reviews["stars"].mean()), 2) if len(reviews) else None
        ),
        "reviews_by_stars_1_to_5": reviews["stars"]
        .value_counts()
        .reindex(range(1, 6), fill_value=0)
        .tolist(),
        "earliest_review_date": (
            dates.min().strftime("%Y-%m-%d") if len(reviews) else None
        ),
        "latest_review_date": (
            dates.max().strftime("%Y-%m-%d") if len(reviews) else None
        ),
        "business_review_count_sum": int(businesses["review_count"].sum()),
    }


def extract_subsets(business_path: Path, review_path: Path, out: Path) -> dict:
    """Write the three specified CSVs and return the expected/actual checks."""
    # Parquet support is optional for the rest of the course.
    import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel

    # Read only local files. No source rows or review texts are cleaned.
    bus = pd.read_parquet(business_path, columns=BUSINESS_COLUMNS[1:])
    mcd = bus[bus["name"].str.match(MCDONALDS_PATTERN, case=False, na=False)]
    ikea = bus[bus["name"].str.match(r"^IKEA", case=False, na=False)]
    selected = pd.concat(
        [mcd.assign(chain="McDonald's"), ikea.assign(chain="IKEA")]
    )
    info, ids = selected.set_index("business_id"), set(selected["business_id"])

    pf = pq.ParquetFile(review_path)
    parts = []
    for group in range(pf.num_row_groups):
        chunk = pf.read_row_group(
            group, columns=REVIEW_INPUT_COLUMNS
        ).to_pandas()
        parts.append(chunk[chunk["business_id"].isin(ids)])
        print(
            f"Row group {group + 1}/{pf.num_row_groups} processed", flush=True
        )
    reviews = pd.concat(parts, ignore_index=True)
    for column, source in [
        ("chain", "chain"),
        ("store", "name"),
        ("city", "city"),
        ("state", "state"),
    ]:
        reviews[column] = reviews["business_id"].map(info[source])

    out.mkdir(parents=True, exist_ok=True)
    selected[BUSINESS_COLUMNS].to_csv(
        out / "businesses_mcdonalds_ikea.csv", index=False, encoding="utf-8"
    )
    actual = {}
    for chain, filename in [
        ("IKEA", "ikea_reviews.csv"),
        ("McDonald's", "mcdonalds_reviews.csv"),
    ]:
        subset = reviews[reviews["chain"] == chain].sort_values("date")[
            REVIEW_OUTPUT_COLUMNS
        ]
        subset.to_csv(out / filename, index=False, encoding="utf-8")
        actual[chain] = chain_stats(
            selected[selected["chain"] == chain], subset
        )

    input_actual = {
        "business_rows": len(bus),
        "review_rows": pf.metadata.num_rows,
    }
    differences = []
    for scope, expected, observed in [
        ("input", EXPECTED_INPUT, input_actual)
    ] + [(chain, target, actual[chain]) for chain, target in EXPECTED.items()]:
        for check, target in expected.items():
            if observed[check] != target:
                differences.append(
                    {
                        "scope": scope,
                        "check": check,
                        "expected": target,
                        "actual": observed[check],
                    }
                )
    report = {
        "input": input_actual,
        "actual": actual,
        "expected": EXPECTED,
        "differences": differences,
        "all_checks_match": not differences,
    }
    (out / "subset_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Run the command-line workflow."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--business", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    for path in [args.business, args.reviews]:
        if not path.is_file():
            parser.error(
                f"Local input file missing: {path}. Nothing will be downloaded."
            )
    report = extract_subsets(args.business, args.reviews, args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["differences"]:
        raise SystemExit(
            (
                (
                    "Counts differ from the specification. "
                    "The selection rules were not "
                    "adjusted."
                )
            )
        )


if __name__ == "__main__":
    main()
