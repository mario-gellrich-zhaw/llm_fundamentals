"""Rebuild the Yelp subset in data/ from the full Yelp Open Dataset.

Only needed if you want to recreate the files; they are already in data/.

Input: the two Parquet files of the Yelp Open Dataset (copy on Hugging Face:
yashraizad/yelp-open-dataset-business, yashraizad/yelp-open-dataset-reviews),
e.g. saved in data/raw/ (ignored by git).

Output:
    data/businesses_mcdonalds_ikea.csv   all McDonald's and IKEA locations
    data/ikea_reviews.csv                all reviews of the IKEA locations

Usage:
    python scripts/prepare_data.py --business data/raw/business.parquet \\
                                   --reviews data/raw/review.parquet

Requires: pandas, pyarrow (pip install pyarrow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--business", type=Path, required=True)
    p.add_argument("--reviews", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "data")
    args = p.parse_args()

    # Business records: exact name match for McDonald's, prefix match for IKEA
    bus = pd.read_parquet(args.business)
    mcd = bus[bus["name"].str.match(r"^McDonald'?s$", case=False, na=False)]
    ikea = bus[bus["name"].str.match(r"^IKEA", case=False, na=False)]
    sel = pd.concat([mcd.assign(chain="McDonald's"), ikea.assign(chain="IKEA")])
    cols = ["chain", "business_id", "name", "city", "state", "stars", "review_count"]
    sel[cols].to_csv(args.out / "businesses_mcdonalds_ikea.csv", index=False)
    print(f"McDonald's locations: {len(mcd)} | IKEA locations: {len(ikea)}")

    # IKEA reviews: read the 3 GB review file one row group at a time
    ids = set(ikea["business_id"])
    pf = pq.ParquetFile(args.reviews)
    parts = []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(
            g, columns=["review_id", "business_id", "stars", "date", "text"]
        ).to_pandas()
        parts.append(t[t["business_id"].isin(ids)])
    rev = pd.concat(parts, ignore_index=True)

    info = ikea.set_index("business_id")
    rev["store"] = rev["business_id"].map(info["name"])
    rev["city"] = rev["business_id"].map(info["city"])
    rev["state"] = rev["business_id"].map(info["state"])
    rev["stars"] = rev["stars"].astype(int)
    cols = ["review_id", "business_id", "store", "city", "state", "stars", "date", "text"]
    rev[cols].to_csv(args.out / "ikea_reviews.csv", index=False)
    print(f"IKEA reviews: {len(rev):,}")


if __name__ == "__main__":
    main()
