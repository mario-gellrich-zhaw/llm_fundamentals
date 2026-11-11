"""Recompute the numbers shown in the lecture from the data in data/.

Prints average ratings, star distribution, token counts, costs and
duplicate reviews for both chains.

Usage:
    python scripts/reproduce_slide_numbers.py
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

DATA = Path(__file__).resolve().parents[1] / "data"

FILES = {"IKEA": "ikea_reviews.csv", "McDonald's": "mcdonalds_reviews.csv"}
TOTAL_YELP_REVIEWS = 6_990_280  # review file of the Yelp Open Dataset
PRICES_PER_M = {
    "gpt-6-luna": 0.10,
    "gpt-6-astra": 10.00,
}  # USD per 1M input tokens, 31 Oct 2026


def chain_stats(df: pd.DataFrame) -> dict:
    """Locations, reviews and average rating weighted by number of reviews."""
    w = df["review_count"]
    return {
        "locations": int(len(df)),
        "review_count_sum": int(w.sum()),
        "stars_weighted_by_review_count": round(
            float((df["stars"] * w).sum() / w.sum()), 2
        ),
    }


def review_stats(rev: pd.DataFrame, enc) -> dict:
    """Review counts, star shares, duplicates and token counts of one chain."""
    tokens = rev["text"].apply(lambda t: len(enc.encode(t)))
    dup = rev[rev.duplicated("text", keep=False)]
    return {
        "n": int(len(rev)),
        "stars": {
            int(k): int(v)
            for k, v in rev["stars"].value_counts().sort_index().items()
        },
        "share_1_star_percent": round(float((rev["stars"] == 1).mean() * 100)),
        "years": [str(rev["date"].min())[:10], str(rev["date"].max())[:10]],
        "duplicate_texts": int(dup["text"].nunique()),
        "duplicate_rows": int(len(dup)),
        "tokens_total": int(tokens.sum()),
        "tokens_mean": round(float(tokens.mean()), 1),
        "tokens_median": int(tokens.median()),
    }


def main() -> None:
    """Run the command-line workflow."""
    bus = pd.read_csv(DATA / "businesses_mcdonalds_ikea.csv")
    enc = tiktoken.get_encoding("o200k_base")
    reviews = {}
    for chain, filename in FILES.items():
        rev = pd.read_csv(DATA / filename, parse_dates=["date"])
        reviews[chain] = review_stats(rev, enc)

    # Token totals: counted for both chains, estimated for all of Yelp with
    # the mean length of an IKEA review (as stated on the slide).
    totals = {chain: stats["tokens_total"] for chain, stats in reviews.items()}
    totals["IKEA + McDonald's"] = sum(totals.values())
    yelp_estimate = TOTAL_YELP_REVIEWS * reviews["IKEA"]["tokens_mean"]

    costs = {}
    for model, price in PRICES_PER_M.items():
        costs[model] = {
            name: round(tokens / 1e6 * price, 2)
            for name, tokens in totals.items()
        }
        costs[model]["all_yelp_est"] = round(yelp_estimate / 1e6 * price)

    numbers = {
        "chains": {chain: chain_stats(g) for chain, g in bus.groupby("chain")},
        "reviews": reviews,
        "tokens_total_o200k_base": totals,
        "cost_read_every_review_usd": costs,
    }
    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
