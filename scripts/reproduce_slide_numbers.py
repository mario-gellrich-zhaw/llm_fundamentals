"""Recompute the numbers shown in the lecture from the data in data/.

Prints average ratings, star distribution, token counts, costs and
duplicate reviews.

Usage:
    python scripts/reproduce_slide_numbers.py
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

DATA = Path(__file__).resolve().parents[1] / "data"

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


def main() -> None:
    """Run the command-line workflow."""
    bus = pd.read_csv(DATA / "businesses_mcdonalds_ikea.csv")
    rev = pd.read_csv(DATA / "ikea_reviews.csv", parse_dates=["date"])

    enc = tiktoken.get_encoding("o200k_base")
    rev["tokens"] = rev["text"].apply(lambda t: len(enc.encode(t)))
    mean_tokens = float(rev["tokens"].mean())
    mcd_reviews = int(
        bus.loc[bus["chain"] == "McDonald's", "review_count"].sum()
    )
    dup = rev[rev.duplicated("text", keep=False)]

    numbers = {
        "chains": {chain: chain_stats(g) for chain, g in bus.groupby("chain")},
        "ikea_reviews": {
            "n": int(len(rev)),
            "stars": {
                int(k): int(v)
                for k, v in rev["stars"].value_counts().sort_index().items()
            },
            "years": [str(rev["date"].min())[:10], str(rev["date"].max())[:10]],
            "duplicate_texts": int(dup["text"].nunique()),
            "duplicate_rows": int(len(dup)),
        },
        "tokens_o200k_base": {
            "total": int(rev["tokens"].sum()),
            "mean": round(mean_tokens, 1),
            "median": int(rev["tokens"].median()),
        },
        "cost_read_every_review_usd": {
            model: {
                "ikea": round(rev["tokens"].sum() / 1e6 * price, 2),
                "mcdonalds_est": round(
                    mcd_reviews * mean_tokens / 1e6 * price, 2
                ),
                "all_yelp_est": round(
                    TOTAL_YELP_REVIEWS * mean_tokens / 1e6 * price
                ),
            }
            for model, price in PRICES_PER_M.items()
        },
    }
    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
