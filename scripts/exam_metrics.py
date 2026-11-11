"""Derive all reported exam statistics from the synthetic CSV files."""

import pandas as pd
from exam_settings import PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, PROMPT_TOKENS


def exam_statistics(directory):
    """Return formatted values for the task text and suggested answers."""
    reviews = pd.read_csv(directory / "reviews.csv")
    runs = pd.read_csv(directory / "llm_runs.csv")
    models = pd.read_csv(directory / "model_comparison.csv")
    stars = reviews["stars"].map(
        {1: "negative", 2: "negative", 3: "mixed", 4: "positive", 5: "positive"}
    )
    matches = int(stars.eq(reviews["gold_label"]).sum())
    repeated = runs.pivot(index="review_id", columns="run", values="label")
    changed = repeated.nunique(axis=1).gt(1).mean()
    injection = runs[runs["review_id"] == "R147"]
    input_tokens = int(reviews["n_tokens"].sum() + len(reviews) * PROMPT_TOKENS)
    lookup = reviews.set_index("review_id")
    examples = []
    for review_id in ("R187", "R188"):
        row = lookup.loc[review_id]
        star_label = {
            1: "negative",
            2: "negative",
            3: "mixed",
            4: "positive",
            5: "positive",
        }[row["stars"]]
        examples.append(
            f"| `{review_id}` | «{row['text']}» | {row['stars']} | "
            f"`{star_label}` | `{row['gold_label']}` |"
        )
    stats = {
        "reviews": str(len(reviews)),
        "stars_equal": str(matches),
        "stars_different": str(len(reviews) - matches),
        "stars_percent": f"{matches / len(reviews) * 100:.0f} %",
        "a2_rows": "\n".join(examples),
        "injection_text": lookup.loc["R147", "text"],
        "injection_stars": str(lookup.loc["R147", "stars"]),
        "injection_positive": str(injection["label"].eq("positive").sum()),
        "injection_runs": str(len(injection)),
        "b1_tokens": f"{input_tokens:,}".replace(",", "'"),
        "b1_cost": f"{input_tokens / 1_000_000 * PRICE_INPUT_PER_M:.4f}",
        "c1_fraction": f"{changed:.3f}",
        "c1_percent": f"{changed * 100:.1f} %",
        "prompt_tokens": str(PROMPT_TOKENS),
        "input_price": f"{PRICE_INPUT_PER_M:.2f}",
        "output_price": f"{PRICE_OUTPUT_PER_M:.2f}",
    }
    rankings = {}
    for model, name in (("gpt-6-luna", "api"), ("qwen3:1.7b", "local")):
        group = models[models["model"] == model]
        gold = group["review_id"].map(lookup["gold_label"])
        stats[f"{name}_accuracy"] = f"{group['label'].eq(gold).mean():.3f}"
        stats[f"{name}_seconds"] = f"{group['seconds'].mean():.2f}"
        cost = 0.0
        if name == "api":
            cost = (
                (
                    group["input_tokens"].sum() * PRICE_INPUT_PER_M
                    + group["output_tokens"].sum() * PRICE_OUTPUT_PER_M
                )
                / 1_000_000
                / len(group)
                * 10_000
            )
        stats[f"{name}_cost"] = f"{cost:.3f}"
        stats[f"{name}_reviews"] = str(len(group))
        rankings[model] = (
            -group["label"].eq(gold).mean(),
            group["seconds"].mean(),
            cost,
        )
    stats["recommended_model"] = min(rankings, key=rankings.get)
    return stats
