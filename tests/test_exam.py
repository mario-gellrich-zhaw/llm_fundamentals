"""Check exam-data consistency, task rendering and reproducible notebooks."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAM = ROOT / "06_Example_Exam_Questions"


def load_script(name, dependencies=None, apparent_path=None):
    """Load a lecturer script with explicit local module dependencies."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    if apparent_path is not None:
        module.__file__ = str(apparent_path)
    previous = {key: sys.modules.get(key) for key in dependencies or {}}
    sys.modules.update(dependencies or {})
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


SETTINGS = load_script("exam_settings")
METRICS = load_script("exam_metrics", {"exam_settings": SETTINGS})
STYLE = load_script("code_style")
BUILDER = load_script(
    "build_ap01_notebooks",
    {"exam_metrics": METRICS, "exam_settings": SETTINGS, "code_style": STYLE},
)


class ExamTests(unittest.TestCase):
    """Verify that reported quantities stay connected to the source CSVs."""

    def test_model_inputs_use_the_same_prompt_budget_as_b1(self):
        """Compare both models' token totals with the per-review assumption."""
        reviews = pd.read_csv(EXAM / "reviews.csv").set_index("review_id")
        models = pd.read_csv(EXAM / "model_comparison.csv")
        expected = models["review_id"].map(reviews["n_tokens"])
        pd.testing.assert_series_equal(
            models["input_tokens"] - expected,
            pd.Series(SETTINGS.PROMPT_TOKENS, index=models.index),
            check_names=False,
        )
        groups = models.groupby("model")["review_id"].apply(set)
        self.assertEqual(groups.iloc[0], groups.iloc[1])
        self.assertFalse((EXAM / "llm_answers.csv").exists())
        self.assertIn("Reasoning", BUILDER.DATA_NOTES)

    def test_changed_csvs_change_the_reported_metrics(self):
        """Alter CSV values and check costs, agreement and attack counts."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "06_Example_Exam_Questions"
            target.mkdir()
            for name in ("reviews.csv", "llm_runs.csv", "model_comparison.csv"):
                (target / name).write_bytes((EXAM / name).read_bytes())
            before = METRICS.exam_statistics(target)
            models = pd.read_csv(target / "model_comparison.csv")
            models.loc[models["model"] == "gpt-6-luna", "output_tokens"] *= 2
            models.loc[models["model"] == "gpt-6-luna", "label"] = "unparsed"
            models.to_csv(target / "model_comparison.csv", index=False)
            runs = pd.read_csv(target / "llm_runs.csv")
            runs.loc[runs["review_id"] == "R147", "label"] = "negative"
            runs.to_csv(target / "llm_runs.csv", index=False)
            after = METRICS.exam_statistics(target)
            self.assertNotEqual(before["api_cost"], after["api_cost"])
            self.assertNotEqual(before["c1_fraction"], after["c1_fraction"])
            self.assertEqual(after["injection_positive"], "0")
            changed_builder = load_script(
                "build_ap01_notebooks",
                {"exam_metrics": METRICS, "code_style": STYLE},
                Path(directory) / "scripts" / "build_ap01_notebooks.py",
            )
            self.assertIn(after["api_cost"], changed_builder.C3["question"])
            self.assertIn(after["c1_fraction"], changed_builder.C1["answer"])
            self.assertIn("0 von 5", changed_builder.A3["question"])
            self.assertIn("**qwen3:1.7b**", changed_builder.C3["answer"])

    def test_task_text_and_completed_skeletons_match_the_solution(self):
        """Run the intended TODO completions and compare their results."""
        namespace = {}
        # Only trusted local exam cells are executed here.
        # pylint: disable=exec-used
        with contextlib.redirect_stdout(io.StringIO()):
            for source in BUILDER.SETUP_CODE:
                exec(source, namespace)
            # Each correction part runs on its own, without the other part.
            for task in (BUILDER.B1, BUILDER.B2):
                for part in task["parts"]:
                    exec(part["code"], dict(namespace))
                    exec(part["code"], namespace)
            self.assertEqual(
                f"{namespace['input_tokens_B1']:,}".replace(",", "'"),
                BUILDER.STATS["b1_tokens"],
            )
            self.assertEqual(
                f"{namespace['cost_B1']:.3f}", BUILDER.STATS["b1_cost"]
            )
            self.assertIn(
                "mixed",
                namespace["schema_B2"]["properties"]["sentiment"]["enum"],
            )
            completions = [
                (BUILDER.C1, "runs_C1.nunique(axis=1) > 1"),
                (BUILDER.C2, "bm25_C2.get_scores(query_words_C2)"),
                (
                    BUILDER.C3,
                    "accuracy_score(test_C3['gold_label'], test_C3['label'])",
                ),
            ]
            for task, expression in completions:
                # Every coding task has exactly one independent TODO.
                self.assertEqual(task["skeleton"].count("# TODO"), 1)
                source = task["skeleton"].replace(
                    "...  # TODO:", expression + "  # TODO:"
                )
                exec(source, dict(namespace))
                exec(source, namespace)
            self.assertEqual(len(namespace["output_C2"]), 5)
            self.assertEqual(
                f"{namespace['changed_C1'].mean():.3f}",
                BUILDER.STATS["c1_fraction"],
            )
        for task, key in (
            (BUILDER.C1, "c1_fraction"),
            (BUILDER.C3, "api_cost"),
        ):
            self.assertIn(BUILDER.STATS[key], task["answer"])
        # The C3 table states the accuracies, so b) to d) need no own code.
        for key in ("api_accuracy", "local_accuracy"):
            self.assertIn(BUILDER.STATS[key], BUILDER.C3["question"])
        for key in ("b1_tokens", "b1_cost"):
            self.assertIn(BUILDER.STATS[key], BUILDER.B1["answer"])

    def test_builds_have_stable_ids_and_exactly_forty_points(self):
        """Check deterministic cells and the unchanged grading structure."""
        for solution in (False, True):
            first = BUILDER.build(solution)
            second = BUILDER.build(solution)
            self.assertEqual(first, second)
            saved_path = BUILDER.SOLUTION_NB if solution else BUILDER.TASKS_NB
            saved = json.loads(saved_path.read_text())
            expected_cells = [
                (c["id"], c["source"], c["metadata"]) for c in first["cells"]
            ]
            saved_cells = [
                (c["id"], "".join(c["source"]), c["metadata"])
                for c in saved["cells"]
            ]
            self.assertEqual(expected_cells, saved_cells)
            ids = [cell["id"] for cell in first["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            questions = [
                cell for cell in first["cells"] if "points" in cell["metadata"]
            ]
            self.assertEqual(len(questions), 8)
            self.assertEqual(
                sum(c["metadata"]["points"] for c in questions), 40
            )
        self.assertEqual(
            BUILDER.build_concept_checks(), BUILDER.build_concept_checks()
        )
        self.assertNotIn("datetime", BUILDER.FOOTER_CODE)
        for path in EXAM.glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertTrue(
                        all(
                            len(line) <= 80
                            for line in "".join(cell["source"]).splitlines()
                        )
                    )

    def test_kappa_example_uses_feasible_counts(self):
        """Recompute the new example from a concrete annotation table."""
        counts = [[3, 3, 0], [0, 3, 0], [0, 0, 3]]
        total = sum(map(sum, counts))
        observed = sum(counts[i][i] for i in range(3)) / total
        expected = sum(
            sum(counts[i]) * sum(row[i] for row in counts) / total**2
            for i in range(3)
        )
        kappa = (observed - expected) / (1 - expected)
        answer = next(
            answer
            for title, _, answer in BUILDER.CONCEPT_CHECKS
            if title == "Gold Labels und Cohen’s Kappa"
        )
        self.assertIn(f"{kappa:.2f}", answer)
        self.assertEqual(total, 12)


if __name__ == "__main__":
    unittest.main()
