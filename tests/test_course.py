"""Offline regression checks of notebook cells: no API or network requests."""

# Exercise repository-owned notebook cells with small fake inputs.
# pylint: disable=exec-used

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def code_cell(notebook, marker):
    (
        (
            "Source of the first code cell of a "
            "notebook that contains the marker "
            "text."
        )
    )
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    return next(
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and marker in "".join(c["source"])
    )


class NotebookTests(unittest.TestCase):
    (
        (
            "The notebook cells that protect data, "
            "guide students and compute "
            "costs."
        )
    )

    def test_notebook_01_saves_only_anonymized_text(self):
        """Check notebook 01 saves only anonymized text."""
        notebook = "01_Explore_Yelp_Data/explore_yelp_data.ipynb"
        with tempfile.TemporaryDirectory() as tmp:
            sample = pd.DataFrame(
                {"id": ["R001"], "stars": [1], "text": ["John Smith was rude."]}
            )
            ns = {
                "DATA": Path(tmp),
                "os": os,
                "pd": pd,
                "sample": sample,
                "anonymize": lambda text: text.replace(
                    "John Smith", "<REDACTED>"
                ),
            }

            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    code_cell(
                        notebook, "# Keep the original text for the comparison"
                    ),
                    ns,
                )
                exec(
                    code_cell(
                        notebook, "# Save the sample without the original texts"
                    ),
                    ns,
                )
            saved = (Path(tmp) / "processed/reviews_sample.csv").read_text()
            self.assertIn("<REDACTED> was rude.", saved)
            self.assertNotIn("John Smith", saved)
            self.assertNotIn("original", saved)

    def test_api_notebooks_never_fall_back_to_raw_reviews(self):
        """Check api notebooks never fall back to raw reviews."""
        cases = [
            (
                "03_Classify_Reviews/classify_reviews.ipynb",
                "sample_path = ",
                "notebook 01",
            ),
            (
                "04_Extract_Information/extract_information.ipynb",
                "sample_path = ",
                "notebook 01",
            ),
            (
                "05_Local_Model/local_model.ipynb",
                "silver_path = ",
                "notebook 03",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # A raw file must never be used as an automatic fallback.
            (Path(tmp) / "ikea_reviews.csv").write_text(
                "review_id,text\nraw-id,raw personal information\n"
            )
            for notebook, marker, _hint in cases:
                with (
                    self.subTest(notebook=notebook),
                    self.assertRaises(FileNotFoundError),
                ):
                    exec(
                        code_cell(notebook, marker),
                        {"DATA": Path(tmp), "pd": pd},
                    )

    def test_notebook_04_builds_both_tables_and_checks_quotes(self):
        """Check notebook 04 keeps known columns and tests every quote."""
        source = code_cell(
            "04_Extract_Information/extract_information.ipynb",
            "# Extract the fields of every review",
        )

        def fake_extract(text, _schema):
            return {
                "chain": "IKEA",
                "places": ["Tampa"],
                "visit_type": "store",
                "items": [],
                "wait_minutes": None,
                "aspects": [
                    {
                        "aspect": "staff",
                        "polarity": "negative",
                        "evidence": "rude staff",
                    },
                    {
                        "aspect": "price",
                        "polarity": "positive",
                        "evidence": "words that are not in " + text,
                    },
                ],
            }

        ns = {
            "pd": pd,
            "extract": fake_extract,
            "full_schema": {},
            "reviews": pd.DataFrame(
                {
                    "id": ["R001", "R002"],
                    "chain": ["IKEA", "McDonald's"],
                    "store": ["IKEA", "McDonald's"],
                    "city": ["Tampa", "Reno"],
                    "stars": [1, 5],
                    "text": ["Very rude staff today.", "Fast and friendly."],
                }
            ),
            "print": lambda *a, **k: None,
        }
        exec(source, ns)
        self.assertEqual(ns["visits"]["city"].tolist(), ["Tampa", "Reno"])
        self.assertEqual(len(ns["aspects"]), 4)
        self.assertEqual(
            ns["aspects"]["quote_found"].tolist(), [True, False, False, False]
        )

    def test_notebook_05_cost_uses_input_and_output_tokens(self):
        """Check notebook 05 adds up both token counts for every request."""
        cell = code_cell(
            "05_Local_Model/local_model.ipynb", "# API model: the same reviews"
        )
        calls = []

        def fake_create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=" Negative.\n",
                usage=SimpleNamespace(input_tokens=100, output_tokens=10),
            )

        ns = {
            "test": pd.DataFrame({"text": ["safe"] * 20}),
            "client": SimpleNamespace(
                responses=SimpleNamespace(create=fake_create)
            ),
            "time": time,
            "MODEL": "fake",
            "PROMPT": "label: ",
            "PRICE_INPUT_PER_M": 0.10,
            "PRICE_OUTPUT_PER_M": 0.50,
            "print": lambda *a, **k: None,
        }
        exec(cell, ns)
        self.assertAlmostEqual(
            ns["api_cost"], (2000 * 0.10 + 200 * 0.50) / 1_000_000
        )
        self.assertEqual(len(calls), 20)
        self.assertTrue(all(call["store"] is False for call in calls))
        self.assertEqual(ns["test"]["api"].tolist(), ["negative"] * 20)


if __name__ == "__main__":
    unittest.main()
