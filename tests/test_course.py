"""Offline regression checks: no real API client or network requests."""

# Exercise repository-owned cells and shared synthetic response fixtures.
# pylint: disable=exec-used,duplicate-code

import concurrent.futures
import contextlib
import io
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from openai.types.decision import Decision
from openai.types.responses import Response

from llm_course import client, privacy, wrap_text
from llm_course.evaluation import parse_label, response_label


def response(
    text='{"sentiment":"negative"}', status="completed", refusal=False
):
    """Construct a synthetic API response for offline checks."""
    return Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 1,
            "model": "gpt-6-luna",
            "object": "response",
            "status": status,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": (
                        [{"type": "refusal", "refusal": "No"}]
                        if refusal
                        else [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ]
                    ),
                }
            ],
        }
    )


class PrivacyTests(unittest.TestCase):
    """Check redaction and protection against unsupported inputs."""

    def tearDown(self):
        """Release fixtures after each test."""
        privacy.anonymize.cache_clear()

    def test_overlapping_spans_leave_no_identifier_fragments(self):
        """Check overlapping spans leave no identifier fragments."""
        detector = SimpleNamespace(
            analyze=lambda **kw: [
                SimpleNamespace(start=0, end=4),
                SimpleNamespace(start=0, end=10),
            ]
        )
        with patch.object(privacy, "_analyzer", return_value=detector):
            self.assertEqual(
                privacy.anonymize("John Smith was helpful."),
                "<REDACTED> was helpful.",
            )

    def test_history_examples_and_decision_instructions_are_redacted(self):
        """Check history examples and decision instructions are redacted."""

        def redact(s):
            return s.replace("john@example.org", "<REDACTED>")

        with patch.object(privacy, "anonymize", side_effect=redact):
            kwargs = privacy.sanitize_request(
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "john@example.org",
                                }
                            ],
                        }
                    ],
                    "instructions": "Example: john@example.org",
                    "questions": [
                        {"instructions": "john@example.org", "type": "choice"}
                    ],
                }
            )
        self.assertNotIn("john@example.org", json.dumps(kwargs))

    def test_non_text_references_and_uploads_are_blocked(self):
        """Check non text references and uploads are blocked."""
        with self.assertRaises(ValueError):
            privacy.sanitize_request(
                {"input": [{"type": "input_file", "file_id": "file_raw"}]}
            )
        with self.assertRaises(ValueError):
            privacy.sanitize_request(
                {"input": "safe", "previous_response_id": "resp_raw"}
            )

    def test_detector_failure_prevents_client_access(self):
        """Check detector failure prevents client access."""
        privacy.anonymize.cache_clear()
        with (
            patch.object(
                privacy, "_analyzer", side_effect=RuntimeError("missing model")
            ),
            patch.object(client, "get_client") as get,
        ):
            with self.assertRaises(RuntimeError):
                client.ask(model="gpt-6-luna", input="John Smith")
            get.assert_not_called()


class CacheTests(unittest.TestCase):
    """Check caching and private request preparation."""

    def setUp(self):
        """Prepare isolated fixtures for each test."""
        # The fixture must survive until tearDown; addCleanup releases it.
        # pylint: disable-next=consider-using-with
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sent = []

        def create(**kwargs):
            self.sent.append(kwargs)
            time.sleep(0.01)
            return response()

        self.fake = SimpleNamespace(
            responses=SimpleNamespace(create=create),
            decisions=SimpleNamespace(create=create),
        )
        self.patches = [
            patch.object(client, "CACHE_DIR", Path(self.tmp.name)),
            patch.object(client, "get_client", return_value=self.fake),
            patch.object(
                privacy,
                "anonymize",
                side_effect=lambda s: s.replace("John Smith", "<REDACTED>"),
            ),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Release fixtures after each test."""
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_duplicate_requests_are_sent_once_and_raw_text_never_reaches_client(
        self,
    ):
        (
            "Check duplicate requests are sent once "
            "and raw text never reaches "
            "client."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: client.ask(
                        model="gpt-6-luna", input="John Smith complains."
                    ),
                    range(8),
                )
            )
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["input"], "<REDACTED> complains.")
        self.assertFalse(self.sent[0]["store"])
        self.assertEqual(len(results), 8)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.json"))), 1)
        self.assertEqual(list(Path(self.tmp.name).glob("*.tmp")), [])
        # ask_many cache hits must work without even constructing an API client.
        with patch.object(
            client,
            "get_client",
            side_effect=AssertionError("cache should need no key"),
        ):
            self.assertEqual(
                len(
                    client.ask_many(
                        ["John Smith complains."] * 2, model="gpt-6-luna"
                    )
                ),
                2,
            )

    def test_corrupt_cache_is_replaced_and_new_run_sends_new_request(self):
        """Check corrupt cache is replaced and new run sends new request."""
        client.ask(model="gpt-6-luna", input="safe")
        path = next(Path(self.tmp.name).glob("*.json"))
        path.write_text("{broken")
        with self.assertWarns(UserWarning):
            client.ask(model="gpt-6-luna", input="safe")
        client.ask(model="gpt-6-luna", input="safe", run=1)
        self.assertEqual(len(self.sent), 3)

    def test_incomplete_and_refused_responses_are_not_cached(self):
        """Check incomplete and refused responses are not cached."""
        for result in [response(status="incomplete"), response(refusal=True)]:
            self.fake.responses.create = lambda result=result, **kw: result
            client.ask(model="gpt-6-luna", input="safe")
            self.assertEqual(list(Path(self.tmp.name).glob("*.json")), [])

    def test_decisions_are_redacted_cached_and_refusals_are_not_cached(self):
        """Check decisions are redacted cached and refusals are not cached."""
        choice = Decision.model_validate(
            {
                "id": "dec_test",
                "object": "decision",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "created_at": 1,
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "type": "choice",
                        "name": "sentiment",
                        "choice": "negative",
                        "confidence": 0.9,
                        "probabilities": [
                            {"value": "negative", "probability": 0.9}
                        ],
                    }
                ],
            }
        )
        calls = []

        def create(**kw):
            calls.append(kw)
            return choice

        self.fake.decisions.create = create
        request = {
            "model": "gpt-6-luna",
            "input": "John Smith",
            "questions": [
                {
                    "type": "choice",
                    "name": "sentiment",
                    "instructions": "John Smith",
                    "choices": [{"value": "negative"}],
                }
            ],
        }
        client.ask_decision(**request)
        client.ask_decision(**request)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("John Smith", json.dumps(calls))
        refused = Decision.model_validate(
            {
                "id": "dec_no",
                "object": "decision",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "created_at": 1,
                "model": "gpt-6-luna",
                "answers": [{"type": "refusal", "name": "sentiment"}],
            }
        )
        self.fake.decisions.create = lambda **kw: refused
        client.ask_decision(**request, run=1)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.json"))), 1)


class OutputTests(unittest.TestCase):
    """Check extraction of model labels and explicit failures."""

    def test_long_output_wraps_without_losing_paragraphs_or_text(self):
        """Wrap reviews, long tokens and existing multi-paragraph answers."""
        text = "First paragraph.\n\n" + "Ä review " * 45 + "\n" + "x" * 241
        wrapped = wrap_text(text)
        self.assertTrue(all(len(line) <= 120 for line in wrapped.splitlines()))
        self.assertTrue(wrapped.startswith("First paragraph.\n\n"))
        self.assertEqual("".join(wrapped.split()), "".join(text.split()))
        self.assertEqual(wrap_text("Short answer.\n"), "Short answer.\n")
        self.assertEqual(wrap_text(""), "")
        self.assertEqual(wrap_text("  Short answer.  "), "  Short answer.  ")

    def test_final_label_parser_does_not_read_explanation_or_substrings(self):
        """Check final label parser does not read explanation or substrings."""
        for text, label in [
            ("Not positive. Overall: negative.", "negative"),
            ("<think>positive or negative?</think>Overall: mixed", "mixed"),
            ("positive.", "positive"),
            ("not positive", "unparsed"),
            ("positively", "unparsed"),
            ("", "unparsed"),
        ]:
            self.assertEqual(parse_label(text), label)

    def test_structured_failures_are_explicit(self):
        """Check structured failures are explicit."""
        for result, label in [
            (response(), "negative"),
            (response(refusal=True), "refused"),
            (response(status="incomplete"), "incomplete"),
            (response("bad json"), "unparsed"),
            (response('{"sentiment": []}'), "unparsed"),
        ]:
            self.assertEqual(response_label(result), label)


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
                "tqdm": lambda values: values,
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
            ("04_Mini_RAG/mini_rag.ipynb", "sample_path = ", "notebook 01"),
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

    def test_notebook_04_retrieve_combines_keyword_and_meaning_ranks(self):
        """Check notebook 04 retrieve combines keyword and meaning ranks."""

        source = code_cell(
            "04_Mini_RAG/mini_rag.ipynb", "# Hybrid search for one question"
        )

        ns = {
            "pd": pd,
            "re": re,
            "df": pd.DataFrame(
                {
                    "id": ["R001", "R002", "R003"],
                    "stars": [1, 2, 3],
                    "text": ["a", "b", "c"],
                }
            ),
            "bm25": SimpleNamespace(
                get_scores=lambda query: [3.0, 2.0, 1.0]
            ),  # ranks 1, 2, 3
            "embeddings": np.array([[0.1], [0.9], [0.5]]),  # ranks 3, 1, 2
            "model": SimpleNamespace(encode=lambda text, **kw: np.array([1.0])),
        }
        exec(source, ns)
        self.assertEqual(ns["hits"]["id"].head(2).tolist(), ["R002", "R001"])

    def test_notebook_05_cost_uses_input_and_output_tokens_without_cache(self):
        """Check notebook 05 cost uses input and output tokens without cache."""
        cell = code_cell(
            "05_Local_Model/local_model.ipynb", "# API model: the same reviews"
        )
        calls = []

        def fake_ask(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="negative",
                usage=SimpleNamespace(input_tokens=100, output_tokens=10),
            )

        ns = {
            "test": pd.DataFrame({"text": ["safe"] * 20}),
            "ask": fake_ask,
            "time": time,
            "MODEL": "fake",
            "PROMPT": "label: ",
            "parse_label": parse_label,
            "PRICE_INPUT_PER_M": 0.10,
            "PRICE_OUTPUT_PER_M": 0.50,
            "print": lambda *a, **k: None,
        }
        exec(cell, ns)
        self.assertAlmostEqual(
            ns["api_cost"], (2000 * 0.10 + 200 * 0.50) / 1_000_000
        )
        self.assertTrue(all(call["use_cache"] is False for call in calls))
        self.assertEqual(ns["test"]["api"].tolist(), ["negative"] * 20)


if __name__ == "__main__":
    unittest.main()
