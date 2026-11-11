(
    (
        "Exercise the five practical notebooks "
        "locally with fake API/embedding/Ollama "
        "clients.\n"
        "\n"
        "No paid requests, tokenizer/model "
        "downloads or shell setup. Real local "
        "anonymization "
        "runs.\n"
        "Token counting uses character tokens "
        "for wiring checks, not real token/cost "
        "estimates.\n"
        "This verifies notebook wiring, not "
        "hosted-model quality or actual "
        "E5/Ollama "
        "speed.\n"
        "Run after installing requirements and "
        "en_core_web_sm: python "
        "scripts/validate_notebooks.py\n"
    )
)

# Execute local notebook cells with fake services for offline validation.
# pylint: disable=exec-used
# Fake responses deliberately mirror the test fixtures.
# pylint: disable=duplicate-code

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import socket
import sys
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np
from openai.types.decision import Decision
from openai.types.responses import Response

import llm_course
from llm_course import client

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    """Character tokens: deterministic, no model files or network."""

    def encode(self, text, add_special_tokens=False):
        """Return deterministic character tokens or fake embedding vectors."""
        return list(text) + ([0, 1] if add_special_tokens else [])

    def decode(self, tokens):
        """Join character tokens back into text."""
        return "".join(tokens)

    def num_special_tokens_to_add(self, _pair=False):
        """Report the two simulated special tokens."""
        return 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Embeddings:
    """Provide deterministic vectors without downloading a model."""

    def __init__(self, *_args, **_kwargs):
        self.tokenizer = Tokenizer()
        self.max_seq_length = 512

    def encode(self, texts, **_kwargs):
        """Return deterministic character tokens or fake embedding vectors."""
        single = isinstance(
            texts, str
        )  # One text gives one vector, a list gives a matrix.
        vectors = np.array(
            [
                list(hashlib.sha256(t.encode()).digest())
                for t in ([texts] if single else texts)
            ],
            dtype=float,
        )
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors[0] if single else vectors


def response(**kwargs):
    """Construct a synthetic API response for offline checks."""
    schema_name = kwargs.get("text", {}).get("format", {}).get("name")
    if schema_name == "aspects":
        text = (
            '{"sentiment":"negative","aspects":[{"asp'
            'ect":"service","polarity":"negative","ev'
            'idence":""}]}'
        )
    else:
        text = (
            '{"sentiment":"negative"}' if schema_name else "Overall: negative."
        )
    return Response.model_validate(
        {
            "id": "resp_offline",
            "object": "response",
            "created_at": 1,
            "model": "gpt-6-luna",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_offline",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
    )


def decision(**_kwargs):
    """Construct a synthetic decision response for offline checks."""
    return Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "type": "choice",
                    "name": "sentiment",
                    "choice": "negative",
                    "confidence": 0.9,
                    "probabilities": [
                        {"value": "negative", "probability": 0.9},
                        {"value": "positive", "probability": 0.05},
                        {"value": "mixed", "probability": 0.05},
                    ],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
    )


def main():
    """Run the command-line workflow."""
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = Embeddings
    fake_api = SimpleNamespace(
        responses=SimpleNamespace(create=response),
        decisions=SimpleNamespace(create=decision),
    )
    fake_local = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="Overall: negative."
                            )
                        )
                    ]
                )
            )
        )
    )
    archive = ROOT / "data/llm_fundamentals_data.zip"
    if not archive.exists():
        raise FileNotFoundError(
            (
                (
                    "Download the Moodle zip into data/ "
                    "first; no API is used by this "
                    "validator."
                )
            )
        )

    matplotlib.use("Agg")
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        data = Path(tmp) / "data"
        data.mkdir()
        (data / "processed").mkdir()
        shutil.copyfile(archive, data / archive.name)
        stack.enter_context(patch.object(llm_course, "DATA", data))
        stack.enter_context(
            patch.object(client, "CACHE_DIR", Path(tmp) / "cache")
        )
        stack.enter_context(
            patch.object(client, "get_client", return_value=fake_api)
        )
        stack.enter_context(patch("openai.OpenAI", return_value=fake_local))
        stack.enter_context(
            patch.dict(sys.modules, {"sentence_transformers": fake_module})
        )
        stack.enter_context(
            patch("tiktoken.get_encoding", return_value=Tokenizer())
        )
        stack.enter_context(
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError(
                    "Network disabled in offline validation"
                ),
            )
        )
        for folder in sorted(ROOT.glob("0[1-5]_*")):
            path = next(folder.glob("*.ipynb"))
            notebook = json.loads(path.read_text())
            namespace = {"__name__": "__main__"}
            count = 0
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if any(line.startswith("!") for line in source.splitlines()):
                    # Never install or start Ollama in this check.
                    continue
                try:
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        exec(
                            compile(source, f"{path.name}:cell{index}", "exec"),
                            namespace,
                        )
                except Exception as exc:
                    raise RuntimeError(
                        (
                            (
                                f"{path.name}"
                                ", cell "
                                f"{index}"
                                " failed: "
                                f"{type(exc).__name__}"
                            )
                        )
                    ) from exc
                count += 1
            if folder.name.startswith("01_"):
                saved = (data / "processed/reviews_sample.csv").read_text()
                assert (
                    len(namespace["sample"]) == 500
                    and "original" not in saved.splitlines()[0]
                )
                assert namespace["sample"].groupby(
                    "chain"
                ).size().to_dict() == {"IKEA": 250, "McDonald's": 250}
                # Notebook 01 must preserve the source download byte-for-byte.
                with zipfile.ZipFile(archive) as source_zip:
                    for filename in (
                        "ikea_reviews.csv",
                        "mcdonalds_reviews.csv",
                        "fewshot_examples.json",
                    ):
                        assert (
                            data / filename
                        ).read_bytes() == source_zip.read(filename)
            if folder.name.startswith("03_"):
                assert not set(namespace["test"].text) & {
                    ex["text"] for ex in namespace["fewshot"]
                }
                assert (data / "processed/silver_labels.csv").exists()
                assert namespace["test"].groupby("chain").size().to_dict() == {
                    "IKEA": 25,
                    "McDonald's": 25,
                }
            if folder.name.startswith("04_"):
                assert set(namespace["df"]["chain"]) == {"McDonald's"}
            if folder.name.startswith("05_"):
                assert namespace["test"].groupby("chain").size().to_dict() == {
                    "IKEA": 5,
                    "McDonald's": 5,
                }
            print(
                (
                    (
                        "PASS "
                        f"{path.relative_to(ROOT)}"
                        " ("
                        f"{count}"
                        " Python cells; fake "
                        "clients)"
                    )
                ),
                flush=True,
            )
    print(
        "No paid calls, remote inference, downloads or original-data changes."
    )


if __name__ == "__main__":
    main()
