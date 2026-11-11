(
    (
        "Exercise the five practical notebooks "
        "locally with fake API/Ollama "
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
        "Ollama "
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
import io
import json
import os
import shutil
import socket
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import pandas as pd
from openai.types.responses import Response

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    """Character tokens: deterministic, no model files or network."""

    def encode(self, text, add_special_tokens=False):
        """Return deterministic character tokens."""
        return list(text) + ([0, 1] if add_special_tokens else [])

    def decode(self, tokens):
        """Join character tokens back into text."""
        return "".join(tokens)

    def num_special_tokens_to_add(self, _pair=False):
        """Report the two simulated special tokens."""
        return 2

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def fake_value(schema):
    """Build a small value that fits a JSON schema (for fake extractions)."""
    kind = schema.get("type")
    if isinstance(kind, list):
        return None if "null" in kind else fake_value({"type": kind[0]})
    if kind == "object":
        return {
            key: fake_value(value)
            for key, value in schema["properties"].items()
        }
    if kind == "array":
        return [fake_value(schema["items"])]
    if kind == "string":
        return schema["enum"][-1] if "enum" in schema else "fries"
    return 1


def response(**kwargs):
    """Construct a synthetic API response for offline checks."""
    answer_format = kwargs.get("text", {}).get("format", {})
    schema_name = answer_format.get("name")
    if schema_name == "extraction":
        text = json.dumps(fake_value(answer_format["schema"]))
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


def main():
    """Run the command-line workflow."""
    # One fake client answers both the API calls and the local Ollama calls.
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=response),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Negative.")
                        )
                    ]
                )
            )
        ),
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
        stack.enter_context(patch("openai.OpenAI", return_value=fake_client))
        # The notebooks read ../data, relative to their own folder.
        previous_directory = os.getcwd()
        stack.callback(os.chdir, previous_directory)
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
            working_directory = Path(tmp) / folder.name
            working_directory.mkdir()
            os.chdir(working_directory)
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
                test = namespace["test"]
                assert test.groupby("chain").size().to_dict() == {
                    "IKEA": 10,
                    "McDonald's": 10,
                }
                assert test.groupby("stars").size().eq(4).all()
                assert test.groupby(["chain", "stars"]).size().eq(2).all()
                # All 40 prepared labels are saved for notebook 05.
                expected = pd.read_csv(data / "silver_reference.csv")
                saved = pd.read_csv(data / "processed/silver_labels.csv")
                pd.testing.assert_frame_equal(
                    saved[["id", "silver"]]
                    .sort_values("id")
                    .reset_index(drop=True),
                    expected.sort_values("id").reset_index(drop=True),
                )
                assert set(test["id"]) <= set(expected["id"])
                assert namespace["runs"].shape == (3, 3)
            if folder.name.startswith("04_"):
                chains = namespace["reviews"].groupby("chain").size()
                assert chains.to_dict() == {"IKEA": 10, "McDonald's": 10}
                assert len(namespace["visits"]) == 20
                assert len(namespace["aspects"]) == 20
                assert "wait_minutes" in namespace["visits"].columns
                assert namespace["aspects"]["quote_found"].dtype == bool
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
