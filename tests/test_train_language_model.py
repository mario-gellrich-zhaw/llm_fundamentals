"""Offline checks for the optional, locally trained language model."""

import ast
import contextlib
import copy
import io
import json
import math
import re
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "06_Train_Language_Model/train_language_model.ipynb"


class LanguageModelTests(unittest.TestCase):
    """Check causal attention and actual learning without downloading a book."""

    @classmethod
    def setUpClass(cls):
        # Keep torch imports close to the model checks.
        import torch  # pylint: disable=import-outside-toplevel
        from torch import nn  # pylint: disable=import-outside-toplevel

        cls.torch = torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "class TinyLanguageModel" in "".join(cell["source"])
        )
        # Exercise the actual teaching implementation, without demo setup.
        namespace = {"torch": torch, "nn": nn}
        exec(  # pylint: disable=exec-used
            source.split("\ntorch.manual_seed(SEED)")[0], namespace
        )
        cls.model_class = namespace["TinyLanguageModel"]

    def test_extracts_complete_tales_without_source_notices(self):
        """Keep full tales and sections, excluding contents and notices."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["id"] == "train-lm-05"
        )
        functions = ast.Module(
            body=[
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
            ],
            type_ignores=[],
        )
        namespace = {"re": re}
        exec(  # pylint: disable=exec-used
            compile(functions, str(NOTEBOOK), "exec"), namespace
        )
        extract = namespace["extract_tales"]
        document = (
            "Source notice\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "PREFACE\n    THE GOLDEN BIRD\n    HANSEL AND GRETEL\n"
            "THE GOLDEN BIRD\nFirst tale.\n"
            "HANSEL AND GRETEL\nSecond tale.\n"
            "THE WEDDING OF MRS FOX\nFIRST STORY\nPart one.\n"
            "SECOND STORY\nPart two.\n"
            "LITTLE RED-CAP [LITTLE RED RIDING HOOD]\nLast tale.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicence"
        )
        tales = extract(document)
        self.assertEqual(len(tales), 4)
        self.assertEqual(tales["THE GOLDEN BIRD"], "First tale.")
        self.assertEqual(tales["HANSEL AND GRETEL"], "Second tale.")
        self.assertIn("SECOND STORY", tales["THE WEDDING OF MRS FOX"])
        self.assertEqual(
            tales["LITTLE RED-CAP [LITTLE RED RIDING HOOD]"], "Last tale."
        )
        with self.assertRaises(ValueError):
            extract(document.replace("THE GOLDEN BIRD", "WRONG TITLE"))

    def test_future_tokens_cannot_change_prefix_predictions(self):
        """Prevent attention from leaking target tokens into predictions."""
        torch = self.torch
        torch.manual_seed(1)
        model = self.model_class(5, context=8, width=16, heads=2, layers=2)
        model.eval()
        first = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
        changed = torch.tensor([[0, 1, 2, 4, 0, 1, 2, 3]])
        with torch.no_grad():
            torch.testing.assert_close(
                model(first)[:, :3], model(changed)[:, :3]
            )

    def test_compact_token_mapping_and_unicode_decoding(self):
        """Keep IDs compact and decode split Unicode bytes correctly."""

        # Use byte tokens as a deterministic stand-in; never download in tests.
        class ByteTokenizer:
            """Expose the established tokenizer interface with sparse IDs."""

            @staticmethod
            def encode_ordinary(text):
                """Return sparse IDs, including separate UTF-8 byte pieces."""
                return [byte + 1000 for byte in text.encode("utf-8")]

            @staticmethod
            def decode_single_token_bytes(token_id):
                """Restore one byte without decoding it separately."""
                return bytes([token_id - 1000])

            @staticmethod
            def decode(token_ids):
                """Restore text from the complete byte sequence."""
                return bytes(i - 1000 for i in token_ids).decode("utf-8")

        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["id"] == "train-lm-07"
        )
        namespace = {
            "train_text": "Hansel and Gretel é🙂. " * 50,
            "val_text": "Hansel " + "Z" * 100,
            "torch": self.torch,
            "tiktoken": SimpleNamespace(
                get_encoding=lambda _name: ByteTokenizer()
            ),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)  # pylint: disable=exec-used

        encode, decode = namespace["encode"], namespace["decode"]
        self.assertEqual(decode(encode("Hansel é🙂")), "Hansel é🙂")
        self.assertEqual(encode("Z"), [0])
        self.assertEqual(decode(encode("Z")), "<UNK>")
        self.assertEqual(namespace["vocabulary"][0], None)
        self.assertEqual(
            set(namespace["token_to_model"].values()),
            set(range(1, namespace["VOCAB_SIZE"])),
        )
        self.assertNotIn(0, namespace["train_data"].tolist())
        self.assertIn(0, namespace["val_data"].tolist())

    def test_early_stopping_restores_best_weights(self):
        """Check patience, its reset, and independent best-state copies."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["id"] == "train-lm-15"
        )
        training = source[source.index("# Set up the optimizer") :]
        torch = self.torch
        cases = [
            ([1.0, 0.8, 0.9, 1.0], 3, 1),
            ([1.0, 1.1, 1.2], 2, 0),
            ([1.0, 0.8, 0.9, 0.7, 0.8, 0.9], 5, 3),
        ]
        for values, expected_stop, expected_best in cases:
            with self.subTest(losses=values):
                model = self.model_class(
                    4, context=8, width=16, heads=2, layers=1
                )
                x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
                snapshots = []
                losses = iter(values)

                def estimate_loss(current_model, records, sequence):
                    records.append(copy.deepcopy(current_model.state_dict()))
                    return {"train": 1.0, "validation": next(sequence)}

                namespace = {
                    "torch": torch,
                    "nn": torch.nn,
                    "F": torch.nn.functional,
                    "copy": copy,
                    "time": time,
                    "math": math,
                    "model": model,
                    "SEED": 42,
                    "VOCAB_SIZE": 4,
                    "LEARNING_RATE": 0.01,
                    "STEPS": 10,
                    "EVAL_EVERY": 1,
                    "PATIENCE": 2,
                    "train_data": x,
                    "get_batch": lambda *_args, batch=x: (
                        batch,
                        (batch + 1) % 4,
                    ),
                    "estimate_loss": partial(
                        estimate_loss, records=snapshots, sequence=losses
                    ),
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(training, namespace)  # pylint: disable=exec-used
                self.assertEqual(namespace["trained_steps"], expected_stop)
                self.assertEqual(namespace["best_step"], expected_best)
                self.assertEqual(namespace["best_loss"], min(values))
                self.assertFalse(model.training)
                for name, parameter in model.state_dict().items():
                    torch.testing.assert_close(
                        parameter, snapshots[expected_best][name]
                    )

    def test_model_learns_and_checkpoint_preserves_predictions(self):
        """Learn a repeated pattern and round-trip the learned parameters."""
        torch = self.torch
        torch.manual_seed(2)
        config = {
            "vocab_size": 4,
            "context": 8,
            "width": 16,
            "heads": 2,
            "layers": 1,
        }
        model = self.model_class(**config)
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(4, 1)
        y = (x + 1) % 4
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        def loss():
            logits = model(x)
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, 4), y.reshape(-1)
            )

        model.eval()
        initial = loss().item()
        for _ in range(40):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            current = loss()
            self.assertTrue(torch.isfinite(current))
            current.backward()
            optimizer.step()
        model.eval()
        self.assertLess(loss().item(), initial * 0.3)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            restored = self.model_class(**config)
            restored.load_state_dict(
                torch.load(checkpoint, weights_only=True)["state_dict"]
            )
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x))


if __name__ == "__main__":
    unittest.main()
