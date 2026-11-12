"""Check the photo notebook offline, including its request payloads."""

# Execute repository-owned cells with temporary photos and fake clients.
# pylint: disable=exec-used

import base64
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "07_Multimodal_Prompting/multimodal_prompting.ipynb"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


class MultimodalTests(unittest.TestCase):
    """Verify missing-photo guidance and the three-question image workflow."""

    def test_photo_workflow_from_both_working_directories(self):
        """Send the image bytes with three independent questions."""
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        cells = [c.source for c in notebook.cells if c.cell_type == "code"]
        for from_root in (True, False):
            with self.subTest(from_root=from_root):
                create = Mock(
                    side_effect=[
                        SimpleNamespace(
                            output_text="A dog yawning.",
                            usage=SimpleNamespace(
                                input_tokens=2000,
                                input_tokens_details=SimpleNamespace(
                                    cached_tokens=500
                                ),
                                output_tokens=150,
                                output_tokens_details=SimpleNamespace(
                                    reasoning_tokens=40
                                ),
                            ),
                        ),
                        SimpleNamespace(
                            output_text="One dog.",
                            usage=SimpleNamespace(
                                input_tokens=800,
                                input_tokens_details=SimpleNamespace(
                                    cached_tokens=0
                                ),
                                output_tokens=20,
                            ),
                        ),
                        SimpleNamespace(
                            output_text="A dog in the foreground.",
                            usage=None,
                        ),
                    ]
                )
                fake_client = SimpleNamespace(
                    responses=SimpleNamespace(create=create)
                )
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    folder = root / "07_Multimodal_Prompting"
                    images = folder / "images"
                    images.mkdir(parents=True)
                    # The alphabetical choice accepts an uppercase extension.
                    (images / "a.PNG").write_bytes(PNG_BYTES)
                    (images / "b.jpg").write_bytes(b"not selected")
                    (images / "README.md").write_text("not an image")
                    previous = Path.cwd()
                    try:
                        os.chdir(root if from_root else folder)
                        with (
                            patch("openai.OpenAI", return_value=fake_client),
                            patch("IPython.display.display"),
                            patch("dotenv.load_dotenv"),
                            contextlib.redirect_stdout(io.StringIO()) as output,
                        ):
                            namespace = {}
                            for source in cells:
                                exec(
                                    source.replace(
                                        'IMAGE_NAME = "photo_02.jpg"',
                                        'IMAGE_NAME = "a.PNG"',
                                    ),
                                    namespace,
                                )
                    finally:
                        os.chdir(previous)
                calls = [call.kwargs for call in create.call_args_list]
                self.assertEqual(len(calls), 3)
                for call in calls:
                    self.assertIs(call["store"], False)
                    self.assertEqual(call["model"], "gpt-6-luna")
                    self.assertEqual(call["service_tier"], "default")
                    content = call["input"][0]["content"]
                    self.assertEqual(content[0]["type"], "input_text")
                    image = content[1]
                    self.assertEqual(image["type"], "input_image")
                    prefix, encoded = image["image_url"].split(",", 1)
                    self.assertEqual(prefix, "data:image/png;base64")
                    self.assertEqual(base64.b64decode(encoded), PNG_BYTES)
                prompts = [
                    call["input"][0]["content"][0]["text"] for call in calls
                ]
                self.assertEqual(
                    prompts,
                    [
                        "Show what you see!",
                        "How many dogs are visible in this image?",
                        (
                            "What is in the foreground, "
                            "and what is in the background?"
                        ),
                    ],
                )
                self.assertTrue(all("text" not in call for call in calls))
                self.assertEqual(namespace["count_answer"], "One dog.")
                # Reasoning is already included in total output tokens.
                self.assertIn("USD 0.00023000", output.getvalue())
                self.assertIn("USD 0.00009000", output.getvalue())
                self.assertIn("Cost estimate unavailable", output.getvalue())

    def test_missing_photo_stops_before_api_client_creation(self):
        """An empty image folder must give actionable guidance without a key."""
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        cells = [c.source for c in notebook.cells if c.cell_type == "code"]
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "07_Multimodal_Prompting/images"
            images.mkdir(parents=True)
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with (
                    patch("openai.OpenAI") as client,
                    patch("dotenv.load_dotenv"),
                    self.assertRaisesRegex(FileNotFoundError, "Add a JPG"),
                ):
                    namespace = {}
                    for source in cells:
                        exec(source, namespace)
                client.assert_not_called()
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
