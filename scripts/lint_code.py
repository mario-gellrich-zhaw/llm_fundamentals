"""Check Python files and exported notebook code with pylint and 80 columns.

Run from the repository root: python scripts/lint_code.py
Notebook exports are temporary and never include outputs or review data.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_notebook(path, destination):
    """Combine notebook code cells so pylint sees their shared namespace."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # These checks conflict with intentional notebook conventions: shared
    # variables, displayed expressions, and imports in later teaching steps.
    lines = [
        '"""Exported notebook code for pylint."""',
        "# pylint: disable=invalid-name,pointless-statement",
        "# pylint: disable=expression-not-assigned,fixme",
        "# pylint: disable=wrong-import-position,wrong-import-order",
        "# pylint: disable=reimported,duplicate-code",
        "",
    ]
    placeholders = set()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines.append(f"# Notebook cell {index}")
        source = "".join(cell["source"])
        pending_members = set()
        if not any(
            line.lstrip().startswith(("!", "%")) for line in source.splitlines()
        ):
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is Ellipsis
                ):
                    placeholders.update(
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    )
            pending_members = {
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in placeholders
            }
        for number, line in enumerate(source.splitlines(), 1):
            # A learner's Ellipsis has no runtime members until completed.
            if number in pending_members:
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(indent + "# pylint: disable-next=no-member")
            if path.stem.endswith("_Aufgaben") and line.startswith(
                "from sklearn.metrics import"
            ):
                # accuracy_score is used by the intended C3 completion.
                lines.append("# pylint: disable-next=unused-import")
            if line.lstrip().startswith(("!", "%")):
                # Shell/model setup is intentionally not Python code.
                lines.append("# " + line)
                for name in re.findall(r"\{([A-Za-z_]\w*)\}", line):
                    lines.append(f"_ = {name}")
            else:
                lines.append(line)
        lines.extend(["", ""])
    destination.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    """Run pylint against the modules and temporary notebook exports."""
    modules = [
        str(path)
        for folder in ("src", "scripts", "tests")
        for path in sorted((ROOT / folder).rglob("*.py"))
    ]
    with tempfile.TemporaryDirectory(prefix="course_lint_") as directory:
        exports = []
        for index, path in enumerate(sorted(ROOT.glob("0*/*.ipynb"))):
            target = Path(directory) / f"notebook_{index:02d}.py"
            export_notebook(path, target)
            exports.append(str(target))
        command = [
            sys.executable,
            "-m",
            "pylint",
            "--rcfile",
            str(ROOT / "pyproject.toml"),
            *modules,
            *exports,
        ]
        result = subprocess.run(command, cwd=ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
