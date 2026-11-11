"""Readable plain-text output for the course notebooks."""

from textwrap import fill


def wrap_text(text, width=120):
    """Wrap long lines while preserving existing line and paragraph breaks."""
    return "\n".join(
        (
            line
            if len(line) <= width
            else fill(
                line, width=width, replace_whitespace=False, expand_tabs=False
            )
        )
        for line in str(text).split("\n")
    )
