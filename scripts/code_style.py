"""Format course Python without changing string values or notebook prompts."""

import ast
import io
import textwrap
import tokenize

import black

LINE_LENGTH = 80


def _literal_parts(value):
    """Split a string into short adjacent literals, keeping every character."""
    parts = []
    for line in value.splitlines(keepends=True):
        while line:
            size = min(40, len(line))
            space = line.rfind(" ", 0, size)
            if space >= size // 2:
                size = space + 1
            while len(repr(line[:size])) > 52 and size > 1:
                size -= 1
            parts.append(repr(line[:size]))
            line = line[size:]
    return parts or [repr("")]


def _short_string(source):
    """Render a constant or formatted string as adjacent short literals."""
    node = ast.parse(source, mode="eval").body
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parts = _literal_parts(node.value)
    elif isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            if isinstance(item, ast.Constant):
                parts.extend(_literal_parts(item.value))
            else:
                parts.append(ast.unparse(ast.JoinedStr(values=[item])))
    else:
        return source
    return "(\n" + "\n".join(parts) + "\n)"


def format_code(source):
    """Apply Black and wrap long comments and strings at 80 characters."""
    original = source
    if not source.strip() or any(
        line.lstrip().startswith(("!", "%")) for line in source.splitlines()
    ):
        return source
    mode = black.Mode(line_length=LINE_LENGTH)
    source = black.format_str(source, mode=mode)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type != tokenize.STRING:
            index += 1
            continue
        group = [token]
        last = index
        cursor = index + 1
        while cursor < len(tokens):
            following = tokens[cursor]
            if following.type == tokenize.STRING:
                group.append(following)
                last = cursor
            elif following.type != tokenize.NL:
                break
            cursor += 1
        index = last + 1
        start_line, start_column = group[0].start
        end_line, end_column = group[-1].end
        if not any(
            len(line.rstrip("\n")) > LINE_LENGTH
            for line in lines[start_line - 1 : end_line]
        ):
            continue
        combined = "(\n" + "\n".join(item.string for item in group) + "\n)"
        replacement = _short_string(combined)
        replacements.append(
            (
                offsets[start_line - 1] + start_column,
                offsets[end_line - 1] + end_column,
                replacement,
            )
        )
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    source = black.format_str(source, mode=mode)
    wrapped = []
    for line in source.splitlines():
        if len(line) > LINE_LENGTH and line.lstrip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip())]
            prefix = indent + "# "
            wrapped.extend(
                textwrap.wrap(
                    line.lstrip()[1:].strip(),
                    width=LINE_LENGTH,
                    initial_indent=prefix,
                    subsequent_indent=prefix,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
            wrapped.append(line)
    source = "\n".join(wrapped) + "\n"
    black.assert_equivalent(original, source)
    return source
