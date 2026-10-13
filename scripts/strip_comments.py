from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

TARGETS = ["backend/app", "backend/run.py", "scripts", "tests"]


def docstring_lines(source: str) -> set[int]:
    tree = ast.parse(source)
    marked: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            start = first.lineno
            end = getattr(first, "end_lineno", start)
            for line in range(start, end + 1):
                marked.add(line)
    return marked


def strip_python(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    try:
        marked = docstring_lines(original)
    except SyntaxError:
        return False
    out_lines = original.splitlines(keepends=True)
    kept: list[str] = []
    for number, line in enumerate(out_lines, start=1):
        if number in marked:
            if line.strip() == "" or line.lstrip().startswith(("'''", '"""')) or line.rstrip().endswith(("'''", '"""')):
                continue
            continue
        kept.append(line)
    intermediate = "".join(kept)

    result: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(intermediate).readline))
    except tokenize.TokenError:
        return False
    lines = intermediate.splitlines(keepends=True)
    removals: list[tuple[int, int, int, int]] = []
    for token in tokens:
        if token.type == tokenize.COMMENT:
            start_row, start_col = token.start
            end_row, end_col = token.end
            text = lines[start_row - 1]
            if start_col == 0 and text.lstrip().startswith("#!"):
                continue
            if start_col == 0 and "coding" in text[:40] and text.lstrip().startswith("#"):
                continue
            removals.append((start_row, start_col, end_row, end_col))
    for start_row, start_col, end_row, end_col in reversed(removals):
        if start_row == end_row:
            line = lines[start_row - 1]
            lines[start_row - 1] = line[:start_col] + line[end_col:]
        else:
            lines[start_row - 1] = lines[start_row - 1][:start_col] + "\n"
            for row in range(start_row, end_row - 1):
                lines[row] = ""
            lines[end_row - 1] = lines[end_row - 1][end_col:]
    for line in lines:
        stripped = line.rstrip()
        result.append(stripped + "\n" if stripped else "")
    text = "".join(result)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    try:
        ast.parse(text)
    except SyntaxError:
        print(f"refused (would break): {path}")
        return False
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    changed = 0
    for target in TARGETS:
        path = root / target
        paths = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for candidate in paths:
            if "__pycache__" in candidate.parts:
                continue
            if candidate.name == "strip_comments.py":
                continue
            if strip_python(candidate):
                changed += 1
                print(f"stripped {candidate.relative_to(root)}")
    print(f"{changed} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
