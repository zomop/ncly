#!/usr/bin/env python3
"""Lint the Markdown for constructs GitHub cannot render.

Two things on GitHub's side fail silently and turn a page into an error box:

* **Math.** GitHub renders ``$…$`` with KaTeX, which rejects some LaTeX macros.
  ``\\Bigl( … \\Bigr)`` fails as "Extra open brace or missing close brace" because
  the opening delimiter is never recognised, so the closing one has no partner.
  ``\\operatorname`` fails as "The following macros are not allowed". Neither
  shows up in a local preview that uses a different renderer, which is why this
  is checked in CI rather than by eye.
* **Mermaid.** ``graph``, ``end``, ``click`` and friends are diagram keywords. A
  node id that collides with one ends the diagram and GitHub reports a parse
  error on that line.

This is a lint, not a TeX engine: it checks the failure modes observed here —
banned macros, unbalanced ``{}``, unmatched ``\\left``/``\\right``,
``\\begin``/``\\end``, and reserved mermaid ids. Rules of thumb: new macro, add
it to ``BANNED_LATEX`` or replace it; new diagram, keep node ids off
``MERMAID_RESERVED``.

Exit status is 1 if any file fails, so it can gate a workflow.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Macros GitHub's math renderer rejects. Each entry is (pattern, why, fix).
BANNED_LATEX = [
    (r"\\operatorname", "`\\operatorname` is not allowed by GitHub", "use `\\mathrm{…}`"),
    (r"\\Bigl", "`\\Bigl` is not allowed by GitHub", "use `\\left(`"),
    (r"\\Bigr", "`\\Bigr` is not allowed by GitHub", "use `\\right)`"),
    (r"\\bigl", "`\\bigl` is not allowed by GitHub", "use `\\left(`"),
    (r"\\bigr", "`\\bigr` is not allowed by GitHub", "use `\\right)`"),
    (r"\\Biggl", "`\\Biggl` is not allowed by GitHub", "use `\\left(`"),
    (r"\\Biggr", "`\\Biggr` is not allowed by GitHub", "use `\\right)`"),
    (r"\\substack", "`\\substack` is not allowed by GitHub", "use `\\begin{aligned}`"),
    (r"\\bm(?![a-zA-Z])", "`\\bm` is not allowed by GitHub", "use `\\boldsymbol{…}`"),
]

# Mermaid keywords that cannot be used as a bare node id.
MERMAID_RESERVED = {
    "graph", "flowchart", "subgraph", "end", "click", "style", "classdef",
    "class", "linkstyle", "direction", "sequenceDiagram", "classDiagram",
    "stateDiagram", "erDiagram", "gantt", "pie", "journey", "gitGraph",
}


def strip_code_spans(text: str) -> str:
    """Blank out fenced blocks and inline code so checks ignore examples."""
    text = re.sub(r"```.*?```", lambda m: " " * len(m.group(0)), text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), text)
    return text


def math_spans(text: str) -> list[tuple[int, str]]:
    """Return (line number, tex) for every display and inline formula."""
    spans: list[tuple[int, str]] = []
    lines = text.split("\n")
    in_display = False
    start = 0
    buf: list[str] = []
    for index, line in enumerate(lines, start=1):
        if line.strip() == "$$":
            if in_display:
                spans.append((start, "\n".join(buf)))
                buf, in_display = [], False
            else:
                in_display, start = True, index
            continue
        if in_display:
            buf.append(line)
            continue
        for match in re.finditer(r"(?<!\\)\$(?!\s)((?:[^$\\]|\\.)+?)(?<!\\)\$", line):
            spans.append((index, match.group(1)))
    if in_display:
        spans.append((start, "\n".join(buf) + "\n(UNCLOSED: no closing $$)"))
    return spans


def check_delimiters(tex: str) -> list[str]:
    """Unbalanced ``{}``, ``\\left``/``\\right`` and ``\\begin``/``\\end``."""
    problems: list[str] = []
    # Braces, ignoring \{ and \}.
    depth = 0
    for match in re.finditer(r"\\.|[{}]", tex):
        token = match.group(0)
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                problems.append("unbalanced `}` (more closers than openers)")
                depth = 0
    if depth > 0:
        problems.append(f"unbalanced `{{` ({depth} unclosed)")

    left = len(re.findall(r"\\left\b", tex))
    right = len(re.findall(r"\\right\b", tex))
    if left != right:
        problems.append(f"`\\left` ×{left} but `\\right` ×{right}")

    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    if sorted(begins) != sorted(ends):
        problems.append(f"`\\begin{{{begins}}}` vs `\\end{{{ends}}}`")
    return problems


def check_mermaid(block: str, first_line: int) -> list[str]:
    """Reserved keywords used as node ids."""
    problems = []
    # node ids appear as `id[`, `id(`, `id{`, or bare `id -->`
    for index, line in enumerate(block.split("\n"), start=first_line):
        stripped = line.strip()
        if stripped.startswith("%%"):
            continue
        for match in re.finditer(r"(?<![\w-])([A-Za-z_][\w-]*)\s*[\[\(\{]", stripped):
            node_id = match.group(1)
            if node_id.lower() in MERMAID_RESERVED:
                problems.append(
                    f"line {index}: `{node_id}` is a mermaid keyword, not a valid node id"
                    " — it ends the diagram; rename the id (e.g. `gph`)"
                )
    return problems


def check_file(path: pathlib.Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    problems: list[str] = []

    code_free = strip_code_spans(raw)

    for line, tex in math_spans(code_free):
        for pattern, why, fix in BANNED_LATEX:
            for match in re.finditer(pattern, tex):
                line_in_tex = tex[: match.start()].count("\n")
                problems.append(f"line {line + line_in_tex}: {why} — {fix}")
        for issue in check_delimiters(tex):
            problems.append(f"line {line}: {issue}")

    for match in re.finditer(r"```mermaid\n(.*?)```", raw, re.S):
        first = raw[: match.start()].count("\n") + 1
        problems += check_mermaid(match.group(1), first)

    return problems


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    files = sorted(
        p for p in root.rglob("*.md")
        if not any(part in {".git", "target", "node_modules"} for part in p.parts)
    )
    total = 0
    for path in files:
        problems = check_file(path)
        if problems:
            total += len(problems)
            print(f"\n{path.relative_to(root)}")
            for problem in problems:
                print(f"  ✗ {problem}")
    if total:
        print(f"\n{total} problem(s) in {len(files)} Markdown file(s).")
        return 1
    print(f"checked {len(files)} Markdown file(s): math and mermaid render on GitHub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
