"""Local syntax gate: parse every Rust file with tree-sitter.

The sandbox has no Rust toolchain, so `cargo check` only runs in CI. This script
catches the cheapest class of error (unbalanced delimiters, stray tokens) before
a push, which keeps the CI round-trip budget for real type errors.
"""

from __future__ import annotations

import pathlib
import sys

import tree_sitter as ts
import tree_sitter_rust as rust

PARSER = ts.Parser(ts.Language(rust.language()))


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    failures = 0
    files = 0
    for path in sorted(root.rglob("*.rs")):
        if "target" in path.parts:
            continue
        files += 1
        source = path.read_bytes()
        tree = PARSER.parse(source)
        if not tree.root_node.has_error:
            continue
        for node in walk(tree.root_node):
            if node.type == "ERROR" or node.is_missing:
                line = node.start_point[0] + 1
                text = source[node.start_byte : node.end_byte].decode("utf8", "replace")
                print(f"{path}:{line}: {node.type}: {text.splitlines()[0][:120] if text else ''}")
                failures += 1
                break
    print(f"checked {files} files, {failures} with syntax errors")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
