"""Shared pytest fixtures.

The tests run against the built extension module. ``maturin develop`` installs it
into the active environment; when it is missing the whole module is skipped with
an actionable message rather than a collection error.
"""

import pathlib
import sys

import pytest

# Allow running from a source checkout where the extension sits in `python/`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ncly = pytest.importorskip(
    "ncly",
    reason="build the extension first: cd python && maturin develop --release",
)
