#!/usr/bin/env python3
"""Validate the `--json` output of the benchmark runner.

`BENCHMARKS.md` documents `cargo run --release -p ncly-benchmarks -- --json`, so
the CI job has to prove that the flag works and that the document is not lying.
The checks are deliberately structural rather than numerical: a benchmark number
is machine-dependent, but the schema is not.

Usage:
    python3 scripts/check_bench_json.py /tmp/bench.json
"""

from __future__ import annotations

import json
import sys

REQUIRED_CASE_KEYS = ("name", "size", "median_ms", "best_ms")
OPTIONAL_CASE_KEYS = ("gflops", "gib_per_second")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    with open(argv[1], encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        print(f"expected a JSON object, got {type(data).__name__}", file=sys.stderr)
        return 1
    if not data.get("kernel"):
        print(f"missing or empty 'kernel': {data!r}", file=sys.stderr)
        return 1

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        print(f"expected a non-empty 'cases' array, got {cases!r}", file=sys.stderr)
        return 1

    for index, case in enumerate(cases):
        for key in REQUIRED_CASE_KEYS:
            if key not in case:
                print(f"case {index} is missing {key!r}: {case!r}", file=sys.stderr)
                return 1
        for key in OPTIONAL_CASE_KEYS:
            if key not in case:
                print(f"case {index} is missing {key!r}: {case!r}", file=sys.stderr)
                return 1
            # Throughput is either a positive number or null for cases where the
            # operation count is not known; a zero would mean the runner lost it.
            value = case[key]
            if value is not None and not value > 0:
                print(f"case {index} has a non-positive {key}: {value!r}", file=sys.stderr)
                return 1
        for key in ("median_ms", "best_ms"):
            if not case[key] > 0:
                print(f"case {index} has a non-positive {key}: {case[key]!r}", file=sys.stderr)
                return 1
        if case["best_ms"] > case["median_ms"]:
            print(
                f"case {index} has best_ms > median_ms, so the statistics are swapped",
                file=sys.stderr,
            )
            return 1

    print(f"json output ok: {len(cases)} case(s), kernel {data['kernel']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
