"""Upload a build log to the commit as chunked Markdown comments.

Used only by CI so that logs written on a runner are retrievable from the
commit itself when live log streaming is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

CHUNK = 55_000
MAX_CHUNKS = 40


def post(token: str, repo: str, sha: str, body: str) -> None:
    url = f"https://api.github.com/repos/{repo}/commits/{sha}/comments"
    payload = json.dumps({"body": body}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    urllib.request.urlopen(request, timeout=30).read()


def main() -> int:
    path = sys.argv[1]
    try:
        log = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        log = "(log file missing)"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (token and repo and sha):
        print("post_log: missing credentials, skipping upload")
        return 0
    body = log[-CHUNK * MAX_CHUNKS :]
    total = max(1, (len(body) + CHUNK - 1) // CHUNK)
    for index in range(total):
        chunk = body[index * CHUNK : (index + 1) * CHUNK]
        try:
            post(token, repo, sha, f"ncly log {index + 1}/{total}\n```\n{chunk}\n```\n")
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            print(f"post_log: upload failed: {exc!r}")
            return 0
    print(f"post_log: uploaded {len(body)} chars in {total} chunk(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
