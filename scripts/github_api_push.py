#!/usr/bin/env python3
"""Publish the project to GitHub via the REST Git Data API.

Some environments allow authenticated calls to ``api.github.com`` but block the
raw git smart-HTTP protocol (``git-receive-pack``). This script pushes a full
commit through the API instead: it uploads every file as a blob, assembles a
tree, creates a commit, and moves the branch ref to it -- the API equivalent of
``git push``.

The token is read **only** from the environment (``GH_TOKEN`` / ``GITHUB_TOKEN``)
and is never written to disk or committed.

Usage
-----
    GH_TOKEN=xxxxx python scripts/github_api_push.py \
        --owner BillKladis --repo "Fuzzy-logic-gearbox-controller-" \
        --branch main --message "Fuzzy logic gear-shift controller" \
        --source .
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
SKIP_SUFFIX = (".pyc",)


def token() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("error: set GH_TOKEN (or GITHUB_TOKEN) in the environment.")
    return tok


def api(method: str, path: str, tok: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "github-api-push")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def collect_files(source: str):
    """Yield (repo_relative_path, absolute_path) for every file to upload."""
    source = os.path.abspath(source)
    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(SKIP_SUFFIX):
                continue
            ap = os.path.join(root, name)
            rel = os.path.relpath(ap, source).replace(os.sep, "/")
            yield rel, ap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--branch", default="main")
    p.add_argument("--message", default="Initial commit")
    p.add_argument("--source", default=".")
    p.add_argument("--name", default="BillKladis")
    p.add_argument("--email", default="kladisbill@gmail.com")
    args = p.parse_args()
    tok = token()
    repo = f"/repos/{args.owner}/{args.repo}"

    status, info = api("GET", repo, tok)
    if status != 200:
        sys.exit(f"cannot access {args.owner}/{args.repo} ({status}): "
                 f"{info.get('message', info)}")

    # 1) upload each file as a blob
    tree = []
    for rel, ap in sorted(collect_files(args.source)):
        with open(ap, "rb") as fh:
            content = base64.b64encode(fh.read()).decode()
        st, blob = api("POST", f"{repo}/git/blobs", tok,
                       {"content": content, "encoding": "base64"})
        if st not in (200, 201):
            sys.exit(f"blob failed for {rel} ({st}): {blob.get('message', blob)}")
        tree.append({"path": rel, "mode": "100644", "type": "blob",
                     "sha": blob["sha"]})
        print(f"  blob  {rel}")

    # 2) build a tree, 3) a commit (clean, parent-less initial commit)
    st, tr = api("POST", f"{repo}/git/trees", tok, {"tree": tree})
    if st not in (200, 201):
        sys.exit(f"tree failed ({st}): {tr.get('message', tr)}")
    commit_payload = {
        "message": args.message,
        "tree": tr["sha"],
        "parents": [],
        "author": {"name": args.name, "email": args.email},
        "committer": {"name": args.name, "email": args.email},
    }
    st, commit = api("POST", f"{repo}/git/commits", tok, commit_payload)
    if st not in (200, 201):
        sys.exit(f"commit failed ({st}): {commit.get('message', commit)}")

    # 4) point the branch ref at the new commit (create or force-update)
    ref = f"refs/heads/{args.branch}"
    st, _ = api("PATCH", f"{repo}/git/{ref}", tok,
                {"sha": commit["sha"], "force": True})
    if st == 404 or st == 422:
        st, body = api("POST", f"{repo}/git/refs", tok,
                       {"ref": ref, "sha": commit["sha"]})
        if st not in (200, 201):
            sys.exit(f"ref create failed ({st}): {body.get('message', body)}")
    elif st not in (200, 201):
        sys.exit(f"ref update failed ({st})")

    # make sure it is the default branch
    api("PATCH", repo, tok, {"default_branch": args.branch})
    print(f"\npushed {len(tree)} files to "
          f"https://github.com/{args.owner}/{args.repo} (branch {args.branch})")
    print(f"commit {commit['sha'][:10]}")


if __name__ == "__main__":
    main()
