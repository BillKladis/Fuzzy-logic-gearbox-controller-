#!/usr/bin/env python3
"""Create the GitHub repository (if needed) and push the current branch.

This is the project's only publishing path: it authenticates with a personal
access token supplied through the environment -- the token is *never* written
to a file, committed, or stored in the git config.

Usage
-----
    GH_TOKEN=xxxxx python scripts/publish.py \
        --owner BillKladis \
        --repo fuzzy-gear-controller \
        --name "BillKladis" \
        --email "kladisbill@gmail.com"

Environment
-----------
    GH_TOKEN   (or GITHUB_TOKEN)   GitHub personal access token. Required.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESCRIPTION = "Fuzzy logic controller for automatic gear changing in a car"


def _token() -> str:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("error: set GH_TOKEN (or GITHUB_TOKEN) in the environment.")
    return tok


def _api(method: str, path: str, token: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "fuzzy-gear-publisher")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def ensure_repo(owner: str, repo: str, token: str) -> None:
    status, _ = _api("GET", f"/repos/{owner}/{repo}", token)
    if status == 200:
        print(f"repository {owner}/{repo} already exists")
        return
    # Create under the authenticated user's account.
    status, body = _api("POST", "/user/repos", token, {
        "name": repo,
        "description": DESCRIPTION,
        "private": False,
        "has_issues": True,
    })
    if status in (200, 201):
        print(f"created repository {owner}/{repo}")
    else:
        sys.exit(f"error creating repo ({status}): {body.get('message', body)}")


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, **kw)


def git_push(owner: str, repo: str, branch: str, name: str, email: str, token: str) -> None:
    run(["git", "config", "user.name", name])
    run(["git", "config", "user.email", email])
    # Build an authenticated URL just for this push; do not persist it.
    auth_url = f"https://{name}:{token}@github.com/{owner}/{repo}.git"
    # Point the named remote at the clean (token-free) URL for everyday use.
    clean_url = f"https://github.com/{owner}/{repo}.git"
    existing = subprocess.run(["git", "remote"], cwd=REPO_ROOT,
                              capture_output=True, text=True).stdout.split()
    run(["git", "remote", "set-url" if "origin" in existing else "add",
         "origin", clean_url])
    run(["git", "push", auth_url, f"HEAD:refs/heads/{branch}", "-u"])
    # Make sure the local branch tracks origin without leaking the token.
    run(["git", "branch", "--set-upstream-to", f"origin/{branch}", branch])
    print(f"\npushed to https://github.com/{owner}/{repo} (branch {branch})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", default="BillKladis")
    p.add_argument("--repo", default="fuzzy-gear-controller")
    p.add_argument("--branch", default="main")
    p.add_argument("--name", default="BillKladis")
    p.add_argument("--email", default="kladisbill@gmail.com")
    args = p.parse_args()

    token = _token()
    ensure_repo(args.owner, args.repo, token)
    git_push(args.owner, args.repo, args.branch, args.name, args.email, token)


if __name__ == "__main__":
    main()
