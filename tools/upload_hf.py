#!/usr/bin/env python3
"""Upload the demo GGUF to a HuggingFace model repo (CORS *, Range — the
browser demo fetches it cross-origin from GitHub Pages).

Auth first (one time):
    tools/.venv/bin/hf auth login          # paste a WRITE token
Then:
    tools/.venv/bin/python tools/upload_hf.py [--repo <namespace>/<name>]

Prints the resolve URL to put in web/worker.js (DEFAULT_MODEL_URL).
"""
import argparse

from huggingface_hub import HfApi, create_repo, upload_file, whoami

GGUF = "parity/qwen3-0.6b-q8_0.gguf"
DEFAULT_NAME = "qwen3-0.6b-q8-gguf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="<namespace>/<name>; default <you>/" + DEFAULT_NAME)
    ap.add_argument("--file", default=GGUF)
    args = ap.parse_args()

    me = whoami()
    user = me["name"]
    orgs = [o["name"] for o in me.get("orgs", [])]
    print(f"authed as {user}; orgs: {orgs}")

    repo_id = args.repo or f"{user}/{DEFAULT_NAME}"
    print(f"repo: {repo_id}")

    import os
    fname = os.path.basename(args.file)
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"uploading {fname} (may take a few minutes)…")
    upload_file(
        path_or_fileobj=args.file,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"{fname} for the nn in-browser demo",
    )
    url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
    print("\nDONE. Model URL (download=true for the raw asset):")
    print(url)
    print(url + "?download=true")


if __name__ == "__main__":
    main()
