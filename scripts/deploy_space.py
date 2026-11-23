# -*- coding: utf-8 -*-
"""Upload this folder to a Hugging Face Space.

    python scripts/deploy_space.py <namespace>/<space-name> [--private]

Uses `huggingface_hub` rather than git push. The pre-built cubes under `cubes/`
are NOT uploaded: they live in the dataset aaguilar90/cdh-demo-cubes, which the
app downloads at start-up.

Needs a login first (`hf auth login`). Secrets are NOT uploaded — set them in
the Space's settings afterwards; README.md lists them.
"""
from __future__ import annotations

import pathlib
import sys

from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parent.parent
IGNORE = [
    ".git/*", ".gradio/*", "__pycache__/*", "*/__pycache__/*", "*.pyc",
    ".env", ".cdsapirc", "runs/*", "*_test.html", "scripts/__pycache__/*",
    "cubes/*",
]


def main(argv: list[str]) -> int:
    if len(argv) < 2 or "/" not in argv[1]:
        print(__doc__)
        return 1
    repo_id = argv[1]
    private = "--private" in argv
    api = HfApi()
    who = api.whoami()
    print(f"logged in as {who.get('name')}; uploading {ROOT} -> {repo_id}")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy Crop Yield Explorer",
    )
    print(f"done: https://huggingface.co/spaces/{repo_id}")
    print("now set GOOGLE_API_KEY and/or OPENROUTER_API_KEY (and optionally "
          "CDSAPI_KEY) under Settings -> Variables and secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
