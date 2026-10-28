#!/usr/bin/env python3
"""Prove authenticated public Hugging Face write and read-back."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("HF token file is empty")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    payload = json.dumps(
        {"schema": "PI0243_HF_PUBLIC_WRITE_PROBE_V1", "repo_id": args.repo_id},
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    commit = api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo=args.path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="PI0243 authenticated public write probe",
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.path,
            repo_type="model",
            token=token,
        )
    )
    observed = downloaded.read_bytes()
    if observed != payload:
        raise SystemExit("HF read-back differs from uploaded probe")
    info = api.model_info(repo_id=args.repo_id)
    if getattr(info, "private", None) is not False:
        raise SystemExit(f"HF repository is not public: {info.private!r}")
    print(
        json.dumps(
            {
                "schema": "PI0243_HF_PUBLIC_WRITE_PROBE_V1",
                "repo_id": args.repo_id,
                "private": info.private,
                "commit_id": getattr(commit, "oid", None),
                "path": args.path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "readback": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
