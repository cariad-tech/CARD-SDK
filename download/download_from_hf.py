#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2025 CARIAD SE
"""
download_from_hf.py

A utility to manage/download the CARD-Data datasets from Hugging Face.

Features:
  - Download specific repos or all of them.
  - Resume interrupted downloads automatically.
  - 'Test' mode: Downloads only metadata/READMEs to verify setup.
  - 'Check' mode: Verifies local files against remote metadata (existence & size).
"""

import argparse
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi

REPO_MAP = {
    "germany_batch2": "CARD-Data/CARD-Germany-Batch2",
    "germany_batch3": "CARD-Data/CARD-Germany-Batch3",
    "italy":          "CARD-Data/CARD-Italy",
    "germany_batch1":  "CARD-Data/CARD-Germany-Batch1", 
}

def get_local_path(base_dir: Path, short_name: str) -> Path:
    """Creates a subdirectory structure: base_dir / short_name"""
    return base_dir / short_name

def download_repo(
    repo_id: str, 
    local_dir: Path, 
    is_test: bool = False, 
    resume: bool = True,
    token: Optional[str] = None
):
    """
    Wraps snapshot_download to handle the heavy lifting.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {repo_id}")
    print(f"Target:     {local_dir}")
    print(f"Mode:       {'TEST (Metadata only)' if is_test else 'FULL DOWNLOAD'}")
    print(f"{'='*60}")

    allow_patterns = None
    if is_test:
        allow_patterns = ["README.md", ".gitattributes", "*.json", "dataset_infos.json"]

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            force_download=not resume,
            token=token,
            allow_patterns=allow_patterns, 
        )
        print(f"✅ Success: {repo_id}")
        return True
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        return False

def check_repo(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None
):
    """
    Compares remote file list (API) vs local files.
    Checks for: Missing files, Extra files, Size mismatches.
    """
    print(f"\n🔎 Checking: {repo_id}...")
    api = HfApi(token=token)

    try:
        print("   Fetching remote file list...")
        remote_files = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
        remote_map = {f.path: f.size for f in remote_files if hasattr(f, "size")}
    except Exception as e:
        print(f"  Failed to fetch remote info: {e}")
        return False

    # 2. Check Local State
    missing = []
    mismatch = []
    ok_count = 0

    print(f"   Verifying {len(remote_map)} files against {local_dir}...")

    for r_path, r_size in remote_map.items():
        l_path = local_dir / r_path
        
        if not l_path.exists():
            missing.append(r_path)
        else:
            l_size = l_path.stat().st_size
            if l_size != r_size:
                mismatch.append((r_path, l_size, r_size))
            else:
                ok_count += 1

    # 3. Report
    print(f"   --- Report for {repo_id} ---")
    print(f"   ✅ OK (Matched): {ok_count}")
    
    if missing:
        print(f"   ⚠️  MISSING ({len(missing)}):")
        for m in missing[:10]: print(f"      - {m}")
        if len(missing) > 10: print(f"      ... and {len(missing)-10} more.")
    
    if mismatch:
        print(f"   ⚠️  SIZE MISMATCH ({len(mismatch)}):")
        for m in mismatch[:10]: print(f"      - {m[0]} (Local: {m[1]} | Remote: {m[2]})")
        if len(mismatch) > 10: print(f"      ... and {len(mismatch)-10} more.")

    if not missing and not mismatch:
        print(" Perfect match!")
    return not missing and not mismatch

def main():
    parser = argparse.ArgumentParser(description="Downloader for CARD-Data repositories.")
    
    # Selection arguments
    parser.add_argument(
        "--repos", 
        nargs="+", 
        choices=list(REPO_MAP.keys()),
        default=list(REPO_MAP.keys()),
        help=f"Which repos to process. Options: {list(REPO_MAP.keys())}"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True, 
        help="Root folder where datasets will be downloaded."
    )

    # Operational arguments
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test mode: Only download README/Config files to verify connectivity."
    )
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check mode: Do not download. Only compare local files vs remote metadata."
    )

    parser.add_argument(
        "--no-resume", 
        action="store_true", 
        help="Force re-download (disable resume)."
    )
    
    parser.add_argument(
        "--token", 
        type=str, 
        default=None, 
        help="HuggingFace token (optional if you are already logged in via `hf auth login`)."
    )

    args = parser.parse_args()
    
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected Repositories: {args.repos}")

    all_successful = True
    for key in args.repos:
        repo_id = REPO_MAP[key]
        local_path = get_local_path(args.output_dir, key) # e.g. ./data/germany_batch2
        
        if args.check:
            successful = check_repo(repo_id, local_path, token=args.token)
        else:
            successful = download_repo(
                repo_id=repo_id,
                local_dir=local_path,
                is_test=args.test,
                resume=not args.no_resume,
                token=args.token
            )
        all_successful = all_successful and successful

    return 0 if all_successful else 1

if __name__ == "__main__":
    raise SystemExit(main())