#!/usr/bin/env python3
"""
One-click helper to decompress in-place `.gz` dataset/log files inside the
PRISM-CPI repository (`data/raw/**/*.csv.gz`, `data/processed/*.pt.gz`, and
`results/logs/training_runs.json.gz`).
"""

import gzip
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def unpack_all() -> None:
    search_dirs = [
        REPO_ROOT / "data",
        REPO_ROOT / "results" / "logs",
    ]
    unpacked = 0
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for gz_path in sorted(sdir.rglob("*.gz")):
            if gz_path.name.endswith(".tar.gz"):
                continue
            out_path = gz_path.with_suffix("")  # strips trailing .gz
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[SKIP-EXISTS] {out_path.relative_to(REPO_ROOT)}")
                continue
            print(f"[UNPACKING] {gz_path.relative_to(REPO_ROOT)} -> {out_path.relative_to(REPO_ROOT)}")
            with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            unpacked += 1
    print(f"Done. Unpacked {unpacked} file(s).")


if __name__ == "__main__":
    unpack_all()

