"""Shared paths and CSV write helpers for scraper update scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_DIR = SCRIPT_DIR.parent
REPO_ROOT = SERVER_DIR.parent
DATA_DIR = SERVER_DIR / "models" / "data"
CACHE_DIR = SCRIPT_DIR / "cache"


def write_csv_atomic(df: pd.DataFrame, output_path: Path, backup: bool = True) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backup and output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        backup_path.write_bytes(output_path.read_bytes())

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)
