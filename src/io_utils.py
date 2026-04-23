"""
File loading utilities for the I-DT eye-tracking project.

Supports .csv, .tsv, .txt, .xlsx and .xls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


SUPPORTED_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


def load_gaze_file(
    path: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
    delimiter: str | None = None,
) -> pd.DataFrame:
    """
    Load a gaze data file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        File path. Format is inferred from the extension.
    sheet_name : str, int or None
        Sheet to load for Excel files (default: first sheet).
    delimiter : str, optional
        Override the delimiter for text-based files. If None, pandas infers it.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTS)}"
        )

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)

    if ext == ".tsv":
        return pd.read_csv(path, sep=delimiter or "\t")

    # .csv / .txt — let pandas sniff the separator if not supplied
    if delimiter is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=delimiter)


def preview_columns(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first ``n`` rows — used by the CLI to help the user pick columns."""
    return df.head(n)
