"""
I-DT (Dispersion-Threshold Identification) Algorithm
=====================================================

Implementation of the I-DT fixation detection algorithm as described in:
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and
    saccades in eye-tracking protocols. In Proceedings of the 2000 symposium
    on Eye tracking research & applications (pp. 71-78). ACM.

The algorithm identifies fixations as groups of consecutive gaze points
whose spatial dispersion stays below a threshold for at least a minimum
duration. All remaining samples are labelled as saccades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults (commonly used values in the eye-tracking literature)
# ---------------------------------------------------------------------------
DEFAULT_DISPERSION_THRESHOLD_PX: float = 25.0   # pixels
DEFAULT_MIN_FIXATION_DURATION_MS: float = 100.0  # milliseconds


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Fixation:
    """A single detected fixation event."""
    index: int              # sequential fixation id (1-based)
    start_idx: int          # first sample index in the original data
    end_idx: int            # last sample index (inclusive)
    start_time: float       # ms
    end_time: float         # ms
    duration: float         # ms
    centroid_x: float       # pixels
    centroid_y: float       # pixels
    dispersion: float       # pixels
    n_samples: int


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def _dispersion(x: np.ndarray, y: np.ndarray) -> float:
    """Salvucci & Goldberg (2000) dispersion: (xmax - xmin) + (ymax - ymin)."""
    return (x.max() - x.min()) + (y.max() - y.min())


def classify_idt(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_PX,
    min_fixation_duration: float = DEFAULT_MIN_FIXATION_DURATION_MS,
) -> tuple[np.ndarray, List[Fixation]]:
    """
    Run the I-DT algorithm on a gaze signal.

    Parameters
    ----------
    x, y : np.ndarray
        Gaze coordinates in pixels. Must be the same length.
    t : np.ndarray
        Timestamps in milliseconds, monotonically increasing.
    dispersion_threshold : float
        Maximum spatial dispersion (px) allowed within a fixation.
    min_fixation_duration : float
        Minimum duration (ms) for a window of samples to qualify as a fixation.

    Returns
    -------
    labels : np.ndarray
        Per-sample classification array of dtype=object containing
        'fixation' or 'saccade' for every input sample.
    fixations : list[Fixation]
        Ordered list of detected fixation events.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    n = len(x)
    if not (len(y) == n == len(t)):
        raise ValueError("x, y and t must all have the same length.")
    if n == 0:
        return np.array([], dtype=object), []

    labels = np.full(n, "saccade", dtype=object)
    fixations: List[Fixation] = []
    fix_counter = 0
    i = 0

    # Salvucci & Goldberg pseudocode, vectorised where possible
    while i < n:
        # Grow an initial window that covers at least min_fixation_duration
        j = i
        while j < n and (t[j] - t[i]) < min_fixation_duration:
            j += 1

        if j >= n:
            # Not enough remaining samples to form a candidate fixation
            break

        win_x = x[i : j + 1]
        win_y = y[i : j + 1]

        if _dispersion(win_x, win_y) <= dispersion_threshold:
            # Extend the window forward while it stays within threshold
            k = j + 1
            while k < n:
                new_x = x[i : k + 1]
                new_y = y[i : k + 1]
                if _dispersion(new_x, new_y) > dispersion_threshold:
                    break
                k += 1
            end = k - 1  # last valid sample in this fixation

            fix_counter += 1
            fx = x[i : end + 1]
            fy = y[i : end + 1]
            fixations.append(
                Fixation(
                    index=fix_counter,
                    start_idx=i,
                    end_idx=end,
                    start_time=float(t[i]),
                    end_time=float(t[end]),
                    duration=float(t[end] - t[i]),
                    centroid_x=float(fx.mean()),
                    centroid_y=float(fy.mean()),
                    dispersion=float(_dispersion(fx, fy)),
                    n_samples=end - i + 1,
                )
            )
            labels[i : end + 1] = "fixation"
            i = end + 1
        else:
            # Window too dispersed — mark first sample as saccade and advance
            labels[i] = "saccade"
            i += 1

    return labels, fixations


# ---------------------------------------------------------------------------
# Convenience wrappers for DataFrames
# ---------------------------------------------------------------------------
def analyse_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    time_unit: str = "ms",
    dispersion_threshold: float = DEFAULT_DISPERSION_THRESHOLD_PX,
    min_fixation_duration: float = DEFAULT_MIN_FIXATION_DURATION_MS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run I-DT on a DataFrame and return an annotated copy plus a fixation summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input gaze data.
    x_col, y_col, time_col : str
        Column names for x (px), y (px) and time.
    time_unit : {'ms', 's', 'us', 'ns'}
        Unit of the time column. Values are converted to milliseconds internally.
    dispersion_threshold : float
        Passed through to the algorithm.
    min_fixation_duration : float
        Passed through to the algorithm.

    Returns
    -------
    annotated : pd.DataFrame
        Copy of ``df`` with two new columns: ``time_ms`` and ``eye_movement_type``.
    summary : pd.DataFrame
        One row per detected fixation.
    """
    for col in (x_col, y_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Drop rows with NaNs in the critical columns (common with eye trackers)
    work = df[[x_col, y_col, time_col]].copy()
    work = work.dropna().reset_index(drop=False)  # keep original index in 'index'

    t_raw = work[time_col].to_numpy(dtype=float)
    t_ms = _to_milliseconds(t_raw, time_unit)

    # Ensure monotonic time — sort if needed
    if not np.all(np.diff(t_ms) >= 0):
        order = np.argsort(t_ms, kind="mergesort")
        work = work.iloc[order].reset_index(drop=True)
        t_ms = t_ms[order]

    labels, fixations = classify_idt(
        x=work[x_col].to_numpy(dtype=float),
        y=work[y_col].to_numpy(dtype=float),
        t=t_ms,
        dispersion_threshold=dispersion_threshold,
        min_fixation_duration=min_fixation_duration,
    )

    annotated = df.copy()
    annotated["time_ms"] = np.nan
    annotated["eye_movement_type"] = pd.NA
    original_indices = work["index"].to_numpy()
    annotated.loc[original_indices, "time_ms"] = t_ms
    annotated.loc[original_indices, "eye_movement_type"] = labels

    summary = pd.DataFrame([f.__dict__ for f in fixations])
    return annotated, summary


def _to_milliseconds(t: np.ndarray, unit: str) -> np.ndarray:
    """Convert a time array to milliseconds."""
    unit = unit.lower()
    factors = {"ms": 1.0, "s": 1e3, "sec": 1e3, "us": 1e-3, "µs": 1e-3, "ns": 1e-6}
    if unit not in factors:
        raise ValueError(
            f"Unsupported time_unit '{unit}'. Use one of: {sorted(factors)}."
        )
    return t * factors[unit]


# ---------------------------------------------------------------------------
# Aggregate metrics (useful for reporting)
# ---------------------------------------------------------------------------
def summarise(summary: pd.DataFrame) -> dict:
    """Return aggregate descriptive statistics for a fixation summary table."""
    if summary.empty:
        return {
            "n_fixations": 0,
            "mean_duration_ms": np.nan,
            "median_duration_ms": np.nan,
            "std_duration_ms": np.nan,
            "total_fixation_time_ms": 0.0,
        }
    return {
        "n_fixations": int(len(summary)),
        "mean_duration_ms": float(summary["duration"].mean()),
        "median_duration_ms": float(summary["duration"].median()),
        "std_duration_ms": float(summary["duration"].std(ddof=1)) if len(summary) > 1 else 0.0,
        "min_duration_ms": float(summary["duration"].min()),
        "max_duration_ms": float(summary["duration"].max()),
        "total_fixation_time_ms": float(summary["duration"].sum()),
        "mean_dispersion_px": float(summary["dispersion"].mean()),
    }
