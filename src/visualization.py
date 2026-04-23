"""
Visualisation utilities for I-DT output.

Produces publication-quality figures (high DPI, journal-style formatting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Journal-style defaults
_PLOT_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _apply_style():
    for k, v in _PLOT_STYLE.items():
        plt.rcParams[k] = v


def plot_scanpath(
    annotated: pd.DataFrame,
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    invert_y: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Gaze scanpath with detected fixations",
) -> plt.Figure:
    """
    Plot the 2-D gaze trajectory with fixations overlaid.

    Fixation markers are scaled by duration.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    valid = annotated.dropna(subset=[x_col, y_col, "eye_movement_type"])
    sac = valid[valid["eye_movement_type"] == "saccade"]
    fix = valid[valid["eye_movement_type"] == "fixation"]

    # Raw trajectory (thin grey line)
    ax.plot(valid[x_col], valid[y_col], "-", color="#BBBBBB", lw=0.6, alpha=0.6,
            label="Scanpath", zorder=1)
    ax.scatter(sac[x_col], sac[y_col], s=6, color="#D55E00", alpha=0.5,
               label="Saccade samples", zorder=2)

    # Fixation circles, scaled by duration
    if not summary.empty:
        sizes = 40 + (summary["duration"] / summary["duration"].max()) * 400
        ax.scatter(
            summary["centroid_x"], summary["centroid_y"],
            s=sizes, facecolor="#0072B2", edgecolor="white",
            alpha=0.65, linewidth=1.2, label="Fixations", zorder=3,
        )
        # Number the fixations in order
        for _, row in summary.iterrows():
            ax.annotate(
                str(int(row["index"])),
                (row["centroid_x"], row["centroid_y"]),
                fontsize=8, color="white", ha="center", va="center",
                fontweight="bold", zorder=4,
            )

    ax.set_xlabel(f"{x_col} (px)")
    ax.set_ylabel(f"{y_col} (px)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()  # screen coordinates: origin top-left

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_timeseries(
    annotated: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot x and y gaze coordinates over time with fixation intervals shaded.
    """
    _apply_style()
    valid = annotated.dropna(subset=[x_col, y_col, "time_ms", "eye_movement_type"])
    t = valid["time_ms"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t, valid[x_col], "-", color="#333333", lw=0.9)
    axes[1].plot(t, valid[y_col], "-", color="#333333", lw=0.9)

    # Shade fixation windows
    is_fix = (valid["eye_movement_type"] == "fixation").to_numpy()
    if is_fix.any():
        edges = np.diff(is_fix.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0]
        if is_fix[0]:
            starts = np.r_[0, starts]
        if is_fix[-1]:
            ends = np.r_[ends, len(is_fix) - 1]
        for s, e in zip(starts, ends):
            for ax in axes:
                ax.axvspan(t[s], t[e], color="#0072B2", alpha=0.15, lw=0)

    axes[0].set_ylabel(f"{x_col} (px)")
    axes[1].set_ylabel(f"{y_col} (px)")
    axes[1].set_xlabel("Time (ms)")
    axes[0].set_title("Gaze coordinates over time (shaded = fixation)")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_duration_histogram(
    summary: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Histogram of fixation durations."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    if summary.empty:
        ax.text(0.5, 0.5, "No fixations detected", ha="center", va="center",
                transform=ax.transAxes)
    else:
        ax.hist(summary["duration"], bins=20, color="#0072B2",
                edgecolor="white", alpha=0.85)
        ax.axvline(summary["duration"].mean(), color="#D55E00", ls="--", lw=1.5,
                   label=f"Mean = {summary['duration'].mean():.0f} ms")
        ax.legend(frameon=False)
    ax.set_xlabel("Fixation duration (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of fixation durations")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
