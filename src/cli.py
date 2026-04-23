"""
Command-line interface for the I-DT eye-tracking analyser.

Usage (interactive)
-------------------
    python -m src.cli

Usage (one-shot)
----------------
    python -m src.cli --file data.csv --x GazeX --y GazeY --time Timestamp \\
                      --time-unit ms --dispersion 25 --min-duration 100 \\
                      --outdir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Support running both as `python -m src.cli` and `python src/cli.py`
try:
    from .idt_algorithm import (
        DEFAULT_DISPERSION_THRESHOLD_PX,
        DEFAULT_MIN_FIXATION_DURATION_MS,
        analyse_dataframe,
        summarise,
    )
    from .io_utils import load_gaze_file
    from .visualization import (
        plot_duration_histogram,
        plot_scanpath,
        plot_timeseries,
    )
except ImportError:  # direct script invocation
    from idt_algorithm import (
        DEFAULT_DISPERSION_THRESHOLD_PX,
        DEFAULT_MIN_FIXATION_DURATION_MS,
        analyse_dataframe,
        summarise,
    )
    from io_utils import load_gaze_file
    from visualization import (
        plot_duration_histogram,
        plot_scanpath,
        plot_timeseries,
    )


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------
def _prompt(msg: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    resp = input(f"{msg}{suffix}: ").strip()
    if not resp and default is not None:
        return default
    return resp


def _prompt_float(msg: str, default: float) -> float:
    resp = _prompt(msg, str(default))
    try:
        return float(resp)
    except ValueError:
        print(f"  ! not a number, using default {default}")
        return default


def _prompt_column(df: pd.DataFrame, label: str) -> str:
    cols = list(df.columns)
    print(f"\nAvailable columns: {cols}")
    while True:
        choice = _prompt(f"Select the {label} column")
        if choice in cols:
            return choice
        # Allow selecting by 1-based index
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(cols):
                return cols[idx]
        print(f"  ! '{choice}' not found. Try again.")


def run_interactive() -> None:
    print("=" * 60)
    print("  I-DT Eye-tracking Fixation Detection")
    print("=" * 60)

    # 1. File
    path_str = _prompt("Path to gaze data file (.csv / .xlsx / .tsv)")
    path = Path(path_str).expanduser()
    print(f"Loading {path} ...")
    df = load_gaze_file(path)
    print(f"  loaded {len(df):,} rows, {len(df.columns)} columns")
    print("\nPreview:")
    print(df.head(5).to_string(index=False))

    # 2. Columns
    x_col = _prompt_column(df, "X (pixels)")
    y_col = _prompt_column(df, "Y (pixels)")
    time_col = _prompt_column(df, "time")
    time_unit = _prompt(
        "Time unit (ms / s / us / ns)", default="ms"
    ).lower()

    # 3. Parameters
    print(f"\nDefaults: dispersion = {DEFAULT_DISPERSION_THRESHOLD_PX} px, "
          f"min duration = {DEFAULT_MIN_FIXATION_DURATION_MS} ms")
    dispersion = _prompt_float(
        "Dispersion threshold (px)", DEFAULT_DISPERSION_THRESHOLD_PX
    )
    min_dur = _prompt_float(
        "Minimum fixation duration (ms)", DEFAULT_MIN_FIXATION_DURATION_MS
    )

    outdir = Path(_prompt("Output directory", default="results")).expanduser()

    _run(df, x_col, y_col, time_col, time_unit, dispersion, min_dur,
         outdir, input_name=path.stem)


# ---------------------------------------------------------------------------
# Core runner — shared between interactive and argparse modes
# ---------------------------------------------------------------------------
def _run(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    time_unit: str,
    dispersion: float,
    min_duration: float,
    outdir: Path,
    input_name: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nRunning I-DT ...")
    annotated, summary = analyse_dataframe(
        df,
        x_col=x_col, y_col=y_col, time_col=time_col,
        time_unit=time_unit,
        dispersion_threshold=dispersion,
        min_fixation_duration=min_duration,
    )

    stats = summarise(summary)
    print("\n--- Results ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<25s} {v:10.2f}")
        else:
            print(f"  {k:<25s} {v:>10}")

    # Save outputs
    annotated_path = outdir / f"{input_name}_annotated.csv"
    summary_path = outdir / f"{input_name}_fixations.csv"
    annotated.to_csv(annotated_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved annotated data  -> {annotated_path}")
    print(f"Saved fixation summary -> {summary_path}")

    # Figures
    scanpath_path = outdir / f"{input_name}_scanpath.png"
    ts_path = outdir / f"{input_name}_timeseries.png"
    hist_path = outdir / f"{input_name}_duration_hist.png"

    plot_scanpath(annotated, summary, x_col=x_col, y_col=y_col,
                  save_path=scanpath_path)
    plot_timeseries(annotated, x_col=x_col, y_col=y_col, save_path=ts_path)
    plot_duration_histogram(summary, save_path=hist_path)
    print(f"Saved figures          -> {outdir}/")


# ---------------------------------------------------------------------------
# argparse entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="I-DT fixation / saccade detection for 2-D gaze data."
    )
    p.add_argument("--file", type=str, help="Path to CSV / Excel file.")
    p.add_argument("--x", dest="x_col", type=str, help="Name of X column (px).")
    p.add_argument("--y", dest="y_col", type=str, help="Name of Y column (px).")
    p.add_argument("--time", dest="time_col", type=str, help="Name of time column.")
    p.add_argument("--time-unit", default="ms",
                   choices=["ms", "s", "us", "ns"], help="Unit of time column.")
    p.add_argument("--dispersion", type=float,
                   default=DEFAULT_DISPERSION_THRESHOLD_PX,
                   help=f"Dispersion threshold in px (default: {DEFAULT_DISPERSION_THRESHOLD_PX}).")
    p.add_argument("--min-duration", type=float,
                   default=DEFAULT_MIN_FIXATION_DURATION_MS,
                   help=f"Min fixation duration in ms (default: {DEFAULT_MIN_FIXATION_DURATION_MS}).")
    p.add_argument("--outdir", type=str, default="results",
                   help="Directory for output files.")
    p.add_argument("--interactive", action="store_true",
                   help="Force interactive prompt mode.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Interactive mode if explicitly asked, or if no file given
    if args.interactive or not args.file:
        run_interactive()
        return 0

    required = {"x_col": args.x_col, "y_col": args.y_col, "time_col": args.time_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"Missing required arguments: {missing}", file=sys.stderr)
        print("Use --interactive for guided column selection.", file=sys.stderr)
        return 2

    df = load_gaze_file(args.file)
    _run(
        df,
        x_col=args.x_col, y_col=args.y_col, time_col=args.time_col,
        time_unit=args.time_unit,
        dispersion=args.dispersion, min_duration=args.min_duration,
        outdir=Path(args.outdir),
        input_name=Path(args.file).stem,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
