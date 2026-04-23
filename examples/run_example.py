"""
End-to-end example: generate synthetic data, run I-DT, save figures.

Run with:
    python examples/run_example.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.generate_sample_data import generate_sample
from src.idt_algorithm import analyse_dataframe, summarise
from src.io_utils import load_gaze_file
from src.visualization import (
    plot_duration_histogram,
    plot_scanpath,
    plot_timeseries,
)


def main():
    sample_path = generate_sample()
    print(f"Loaded synthetic data: {sample_path}")

    df = load_gaze_file(sample_path)

    annotated, summary = analyse_dataframe(
        df,
        x_col="GazeX_px",
        y_col="GazeY_px",
        time_col="Timestamp_ms",
        time_unit="ms",
        dispersion_threshold=25.0,
        min_fixation_duration=100.0,
    )

    stats = summarise(summary)
    print("\nSummary statistics:")
    for k, v in stats.items():
        print(f"  {k:<25s} {v}")

    out_dir = ROOT / "examples" / "output"
    out_dir.mkdir(exist_ok=True)

    annotated.to_csv(out_dir / "sample_annotated.csv", index=False)
    summary.to_csv(out_dir / "sample_fixations.csv", index=False)

    plot_scanpath(annotated, summary, x_col="GazeX_px", y_col="GazeY_px",
                  save_path=out_dir / "scanpath.png")
    plot_timeseries(annotated, x_col="GazeX_px", y_col="GazeY_px",
                    save_path=out_dir / "timeseries.png")
    plot_duration_histogram(summary, save_path=out_dir / "duration_hist.png")

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
