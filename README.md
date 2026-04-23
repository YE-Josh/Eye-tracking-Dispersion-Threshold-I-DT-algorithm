# I-DT Eye-tracking Fixation Detection

A Python toolkit for detecting **fixations** and **saccades** in 2-D eye-tracking data using the **Dispersion-Threshold Identification (I-DT) algorithm** (Salvucci & Goldberg, 2000).

Designed for researchers who work with gaze data from eye trackers such as Tobii Pro Glasses, HTC Vive Pro Eye, EyeLink and SMI. Accepts CSV or Excel input, lets the user interactively pick the X / Y / time columns, applies sensible defaults, and outputs an annotated dataset plus publication-quality figures.

---

## Features

- **I-DT algorithm** faithfully implemented from Salvucci & Goldberg (2000).
- **Interactive CLI** for picking the X / Y / time columns at runtime.
- **Flexible input** — `.csv`, `.tsv`, `.xlsx`, `.xls`.
- **Automatic time-unit conversion** — input time can be in `ms`, `s`, `us` or `ns`.
- **Configurable thresholds** with research-based defaults.
- **Per-sample labelling** — every row gets an `eye_movement_type` column (`fixation` / `saccade`).
- **Fixation summary table** — centroid, duration, dispersion, sample count.
- **Publication-quality figures** — scanpath overlay, time-series with fixation shading, duration histogram. 300 DPI, journal-style formatting.
- **Programmatic API** — use as a library or a command-line tool.
- **Unit-tested** core algorithm.

---

## Installation

```bash
git clone https://github.com/<your-username>/idt-eyetracking.git
cd idt-eyetracking
pip install -r requirements.txt
```

Or install as an editable package (this also exposes the `idt-analyse` command):

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy, pandas, matplotlib, openpyxl.

---

## Quick start

### Option 1 — Interactive mode

```bash
python -m src.cli
```

You will be prompted to:

1. Enter the path to your data file (CSV or Excel).
2. Select the X, Y and time columns from a preview.
3. Specify the time unit (`ms` / `s` / `us` / `ns`).
4. Set the dispersion threshold and minimum fixation duration (press Enter to accept defaults).
5. Choose an output directory.

### Option 2 — One-shot command line

```bash
python -m src.cli \
    --file data/trial1.csv \
    --x GazeX_px --y GazeY_px --time Timestamp_ms \
    --time-unit ms \
    --dispersion 25 --min-duration 100 \
    --outdir results/
```

### Option 3 — Python API

```python
from src.io_utils import load_gaze_file
from src.idt_algorithm import analyse_dataframe, summarise
from src.visualization import plot_scanpath

df = load_gaze_file("data/trial1.csv")

annotated, fixations = analyse_dataframe(
    df,
    x_col="GazeX_px",
    y_col="GazeY_px",
    time_col="Timestamp_ms",
    time_unit="ms",
    dispersion_threshold=25.0,     # pixels
    min_fixation_duration=100.0,   # milliseconds
)

print(summarise(fixations))
plot_scanpath(annotated, fixations, x_col="GazeX_px", y_col="GazeY_px",
              save_path="scanpath.png")
```

---

## The I-DT algorithm

I-DT identifies a fixation as a temporally contiguous cluster of gaze samples whose spatial dispersion stays below a threshold for at least a minimum duration. Every remaining sample is labelled a saccade.

**Dispersion** is defined as:

$$
D = (\max(x) - \min(x)) + (\max(y) - \min(y))
$$

**Pseudocode:**

```
while samples remaining:
    initialise window covering duration >= min_fixation_duration
    if dispersion(window) <= dispersion_threshold:
        extend window forward while dispersion stays <= threshold
        record fixation (start, end, centroid, duration)
        advance past the fixation
    else:
        label first sample as saccade
        advance by one sample
```

**Default parameters** (configurable):

| Parameter               | Default | Typical range in literature |
|-------------------------|---------|-----------------------------|
| Dispersion threshold    | 25 px   | 20–50 px                    |
| Minimum fixation length | 100 ms  | 80–200 ms                   |

### Choosing thresholds

The dispersion threshold depends on screen resolution and viewing distance. A common rule is to set it to roughly 1° of visual angle in screen pixels. For a typical lab setup (24" monitor at 60 cm, 1920×1080), 1° ≈ 35–40 px.

Minimum fixation durations below 60 ms are physiologically implausible; values of 100–150 ms are typical for reading and visual search tasks.

---

## Input format

Your CSV or Excel file should contain at minimum:

- An **X** column — gaze X in pixels (screen coordinates).
- A **Y** column — gaze Y in pixels (screen coordinates).
- A **time** column — sample timestamps.

Additional columns (participant ID, trial number, etc.) are preserved untouched in the annotated output.

Example:

| Timestamp_ms | GazeX_px | GazeY_px | ParticipantID |
|--------------|----------|----------|---------------|
| 0.0          | 401.2    | 298.7    | P001          |
| 5.0          | 400.5    | 301.1    | P001          |
| 10.0         | 399.8    | 299.5    | P001          |
| ...          | ...      | ...      | ...           |

NaN values (e.g., during blinks) are dropped before analysis.

---

## Output

Two tables and three figures are saved per run.

### `<input>_annotated.csv`

Copy of the input with two added columns:

- `time_ms` — timestamp in milliseconds (converted from the input unit).
- `eye_movement_type` — `"fixation"` or `"saccade"`.

### `<input>_fixations.csv`

One row per detected fixation:

| Column        | Description                                    |
|---------------|------------------------------------------------|
| `index`       | Sequential fixation ID (1-based)               |
| `start_idx`   | First sample index of the fixation             |
| `end_idx`     | Last sample index (inclusive)                  |
| `start_time`  | Start time (ms)                                |
| `end_time`    | End time (ms)                                  |
| `duration`    | Fixation duration (ms)                         |
| `centroid_x`  | Mean X of samples in the fixation (px)         |
| `centroid_y`  | Mean Y of samples in the fixation (px)         |
| `dispersion`  | Final dispersion within the fixation (px)      |
| `n_samples`   | Number of samples in the fixation              |

### Figures

- `<input>_scanpath.png` — 2-D scanpath with numbered fixation circles (size ∝ duration).
- `<input>_timeseries.png` — X and Y coordinates over time, fixation intervals shaded.
- `<input>_duration_hist.png` — histogram of fixation durations with mean line.

---

## Example run with synthetic data

A synthetic data generator is included. From the project root:

```bash
python examples/run_example.py
```

This creates `examples/sample_gaze.csv`, runs I-DT, prints summary statistics, and writes annotated CSVs and figures to `examples/output/`.

---

## Project structure

```
idt-eyetracking/
├── src/
│   ├── __init__.py
│   ├── idt_algorithm.py      # Core I-DT implementation
│   ├── io_utils.py           # CSV / Excel loaders
│   ├── visualization.py      # Figure generation
│   └── cli.py                # Interactive & argparse CLI
├── examples/
│   ├── generate_sample_data.py
│   └── run_example.py
├── tests/
│   └── test_idt.py
├── docs/
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## Citation

If you use this implementation in academic work, please cite the original algorithm:

> Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. In *Proceedings of the 2000 Symposium on Eye Tracking Research & Applications* (pp. 71–78). ACM. https://doi.org/10.1145/355017.355028

---

## Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue first to discuss the proposal.

---

## License

MIT — see [LICENSE](LICENSE).
