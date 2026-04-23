"""
Generate a synthetic gaze trace for demo / testing.

Saves a CSV to examples/sample_gaze.csv with three fixations connected by
saccades — similar in shape to a real eye-tracker recording.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(
    seed: int = 2025,
    sampling_rate_hz: int = 200,
    duration_s: float = 6.0,
    out_path: Path = Path(__file__).parent / "sample_gaze.csv",
) -> Path:
    rng = np.random.default_rng(seed)
    dt_ms = 1000.0 / sampling_rate_hz
    n = int(duration_s * sampling_rate_hz)
    t = np.arange(n) * dt_ms

    # Define three fixation targets (screen px) and saccades between them
    targets = [(400, 300), (900, 250), (600, 550), (300, 500)]
    # Divide time across segments: fix, sac, fix, sac, fix, sac, fix
    seg_durations_ms = [800, 60, 1200, 80, 900, 70, 1500]  # ~ 4.6 s of actual use
    # Normalise to total samples
    total = sum(seg_durations_ms)
    seg_samples = [int(round(d / total * n)) for d in seg_durations_ms]
    diff = n - sum(seg_samples)
    seg_samples[-1] += diff  # fix rounding

    xs = np.empty(n)
    ys = np.empty(n)
    idx = 0
    current_target = 0
    for seg_i, ns in enumerate(seg_samples):
        if seg_i % 2 == 0:  # fixation segment
            cx, cy = targets[current_target]
            xs[idx:idx + ns] = cx + rng.normal(0, 1.5, ns)
            ys[idx:idx + ns] = cy + rng.normal(0, 1.5, ns)
        else:  # saccade segment — linear ramp with noise
            cx_from, cy_from = targets[current_target]
            current_target += 1
            cx_to, cy_to = targets[current_target]
            xs[idx:idx + ns] = np.linspace(cx_from, cx_to, ns) + rng.normal(0, 3, ns)
            ys[idx:idx + ns] = np.linspace(cy_from, cy_to, ns) + rng.normal(0, 3, ns)
        idx += ns

    # Inject a few NaNs (blinks)
    blink_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    xs[blink_idx] = np.nan
    ys[blink_idx] = np.nan

    df = pd.DataFrame({
        "Timestamp_ms": t,
        "GazeX_px": xs,
        "GazeY_px": ys,
        "ParticipantID": "P001",
        "TrialNumber": 1,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = generate_sample()
    print(f"Wrote {out}")
