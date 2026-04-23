"""Tests for the I-DT algorithm."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.idt_algorithm import (
    _dispersion,
    analyse_dataframe,
    classify_idt,
    summarise,
)


def test_dispersion_basic():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.5, 1.0])
    assert _dispersion(x, y) == pytest.approx(3.0)


def test_single_long_fixation():
    # 500 ms of tightly clustered gaze -> one fixation, no saccades
    n = 100
    t = np.linspace(0, 500, n)  # 500 ms, 5 ms per sample
    x = 100 + np.random.default_rng(0).normal(0, 0.5, n)
    y = 200 + np.random.default_rng(1).normal(0, 0.5, n)

    labels, fixations = classify_idt(x, y, t, dispersion_threshold=10, min_fixation_duration=100)

    assert len(fixations) == 1
    assert fixations[0].duration >= 100
    assert np.all(labels == "fixation")


def test_saccade_then_fixation():
    # First 50 ms: wildly dispersed samples (saccade)
    # Next 300 ms: tight cluster (fixation)
    t1 = np.linspace(0, 50, 10)
    x1 = np.linspace(0, 500, 10)
    y1 = np.linspace(0, 500, 10)

    t2 = np.linspace(51, 350, 60)
    x2 = np.full(60, 500.0) + np.random.default_rng(0).normal(0, 0.1, 60)
    y2 = np.full(60, 500.0) + np.random.default_rng(1).normal(0, 0.1, 60)

    t = np.concatenate([t1, t2])
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    labels, fixations = classify_idt(x, y, t, dispersion_threshold=10, min_fixation_duration=100)

    assert len(fixations) == 1
    # At least some of the saccade samples at the start
    assert (labels[:10] == "saccade").sum() > 0
    # The steady-state samples at the end are fixation
    assert (labels[-20:] == "fixation").all()


def test_no_fixation_when_too_short():
    # A 50 ms cluster cannot be a fixation if min duration is 100 ms
    t = np.linspace(0, 50, 20)
    x = np.full(20, 100.0)
    y = np.full(20, 100.0)

    _, fixations = classify_idt(x, y, t, dispersion_threshold=5, min_fixation_duration=100)
    assert len(fixations) == 0


def test_empty_input():
    labels, fixations = classify_idt(np.array([]), np.array([]), np.array([]))
    assert len(labels) == 0
    assert fixations == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        classify_idt(np.array([1.0, 2.0]), np.array([1.0]), np.array([0.0, 10.0]))


def test_analyse_dataframe_roundtrip():
    n = 200
    t = np.linspace(0, 1000, n)
    rng = np.random.default_rng(42)
    # Two fixations separated by a saccade
    x = np.concatenate([
        100 + rng.normal(0, 0.3, 80),
        np.linspace(100, 400, 40),
        400 + rng.normal(0, 0.3, 80),
    ])
    y = np.concatenate([
        200 + rng.normal(0, 0.3, 80),
        np.linspace(200, 200, 40),
        200 + rng.normal(0, 0.3, 80),
    ])
    df = pd.DataFrame({"GazeX": x, "GazeY": y, "T": t})

    annotated, summary = analyse_dataframe(
        df, x_col="GazeX", y_col="GazeY", time_col="T",
        time_unit="ms", dispersion_threshold=10, min_fixation_duration=100,
    )
    assert "eye_movement_type" in annotated.columns
    assert "time_ms" in annotated.columns
    assert len(summary) >= 1
    assert set(summary.columns) >= {"index", "start_time", "end_time",
                                     "duration", "centroid_x", "centroid_y"}


def test_time_unit_conversion():
    n = 100
    t_sec = np.linspace(0, 1.0, n)  # 1 second in seconds
    x = np.full(n, 50.0)
    y = np.full(n, 50.0)
    df = pd.DataFrame({"x": x, "y": y, "t": t_sec})

    annotated, summary = analyse_dataframe(
        df, x_col="x", y_col="y", time_col="t",
        time_unit="s", dispersion_threshold=5, min_fixation_duration=100,
    )
    # time_ms should range 0..1000
    assert annotated["time_ms"].max() == pytest.approx(1000.0)
    assert len(summary) == 1
    assert summary.iloc[0]["duration"] == pytest.approx(1000.0, rel=0.05)


def test_summarise_empty():
    stats = summarise(pd.DataFrame())
    assert stats["n_fixations"] == 0
    assert stats["total_fixation_time_ms"] == 0.0
