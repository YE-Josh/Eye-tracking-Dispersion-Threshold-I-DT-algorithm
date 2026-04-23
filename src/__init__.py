"""I-DT eye-tracking fixation detection package."""

from .idt_algorithm import (
    DEFAULT_DISPERSION_THRESHOLD_PX,
    DEFAULT_MIN_FIXATION_DURATION_MS,
    Fixation,
    analyse_dataframe,
    classify_idt,
    summarise,
)
from .io_utils import load_gaze_file
from .visualization import (
    plot_duration_histogram,
    plot_scanpath,
    plot_timeseries,
)

__version__ = "1.0.0"
__all__ = [
    "analyse_dataframe",
    "classify_idt",
    "summarise",
    "Fixation",
    "load_gaze_file",
    "plot_scanpath",
    "plot_timeseries",
    "plot_duration_histogram",
    "DEFAULT_DISPERSION_THRESHOLD_PX",
    "DEFAULT_MIN_FIXATION_DURATION_MS",
]
