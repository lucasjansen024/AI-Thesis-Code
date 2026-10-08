"""Integer 1 Hz grid on the CPET clock, 0 s -> exercise end, with `phase` and `unit` from the nominal timeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inventory.catalog import Timeline

PHASES = ("rest", "warmup", "exercise")  # code = index
REST, WARMUP, EXERCISE = range(len(PHASES))
UNITS = ("visit", "rep1", "rep2")  # code = index; same names as inclusion.csv
NO_UNIT = -1  # rest and warm-up seconds belong to no unit


@dataclass(frozen=True)
class Grid:
    t: np.ndarray  # int64 seconds 0 .. exercise_end_s - 1
    phase: np.ndarray  # int8 code into PHASES
    unit: np.ndarray  # int8 code into UNITS, NO_UNIT outside exercise


def build_grid(timeline: Timeline) -> Grid:
    """Seconds [0, exercise_end_s). Units: PRBS repetitions [start_k, start_k+1), else the whole exercise phase."""
    if timeline.exercise_end_s is None:
        raise ValueError(f"exercise end unknown for timeline {timeline}")
    t = np.arange(int(timeline.exercise_end_s), dtype=np.int64)
    phase = np.full(len(t), REST, dtype=np.int8)
    phase[t >= timeline.rest_end_s] = WARMUP
    phase[t >= timeline.warmup_end_s] = EXERCISE
    unit = np.full(len(t), NO_UNIT, dtype=np.int8)
    unit[phase == EXERCISE] = UNITS.index("visit")
    for number, start in enumerate(timeline.repetition_starts_s, start=1):
        unit[(t >= start) & (phase == EXERCISE)] = UNITS.index(f"rep{number}")
    return Grid(t, phase, unit)
