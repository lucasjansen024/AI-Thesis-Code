"""Tiny synthetic processed caches and inclusion tables (no raw data needed)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from inventory.build_inclusion import OUTPUT_COLUMNS, SENSOR_GROUPS
from pipeline.cache import ProcessedCache, from_visits
from pipeline.prepare.grid import EXERCISE, NO_UNIT, REST, UNITS, WARMUP
from pipeline.prepare.visit import ProcessedVisit
from pipeline.runs import Run, TrainingSettings

REST_END_S, WARMUP_END_S, VISIT_S = 120, 330, 600  # 270 exercise seconds per visit
VISIT_UNIT = UNITS.index("visit")
STEP_S, LOW_W = 30, 25.0
SYNTHETIC_HR_REST_BPM, SYNTHETIC_HR_PEAK_BPM = 70.0, 185.0
SYNTHETIC_VE_REST_L_MIN, SYNTHETIC_BR_REST_PER_MIN = 10.0, 14.0
SYNTHETIC_TSI_REST_PCT, SYNTHETIC_HBDIFF_REST_UM = 68.0, 5.0
SYNTHETIC_SMM_KG, SYNTHETIC_SMM_STEP_KG = 30.0, 2.5  # per participant; sex alternates male, female, male
# (participant_id, cohort, protocol): single-unit protocols keep the unit grid trivial
PARTICIPANTS = (("G01", "Gies", "Ramp"), ("G03", "Gies", "Ramp"), ("L01", "Lucas", "Step"))
SYNTHETIC_RUN = Run("synthetic", ("Gies", "Lucas"), ("WR",), seeds=(0,), training=TrainingSettings(epochs=2))

ColumnEdit = Callable[[str, dict[str, np.ndarray]], dict[str, np.ndarray]]


@dataclass(frozen=True)
class Synthetic:
    cache: ProcessedCache
    inclusion: pd.DataFrame


def _columns(number: int) -> dict[str, np.ndarray]:
    t = np.arange(VISIT_S)
    phase = np.where(t < REST_END_S, REST, np.where(t < WARMUP_END_S, WARMUP, EXERCISE))
    unit = np.where(phase == EXERCISE, VISIT_UNIT, NO_UNIT)
    rng = np.random.default_rng(number)
    blocks = np.arange(VISIT_S // STEP_S + 1)
    levels = np.where((blocks * 7 + number) % 3 == 0, LOW_W, 100.0 + 20 * number)  # both levels in every visit
    wr = np.where(t < REST_END_S, 0.0, np.repeat(levels, STEP_S)[:VISIT_S])
    vo2 = 400.0 + 10.0 * pd.Series(wr).rolling(40, min_periods=1).mean().to_numpy() + rng.normal(0, 30, VISIT_S)
    ones = np.ones(VISIT_S)
    hr = SYNTHETIC_HR_REST_BPM + 0.4 * pd.Series(wr).rolling(60, min_periods=1).mean().to_numpy()
    hrr = (hr - hr[:REST_END_S].mean()) / (SYNTHETIC_HR_PEAK_BPM - hr[:REST_END_S].mean())
    wr_30s = pd.Series(wr).rolling(30, min_periods=1).mean().to_numpy()  # no rng draw: VO2 noise unchanged
    ve = SYNTHETIC_VE_REST_L_MIN + 0.25 * wr_30s
    br = SYNTHETIC_BR_REST_PER_MIN + 0.08 * wr_30s
    tsi = SYNTHETIC_TSI_REST_PCT - 0.05 * wr_30s  # muscle desaturates with load; no rng draw
    hbdiff = SYNTHETIC_HBDIFF_REST_UM - 0.04 * wr_30s
    sex = np.full(VISIT_S, 1.0 if number % 2 == 0 else 0.0)  # static: constant per visit, no rng draw
    smm = np.full(VISIT_S, SYNTHETIC_SMM_KG + SYNTHETIC_SMM_STEP_KG * number)
    return {"t": t, "phase": phase, "unit": unit, "VO2": vo2, "present_VO2": ones, "WR": wr, "present_WR": ones,
            "HR": hr, "HRR": hrr, "present_HR": ones, "VE": ve, "BR": br, "present_VENT": ones,
            "TSI": tsi, "HBDIFF": hbdiff, "present_NIRS": ones,
            "sex": sex, "SMM": smm, "present_static": ones}


def processed_visits(edit: ColumnEdit | None = None) -> list[ProcessedVisit]:
    visits = []
    for number, (pid, cohort, protocol) in enumerate(PARTICIPANTS):
        columns = _columns(number)
        if edit is not None:
            columns = edit(pid, columns)
        ids = {"participant_id": pid, "cohort": cohort, "protocol": protocol,
               "visit_id": f"{pid}_{protocol}", "source_visit": protocol}
        visits.append(ProcessedVisit(ids, columns))
    return visits


def inclusion_table(invalid: frozenset[tuple[str, str]] = frozenset()) -> pd.DataFrame:
    """Every (visit, group) valid except the listed (visit_id, sensor_group) pairs."""
    rows = [
        {"participant_id": pid, "cohort": cohort, "protocol": protocol, "visit_id": f"{pid}_{protocol}",
         "unit": "visit", "sensor_group": group, "valid": (f"{pid}_{protocol}", group) not in invalid,
         "reason": "", "decided_by": "rule", "hr_source": ""}
        for pid, cohort, protocol in PARTICIPANTS
        for group in SENSOR_GROUPS
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def make_synthetic(edit: ColumnEdit | None = None,
                   invalid: frozenset[tuple[str, str]] = frozenset()) -> Synthetic:
    return Synthetic(from_visits(processed_visits(edit), key="synthetic"), inclusion_table(invalid))


def make_synthetic_visits(visits: tuple[tuple[str, str, str], ...],
                          invalid: frozenset[tuple[str, str]] = frozenset()) -> Synthetic:
    """Like `make_synthetic` for any (participant_id, cohort, protocol) list, e.g. several protocols per participant
    (Aim 2). Every visit has the same 270 exercise seconds; signals differ by list position, while sex and SMM are
    those of the participant's position among the distinct participants (constant across its visits)."""
    participant_number = {pid: n for n, pid in enumerate(dict.fromkeys(pid for pid, _, _ in visits))}

    def visit_columns(number: int, pid: str) -> dict[str, np.ndarray]:
        static = _columns(participant_number[pid])
        return {**_columns(number), "sex": static["sex"], "SMM": static["SMM"]}

    processed = [ProcessedVisit({"participant_id": pid, "cohort": cohort, "protocol": protocol,
                                 "visit_id": f"{pid}_{protocol}", "source_visit": protocol}, visit_columns(number, pid))
                 for number, (pid, cohort, protocol) in enumerate(visits)]
    rows = [
        {"participant_id": pid, "cohort": cohort, "protocol": protocol, "visit_id": f"{pid}_{protocol}",
         "unit": "visit", "sensor_group": group, "valid": (f"{pid}_{protocol}", group) not in invalid,
         "reason": "", "decided_by": "rule", "hr_source": ""}
        for pid, cohort, protocol in visits
        for group in SENSOR_GROUPS
    ]
    return Synthetic(from_visits(processed, key="synthetic_visits"), pd.DataFrame(rows, columns=OUTPUT_COLUMNS))
