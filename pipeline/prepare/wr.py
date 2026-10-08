"""Ideal protocol work rate on the grid (never the breath-tied measured power as a feature).

- Submax with its own xml, and ramps (CPET protocol name): as `inventory.sensors` builds it (`visit.protocol`).
- A repeated visit without an xml (G08 Medium2, G09 Hard2): the superseded sibling's xml (same CPET protocol).
- L04 PRBS (no xml): another Lucas PRBS xml's PRBSV2 timing, 25 W low, warm-up and high level = median
  measured power over those phases of L04's own CPET file.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from inventory.catalog import VisitRef, choose_protocol_xml
from inventory.sensors import load_protocol_xml
from pipeline.prepare.group import GroupColumns, PrepInput

logger = logging.getLogger(__name__)

WR_GROUP = "WR"
WR_COLUMN = "WR"
SUPERSEDED_ROLE = "superseded"
CANDIDATE_ROLE = "candidate"
REBUILT_PRBS_VISITS = frozenset({"L04_PRBS"})  # decided in data validity (inclusion.csv WR note)
LUCAS_PRBS_LOW_W = 25.0
PHASE_EDGE_SKIP_S = 15  # skip the load transitions when taking a median of measured power


def _on_grid(protocol_wr: pd.DataFrame, grid_t: np.ndarray) -> np.ndarray:
    """protocol_wr has one row per second from 0 s; seconds past its end are NaN."""
    t = protocol_wr["t"].to_numpy(dtype=np.int64)
    if not np.array_equal(t, np.arange(len(t))):
        raise ValueError("protocol WR is not a contiguous 1 Hz series from 0 s")
    w = protocol_wr["W"].to_numpy(dtype=float)
    out = np.full(len(grid_t), np.nan)
    inside = grid_t < len(w)
    out[inside] = w[grid_t[inside]]
    return out


def _xml_wr(ref: VisitRef) -> pd.DataFrame | None:
    choice = choose_protocol_xml(ref.path, ref.protocol)
    return None if choice.selected is None else load_protocol_xml(choice.selected)


def _sibling_wr(ref: VisitRef, refs: tuple[VisitRef, ...]) -> pd.DataFrame | None:
    """The superseded visit of the same participant and protocol, if it has an xml."""
    for other in refs:
        if (other.role == SUPERSEDED_ROLE and other.cohort == ref.cohort
                and other.participant_id == ref.participant_id and other.protocol == ref.protocol):
            wr = _xml_wr(other)
            if wr is not None:
                logger.info("%s: WR from the superseded sibling %s", ref.visit_id, other.source_visit)
                return wr
    return None


def _template_wr(ref: VisitRef, refs: tuple[VisitRef, ...]) -> pd.DataFrame:
    """The first other candidate visit of the same cohort and protocol that has an xml (shared PRBSV2 timing)."""
    for other in refs:
        if (other.role == CANDIDATE_ROLE and other.cohort == ref.cohort and other.protocol == ref.protocol
                and other.visit_id != ref.visit_id):
            wr = _xml_wr(other)
            if wr is not None:
                logger.info("%s: PRBS timing from %s", ref.visit_id, other.visit_id)
                return wr
    raise ValueError(f"{ref.visit_id}: no {ref.cohort} {ref.protocol} xml to take the PRBS timing from")


def _median_power(cpet: pd.DataFrame, seconds: np.ndarray, label: str) -> float:
    breath_s = np.floor(cpet["t"].to_numpy(dtype=float))
    power = cpet["Power"].to_numpy(dtype=float)
    selected = power[np.isin(breath_s, seconds) & np.isfinite(power)]
    if len(selected) == 0:
        raise ValueError(f"no measured power in the {label} seconds")
    return float(np.median(selected))


def rebuild_prbs_wr(template: pd.DataFrame, cpet: pd.DataFrame, rest_end_s: int, warmup_end_s: int) -> pd.DataFrame:
    """Template timing; rest 0 W, low 25 W, warm-up and high from the median measured power of this visit."""
    t = template["t"].to_numpy(dtype=np.int64)
    w = template["W"].to_numpy(dtype=float)
    exercise = t >= warmup_end_s
    low = w[exercise].min()
    if low != LUCAS_PRBS_LOW_W:
        raise ValueError(f"template PRBS low level {low} W, expected {LUCAS_PRBS_LOW_W} W")
    high_s = t[exercise & (w > low)]
    warmup_s = t[(t >= rest_end_s + PHASE_EDGE_SKIP_S) & (t < warmup_end_s - PHASE_EDGE_SKIP_S)]
    high_w = _median_power(cpet, high_s, "PRBS high")
    warmup_w = _median_power(cpet, warmup_s, "warm-up")
    rebuilt = np.where(t < rest_end_s, 0.0, warmup_w)
    rebuilt = np.where(exercise, np.where(w > low, high_w, LUCAS_PRBS_LOW_W), rebuilt)
    return pd.DataFrame({"t": t, "W": rebuilt})


def _protocol_wr(inp: PrepInput) -> pd.DataFrame | None:
    visit = inp.visit
    ref = visit.ref
    if visit.protocol.ok:
        return visit.protocol.data
    if visit.protocol.error is not None:
        raise ValueError(f"{ref.visit_id}: protocol WR failed to load: {visit.protocol.error}")
    sibling = _sibling_wr(ref, inp.refs)
    if sibling is not None:
        return sibling
    if ref.visit_id in REBUILT_PRBS_VISITS:
        if not visit.cpet.ok:
            raise ValueError(f"{ref.visit_id}: rebuilding PRBS WR needs the CPET measured power")
        template = _template_wr(ref, inp.refs)
        timeline = visit.timeline
        return rebuild_prbs_wr(template, visit.cpet.data.df, timeline.rest_end_s, timeline.warmup_end_s)
    logger.warning("%s: no protocol WR file; WR not present", ref.visit_id)
    return None


def build_wr(inp: PrepInput) -> GroupColumns:
    protocol_wr = _protocol_wr(inp)
    if protocol_wr is None:
        n = len(inp.grid.t)
        return GroupColumns(WR_GROUP, {WR_COLUMN: np.full(n, np.nan)}, np.zeros(n, dtype=bool))
    wr = _on_grid(protocol_wr, inp.grid.t)
    return GroupColumns(WR_GROUP, {WR_COLUMN: wr}, np.isfinite(wr))
