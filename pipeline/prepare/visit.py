"""Assemble one processed 1 Hz table per candidate visit: IDs, t, phase, unit, features, presence flags.

Inclusion verdicts are never baked in here; `pipeline.inclusion` joins them at load time.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from inventory.annotations import annotation_for, load_annotations
from inventory.catalog import RAW_ROOT, VisitRef, enumerate_visits
from inventory.sensors import Loaded, VisitData, load_participant_fitdays, load_visit
from pipeline.prepare.anthro import build_anthro, candidate_smm
from pipeline.prepare.breath import build_vo2
from pipeline.prepare.grid import build_grid
from pipeline.prepare.group import GroupColumns, PrepInput, presence_column
from pipeline.prepare.hr import build_hr
from pipeline.prepare.nirs import build_nirs
from pipeline.prepare.settings import PrepSettings
from pipeline.prepare.vent import build_vent
from pipeline.prepare.wr import build_wr

logger = logging.getLogger(__name__)

CANDIDATE_ROLE = "candidate"
META_COLUMNS = ("t", "phase", "unit")
ID_FIELDS = ("participant_id", "cohort", "protocol", "visit_id", "source_visit")

# One builder per sensor group. Later domains append theirs (HR, VENT, NIRS, static); order fixes column order.
GROUP_BUILDERS: tuple[Callable[[PrepInput], GroupColumns], ...] = (build_vo2, build_wr, build_hr,
                                                                    build_vent, build_nirs, build_anthro)


@dataclass(frozen=True)
class ProcessedVisit:
    ids: dict[str, str]  # ID_FIELDS
    columns: dict[str, np.ndarray]  # META_COLUMNS, features, presence flags; all on the same grid


def prepare_visit(visit: VisitData, refs: tuple[VisitRef, ...], settings: PrepSettings,
                  hr_source: str = "", smm_kg: float = math.nan) -> ProcessedVisit:
    ref = visit.ref
    if not visit.cpet.ok and visit.timeline.exercise_end_s is None:
        raise ValueError(f"{ref.visit_id}: ramp end unknown without a CPET file ({visit.cpet.error or 'missing'})")
    grid = build_grid(visit.timeline)
    inp = PrepInput(visit, grid, settings, refs, hr_source, smm_kg)
    columns: dict[str, np.ndarray] = {"t": grid.t, "phase": grid.phase, "unit": grid.unit}
    for build in GROUP_BUILDERS:
        group = build(inp)
        duplicates = (set(group.values) | {presence_column(group.sensor_group)}) & set(columns)
        if duplicates:
            raise ValueError(f"{ref.visit_id}: builder {build.__name__} repeats columns {sorted(duplicates)}")
        columns = {**columns, **group.values, presence_column(group.sensor_group): group.present}
    ids = {field: getattr(ref, field) for field in ID_FIELDS}
    return ProcessedVisit(ids, columns)


def candidate_refs(raw_root: Path = RAW_ROOT) -> tuple[tuple[VisitRef, ...], tuple[VisitRef, ...]]:
    """(all visit folders, candidate visits only). Superseded and failed visits are never prepared."""
    refs = tuple(enumerate_visits(raw_root))
    return refs, tuple(ref for ref in refs if ref.role == CANDIDATE_ROLE)


def prepare_all(settings: PrepSettings, raw_root: Path = RAW_ROOT, *,
                hr_sources: Mapping[str, str]) -> list[ProcessedVisit]:
    """Every candidate visit of both cohorts, rebuilt from raw (~3 s per visit).

    `hr_sources`: visit_id -> HR source (`prepare.hr.read_hr_sources`); every candidate must be listed.
    """
    refs, candidates = candidate_refs(raw_root)
    unlisted = sorted(ref.visit_id for ref in candidates if ref.visit_id not in hr_sources)
    if unlisted:
        raise ValueError(f"candidate visits without HR rows in inclusion.csv: {unlisted}")
    annotations = load_annotations()
    participant_dirs = dict.fromkeys(ref.path.parent for ref in candidates)
    fitdays: dict[Path, Loaded] = {d: load_participant_fitdays(d) for d in participant_dirs}
    smm = candidate_smm(candidates, fitdays, settings.smm)  # before any visit: raises on an unmatched visit
    processed = []
    for number, ref in enumerate(candidates, start=1):
        logger.info("[%d/%d] preparing %s", number, len(candidates), ref.visit_id)
        visit = load_visit(ref, annotation_for(ref, annotations), fitdays[ref.path.parent])
        processed.append(prepare_visit(visit, refs, settings, hr_sources[ref.visit_id], smm[ref.visit_id]))
    return processed
