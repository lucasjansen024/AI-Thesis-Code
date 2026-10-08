"""ANTHRO domain (sensor group `static`): sex and SMM, constant within a visit.

- sex: CPET header ("Male"/"Female"), male = 1, female = 0; never scaled. Every column and output says "sex".
- SMM (kg) = Fitdays `Skeletal muscle` (%) / 100 x `Weight` (kg). Each candidate visit is matched to a Fitdays
  row by date: the row's date (Gies `17:29 Jul.24,2024`, Lucas `12:13 17/02/2026`) equals the CPET test date
  (header cell after "Test date", `31/07/2024`). Several participants were measured twice on one day, so
  same-day rows are narrowed by the Fitdays `Visit` label (= the raw visit folder) where the file has one; a
  single same-day row whose label names another visit fails the cross-check. No row, or no unique row, raises
  with the visit named.
- `smm="per_visit"`: the visit's own SMM. `"participant_mean"`: the mean of the per-visit SMM over the
  participant's candidate visits, so one SMM per participant.
- The static group is present on every grid second when both values exist, else on none.

SMM is resolved for all candidates before any visit is prepared (`candidate_smm`), because the participant mean
needs every visit of the participant; `build_anthro` reads the result from `PrepInput.smm_kg`.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from inventory.catalog import VisitRef, choose_cpet_file
from inventory.sensors import Loaded, VisitData
from pipeline.prepare.group import GroupColumns, PrepInput
from pipeline.prepare.settings import SMM_PARTICIPANT_MEAN

logger = logging.getLogger(__name__)

STATIC_GROUP = "static"
SEX_COLUMN, SMM_COLUMN = "sex", "SMM"
ANTHRO_COLUMNS = (SEX_COLUMN, SMM_COLUMN)
SEX_CODES = {"male": 1.0, "female": 0.0}
CPET_TEST_DATE_LABEL = "test date"
CPET_DATE_FORMAT = "%d/%m/%Y"
FITDAYS_DATE_FORMATS = ("%H:%M %b.%d,%Y", "%H:%M %d/%m/%Y")  # Gies (English app), Lucas (Dutch app)
FITDAYS_DATE_COLUMNS = ("Date", "Datum")
FITDAYS_WEIGHT_COLUMNS = ("Weight", "Gewicht")
FITDAYS_SKELETAL_COLUMNS = ("Skeletal muscle", "Skeletspier")
FITDAYS_VISIT_COLUMN = "Visit"
PERCENT = 100.0
LEADING_NUMBER = re.compile(r"\s*(-?\d+(?:[.,]\d+)?)")  # "81.90kg", "42.7%"; "- -" has none


@dataclass(frozen=True)
class SmmVisit:
    visit_id: str
    source_visit: str  # raw folder name, as in the Fitdays `Visit` label
    test_date: date


# --- parsing ---------------------------------------------------------------------------------------------------

def _as_date(value: object, formats: Sequence[str], what: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unreadable {what} {value!r} (formats {list(formats)})")


def parse_fitdays_date(value: object) -> date:
    return _as_date(value, FITDAYS_DATE_FORMATS, "Fitdays date")


def fitdays_number(value: object) -> float:
    """A Fitdays cell as a number: units ("kg", "%") stripped; NaN for "- -" or empty."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = LEADING_NUMBER.match(str(value))
    return float(match.group(1).replace(",", ".")) if match else math.nan


def read_cpet_test_date(path: Path) -> date:
    """The CPET export's test date: the cell right of "Test date" in the first row."""
    first_row = pd.read_excel(path, header=None, nrows=1).iloc[0].tolist()
    labels = [str(cell).strip().lower() for cell in first_row]
    if CPET_TEST_DATE_LABEL not in labels or labels.index(CPET_TEST_DATE_LABEL) + 1 >= len(first_row):
        raise ValueError(f"{path.name}: no '{CPET_TEST_DATE_LABEL}' value in the first row")
    return _as_date(first_row[labels.index(CPET_TEST_DATE_LABEL) + 1], (CPET_DATE_FORMAT,), "CPET test date")


def visit_sex(visit: VisitData) -> float:
    """1.0 male, 0.0 female, from this visit's CPET header."""
    sex = visit.cpet.data.header.get("sex") if visit.cpet.ok else None
    code = SEX_CODES.get(str(sex).strip().lower())
    if code is None:
        raise ValueError(f"{visit.ref.visit_id}: unknown sex {sex!r} in the CPET header")
    return code


# --- matching --------------------------------------------------------------------------------------------------

def _column(fitdays: pd.DataFrame, names: Sequence[str]) -> str:
    found = next((name for name in names if name in fitdays.columns), None)
    if found is None:
        raise ValueError(f"Fitdays has none of the columns {list(names)} (has {list(fitdays.columns)})")
    return found


def _labels(rows: pd.DataFrame) -> pd.Series:
    return rows[FITDAYS_VISIT_COLUMN].fillna("").astype(str).str.strip()


def match_fitdays_row(fitdays: pd.DataFrame, visit: SmmVisit) -> pd.Series:
    """The visit's Fitdays row: same date as the CPET test, narrowed or cross-checked by the `Visit` label."""
    raw_dates = fitdays[_column(fitdays, FITDAYS_DATE_COLUMNS)]
    dates = raw_dates.map(lambda value: parse_fitdays_date(value) if pd.notna(value) else None)
    same_day = fitdays[dates == visit.test_date]
    has_labels = FITDAYS_VISIT_COLUMN in fitdays.columns
    if len(same_day) > 1 and has_labels:
        same_day = same_day[_labels(same_day) == visit.source_visit]
    if len(same_day) != 1:
        problem = "no Fitdays row" if same_day.empty else f"{len(same_day)} Fitdays rows, no unique Visit label"
        raise ValueError(f"{visit.visit_id}: {problem} on the CPET test date {visit.test_date.isoformat()}")
    row = same_day.iloc[0]
    if has_labels:
        label = _labels(same_day).iloc[0]
        if label and label != visit.source_visit:
            raise ValueError(f"{visit.visit_id}: the Fitdays row of {visit.test_date.isoformat()} is labelled "
                             f"{label!r}, not {visit.source_visit!r}")
    return row


def fitdays_smm_kg(fitdays: pd.DataFrame, visit: SmmVisit) -> float:
    row = match_fitdays_row(fitdays, visit)
    skeletal_pct = fitdays_number(row[_column(fitdays, FITDAYS_SKELETAL_COLUMNS)])
    weight_kg = fitdays_number(row[_column(fitdays, FITDAYS_WEIGHT_COLUMNS)])
    smm = skeletal_pct / PERCENT * weight_kg
    if not (math.isfinite(smm) and smm > 0):
        raise ValueError(f"{visit.visit_id}: no SMM from Fitdays (skeletal muscle {skeletal_pct} %, "
                         f"weight {weight_kg} kg)")
    return smm


def participant_smm(fitdays: pd.DataFrame, visits: Sequence[SmmVisit], mode: str) -> dict[str, float]:
    """visit_id -> SMM (kg) for one participant's visits: per visit, or their mean (`participant_mean`)."""
    per_visit = {v.visit_id: fitdays_smm_kg(fitdays, v) for v in visits}
    if mode != SMM_PARTICIPANT_MEAN:
        return per_visit
    mean = float(np.mean(list(per_visit.values())))
    return {visit_id: mean for visit_id in per_visit}


def candidate_smm(candidates: Sequence[VisitRef], fitdays: Mapping[Path, Loaded], mode: str) -> dict[str, float]:
    """visit_id -> SMM (kg) for every candidate (reads each CPET test date). Raises on the first unmatched visit."""
    by_participant: dict[Path, list[SmmVisit]] = defaultdict(list)
    for ref in candidates:
        cpet_path = choose_cpet_file(ref.path).selected
        if cpet_path is None:
            raise ValueError(f"{ref.visit_id}: no CPET file, so no test date to match Fitdays")
        by_participant[ref.path.parent].append(SmmVisit(ref.visit_id, ref.source_visit,
                                                        read_cpet_test_date(cpet_path)))
    smm: dict[str, float] = {}
    for participant_dir, visits in by_participant.items():
        loaded = fitdays[participant_dir]
        if not loaded.ok:
            raise ValueError(f"{participant_dir.name}: no Fitdays data ({loaded.error or 'file missing'}) for "
                             f"{[v.visit_id for v in visits]}")
        smm = {**smm, **participant_smm(loaded.data, visits, mode)}
    return smm


# --- builder ---------------------------------------------------------------------------------------------------

def build_anthro(inp: PrepInput) -> GroupColumns:
    n = len(inp.grid.t)
    visit_id = inp.visit.ref.visit_id
    if not inp.visit.cpet.ok or not math.isfinite(inp.smm_kg):
        logger.warning("%s: %s; static group absent", visit_id,
                       "no CPET header" if not inp.visit.cpet.ok else "no SMM resolved")
        return GroupColumns(STATIC_GROUP, {c: np.full(n, np.nan) for c in ANTHRO_COLUMNS}, np.zeros(n, dtype=bool))
    values = {SEX_COLUMN: np.full(n, visit_sex(inp.visit)), SMM_COLUMN: np.full(n, inp.smm_kg)}
    return GroupColumns(STATIC_GROUP, values, np.ones(n, dtype=bool))
