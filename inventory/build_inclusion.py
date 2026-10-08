"""Build the committed inclusion CSV: validity per visit x unit x sensor group.

Run from the repo root, after a fresh `python -m inventory`:

    python -m inventory.build_inclusion

Reads outputs/inventory/inventory_{visits,signals}.csv (layer 1: automatic checks) and the hand-edited
inventory/inclusion/nirs_review.csv (layer 2: the dev's NIRS review). Writes inventory/inclusion/inclusion.csv,
the single source of inclusion verdicts; the pipeline only reads it.
"""
from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from inventory.catalog import NOMINAL_TIMELINES, OUTPUT_ROOT

logger = logging.getLogger("inventory.inclusion")

INCLUSION_DIR = Path(__file__).resolve().parent / "inclusion"
INCLUSION_CSV = INCLUSION_DIR / "inclusion.csv"
REVIEW_CSV = INCLUSION_DIR / "nirs_review.csv"
VISITS_CSV = OUTPUT_ROOT / "inventory_visits.csv"
SIGNALS_CSV = OUTPUT_ROOT / "inventory_signals.csv"

# Layer-1 thresholds, named after the pipeline flags they become.
MAX_INTERP_GAP_S = 20  # max_interp_gap_s: a longer gap on raw timestamps invalidates the unit
HR_MIN_SPAN_BPM = 20.0  # hr_min_span_bpm: p95 - p5, rest to exercise end
HR_MIN_R_VO2 = 0.5  # hr_min_r_vo2: Pearson r vs smoothed VO2, rest to exercise end
HR_MIN_REST_COVERAGE = 0.8  # hr_min_rest_coverage: HRrest needs the rest phase (same level as the inventory flag)
NIRS_MIN_FIT_SHARE = 0.95  # nirs_min_fit_share: share of seconds with fit factor >= nirs_min_fit_factor
NIRS_FIT_SHARE_COLUMN = "ff_ge99"  # inventory column prefix for nirs_min_fit_factor = 0.99

CANDIDATE_ROLE = "candidate"  # superseded and failed visits never enter the CSV
VISIT_UNIT = "visit"
SENSOR_GROUPS = ("VO2", "WR", "HR", "VENT", "NIRS", "static")
REVIEWED_GROUP = "NIRS"  # the review list may only decide NIRS
HR_SOURCES = (("tymewear", "tymewear_hr"), ("cpet", "cpet_hr"))  # preference order
# Visits without their own protocol xml whose ideal WR is still defined (decided during data validity).
WR_WITHOUT_OWN_XML = {
    "G08_Medium": "no xml in Medium2; reuses the superseded Medium visit's xml (same CPET protocol)",
    "G09_Hard": "no xml in Hard2; reuses the superseded Hard visit's xml (same CPET protocol)",
    "L04_PRBS": "no xml; rebuilt from the shared PRBSV2 timing (25 W low, median measured high)",
}
REVIEW_COLUMNS = ["visit_id", "unit", "sensor_group", "valid", "note"]
REVIEW_BOOLEANS = {"true": True, "false": False}
OUTPUT_COLUMNS = ["participant_id", "cohort", "protocol", "visit_id", "unit", "sensor_group",
                  "valid", "reason", "decided_by", "hr_source"]
PER_REP_COLUMNS = ("exercise_coverage_rep1", "max_gap_s_rep1", "exercise_coverage_rep2", "max_gap_s_rep2")

Signals = dict[tuple[str, str], dict]  # (visit_id, signal) -> inventory signal row


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str = ""  # why invalid, or how a non-default valid verdict was reached
    decided_by: str = "rule"
    hr_source: str = ""


@dataclass(frozen=True)
class ReviewEntry:
    valid: bool
    note: str


class ReviewError(ValueError):
    """The review list is malformed."""


def units_of(cohort: str, protocol: str) -> tuple[str, ...]:
    reps = NOMINAL_TIMELINES[(cohort, protocol)].repetition_starts_s
    return tuple(f"rep{n}" for n in range(1, len(reps) + 1)) or (VISIT_UNIT,)


def _suffix(unit: str) -> str:
    return "" if unit == VISIT_UNIT else f"_{unit}"


def _data_problem(row: dict | None, unit: str, label: str) -> str | None:
    """Missing source or a gap longer than MAX_INTERP_GAP_S within the unit's exercise seconds."""
    if row is None or not row["loaded"] or not row[f"exercise_coverage{_suffix(unit)}"] > 0:
        return f"{label} missing"
    gap = row[f"max_gap_s{_suffix(unit)}"]
    if not gap <= MAX_INTERP_GAP_S:
        return f"{label} gap {gap:.0f} s > {MAX_INTERP_GAP_S} s"
    return None


def _verdict(problems: list[str | None]) -> Verdict:
    found = [p for p in problems if p]
    return Verdict(not found, "; ".join(found))


def _hr_source_problem(row: dict | None, label: str) -> str | None:
    """Per visit: is this HR source present and plausible? Rest is part of the check (needed for HRrest)."""
    if row is None or not row["loaded"] or not row["exercise_coverage"] > 0:
        return f"{label} missing"
    if not row["hr_span_bpm"] >= HR_MIN_SPAN_BPM:
        return f"{label} span {row['hr_span_bpm']:.0f} < {HR_MIN_SPAN_BPM:.0f} bpm"
    if not row["hr_vo2_r"] >= HR_MIN_R_VO2:
        return f"{label} r vs VO2 {row['hr_vo2_r']:.2f} < {HR_MIN_R_VO2}"
    if not row["rest_coverage"] >= HR_MIN_REST_COVERAGE:
        return f"{label} rest coverage {row['rest_coverage']:.0%} < {HR_MIN_REST_COVERAGE:.0%}"
    return None


def _hr_verdict(signals: Signals, visit_id: str, unit: str) -> Verdict:
    """The first plausible source (Tymewear, else CPET) is the visit's HR; the unit then needs data from it."""
    rejected = []
    for source, signal in HR_SOURCES:
        row = signals.get((visit_id, signal))
        problem = _hr_source_problem(row, signal)
        if problem is not None:
            rejected.append(problem)
            continue
        unit_problem = _data_problem(row, unit, signal)
        if unit_problem is not None:
            return Verdict(False, "; ".join([*rejected, unit_problem]), hr_source=source)
        note = f"{'; '.join(rejected)}; using {signal}" if rejected else ""
        return Verdict(True, note, hr_source=source)
    return Verdict(False, "; ".join(rejected))


def _wr_verdict(signals: Signals, visit_id: str) -> Verdict:
    row = signals.get((visit_id, "protocol_wr_w"))
    if row is not None and row["loaded"]:
        return Verdict(True)
    if visit_id in WR_WITHOUT_OWN_XML:
        return Verdict(True, WR_WITHOUT_OWN_XML[visit_id])
    return Verdict(False, "protocol WR missing")


def _fit_problem(row: dict | None, unit: str) -> str | None:
    if row is None:  # NIRS not loaded: already reported as missing
        return None
    share = row[f"{NIRS_FIT_SHARE_COLUMN}_{'exercise' if unit == VISIT_UNIT else unit}"]
    if not share >= NIRS_MIN_FIT_SHARE:
        return f"fit factor >= 99 % in {share:.0%} < {NIRS_MIN_FIT_SHARE:.0%} of seconds"
    return None


def _static_verdict(visit: dict) -> Verdict:
    return _verdict([None if pd.notna(visit["sex"]) else "sex missing",
                     None if pd.notna(visit["skeletal_muscle"]) else "SMM missing"])


def rule_verdict(signals: Signals, visit: dict, unit: str, group: str) -> Verdict:
    """Layer 1 for one unit and sensor group."""
    visit_id = visit["visit_id"]

    def problem(signal: str) -> str | None:
        return _data_problem(signals.get((visit_id, signal)), unit, signal)

    if group == "VO2":
        return _verdict([problem("cpet_vo2")])
    if group == "WR":
        return _wr_verdict(signals, visit_id)
    if group == "HR":
        return _hr_verdict(signals, visit_id, unit)
    if group == "VENT":
        return _verdict([problem("tymewear_ve"), problem("tymewear_br")])
    if group == "NIRS":
        fit_row = signals.get((visit_id, "nirs_fit_factor"))
        return _verdict([problem("nirs_tsi"), problem("nirs_hbdiff"), _fit_problem(fit_row, unit)])
    if group == "static":
        return _static_verdict(visit)
    raise ValueError(f"unknown sensor group {group!r}")


def _field_count_errors(path: Path) -> list[str]:
    """pandas guesses an index (or with index_col=False drops trailing fields) when a row is too long."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    return [
        f"line {number}: {len(row)} fields, expected {len(REVIEW_COLUMNS)} (quote a note that contains a comma)"
        for number, row in enumerate(rows, start=1)
        if row and len(row) != len(REVIEW_COLUMNS)
    ]


def load_review(path: Path, units: set[tuple[str, str]]) -> dict[tuple[str, str], ReviewEntry]:
    """(visit_id, unit) -> the dev's NIRS call. Every entry must name an existing candidate unit, once."""
    field_errors = _field_count_errors(path)
    if field_errors:
        raise ReviewError(f"{path.name}: " + " | ".join(field_errors))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
    if list(df.columns) != REVIEW_COLUMNS:
        raise ReviewError(f"{path.name}: columns {list(df.columns)}, expected {REVIEW_COLUMNS}")
    entries, errors = {}, []
    for number, row in enumerate(df.to_dict("records"), start=2):  # line 1 is the header
        key = (row["visit_id"], row["unit"])
        valid = REVIEW_BOOLEANS.get(row["valid"].strip().lower())
        if row["sensor_group"] != REVIEWED_GROUP:
            errors.append(f"line {number}: sensor_group {row['sensor_group']!r}, only {REVIEWED_GROUP} is reviewed")
        elif key not in units:
            errors.append(f"line {number}: {key} is not a candidate visit unit")
        elif valid is None:
            errors.append(f"line {number}: valid {row['valid']!r} is not true/false")
        elif key in entries:
            errors.append(f"line {number}: {key} listed twice")
        else:
            entries[key] = ReviewEntry(valid, row["note"])
    if errors:
        raise ReviewError(f"{path.name}: " + " | ".join(errors))
    return entries


def apply_review(key: tuple[str, str], rule: Verdict, review: ReviewEntry) -> Verdict:
    """The dev's exclusions always win for NIRS; an include never overrides a layer-1 failure."""
    if review.valid and not rule.valid:
        logger.warning("%s: review includes NIRS but layer 1 fails it (%s); layer 1 wins", key, rule.reason)
        return replace(rule, reason=f"{rule.reason}; review include overridden ({review.note})")
    reason = review.note if rule.valid else f"{rule.reason}; {review.note}"
    return replace(rule, valid=review.valid, reason=reason, decided_by="review")


def build_table(visits: pd.DataFrame, signals: Signals, review: dict[tuple[str, str], ReviewEntry]) -> pd.DataFrame:
    rows = []
    for visit in visits.to_dict("records"):
        for unit in units_of(visit["cohort"], visit["protocol"]):
            for group in SENSOR_GROUPS:
                verdict = rule_verdict(signals, visit, unit, group)
                key = (visit["visit_id"], unit)
                if group == REVIEWED_GROUP and key in review:
                    verdict = apply_review(key, verdict, review[key])
                rows.append({**{k: visit[k] for k in ("participant_id", "cohort", "protocol", "visit_id")},
                             "unit": unit, "sensor_group": group, "valid": verdict.valid, "reason": verdict.reason,
                             "decided_by": verdict.decided_by,
                             "hr_source": verdict.hr_source if group == "HR" else ""})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _status(rows: pd.DataFrame) -> str:
    """'ok', 'no', or the valid units, e.g. 'rep1 only'."""
    valid_units = rows.loc[rows["valid"], "unit"].tolist()
    if len(valid_units) == len(rows):
        return "ok"
    return f"{'+'.join(valid_units)} only" if valid_units else "no"


def summarise(table: pd.DataFrame) -> list[str]:
    """Shared-mask counts per cohort and the per-group status of every visit with an invalid unit."""
    lines = []
    for cohort, rows in table.groupby("cohort", sort=False):
        unit_ok = rows.groupby(["visit_id", "unit"], sort=False)["valid"].all()
        visit_ok = unit_ok.groupby(level="visit_id", sort=False)
        included, complete = visit_ok.any(), visit_ok.all()
        lines.append(f"{cohort}: shared mask includes {int(included.sum())} of {len(included)} visits")
        for visit_id in included[included & ~complete].index:
            kept = unit_ok[visit_id]
            lines.append(f"  partial: {visit_id} keeps {'+'.join(kept[kept].index)} only")
        lines.append(f"  excluded: {', '.join(included[~included].index) or 'none'}")
    lines.append("Per-group status of visits with any invalid unit (per_config):")
    for visit_id, rows in table.groupby("visit_id", sort=False):
        if rows["valid"].all():
            continue
        groups = rows.groupby("sensor_group", sort=False)
        lines.append(f"  {visit_id}: " + " | ".join(f"{group} {_status(g)}" for group, g in groups))
    return lines


def _read_inventory() -> tuple[pd.DataFrame, Signals]:
    visits = pd.read_csv(VISITS_CSV)
    signals = pd.read_csv(SIGNALS_CSV)
    stale = [c for c in PER_REP_COLUMNS if c not in signals.columns]
    if stale:
        raise SystemExit(f"{SIGNALS_CSV} lacks {stale}: re-run `python -m inventory` first")
    lookup = {(row["visit_id"], row["signal"]): row for row in signals.to_dict("records")}
    if len(lookup) != len(signals):
        raise SystemExit(f"{SIGNALS_CSV} has duplicate (visit_id, signal) rows")
    return visits, lookup


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    visits, signals = _read_inventory()
    candidates = visits[visits["role"] == CANDIDATE_ROLE]
    dropped = visits.loc[visits["role"] != CANDIDATE_ROLE, "visit_id"]
    logger.info("Not candidates, left out: %s", ", ".join(dropped) or "none")
    units = {(v, u) for v, c, p in zip(candidates["visit_id"], candidates["cohort"], candidates["protocol"])
             for u in units_of(c, p)}
    table = build_table(candidates, signals, load_review(REVIEW_CSV, units))
    table.to_csv(INCLUSION_CSV, index=False)
    logger.info("Wrote %s (%d rows)", INCLUSION_CSV, len(table))
    for line in summarise(table):
        logger.info(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
