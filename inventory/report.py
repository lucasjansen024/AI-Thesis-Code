"""Write the inventory CSVs and a markdown summary with the ticket's cross-checks."""
from __future__ import annotations

import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from inventory.annotations import EXPECTED_EXCLUSIONS, NIRS_SECOND_HALF
from inventory.metrics import COVERAGE_FLAG, HR_MIN_SPAN_BPM, HR_MIN_VO2_R, hr_is_plausible
from inventory.sensors import VisitData

Identity = tuple[str, str]  # (dob, sex) as text; kept in memory only, never written

VISITS_CSV = "inventory_visits.csv"
SIGNALS_CSV = "inventory_signals.csv"
SUMMARY_MD = "summary.md"
OBSERVED_MISSING_COVERAGE = 0.5
REPEATED_PAIRS = (("G08_Medium_superseded", "G08_Medium"), ("G09_Hard_superseded", "G09_Hard"))
REPEATED_COLUMNS = ("cpet_vo2", "tymewear_hr", "nirs_tsi")
PRESENCE_CSV = "inventory_presence.csv"
PRESENCE_MD = "presence.md"
PRESENCE_SIGNALS = {  # presence column -> signal row
    "cpet_vo2": "cpet_vo2",
    "tymewear_hr": "tymewear_hr",
    "cpet_hr": "cpet_hr",
    "tymewear_ve": "tymewear_ve",
    "tymewear_br": "tymewear_br",
    "nirs": "nirs_tsi",
    "protocol_wr": "protocol_wr_w",
}


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _coverage(signals: pd.DataFrame, visit_id: str, signal: str) -> float:
    hit = signals[(signals["visit_id"] == visit_id) & (signals["signal"] == signal)]
    if hit.empty or "exercise_coverage" not in hit.columns:
        return float("nan")
    return float(hit["exercise_coverage"].iloc[0])


def _missing_section(visits: pd.DataFrame) -> list[str]:
    rows = [[v.visit_id, v.role, v.flags or "—"] for v in visits.itertuples()]
    return ["## Missing data per visit", "", *_table(rows, ["visit", "role", "flags"]), ""]


def _gies_section(visits: pd.DataFrame, signals: pd.DataFrame) -> list[str]:
    gies = visits[(visits["cohort"] == "Gies") & (visits["role"] == "candidate")]
    rows = []
    for v in gies.itertuples():
        tw_cov = _coverage(signals, v.visit_id, "tymewear_hr")
        observed_tw = "missing" if not tw_cov >= OBSERVED_MISSING_COVERAGE else f"{tw_cov:.0%}"
        if v.expected_exclusions_handoff or v.annotation_exclusions:
            agree = "yes" if v.expected_exclusions_handoff == v.annotation_exclusions else "**no**"
            rows.append([v.visit_id, v.expected_exclusions_handoff or "—", v.annotation_exclusions or "—",
                         agree, observed_tw])
    present = set(zip(gies["participant_id"], gies["protocol"]))
    in_run = set(gies["participant_id"])  # a --visits subset must not list every other participant's visits
    absent = [f"{pid} {protocol}" for pid, protocol in EXPECTED_EXCLUSIONS
              if pid in in_run and (pid, protocol) not in present]
    blocking = gies["annotation_exclusions"].fillna("").apply(
        lambda s: bool(s) and s != NIRS_SECOND_HALF)
    lines = ["### Gies: handoff §2.8 vs annotations vs files", "",
             *_table(rows, ["visit", "handoff", "annotation", "agree", "Tymewear HR coverage"]), "",
             f"- Expected exclusions with no visit folder in this run: {', '.join(absent) or 'none'}",
             f"- Candidates in this run: {len(gies)}; included by annotations: {int((~blocking).sum())} "
             "(charting predicted 44 of 55 for the full run)", ""]
    return lines


def _lucas_section(visits: pd.DataFrame) -> list[str]:
    lucas = visits[visits["cohort"] == "Lucas"]
    flagged = lucas[lucas["flags"].fillna("") != ""]
    lines = ["### Lucas: are all visits valid?", "", f"- {len(lucas)} visits in this run, {len(flagged)} with flags:"]
    lines += [f"  - {v.visit_id}: {v.flags}" for v in flagged.itertuples()]
    return [*lines, ""]


def _repeated_section(visits: pd.DataFrame, signals: pd.DataFrame) -> list[str]:
    rows = []
    for pair in REPEATED_PAIRS:
        for visit_id in pair:
            hit = visits[visits["visit_id"] == visit_id]
            if hit.empty:
                continue
            v = hit.iloc[0]
            covs = [f"{_coverage(signals, visit_id, s):.0%}" for s in REPEATED_COLUMNS]
            rows.append([visit_id, v.source_visit, v.exercise_duration_s, *covs])
    header = ["visit", "source", "exercise s", *REPEATED_COLUMNS]
    return ["### Repeated visits (expected complete: Medium2, Hard2)", "", *_table(rows, header), ""]


def _hr_cell(signals: pd.DataFrame, visit_id: str, signal: str) -> str:
    hit = signals[(signals["visit_id"] == visit_id) & (signals["signal"] == signal)]
    if hit.empty or not hit["exercise_coverage"].iloc[0] > 0:
        return "missing"
    row = hit.iloc[0].to_dict()
    verdict = "ok" if hr_is_plausible(row) else "**not representative**"
    return f"{row['exercise_coverage']:.0%}, span {row['hr_span_bpm']:.0f}, r {row['hr_vo2_r']:.2f}: {verdict}"


def _hr_section(visits: pd.DataFrame, signals: pd.DataFrame) -> list[str]:
    """Visits whose Tymewear HR is not usable, with the CPET HR backup next to it."""
    rows = []
    for v in visits.itertuples():
        tymewear = _hr_cell(signals, v.visit_id, "tymewear_hr")
        if tymewear.endswith(": ok") and _coverage(signals, v.visit_id, "tymewear_hr") >= COVERAGE_FLAG:
            continue
        rows.append([v.visit_id, v.hr_source, tymewear, _hr_cell(signals, v.visit_id, "cpet_hr")])
    header = ["visit", "annotated HR_source", "Tymewear HR (coverage, span bpm, r vs VO2)", "CPET HR backup"]
    return [f"### HR sources (span < {HR_MIN_SPAN_BPM:.0f} bpm or r < {HR_MIN_VO2_R} = not representative)", "",
            *_table(rows, header), ""]


def _presence_cell(signals: pd.DataFrame, visit_id: str, signal: str) -> str:
    hit = signals[(signals["visit_id"] == visit_id) & (signals["signal"] == signal)]
    if hit.empty:
        return "missing"
    row = hit.iloc[0].to_dict()
    if signal == "protocol_wr_w":
        return "present" if row["loaded"] else "missing"
    coverage = row.get("exercise_coverage", float("nan"))
    if not coverage > 0:
        return "missing"
    if not hr_is_plausible(row):
        return "not representative"
    return "present" if coverage >= COVERAGE_FLAG else f"partial {coverage:.0%}"


def presence_table(visits: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Per visit: is each data source there (exercise phase)? Raw availability only, no annotations or rules."""
    rows = []
    for v in visits.itertuples():
        row = {"visit_id": v.visit_id, "role": v.role}
        row.update({name: _presence_cell(signals, v.visit_id, signal) for name, signal in PRESENCE_SIGNALS.items()})
        row["age_sex"] = "present" if pd.notna(v.age) and pd.notna(v.sex) else "missing"
        row["smm"] = "present" if pd.notna(v.skeletal_muscle) else "missing"
        rows.append(row)
    return pd.DataFrame(rows)


def identity(visit: VisitData) -> Identity | None:
    if not visit.cpet.ok:
        return None
    header = visit.cpet.data.header
    return str(header.get("dob")), str(header.get("sex"))


def _duplicate_section(people: dict[str, Identity]) -> list[str]:
    """Same DOB and sex across cohorts. DOB is compared in memory only and never written."""
    pairs = [
        f"{a} ~ {b}" for a, b in combinations(sorted(people), 2)
        if a[0] != b[0] and people[a] == people[b] and people[a][0] != "None"
    ]
    no_dob = sorted(pid for pid, (dob, _) in people.items() if dob == "None")
    return ["### Cross-cohort duplicate persons (same DOB and sex)", "",
            f"- Pairs: {', '.join(pairs) or 'none'}",
            f"- Participants without a DOB in the CPET header: {', '.join(no_dob) or 'none'}", ""]


def write_reports(
    people: dict[str, Identity], visit_rows: list[dict], signal_rows: list[dict], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    visits = pd.DataFrame(visit_rows)
    signals = pd.DataFrame(signal_rows)
    visits.to_csv(out_dir / VISITS_CSV, index=False)
    signals.to_csv(out_dir / SIGNALS_CSV, index=False)
    presence = presence_table(visits, signals)
    presence.to_csv(out_dir / PRESENCE_CSV, index=False)
    legend = (f"Exercise phase only. present = coverage >= {COVERAGE_FLAG:.0%}; partial = less; "
              "not representative = HR fails the span/VO2 check. No annotations or inclusion rules applied.")
    (out_dir / PRESENCE_MD).write_text(
        "\n".join(["# Data presence per visit", "", legend, "",
                   *_table(presence.astype(str).values.tolist(), list(presence.columns)), ""]),
        encoding="utf-8")
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# Data inventory summary ({stamp})", "",
             f"{len(visits)} visits: " + ", ".join(f"{'/'.join(k)} {n}" for k, n in visits.groupby(["cohort", "role"]).size().items()),
             "", *_missing_section(visits), "## Cross-checks", "",
             *_gies_section(visits, signals), *_lucas_section(visits), *_hr_section(visits, signals),
             *_repeated_section(visits, signals), *_duplicate_section(people)]
    (out_dir / SUMMARY_MD).write_text("\n".join(lines), encoding="utf-8")
