"""Enumerate the visits of both cohorts and pick each sensor's raw file.

File selection is explicit here instead of the legacy first-match lookup
(`Utils.get_file`), so the inventory can report which files were ignored and why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "inventory"
REFERENCE_PREPARED_ROOTS = {
    "Gies": REPO_ROOT / "data" / "prepared_gies2024",
    "Lucas": REPO_ROOT / "data" / "prepared",
}
COHORT_PREFIX = {"Gies": "G", "Lucas": "L"}
PROTOCOL_ORDER = ("Ramp", "Easy", "Medium", "Hard", "PRBS", "Interval", "Step")

# Repeated / failed Gies visits as settled during charting; the inventory verifies them.
SUPERSEDED_VISITS = {("Gies", "P8", "Medium"), ("Gies", "P9", "Hard")}
FAILED_VISITS = {("Gies", "P13", "FailedRamp")}

LOCK_FILE_PREFIX = "~$"
TEN_SECOND_EXPORT = re.compile(r"(?<!\d)10[ _]?s(ec)?(?![a-z])", re.IGNORECASE)
NIRS_SUFFIXES = {".xlsx", ".txt", ".oxy4", ".oxy5"}
TYMEWEAR_SUFFIXES = {".csv", ".fit", ".xlsx", ".xls"}

RAMP_REST_END_S = 120
RAMP_WARMUP_END_S = 360  # 2 min rest + 4 min at 30 W (both cohorts)
SUBMAX_REST_END_S = 120
SUBMAX_WARMUP_END_S = 330  # 2 min rest + 3.5 min warm-up
GIES_REPETITION_S = 960  # 32 PRBS units of 30 s
LUCAS_SEGMENT_HALF_S = 930  # 31 PRBS units of 30 s


@dataclass(frozen=True)
class Timeline:
    """Nominal protocol timeline in CPET seconds. `exercise_end_s` is None for ramps."""

    rest_end_s: int
    warmup_end_s: int
    exercise_end_s: int | None
    repetition_starts_s: tuple[int, ...] = ()


GIES_PRBS_TIMELINE = Timeline(
    SUBMAX_REST_END_S,
    SUBMAX_WARMUP_END_S,
    SUBMAX_WARMUP_END_S + 2 * GIES_REPETITION_S,
    (SUBMAX_WARMUP_END_S, SUBMAX_WARMUP_END_S + GIES_REPETITION_S),
)
LUCAS_SUBMAX_END_S = SUBMAX_WARMUP_END_S + 2 * LUCAS_SEGMENT_HALF_S
NOMINAL_TIMELINES = {
    ("Gies", "Ramp"): Timeline(RAMP_REST_END_S, RAMP_WARMUP_END_S, None),
    ("Gies", "Easy"): GIES_PRBS_TIMELINE,
    ("Gies", "Medium"): GIES_PRBS_TIMELINE,
    ("Gies", "Hard"): GIES_PRBS_TIMELINE,
    ("Lucas", "Ramp"): Timeline(RAMP_REST_END_S, RAMP_WARMUP_END_S, None),
    ("Lucas", "PRBS"): Timeline(
        SUBMAX_REST_END_S,
        SUBMAX_WARMUP_END_S,
        LUCAS_SUBMAX_END_S,
        (SUBMAX_WARMUP_END_S, SUBMAX_WARMUP_END_S + LUCAS_SEGMENT_HALF_S),
    ),
    ("Lucas", "Interval"): Timeline(SUBMAX_REST_END_S, SUBMAX_WARMUP_END_S, LUCAS_SUBMAX_END_S),
    ("Lucas", "Step"): Timeline(SUBMAX_REST_END_S, SUBMAX_WARMUP_END_S, LUCAS_SUBMAX_END_S),
}


@dataclass(frozen=True)
class VisitRef:
    cohort: str
    source_participant: str  # raw folder name, e.g. "P8"
    source_visit: str  # raw folder name, e.g. "Medium2"
    participant_id: str  # e.g. "G08"
    protocol: str  # e.g. "Medium"
    visit_id: str  # e.g. "G08_Medium"
    role: str  # "candidate" | "superseded" | "failed"
    path: Path


@dataclass(frozen=True)
class FileChoice:
    selected: Path | None
    ignored: tuple[str, ...]
    problem: str | None


def participant_id(cohort: str, source_participant: str) -> str:
    number = int(re.sub(r"\D", "", source_participant))
    return f"{COHORT_PREFIX[cohort]}{number:02d}"


def protocol_of(source_visit: str) -> str:
    return re.sub(r"^Failed|\d+$", "", source_visit)


def _role(cohort: str, source_participant: str, source_visit: str) -> str:
    key = (cohort, source_participant, source_visit)
    if key in FAILED_VISITS:
        return "failed"
    if key in SUPERSEDED_VISITS:
        return "superseded"
    return "candidate"


def _visit_ref(cohort: str, participant_dir: Path, visit_dir: Path) -> VisitRef:
    pid = participant_id(cohort, participant_dir.name)
    protocol = protocol_of(visit_dir.name)
    role = _role(cohort, participant_dir.name, visit_dir.name)
    suffix = "" if role == "candidate" else f"_{role}"
    return VisitRef(
        cohort=cohort,
        source_participant=participant_dir.name,
        source_visit=visit_dir.name,
        participant_id=pid,
        protocol=protocol,
        visit_id=f"{pid}_{protocol}{suffix}",
        role=role,
        path=visit_dir,
    )


def _sort_key(ref: VisitRef) -> tuple:
    protocol_rank = PROTOCOL_ORDER.index(ref.protocol) if ref.protocol in PROTOCOL_ORDER else len(PROTOCOL_ORDER)
    return (ref.cohort, ref.participant_id, protocol_rank, ref.role != "candidate", ref.source_visit)


def enumerate_visits(raw_root: Path = RAW_ROOT) -> list[VisitRef]:
    refs = []
    for cohort in COHORT_PREFIX:
        for participant_dir in sorted((raw_root / cohort).iterdir()):
            if not participant_dir.is_dir() or not re.fullmatch(r"P\d+", participant_dir.name):
                continue
            refs.extend(
                _visit_ref(cohort, participant_dir, visit_dir)
                for visit_dir in sorted(participant_dir.iterdir())
                if visit_dir.is_dir()
            )
    return sorted(refs, key=_sort_key)


def _choose(candidates: list[Path], is_preferred: Callable[[Path], bool]) -> FileChoice:
    preferred = sorted((c for c in candidates if is_preferred(c)), key=lambda p: p.name)
    ignored = tuple(sorted(c.name for c in candidates if c not in preferred[:1]))
    if not preferred:
        problem = "no usable file" if not candidates else "only ignored variants present"
        return FileChoice(None, ignored, problem)
    problem = None
    if len(preferred) > 1:
        problem = f"ambiguous, took first of {len(preferred)}"
    return FileChoice(preferred[0], ignored, problem)


def _is_lock(path: Path) -> bool:
    return path.name.startswith(LOCK_FILE_PREFIX)


def _files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file()]


def choose_cpet_file(visit_dir: Path) -> FileChoice:
    candidates = [
        p for p in _files(visit_dir)
        if p.suffix.lower() in {".xlsx", ".xls"}
        and ("BxB" in p.name or "CPET" in p.name or TEN_SECOND_EXPORT.search(p.name))
    ]
    return _choose(
        candidates,
        lambda p: not _is_lock(p) and "BxB" in p.name and not TEN_SECOND_EXPORT.search(p.name),
    )


def choose_nirs_file(visit_dir: Path) -> FileChoice:
    candidates = [
        p for p in _files(visit_dir)
        if p.suffix.lower() in NIRS_SUFFIXES
        and ("Oxy" in p.name or "_NIRS" in p.name or p.suffix.lower() in {".oxy4", ".oxy5"})
    ]
    return _choose(candidates, lambda p: not _is_lock(p) and p.suffix.lower() == ".xlsx")


def choose_tymewear_file(visit_dir: Path) -> FileChoice:
    candidates = [
        p for p in _files(visit_dir)
        if p.suffix.lower() in TYMEWEAR_SUFFIXES
        and "VE" in p.name
        and not any(k in p.name for k in ("BxB", "CPET", "Oxy", "_NIRS"))
    ]
    return _choose(candidates, lambda p: not _is_lock(p) and p.suffix.lower() == ".csv")


def choose_protocol_xml(visit_dir: Path, protocol: str) -> FileChoice:
    """Ergometer protocol export. Lucas PRBS folders also hold a 'PRBSV2 STEP' variant."""
    candidates = [p for p in _files(visit_dir) if p.suffix.lower() == ".xml"]

    def is_preferred(p: Path) -> bool:
        name = p.name.lower()
        is_other_step = "step" in name and protocol != "Step"
        return protocol.lower() in name and not is_other_step

    return _choose(candidates, is_preferred)


def choose_fitdays_file(participant_dir: Path) -> FileChoice:
    candidates = [p for p in _files(participant_dir) if p.name.lower().startswith("fitdays")]
    return _choose(candidates, lambda p: not _is_lock(p))


def reference_prepared_file(ref: VisitRef, kind: str) -> Path:
    """The legacy preparer's output for this visit, e.g. kind='WR' -> P11HardWR.csv."""
    root = REFERENCE_PREPARED_ROOTS[ref.cohort]
    name = f"{ref.source_participant}{ref.source_visit}{kind}.csv"
    return root / ref.source_participant / ref.source_visit / name
