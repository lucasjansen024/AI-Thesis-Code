"""Display item 1 (paper Table 1): participant characteristics per cohort and pooled (METHODOLOGY §2.1).

Sources, one row per participant:
- sex, age, height: CPET header of the ramp visit; body mass and SMM (= skeletal-muscle % x weight): the Fitdays
  reading matched to the ramp visit. Both from `outputs/inventory/inventory_visits.csv` (candidate Ramp rows).
- peak V̇O₂, peak power, GET (V̇O₂ and W): the ramp analysis sheet `Ramp/MRT estimation <P>.xlsx` (the values that
  prescribed the intensities), cells checked against their labels before reading.

Files in `outputs/paper/`: `table1_participants.csv` (per participant, unrounded) and `table1.md` (mean ± SD).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from figures.style import COHORT_LABELS, PAPER_ROOT
from inventory.catalog import OUTPUT_ROOT as INVENTORY_ROOT
from inventory.catalog import RAW_ROOT
from inventory.report import VISITS_CSV

PARTICIPANTS_NAME, TABLE_NAME = "table1_participants.csv", "table1.md"
RAMP_PROTOCOL, CANDIDATE_ROLE = "Ramp", "candidate"
MRT_SHEET_PREFIX = "mrt estimation"
ML_PER_L = 1000.0
PERCENT = 100.0
# value cell -> (label cell, label); labels are compared case- and whitespace-insensitively
MRT_CELLS = {
    "get_ml_min": ("J21", ("J20", "GET VO2")),
    "peak_vo2_ml_min": ("K21", ("K20", "VO2max")),
    "peak_power_w": ("L21", ("L20", "Peak WR")),
    "get_w": ("E22", ("D22", "(W)")),
}
SEX_LABELS = {"Male": "men", "Female": "women"}
# (column, label, decimals) in table order; sex is formatted separately
TABLE_ROWS = (
    ("age_y", "Age (y)", 1),
    ("height_cm", "Height (cm)", 1),
    ("body_mass_kg", "Body mass (kg)", 1),
    ("smm_kg", "Skeletal muscle mass (kg)", 1),
    ("peak_vo2_l_min", "Peak V̇O₂ (L·min⁻¹)", 2),
    ("peak_vo2_ml_kg_min", "Peak V̇O₂ (mL·kg⁻¹·min⁻¹)", 1),
    ("peak_power_w", "Peak power (W)", 0),
    ("get_l_min", "GET (L·min⁻¹)", 2),
    ("get_w", "GET (W)", 0),
    ("get_pct_peak_vo2", "GET (% peak V̇O₂)", 0),
)


def _strip_unit(values: pd.Series, unit: str) -> pd.Series:
    text = values.astype(str).str.strip()
    if not text.str.endswith(unit).all():
        raise ValueError(f"expected every value to end in {unit!r}, got {text[~text.str.endswith(unit)].tolist()}")
    return pd.to_numeric(text.str.removesuffix(unit))


def read_anthropometrics(visits_csv: Path = INVENTORY_ROOT / VISITS_CSV) -> pd.DataFrame:
    visits = pd.read_csv(visits_csv)
    ramps = visits[(visits["protocol"] == RAMP_PROTOCOL) & (visits["role"] == CANDIDATE_ROLE)]
    repeated = ramps["participant_id"][ramps["participant_id"].duplicated()]
    if len(repeated):
        raise ValueError(f"more than one candidate Ramp for {repeated.tolist()}")
    weight = _strip_unit(ramps["weight"], "kg")
    return pd.DataFrame({
        "participant_id": ramps["participant_id"], "cohort": ramps["cohort"],
        "source_participant": ramps["source_participant"], "sex": ramps["sex"], "age_y": ramps["age"],
        "height_cm": ramps["height"], "body_mass_kg": weight,
        "smm_kg": _strip_unit(ramps["skeletal_muscle"], "%") / PERCENT * weight,
    }).set_index("participant_id")


def _normalised(label: object) -> str:
    return " ".join(str(label).split()).lower()


def _mrt_file(cohort: str, source_participant: str, raw_root: Path) -> Path:
    ramp_dir = raw_root / cohort / source_participant / RAMP_PROTOCOL
    files = [f for f in ramp_dir.glob("*.xlsx") if f.name.lower().startswith(MRT_SHEET_PREFIX)]  # G09: "Estimation_"
    if len(files) != 1:
        raise FileNotFoundError(f"{ramp_dir}: expected one 'MRT estimation' xlsx, found {[f.name for f in files]}")
    return files[0]


def read_ramp_sheet(path: Path) -> dict[str, float]:
    """Peak V̇O₂, peak power and GET from one ramp analysis sheet (cached values of its first worksheet)."""
    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        sheet = book.worksheets[0]
        values = {}
        for name, (cell, (label_cell, label)) in MRT_CELLS.items():
            found = sheet[label_cell].value
            if _normalised(found) != _normalised(label):
                raise ValueError(f"{path}: {label_cell} is {found!r}, expected {label!r}; the sheet layout differs")
            value = sheet[cell].value
            if not isinstance(value, (int, float)):
                raise ValueError(f"{path}: {cell} ({label}) is {value!r}, expected a number")
            values[name] = float(value)
    finally:
        book.close()
    return values


def read_characteristics(visits_csv: Path = INVENTORY_ROOT / VISITS_CSV, raw_root: Path = RAW_ROOT) -> pd.DataFrame:
    """One row per participant (index participant_id): cohort, sex and the numeric columns of TABLE_ROWS."""
    people = read_anthropometrics(visits_csv)
    sheets = pd.DataFrame.from_dict({pid: read_ramp_sheet(_mrt_file(row.cohort, row.source_participant, raw_root))
                                     for pid, row in people.iterrows()}, orient="index")
    return people.join(sheets).assign(
        peak_vo2_l_min=lambda d: d["peak_vo2_ml_min"] / ML_PER_L,
        peak_vo2_ml_kg_min=lambda d: d["peak_vo2_ml_min"] / d["body_mass_kg"],
        get_l_min=lambda d: d["get_ml_min"] / ML_PER_L,
        get_pct_peak_vo2=lambda d: d["get_ml_min"] / d["peak_vo2_ml_min"] * PERCENT,
    ).sort_index()


def _mean_sd(values: pd.Series, decimals: int) -> str:
    return f"{values.mean():.{decimals}f} ± {values.std(ddof=1):.{decimals}f}"


def _sex_counts(sex: pd.Series) -> str:
    unknown = set(sex) - set(SEX_LABELS)
    if unknown:
        raise ValueError(f"unexpected sex values {sorted(unknown)}")
    return "/".join(str(int((sex == value).sum())) for value in SEX_LABELS)


def format_table(characteristics: pd.DataFrame) -> str:
    """Markdown Table 1: one column per cohort, then pooled."""
    groups = [(f"{COHORT_LABELS[c]} (n = {len(g)})", g) for c, g in characteristics.groupby("cohort", sort=True)]
    groups.append((f"Pooled (n = {len(characteristics)})", characteristics))
    lines = ["| | " + " | ".join(name for name, _ in groups) + " |", "|---" * (len(groups) + 1) + "|",
             f"| Sex ({'/'.join(SEX_LABELS.values())}) | " + " | ".join(_sex_counts(g["sex"]) for _, g in groups) + " |"]
    lines += [f"| {label} | " + " | ".join(_mean_sd(g[column], decimals) for _, g in groups) + " |"
              for column, label, decimals in TABLE_ROWS]
    return "\n".join(lines) + "\n\nValues are mean ± SD.\n"


def write_table1(out_dir: Path = PAPER_ROOT) -> tuple[Path, Path]:
    """Table 1 → `<out_dir>/table1.md`, per-participant values → `<out_dir>/table1_participants.csv`."""
    characteristics = read_characteristics()
    out_dir.mkdir(parents=True, exist_ok=True)
    participants, table = out_dir / PARTICIPANTS_NAME, out_dir / TABLE_NAME
    characteristics.to_csv(participants)
    table.write_text(format_table(characteristics), encoding="utf-8")
    return participants, table
