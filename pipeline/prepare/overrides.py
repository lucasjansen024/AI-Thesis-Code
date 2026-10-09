"""Per-visit preparation overrides: committed fixes for raw-data gaps the matching rules can't resolve.

`inventory/inclusion/prep_overrides.csv` (columns `visit_id, field, value, note`) is hand-edited. Fields:
- `fitdays_row`: the Fitdays reading to use for the visit's SMM, named by its exact date-time cell
  (e.g. `08:28 09/04/2026`), instead of the same-day match. For a test day without a reading, or with repeats.
- `age_from_visit`: take the HRpeak age from another visit's CPET header (same participant), for a visit whose
  header carries another person's profile.

Every other visit keeps the strict rules. Each row needs a note saying why. The overrides are part of the cache key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from inventory.build_inclusion import INCLUSION_DIR
from inventory.catalog import VisitRef

OVERRIDES_CSV = INCLUSION_DIR / "prep_overrides.csv"
OVERRIDE_COLUMNS = ["visit_id", "field", "value", "note"]
FITDAYS_ROW = "fitdays_row"
AGE_FROM_VISIT = "age_from_visit"
OVERRIDE_FIELDS = (FITDAYS_ROW, AGE_FROM_VISIT)


@dataclass(frozen=True)
class PrepOverrides:
    fitdays_rows: Mapping[str, str] = field(default_factory=dict)  # visit_id -> Fitdays date-time cell
    age_from: Mapping[str, str] = field(default_factory=dict)  # visit_id -> visit_id whose CPET age to use

    def as_key(self) -> dict[str, dict[str, str]]:
        """A JSON-ready, order-stable form for the cache key and meta."""
        return {FITDAYS_ROW: dict(sorted(self.fitdays_rows.items())),
                AGE_FROM_VISIT: dict(sorted(self.age_from.items()))}


NO_OVERRIDES = PrepOverrides()


def read_overrides(path: Path = OVERRIDES_CSV) -> PrepOverrides:
    """The override file, schema-checked. A missing file means no overrides."""
    if not path.exists():
        return NO_OVERRIDES
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(table.columns) != OVERRIDE_COLUMNS:
        raise ValueError(f"{path.name}: columns {list(table.columns)}, expected {OVERRIDE_COLUMNS}")
    table = table.apply(lambda column: column.str.strip())
    unknown = sorted(set(table["field"]) - set(OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(f"{path.name}: unknown fields {unknown} (allowed {list(OVERRIDE_FIELDS)})")
    blank = table[(table[OVERRIDE_COLUMNS] == "").any(axis=1)]
    if not blank.empty:
        raise ValueError(f"{path.name}: empty cells (every override needs a value and a note) in rows "
                         f"{(blank.index + 2).tolist()}")
    repeated = table[table.duplicated(["visit_id", "field"], keep=False)]
    if not repeated.empty:
        pairs = sorted(set(zip(repeated["visit_id"], repeated["field"])))
        raise ValueError(f"{path.name}: repeated visit_id/field pairs {pairs}")
    by_field = {name: dict(zip(rows["visit_id"], rows["value"])) for name, rows in table.groupby("field")}
    return PrepOverrides(by_field.get(FITDAYS_ROW, {}), by_field.get(AGE_FROM_VISIT, {}))


def check_overrides(overrides: PrepOverrides, refs: Iterable[VisitRef], candidates: Iterable[VisitRef]) -> None:
    """Every override names a candidate visit; an age source is another visit of the same participant."""
    by_id = {ref.visit_id: ref for ref in refs}
    candidate_ids = {ref.visit_id for ref in candidates}
    named = set(overrides.fitdays_rows) | set(overrides.age_from)
    not_candidates = sorted(named - candidate_ids)
    if not_candidates:
        raise ValueError(f"{OVERRIDES_CSV.name}: overrides for non-candidate visits {not_candidates}")
    for visit_id, source_id in sorted(overrides.age_from.items()):
        source = by_id.get(source_id)
        if source is None or source_id == visit_id:
            raise ValueError(f"{OVERRIDES_CSV.name}: {visit_id} takes its age from {source_id!r}, "
                             "which is not another visit")
        if source.participant_id != by_id[visit_id].participant_id:
            raise ValueError(f"{OVERRIDES_CSV.name}: {visit_id} takes its age from {source_id}, "
                             "a different participant")
