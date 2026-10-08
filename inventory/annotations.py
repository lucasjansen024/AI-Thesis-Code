"""Per-visit annotations of both cohorts and the exclusions the handoff expects.

Gies: `docs/Gies/config.py` (VE_lag, HR_source, NIRS_repetition, notes).
Lucas: `config_lucas` (HR_source, NIRS_repetition, ramp_rate); no lag, Tymewear is on the CPET clock.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

from inventory.catalog import REPO_ROOT, VisitRef, participant_id
from src2_training_ready.preparation.config import config_lucas

GIES_CONFIG_PATH = REPO_ROOT / "docs" / "Gies" / "config.py"

TYMEWEAR = "tymewear missing"
NIRS_WHOLE = "NIRS missing, whole visit"
NIRS_SECOND_HALF = "NIRS missing, second half"

# Handoff §2.8, keyed by (participant_id, protocol).
EXPECTED_EXCLUSIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("G06", "Ramp"): (TYMEWEAR,),
    ("G07", "Ramp"): (TYMEWEAR, NIRS_WHOLE),
    ("G07", "Medium"): (NIRS_WHOLE,),
    ("G07", "Hard"): (TYMEWEAR, NIRS_SECOND_HALF),
    ("G08", "Easy"): (NIRS_WHOLE,),
    ("G09", "Ramp"): (TYMEWEAR,),
    ("G11", "Ramp"): (TYMEWEAR, NIRS_WHOLE),
    ("G11", "Easy"): (NIRS_WHOLE,),
    ("G11", "Hard"): (NIRS_SECOND_HALF,),
    ("G13", "Ramp"): (TYMEWEAR,),
    ("G13", "Easy"): (TYMEWEAR,),
    ("G14", "Medium"): (TYMEWEAR, NIRS_WHOLE),
    ("G15", "Ramp"): (NIRS_WHOLE,),
}


@dataclass(frozen=True)
class Annotation:
    present: bool
    tymewear_lag_s: float  # NaN: no Tymewear recording annotated
    hr_source: str | None  # 'TW'/'HR' Tymewear, 'VO2' CPET, None no HR
    nirs_repetition: str | None  # 'Both', 'One' (first repetition only), None
    ramp_rate_w_per_min: float
    notes: str

    def predicted_exclusions(self) -> tuple[str, ...]:
        """What the annotation alone says, phrased like the handoff table."""
        if not self.present:
            return ()
        reasons = []
        if math.isnan(self.tymewear_lag_s) or self.hr_source is None:
            reasons.append(TYMEWEAR)
        if self.nirs_repetition is None:
            reasons.append(NIRS_WHOLE)
        elif self.nirs_repetition == "One":
            reasons.append(NIRS_SECOND_HALF)
        return tuple(reasons)


MISSING_ANNOTATION = Annotation(False, math.nan, None, None, math.nan, "no annotation for this visit")


class _NanToConstant(ast.NodeTransformer):
    """Rewrite `np.nan` to a literal so the config can go through `ast.literal_eval`."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "nan":
            return ast.copy_location(ast.Constant(math.nan), node)
        return self.generic_visit(node)


def load_gies_config(path: Path = GIES_CONFIG_PATH) -> dict:
    """Read the `config` dict literal without importing the file.

    The file imports a `data_types` module that is not importable from the repo root.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [None])[0]
        if isinstance(target, ast.Name) and target.id == "config":
            return ast.literal_eval(_NanToConstant().visit(node.value))
    raise ValueError(f"No `config = {{...}}` assignment found in {path}")


def _notes(visit_config: dict) -> str:
    occlusion = visit_config.get("Occlusion") or {}
    parts = [visit_config.get("notes"), occlusion.get("notes")]
    return " | ".join(p for p in parts if p)


def _gies_annotation(visit_config: dict) -> Annotation:
    return Annotation(
        present=True,
        tymewear_lag_s=float(visit_config.get("VE_lag", math.nan)),
        hr_source=visit_config.get("HR_source"),
        nirs_repetition=visit_config.get("NIRS_repetition"),
        ramp_rate_w_per_min=math.nan,
        notes=_notes(visit_config),
    )


def _lucas_annotation(visit_config: dict) -> Annotation:
    hr_source = visit_config.get("HR_source")
    nirs_repetition = visit_config.get("NIRS_repetition")
    return Annotation(
        present=True,
        tymewear_lag_s=0.0,
        hr_source=hr_source.value if hr_source is not None else None,
        nirs_repetition=nirs_repetition.value if nirs_repetition is not None else None,
        ramp_rate_w_per_min=float(visit_config.get("ramp_rate", math.nan)),
        notes=_notes(visit_config),
    )


def load_annotations() -> dict[tuple[str, str], Annotation]:
    """Annotations keyed by (participant_id, source_visit)."""
    annotations = {}
    for source_participant, visits in load_gies_config().items():
        pid = participant_id("Gies", source_participant)
        for source_visit, visit_config in visits.items():
            annotations[(pid, source_visit)] = _gies_annotation(visit_config)
    for source_participant, visits in config_lucas.items():
        pid = participant_id("Lucas", source_participant)
        for source_visit, visit_config in visits.items():
            annotations[(pid, source_visit)] = _lucas_annotation(visit_config)
    return annotations


def annotation_for(ref: VisitRef, annotations: dict[tuple[str, str], Annotation]) -> Annotation:
    return annotations.get((ref.participant_id, ref.source_visit), MISSING_ANNOTATION)


def expected_exclusions(ref: VisitRef) -> tuple[str, ...]:
    return EXPECTED_EXCLUSIONS.get((ref.participant_id, ref.protocol), ())
