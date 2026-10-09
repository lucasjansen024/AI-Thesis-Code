"""`per_config` sensitivity runs: per-configuration masks, mask groups, packed training, resume, evaluate."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import train as train_module
from pipeline.cache import ProcessedCache
from pipeline.domains import feature_union, parse_configuration
from pipeline.evaluate import evaluate
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import WINDOW_S, config_mask, per_config_masks, shared_mask
from pipeline.prepare.grid import UNITS
from pipeline.prepare.group import presence_column
from pipeline.runs import (A1_CONFIGURATIONS, A1_GIES, A1_LUCAS, A1_PER_CONFIG_RUNS, A1_POOLED, A1_RUNS,
                           A1_TRAINING, PER_CONFIG_MASK, Run)
from pipeline.train import (EPOCHS_NAME, MANIFEST_NAME, PREDICTIONS_NAME, RunData, build_run_data, fold_row_mask,
                            mask_groups, train_folds)
from tests.synthetic import PARTICIPANTS, SYNTHETIC_RUN, VISIT_S, WARMUP_END_S, Synthetic, make_synthetic

CPU = torch.device("cpu")
GAP_VISIT, GAP_S = "G03_Ramp", 400  # present_HR = 0 here: drops HR configurations' windows covering it
PER_CONFIG_RUN = replace(SYNTHETIC_RUN, run_id="synthetic_per_config", mask=PER_CONFIG_MASK,
                         configurations=("WR", "WR+HR", "WR+VENT", "WR+HR+ANTHRO"))
EXPECTED_GROUPS = (("WR", "WR+VENT"), ("WR+HR", "WR+HR+ANTHRO"))  # with the HR gap, in every fold
ALL_EXERCISE = np.arange(WARMUP_END_S, VISIT_S)
# Lucas: every unit is valid for every group, but VENT is absent in the last 1-11 s of exercise in 10 visits
# (Tymewear end / trailing median), so configurations without VENT keep a few final targets: 2 groups.
MAX_POOLED_GROUPS, LUCAS_GROUPS = 8, 2
N_GIES_VISITS = 55


def _hr_gap(pid: str, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not GAP_VISIT.startswith(pid):
        return columns
    flags = columns[presence_column("HR")].copy()
    flags[GAP_S] = 0
    return {**columns, presence_column("HR"): flags}


def _targets(cache: ProcessedCache, mask: np.ndarray, visit_id: str) -> np.ndarray:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    rows = slice(int(row["start"]), int(row["stop"]))
    return cache.column("t")[rows].astype(int)[mask[rows]]


def _masks(data: Synthetic, *config_ids: str) -> dict[str, np.ndarray]:
    return per_config_masks(data.cache, data.inclusion, [parse_configuration(c) for c in config_ids])


# --- per-configuration masks (synthetic) ------------------------------------------------------------------------

def test_unit_validity_uses_only_the_configurations_own_groups() -> None:
    data = make_synthetic(invalid=frozenset({("G03_Ramp", "NIRS"), ("G01_Ramp", "static")}))
    masks = _masks(data, "WR", "WR+TSI", "WR+ANTHRO")
    np.testing.assert_array_equal(_targets(data.cache, masks["WR"], "G03_Ramp"), ALL_EXERCISE)
    np.testing.assert_array_equal(_targets(data.cache, masks["WR"], "G01_Ramp"), ALL_EXERCISE)
    assert len(_targets(data.cache, masks["WR+TSI"], "G03_Ramp")) == 0
    assert len(_targets(data.cache, masks["WR+ANTHRO"], "G01_Ramp")) == 0
    np.testing.assert_array_equal(_targets(data.cache, masks["WR+TSI"], "G01_Ramp"), ALL_EXERCISE)
    shared = shared_mask(data.cache, data.inclusion)
    assert len(_targets(data.cache, shared, "G03_Ramp")) == 0 and len(_targets(data.cache, shared, "G01_Ramp")) == 0


def test_vo2_validity_is_always_required() -> None:
    data = make_synthetic(invalid=frozenset({("G03_Ramp", "VO2")}))
    assert len(_targets(data.cache, _masks(data, "WR")["WR"], "G03_Ramp")) == 0


def test_context_presence_uses_only_the_configurations_input_groups() -> None:
    data = make_synthetic(edit=_hr_gap)
    masks = _masks(data, "WR", "WR+HR")
    np.testing.assert_array_equal(_targets(data.cache, masks["WR"], GAP_VISIT), ALL_EXERCISE)
    covered = np.arange(GAP_S + 1, GAP_S + WINDOW_S + 1)
    np.testing.assert_array_equal(_targets(data.cache, masks["WR+HR"], GAP_VISIT),
                                  np.setdiff1d(ALL_EXERCISE, covered))


def test_configurations_with_the_same_groups_share_one_mask() -> None:
    data = make_synthetic(edit=_hr_gap)
    masks = _masks(data, "WR+TSI", "WR+HBDIFF", "WR+HR")
    assert masks["WR+TSI"] is masks["WR+HBDIFF"]
    np.testing.assert_array_equal(masks["WR+HR"], config_mask(data.cache, data.inclusion,
                                                               parse_configuration("WR+HR")))


# --- mask groups (synthetic) ------------------------------------------------------------------------------------

def _run_data(data: Synthetic, run: Run = PER_CONFIG_RUN) -> RunData:
    return build_run_data(run, data.cache, data.inclusion, "synthetic-sha")


def test_identical_target_sets_form_one_group_in_every_fold() -> None:
    data = _run_data(make_synthetic(edit=_hr_gap))
    configurations = tuple(parse_configuration(c) for c in PER_CONFIG_RUN.configurations)
    for fold in run_folds(PER_CONFIG_RUN, data.cache):
        groups = mask_groups(PER_CONFIG_RUN, configurations, data.mask, config_masks=data.config_masks,
                             fold_rows=fold_row_mask(data.cache, fold))
        assert [g.index for g in groups] == list(range(len(EXPECTED_GROUPS)))
        assert tuple(tuple(c.config_id for c in g.configurations) for g in groups) == EXPECTED_GROUPS


def test_without_differences_per_config_gives_one_group() -> None:
    data = _run_data(make_synthetic())
    configurations = tuple(parse_configuration(c) for c in PER_CONFIG_RUN.configurations)
    fold = run_folds(PER_CONFIG_RUN, data.cache)[0]
    groups = mask_groups(PER_CONFIG_RUN, configurations, data.mask, config_masks=data.config_masks,
                         fold_rows=fold_row_mask(data.cache, fold))
    assert len(groups) == 1 and groups[0].configurations == configurations


def test_each_configuration_gets_the_targets_of_an_unpacked_run_with_its_own_mask() -> None:
    synthetic = make_synthetic(edit=_hr_gap)
    data = _run_data(synthetic)
    configurations = tuple(parse_configuration(c) for c in PER_CONFIG_RUN.configurations)
    for fold in run_folds(PER_CONFIG_RUN, data.cache):
        groups = mask_groups(PER_CONFIG_RUN, configurations, data.mask, config_masks=data.config_masks,
                             fold_rows=fold_row_mask(data.cache, fold))
        for group in groups:
            packed = build_fold_data(data.cache, group.mask, fold, feature_union(group.configurations))
            for config in group.configurations:
                own_mask = config_mask(synthetic.cache, synthetic.inclusion, config)
                own = build_fold_data(synthetic.cache, own_mask, fold, config.columns)
                np.testing.assert_array_equal(packed.train_targets, own.train_targets)
                np.testing.assert_array_equal(packed.test_targets, own.test_targets)
                for column in (*config.columns, "VO2"):
                    assert packed.scalers[column] == own.scalers[column], (fold, config.config_id, column)


def test_run_data_mask_is_the_union_under_per_config() -> None:
    data = _run_data(make_synthetic(edit=_hr_gap))
    assert set(data.config_masks) == set(PER_CONFIG_RUN.configurations)
    np.testing.assert_array_equal(data.mask, data.config_masks["WR"])  # WR's targets contain every other's
    shared = _run_data(make_synthetic(edit=_hr_gap), SYNTHETIC_RUN)
    assert shared.config_masks == {}


# --- training, resume, evaluate (synthetic, CPU) ----------------------------------------------------------------

@pytest.fixture
def per_config_trained(tmp_path: Path) -> tuple[RunData, Path]:
    data = _run_data(make_synthetic(edit=_hr_gap))
    train_folds(PER_CONFIG_RUN, data, output_root=tmp_path, device=CPU)
    return data, tmp_path / PER_CONFIG_RUN.run_id


def _manifest(fold_dir: Path) -> dict:
    return json.loads((fold_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_per_config_fold_trains_every_group_into_one_folder(per_config_trained: tuple[RunData, Path]) -> None:
    data, run_dir = per_config_trained
    visit_ids, t = data.cache.row_ids("visit_id"), data.cache.column("t").astype(int)
    for pid, _, _ in PARTICIPANTS:
        fold_dir = run_dir / pid
        manifest = _manifest(fold_dir)
        jobs = manifest["packing"]["jobs"]
        assert manifest["mask"] == PER_CONFIG_MASK and manifest["mask_groups"] == len(EXPECTED_GROUPS)
        assert tuple(tuple(j["config_order"]) for j in jobs) == EXPECTED_GROUPS
        assert len({j["dropout_seed"] for j in jobs}) == len(jobs)
        assert manifest["scalers"] is None and all("VO2" in j["scalers"] for j in jobs)
        predictions = pd.read_parquet(fold_dir / PREDICTIONS_NAME)
        assert set(predictions["config_id"]) == set(PER_CONFIG_RUN.configurations)
        epochs = pd.read_csv(fold_dir / EPOCHS_NAME)
        assert len(epochs) == len(PER_CONFIG_RUN.configurations) * PER_CONFIG_RUN.training.epochs
        fold = next(f for f in run_folds(PER_CONFIG_RUN, data.cache) if f.test_participant == pid)
        union_train, union_test = set(), set()
        for config_id, mask in data.config_masks.items():
            own = build_fold_data(data.cache, mask, fold, parse_configuration(config_id).columns)
            stored = predictions[predictions["config_id"] == config_id]
            assert set(zip(stored["visit_id"], stored["t_s"])) == set(zip(visit_ids[own.test_targets],
                                                                           t[own.test_targets]))
            union_train |= set(own.train_targets.tolist())
            union_test |= set(own.test_targets.tolist())
        assert (manifest["n_train_targets"], manifest["n_test_targets"]) == (len(union_train), len(union_test))
        assert manifest["n_train_participants"] == len(PARTICIPANTS) - 1


def test_groups_differ_in_their_targets(per_config_trained: tuple[RunData, Path]) -> None:
    _, run_dir = per_config_trained
    wr_group, hr_group = _manifest(run_dir / "G01")["packing"]["jobs"]
    assert hr_group["n_train_targets"] == wr_group["n_train_targets"] - WINDOW_S
    predictions = pd.read_parquet(run_dir / "G03" / PREDICTIONS_NAME)
    counts = predictions.groupby("config_id").size()
    assert counts["WR"] == counts["WR+VENT"] == len(ALL_EXERCISE)
    assert counts["WR+HR"] == counts["WR+HR+ANTHRO"] == len(ALL_EXERCISE) - WINDOW_S


def test_per_config_resume_skips_finished_folds_and_retrains_a_leftover(
        per_config_trained: tuple[RunData, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    data, run_dir = per_config_trained
    finished = (run_dir / "G01" / MANIFEST_NAME).read_text(encoding="utf-8")

    def must_not_train(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished fold was retrained")

    monkeypatch.setattr(train_module, "build_fold_data", must_not_train)
    train_folds(PER_CONFIG_RUN, data, folds=["G01"], output_root=run_dir.parent, device=CPU)
    assert (run_dir / "G01" / MANIFEST_NAME).read_text(encoding="utf-8") == finished
    monkeypatch.undo()

    leftover = run_dir / "G03.tmp"
    (run_dir / "G03").rename(leftover)
    (leftover / "partial.bin").write_bytes(b"junk")
    train_folds(PER_CONFIG_RUN, data, folds=["G03"], output_root=run_dir.parent, device=CPU)
    assert _manifest(run_dir / "G03")["mask_groups"] == len(EXPECTED_GROUPS)
    assert not leftover.exists() and not (run_dir / "G03" / "partial.bin").exists()


def test_per_config_run_evaluates(per_config_trained: tuple[RunData, Path]) -> None:
    data, run_dir = per_config_trained
    summary_dir = evaluate(PER_CONFIG_RUN, output_root=run_dir.parent, data=data)
    summary = pd.read_csv(summary_dir / "summary.csv")
    assert set(summary["config_id"]) == set(PER_CONFIG_RUN.configurations)
    folds = pd.read_csv(summary_dir / "folds.csv").set_index("fold")
    for pid, _, _ in PARTICIPANTS:
        assert folds.loc[pid, "n_train_targets"] == _manifest(run_dir / pid)["n_train_targets"]
    assert int(pd.read_csv(summary_dir / "targets.csv")["target_s"].sum()) == int(data.mask.sum())


# --- registry ---------------------------------------------------------------------------------------------------

def test_per_config_runs_mirror_the_a1_runs_under_their_own_ids() -> None:
    assert len(A1_PER_CONFIG_RUNS) == len(A1_RUNS)
    for per_config, a1 in zip(A1_PER_CONFIG_RUNS, A1_RUNS):
        assert per_config.mask == PER_CONFIG_MASK and per_config.run_id == f"{a1.run_id}_per_config"
        assert per_config.configurations == A1_CONFIGURATIONS and per_config.cohorts == a1.cohorts
        assert asdict(per_config.training) == asdict(A1_TRAINING) and per_config.fold_ids is None
    run_ids = [r.run_id for r in (*A1_RUNS, *A1_PER_CONFIG_RUNS)]
    assert len(set(run_ids)) == len(run_ids)


def test_unknown_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported"):
        replace(A1_POOLED, mask="per_visit")


# --- real data --------------------------------------------------------------------------------------------------

PER_GROUP_TABLE = {  # docs/MAPPING.md §8 and the per_config validity table: visit -> (HR, VENT, NIRS)
    "G06_Ramp": ("all", "none", "all"),
    "G09_Ramp": ("all", "none", "all"),
    "G11_Ramp": ("none", "none", "all"),
    "G07_Ramp": ("none", "none", "all"),
    "G13_Ramp": ("none", "all", "all"),
    "G13_Easy": ("none", "all", "all"),
    "G07_Hard": ("none", "all", "rep1"),
    "G08_Easy": ("all", "all", "none"),
}
TABLE_CONFIGURATIONS = ("WR+HR", "WR+VENT", "WR+TSI")


def _units_with_targets(cache: ProcessedCache, mask: np.ndarray, visit_id: str) -> set[str]:
    row = cache.visits.set_index("visit_id").loc[visit_id]
    rows = slice(int(row["start"]), int(row["stop"]))
    return {UNITS[u] for u in np.unique(cache.column("unit")[rows][mask[rows]].astype(int))}


def _expected_units(verdict: str, all_units: set[str]) -> set[str]:
    return {"all": all_units, "none": set(), "rep1": {"rep1"}}[verdict]


@pytest.mark.real
def test_wr_uses_every_gies_visit(real_cache: ProcessedCache, committed_inclusion: pd.DataFrame) -> None:
    mask = per_config_masks(real_cache, committed_inclusion, [parse_configuration("WR")])["WR"]
    gies = committed_inclusion[committed_inclusion["cohort"] == "Gies"]
    expected = set(zip(gies["visit_id"], gies["unit"])) - {("G04_Hard", "rep2")}
    used = {(v, u) for v in gies["visit_id"].unique() for u in _units_with_targets(real_cache, mask, v)}
    assert len({v for v, _ in used}) == N_GIES_VISITS
    assert used == expected


@pytest.mark.real
@pytest.mark.parametrize("visit_id", sorted(PER_GROUP_TABLE))
def test_per_group_validity_matches_the_documented_table(real_cache: ProcessedCache,
                                                         committed_inclusion: pd.DataFrame, visit_id: str) -> None:
    configurations = [parse_configuration(c) for c in TABLE_CONFIGURATIONS]
    masks = per_config_masks(real_cache, committed_inclusion, configurations)
    all_units = set(committed_inclusion.loc[committed_inclusion["visit_id"] == visit_id, "unit"])
    found = tuple(_units_with_targets(real_cache, masks[c], visit_id) for c in TABLE_CONFIGURATIONS)
    assert found == tuple(_expected_units(v, all_units) for v in PER_GROUP_TABLE[visit_id])


@pytest.mark.real
def test_g04_hard_uses_rep1_only_for_every_configuration(real_cache: ProcessedCache,
                                                         committed_inclusion: pd.DataFrame) -> None:
    configurations = [parse_configuration(c) for c in A1_CONFIGURATIONS]
    masks = per_config_masks(real_cache, committed_inclusion, configurations)
    assert all(_units_with_targets(real_cache, m, "G04_Hard") == {"rep1"} for m in masks.values())


@pytest.mark.real
@pytest.mark.parametrize(("run", "max_groups"), [(A1_POOLED, MAX_POOLED_GROUPS), (A1_GIES, MAX_POOLED_GROUPS),
                                                 (A1_LUCAS, LUCAS_GROUPS)],
                         ids=lambda x: x.run_id if isinstance(x, Run) else str(x))
def test_per_config_mask_group_counts(real_cache: ProcessedCache, committed_inclusion: pd.DataFrame, run: Run,
                                      max_groups: int) -> None:
    per_config = replace(run, run_id=f"{run.run_id}_per_config", mask=PER_CONFIG_MASK)
    data = build_run_data(per_config, real_cache, committed_inclusion, "committed")
    configurations = tuple(parse_configuration(c) for c in per_config.configurations)
    counts = {fold.test_participant: len(mask_groups(per_config, configurations, data.mask,
                                                     config_masks=data.config_masks,
                                                     fold_rows=fold_row_mask(real_cache, fold)))
              for fold in run_folds(per_config, real_cache)}
    assert max(counts.values()) <= max_groups, counts
