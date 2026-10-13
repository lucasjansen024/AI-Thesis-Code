"""Aim 2 (METHODOLOGY Appendix B): left-out folds, the seen arm's subsample, cohort transfer, the outcome table and
the registry. Synthetic data on CPU; one real-data check of the registered test participants."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from pipeline import aim2 as aim2_module
from pipeline.aim2 import (PartialTransferError, UnscorableParticipantError, check_test_participants,
                           left_out_folds, subsampled_mask, train_left_out, train_transfer, transfer_fold)
from pipeline.aim2_outcome import PARTICIPANTS_NAME, SUMMARY_TABLE_NAME, UnpairedVisitsError, write_outcome
from pipeline.cache import ProcessedCache
from pipeline.evaluate import evaluate
from pipeline.inclusion import shared_mask
from pipeline.runs import (A1_RUNS, A2_CONFIGURATIONS, A2_SMOKE_ARMS, A2_SMOKE_TRANSFER, A2A_ARMS, A2A_POOL,
                           A2A_STUDY, A2B_ARMS, A2C_TRANSFERS, COHORTS, PARTICIPANTS, CohortTransfer, LeftOutArm, Run,
                           TrainingSettings, left_out_arms)
from pipeline.train import MANIFEST_NAME, MODELS_NAME, PREDICTIONS_NAME, RunData
from tests.synthetic import Synthetic, make_synthetic_visits

CPU = torch.device("cpu")
TINY = TrainingSettings(epochs=1)
GIES_TEST = ("G01", "G03", "G04")
VISITS = (
    ("G01", "Gies", "Ramp"), ("G01", "Gies", "Easy"), ("G01", "Gies", "Medium"), ("G01", "Gies", "Hard"),
    ("G03", "Gies", "Easy"), ("G03", "Gies", "Medium"), ("G03", "Gies", "Hard"),
    ("G04", "Gies", "Easy"), ("G04", "Gies", "Medium"), ("G04", "Gies", "Hard"),
    ("L01", "Lucas", "Step"), ("L02", "Lucas", "Step"),
)
VISIT_METRIC_COLUMNS = ["config_id", "participant_id", "cohort", "protocol", "visit_id", "n_s", "rmse", "mae", "bias"]


def _tiny(arm: LeftOutArm, fold_ids: tuple[str, ...] = GIES_TEST) -> LeftOutArm:
    return replace(arm, run=replace(arm.run, configurations=("WR",), seeds=(0,), training=TINY, fold_ids=fold_ids))


def _arm(left_out: str, is_seen: bool, fold_ids: tuple[str, ...] = GIES_TEST) -> LeftOutArm:
    arms = left_out_arms(A2A_STUDY, "Gies", A2A_POOL, {c: fold_ids for c in A2A_POOL})
    return _tiny(next(a for a in arms if a.left_out == left_out and a.is_seen == is_seen), fold_ids)


def _run_data(data: Synthetic) -> RunData:
    return RunData(data.cache, data.inclusion, "synthetic-sha", shared_mask(data.cache, data.inclusion))


def _manifest(root: Path, run_id: str, participant: str) -> dict:
    return json.loads((root / run_id / participant / MANIFEST_NAME).read_text(encoding="utf-8"))


# --- left-out folds and the seen arm's subsample ----------------------------------------------------------------

def test_folds_drop_the_left_out_level_the_ramp_and_the_test_participant() -> None:
    visits = make_synthetic_visits(VISITS).cache.visits
    unseen = {f.test_participant: f.train_visits for f in left_out_folds(_arm("Easy", False), visits)}
    seen = {f.test_participant: f.train_visits for f in left_out_folds(_arm("Easy", True), visits)}
    assert unseen["G01"] == {"G03_Medium", "G03_Hard", "G04_Medium", "G04_Hard"}
    assert seen["G01"] == unseen["G01"] | {"G03_Easy", "G04_Easy"}
    assert all("G01_Ramp" not in v for v in [*unseen.values(), *seen.values()])


def test_seen_mask_keeps_exactly_the_unseen_count_and_changes_only_training_rows() -> None:
    data = make_synthetic_visits(VISITS)
    mask = shared_mask(data.cache, data.inclusion)
    before = mask.copy()
    arm = _arm("Easy", True)
    fold = left_out_folds(arm, data.cache.visits)[0]
    subsampled = subsampled_mask(arm, data.cache, mask, fold)
    visit_ids = data.cache.row_ids("visit_id")
    training = np.isin(visit_ids, list(fold.train_visits))
    unseen_training = np.isin(visit_ids, ["G03_Medium", "G03_Hard", "G04_Medium", "G04_Hard"])
    assert (subsampled & training).sum() == (mask & unseen_training).sum()
    assert np.array_equal(subsampled[~training], mask[~training])
    assert not (subsampled & ~mask).any()
    assert (subsampled & np.isin(visit_ids, ["G03_Easy", "G04_Easy"])).any()
    assert np.array_equal(subsampled, subsampled_mask(arm, data.cache, mask, fold))
    assert np.array_equal(mask, before)


def test_seen_mask_raises_when_the_pool_adds_no_left_out_targets() -> None:
    data = make_synthetic_visits(VISITS, invalid=frozenset({("G03_Easy", "VO2"), ("G04_Easy", "VO2")}))
    arm = _arm("Easy", True)
    with pytest.raises(ValueError, match="seen pool"):
        subsampled_mask(arm, data.cache, shared_mask(data.cache, data.inclusion),
                        left_out_folds(arm, data.cache.visits)[0])


def test_test_participants_need_left_out_targets() -> None:
    data = make_synthetic_visits(VISITS, invalid=frozenset({("G04_Easy", "VO2")}))
    with pytest.raises(UnscorableParticipantError, match="G04"):
        check_test_participants(_arm("Easy", False), data.cache, shared_mask(data.cache, data.inclusion))


def test_test_participants_must_be_in_the_arms_cohort() -> None:
    data = make_synthetic_visits((*VISITS, ("L01", "Lucas", "Easy")))
    with pytest.raises(UnscorableParticipantError, match="L01"):
        check_test_participants(_arm("Easy", False, ("G01", "L01")), data.cache,
                                shared_mask(data.cache, data.inclusion))


def test_arms_train_on_equal_sizes_and_evaluate(tmp_path: Path) -> None:
    data = _run_data(make_synthetic_visits(VISITS))
    arms = (_arm("Easy", False, ("G01",)), _arm("Easy", True, ("G01",)))
    for arm in arms:
        train_left_out(arm, data, output_root=tmp_path, device=CPU)
        evaluate(arm.run, output_root=tmp_path, data=data)
    unseen, seen = (_manifest(tmp_path, a.run.run_id, "G01") for a in arms)
    assert unseen["n_train_targets"] == seen["n_train_targets"] > 0
    assert unseen["n_train_participants"] == seen["n_train_participants"] == 2
    predictions = pd.read_parquet(tmp_path / arms[1].run.run_id / "G01" / PREDICTIONS_NAME)
    assert set(predictions["participant_id"]) == {"G01"} and "Easy" in set(predictions["protocol"])
    outcome = pd.read_csv(write_outcome(arms, runs_root=tmp_path, out_dir=tmp_path / "aim2") / PARTICIPANTS_NAME)
    assert outcome[["participant_id", "scope", "n_visits"]].values.tolist() == [["G01", "Easy", 1]]


# --- cohort transfer --------------------------------------------------------------------------------------------

def _transfer(fold_ids: tuple[str, ...] = ("L01", "L02")) -> CohortTransfer:
    return CohortTransfer(Run("transfer", COHORTS, ("WR",), seeds=(0,), training=TINY, fold_ids=fold_ids),
                          "Gies", "reference")


def test_transfer_writes_one_folder_per_test_participant_from_one_training(tmp_path: Path) -> None:
    data = _run_data(make_synthetic_visits(VISITS))
    spec = _transfer()
    train_transfer(spec, data, output_root=tmp_path, device=CPU)
    gies_targets = int((data.mask & (data.cache.row_ids("cohort") == "Gies")).sum())
    for participant in spec.run.fold_ids:
        folder = tmp_path / spec.run.run_id / participant
        predictions = pd.read_parquet(folder / PREDICTIONS_NAME)
        assert set(predictions["participant_id"]) == set(predictions["fold"]) == {participant}
        manifest = _manifest(tmp_path, spec.run.run_id, participant)
        assert manifest["n_train_targets"] == gies_targets and manifest["transfer"]["train_cohort"] == "Gies"
    saved = [torch.load(tmp_path / spec.run.run_id / p / MODELS_NAME, weights_only=True) for p in spec.run.fold_ids]
    first, second = (s["state_dicts"]["WR"][0] for s in saved)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert (evaluate(spec.run, output_root=tmp_path, data=data) / "summary.csv").exists()


def test_transfer_skips_when_finished_and_refuses_a_partial_set(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    data = _run_data(make_synthetic_visits(VISITS))
    spec = _transfer()
    train_transfer(spec, data, output_root=tmp_path, device=CPU)

    def must_not_train(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished transfer was retrained")

    monkeypatch.setattr(aim2_module, "fit_packed", must_not_train)
    train_transfer(spec, data, output_root=tmp_path, device=CPU)
    shutil.rmtree(tmp_path / spec.run.run_id / "L02")
    with pytest.raises(PartialTransferError, match="L01"):
        train_transfer(spec, data, output_root=tmp_path, device=CPU)


def test_transfer_rejects_test_participants_outside_the_test_cohort() -> None:
    visits = make_synthetic_visits(VISITS).cache.visits
    with pytest.raises(UnscorableParticipantError, match="G01"):
        transfer_fold(_transfer(("L01", "G01")), visits)
    assert transfer_fold(_transfer(), visits).train_visits == set(visits.loc[visits["cohort"] == "Gies", "visit_id"])


# --- outcome table ----------------------------------------------------------------------------------------------

def _write_visit_metrics(root: Path, run_id: str, rows: list[tuple[str, str, str, float]], n_s: int = 100) -> None:
    """rows: (config_id, participant_id, protocol, rmse); one visit per participant x protocol."""
    folder = root / run_id / "summary"
    folder.mkdir(parents=True)
    frame = pd.DataFrame([(c, p, "Gies", pr, f"{p}_{pr}", n_s, r, r, 0.0) for c, p, pr, r in rows],
                         columns=VISIT_METRIC_COLUMNS)
    frame.to_csv(folder / "visit_metrics_seedavg.csv", index=False)


def test_outcome_differences_and_t_ci(tmp_path: Path) -> None:
    unseen, seen = _arm("Easy", False), _arm("Easy", True)
    _write_visit_metrics(tmp_path, seen.run.run_id, [("WR", "G01", "Easy", 100.0), ("WR", "G03", "Easy", 110.0),
                                                     ("WR", "G04", "Easy", 120.0), ("WR", "G01", "Hard", 999.0)])
    _write_visit_metrics(tmp_path, unseen.run.run_id, [("WR", "G01", "Easy", 130.0), ("WR", "G03", "Easy", 120.0),
                                                       ("WR", "G04", "Easy", 170.0), ("WR", "G01", "Hard", 1.0)])
    summary = pd.read_csv(write_outcome((unseen, seen), runs_root=tmp_path, out_dir=tmp_path / "o")
                          / SUMMARY_TABLE_NAME)
    diffs = np.array([30.0, 10.0, 50.0])
    half = stats.t.ppf(0.975, 2) * diffs.std(ddof=1) / np.sqrt(3)
    row = summary.iloc[0]
    assert (row["scope"], row["n_participants"]) == ("Easy", 3)
    assert row["diff_mean"] == pytest.approx(30.0)
    assert (row["diff_ci_lo"], row["diff_ci_hi"]) == (pytest.approx(30.0 - half), pytest.approx(30.0 + half))


def test_outcome_transfer_filters_the_reference_and_scopes_by_protocol(tmp_path: Path) -> None:
    spec = _transfer()
    _write_visit_metrics(tmp_path, "reference", [("WR", "L01", "Step", 100.0), ("WR", "L02", "Step", 100.0),
                                                 ("HR", "L01", "Step", 5.0), ("WR", "L03", "Step", 5.0)])
    _write_visit_metrics(tmp_path, "transfer", [("WR", "L01", "Step", 150.0), ("WR", "L02", "Step", 130.0)])
    summary = pd.read_csv(write_outcome(transfers=(spec,), runs_root=tmp_path, out_dir=tmp_path / "o")
                          / SUMMARY_TABLE_NAME)
    assert summary["scope"].tolist() == ["all", "Step"]
    assert summary["condition"].unique().tolist() == ["Gies_to_Lucas"]
    assert summary["diff_mean"].tolist() == [pytest.approx(40.0)] * 2


def test_outcome_raises_on_unpaired_visits(tmp_path: Path) -> None:
    unseen, seen = _arm("Easy", False), _arm("Easy", True)
    _write_visit_metrics(tmp_path, seen.run.run_id, [("WR", "G01", "Easy", 100.0), ("WR", "G03", "Easy", 110.0)])
    _write_visit_metrics(tmp_path, unseen.run.run_id, [("WR", "G01", "Easy", 130.0)])
    with pytest.raises(UnpairedVisitsError, match="G03_Easy"):
        write_outcome((unseen, seen), runs_root=tmp_path, out_dir=tmp_path / "o")


def test_outcome_raises_when_the_sides_were_scored_on_different_seconds(tmp_path: Path) -> None:
    unseen, seen = _arm("Easy", False), _arm("Easy", True)
    _write_visit_metrics(tmp_path, seen.run.run_id, [("WR", "G01", "Easy", 100.0)], n_s=100)
    _write_visit_metrics(tmp_path, unseen.run.run_id, [("WR", "G01", "Easy", 130.0)], n_s=99)
    with pytest.raises(UnpairedVisitsError, match="G01_Easy"):
        write_outcome((unseen, seen), runs_root=tmp_path, out_dir=tmp_path / "o")


def test_outcome_refuses_two_arms_of_the_same_kind(tmp_path: Path) -> None:
    unseen, seen = _arm("Easy", False), _arm("Easy", True)
    smoke_seen = replace(seen, run=replace(seen.run, run_id="smoke_seen"))
    with pytest.raises(ValueError, match="two seen arms"):
        write_outcome((unseen, seen, smoke_seen), runs_root=tmp_path, out_dir=tmp_path / "o")


# --- registry ---------------------------------------------------------------------------------------------------

def test_aim2_registry() -> None:
    specs = [*A2A_ARMS, *A2B_ARMS, *A2C_TRANSFERS, *A2_SMOKE_ARMS, A2_SMOKE_TRANSFER]
    run_ids = [s.run.run_id for s in specs] + [r.run_id for r in A1_RUNS]
    assert len(run_ids) == len(set(run_ids))
    assert (len(A2A_ARMS), len(A2B_ARMS)) == (6, 8)
    assert all(s.run.configurations == A2_CONFIGURATIONS for s in specs)
    assert "Ramp" not in A2A_POOL
    assert all(t.run.fold_ids == PARTICIPANTS[t.test_cohort] for t in A2C_TRANSFERS)
    assert sum(len(a.run.fold_ids) for a in A2A_ARMS) == 76 and sum(len(a.run.fold_ids) for a in A2B_ARMS) == 104


@pytest.mark.real
def test_registered_test_participants_have_left_out_targets(real_cache: ProcessedCache,
                                                            committed_inclusion: pd.DataFrame) -> None:
    mask = shared_mask(real_cache, committed_inclusion)
    participants, protocols = real_cache.row_ids("participant_id"), real_cache.row_ids("protocol")
    cohorts = real_cache.row_ids("cohort")
    for arm in (*A2A_ARMS, *A2B_ARMS):
        scored = mask & (protocols == arm.left_out) & (cohorts == arm.run.cohorts[0])  # Ramp is in both cohorts
        assert set(arm.run.fold_ids) == set(participants[scored]), arm.run.run_id
