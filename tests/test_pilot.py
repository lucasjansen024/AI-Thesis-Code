"""Training-settings pilot (slice 10): per-epoch hook, validation splits, pilot smoke path, and `decide()`."""
from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import pilot as pilot_module
from pipeline.cache import ProcessedCache
from pipeline.fit import configure_determinism, fit_packed, pack_layout, to_device
from pipeline.folds import FoldIntegrityError, build_validation_data, validation_fold
from pipeline.inclusion import WINDOW_S, shared_mask
from pipeline.pilot import (ARMS, DECISION_JSON, DECISION_MD, EPOCHS_NAME, OUTPUTS_ROOT, PILOT_EPOCH_COLUMNS,
                            VAL_VISIT_COLUMNS, VAL_VISITS_NAME, PilotLogError, arm_checkpoints, check_loso_coverage,
                            check_splits, choose_arm, decide, participant_rmse, pilot_run_spec, train_pilot)
from pipeline.runs import (COHORTS, PILOT, PILOT_CONFIGURATIONS, PILOT_SMOKE, PILOT_SPLITS,
                           PILOT_TRAJECTORIES, Pilot, PilotSplit, PilotTrajectory, TrainingSettings)
from pipeline.train import MANIFEST_NAME, RUNS_ROOT, build_run_data
from tests.synthetic import ColumnEdit, make_synthetic

CPU = torch.device("cpu")
ARM = {a.name: a for a in ARMS}
EXTREME = 1e6
SYNTHETIC_SPLIT = PilotSplit("pooled_1", "pooled", ("G01", "L01"))
SYNTHETIC_PILOT = Pilot("synthetic_pilot", (SYNTHETIC_SPLIT,),
                        (PilotTrajectory("lr5e-4_ep2", TrainingSettings(learning_rate=5e-4, epochs=2)),),
                        ("WR", "WR+HR+VENT+TSI"), (0,))  # one training participant: no ANTHRO (SMM spread 0)


# --- per-epoch hook in fit_packed -------------------------------------------------------------------------------

def _tiny_fit(on_epoch: Callable | None) -> tuple:
    rng = np.random.default_rng(0)
    data = to_device(rng.normal(size=(400, 3)).astype(np.float32), rng.normal(size=400).astype(np.float32), CPU)
    branches = pack_layout((("a", (0,)), ("b", (1, 2))), (0, 1))
    settings = TrainingSettings(learning_rate=1e-2, epochs=3, batch_size=32)
    return branches, fit_packed(data, np.arange(WINDOW_S, WINDOW_S + 100), branches, settings, dropout_seed=9,
                                on_epoch=on_epoch)


def test_epoch_hook_sees_every_branch_and_leaves_training_unchanged() -> None:
    configure_determinism(CPU)
    calls: list[tuple[int, tuple]] = []
    branches, hooked = _tiny_fit(lambda epoch, states: calls.append((epoch, states)))
    _, plain = _tiny_fit(None)
    assert [epoch for epoch, _ in calls] == [1, 2, 3]
    assert all(len(states) == len(branches) for _, states in calls)
    for a, b in zip(hooked, plain):
        assert a.epoch_losses == b.epoch_losses and a.is_best == b.is_best
        assert all(torch.equal(a.state_dict[n], b.state_dict[n]) for n in a.state_dict)
    for k, fit in enumerate(hooked):  # the checkpoint epoch's current weights are the kept best weights
        _, states = calls[fit.checkpoint_epoch - 1]
        assert all(torch.equal(states[k][n], fit.state_dict[n]) for n in fit.state_dict)


# --- splits -----------------------------------------------------------------------------------------------------

def _cohort_participants(inclusion: pd.DataFrame) -> dict[str, list[str]]:
    pairs = inclusion[["participant_id", "cohort"]].drop_duplicates()
    return {c: sorted(pairs.loc[pairs["cohort"] == c, "participant_id"]) for c in COHORTS}


def test_pilot_is_loso_over_every_participant_of_each_family(committed_inclusion: pd.DataFrame) -> None:
    participants = _cohort_participants(committed_inclusion)
    by_family = {f: [s for s in PILOT_SPLITS if s.family == f] for f in ("pooled", *COHORTS)}
    assert all(len(s.validation) == 1 for s in PILOT_SPLITS)
    assert sorted(p for s in by_family["pooled"] for p in s.validation) == sorted(participants["Gies"]
                                                                                  + participants["Lucas"])
    for cohort in COHORTS:
        assert sorted(p for s in by_family[cohort] for p in s.validation) == participants[cohort]
    assert [len(by_family[f]) for f in ("pooled", *COHORTS)] == [27, 14, 13]
    assert PILOT.is_loso
    check_splits(PILOT_SPLITS, committed_inclusion)
    check_loso_coverage(PILOT_SPLITS, committed_inclusion)


@pytest.mark.parametrize("splits", [
    (PilotSplit("pooled_1", "pooled", ("G01", "G03", "L01")), PilotSplit("pooled_2", "pooled", ("G03", "L02", "G04"))),
    (PilotSplit("gies_1", "Gies", ("G01", "L01")),),
])
def test_check_splits_rejects_overlap_and_wrong_cohort(splits: tuple[PilotSplit, ...],
                                                       committed_inclusion: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="pilot splits"):
        check_splits(splits, committed_inclusion)


@pytest.mark.parametrize("splits", [
    PILOT_SPLITS[:-1],  # Lucas misses its last participant
    (*PILOT_SPLITS[:-1], PilotSplit("lucas_pair", "Lucas", ("L13", "L14"))),  # two held out at once
])
def test_loso_coverage_rejects_missing_or_grouped_participants(splits: tuple[PilotSplit, ...],
                                                               committed_inclusion: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="LOSO pilot splits"):
        check_loso_coverage(splits, committed_inclusion)


def test_pilot_registry() -> None:
    assert PILOT.configurations == PILOT_CONFIGURATIONS and len(PILOT_CONFIGURATIONS) == 5
    assert PILOT.seeds == (0, 1) and len(PILOT.splits) == 54
    assert [(t.training.learning_rate, t.training.epochs) for t in PILOT_TRAJECTORIES] == \
        [(5e-5, 30), (2e-4, 10), (5e-4, 3)]
    assert [s.split_id for s in PILOT_SMOKE.splits] == ["pooled_G01"]
    assert [t.training.epochs for t in PILOT_SMOKE.trajectories] == [3]
    assert PILOT_SMOKE.configurations == ("WR", "WR+HR+VENT+TSI+ANTHRO") and PILOT_SMOKE.seeds == (0,)
    trajectories = {t.trajectory_id: t.training.epochs for t in PILOT_TRAJECTORIES}
    assert all(arm.epochs <= trajectories[arm.trajectory_id] for arm in ARMS)


# --- validation folds -------------------------------------------------------------------------------------------

def _validation_scalers(edit: ColumnEdit | None = None) -> dict:
    data = make_synthetic(edit=edit)
    fold = validation_fold("s", SYNTHETIC_SPLIT.validation, COHORTS, data.cache.visits)
    mask = shared_mask(data.cache, data.inclusion)
    return build_validation_data(data.cache, mask, fold, ("WR",)).scalers


def test_validation_participants_never_train_or_fit_scalers() -> None:
    data = make_synthetic()
    fold = validation_fold("s", SYNTHETIC_SPLIT.validation, COHORTS, data.cache.visits)
    assert fold.train_visits == frozenset({"G03_Ramp"})
    fold_data = build_validation_data(data.cache, shared_mask(data.cache, data.inclusion), fold, ("WR",))
    held_out = set(data.cache.row_ids("participant_id")[fold_data.test_targets])
    assert held_out == {"G01", "L01"} and fold_data.n_train_participants == 1

    def extreme(pid: str, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if pid not in SYNTHETIC_SPLIT.validation:
            return columns
        n = len(columns["t"])
        return {**columns, "WR": np.full(n, EXTREME), "VO2": np.linspace(-EXTREME, EXTREME, n)}

    assert _validation_scalers(extreme) == _validation_scalers()


def test_within_cohort_split_trains_on_its_cohort_only() -> None:
    data = make_synthetic()
    fold = validation_fold("gies_1", ("G01",), ("Gies",), data.cache.visits)
    assert fold.train_visits == frozenset({"G03_Ramp"})
    with pytest.raises(FoldIntegrityError, match="not in cohorts"):
        validation_fold("gies_1", ("L01",), ("Gies",), data.cache.visits)


@pytest.mark.real
@pytest.mark.parametrize("split", PILOT_SPLITS, ids=lambda s: s.split_id)
def test_real_split_never_trains_on_its_validation_participants(real_cache: ProcessedCache,
                                                                split: PilotSplit) -> None:
    fold = validation_fold(split.split_id, split.validation, split.cohorts, real_cache.visits)
    participant_of = dict(zip(real_cache.visits["visit_id"], real_cache.visits["participant_id"]))
    trained = {participant_of[v] for v in fold.train_visits}
    assert not trained & set(split.validation)
    expected = 26 if split.family == "pooled" else {"Gies": 13, "Lucas": 12}[split.family]
    assert len(trained) == expected


# --- pilot smoke path (synthetic) -------------------------------------------------------------------------------

def test_synthetic_pilot_writes_validation_logs_and_resumes(tmp_path: Path) -> None:
    synthetic = make_synthetic()
    data = build_run_data(pilot_run_spec(SYNTHETIC_PILOT), synthetic.cache, synthetic.inclusion, "synthetic-sha")
    [job_dir] = train_pilot(SYNTHETIC_PILOT, data, output_root=tmp_path, device=CPU)
    assert job_dir == tmp_path / "synthetic_pilot" / "pooled_1" / "lr5e-4_ep2"
    epochs = pd.read_csv(job_dir / EPOCHS_NAME)
    assert list(epochs.columns) == PILOT_EPOCH_COLUMNS and len(epochs) == 2 * 1 * 2
    assert np.isfinite(epochs["val_mse_scaled"]).all() and (epochs["val_mse_scaled"] > 0).all()
    visits = pd.read_csv(job_dir / VAL_VISITS_NAME)
    assert list(visits.columns) == VAL_VISIT_COLUMNS
    assert set(visits["participant_id"]) == {"G01", "L01"}
    assert len(visits) == 2 * 2 * 2  # configurations x epochs x validation visits
    assert np.isfinite(visits["rmse"]).all() and (visits["n_s"] > 0).all()
    manifest = json.loads((job_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["validation_participants"] == ["G01", "L01"] and manifest["K"] == 2
    assert manifest["n_train_participants"] == 1 and manifest["mask"] == "shared"
    assert not list(job_dir.parent.glob("*.tmp"))
    first = (job_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    train_pilot(SYNTHETIC_PILOT, data, output_root=tmp_path, device=CPU)
    assert (job_dir / MANIFEST_NAME).read_text(encoding="utf-8") == first


def test_pilot_never_reads_run_outputs() -> None:
    source = inspect.getsource(pilot_module)
    assert "RUNS_ROOT" not in source and "pipeline.evaluate" not in source and "outputs/runs" not in source
    assert RUNS_ROOT not in (OUTPUTS_ROOT / PILOT.pilot_id).parents and OUTPUTS_ROOT / PILOT.pilot_id != RUNS_ROOT


# --- decide() on hand-built logs --------------------------------------------------------------------------------

RmseOf = Callable[[str, str, str], float]  # (family, config_id, arm) -> RMSE at the arm's checkpoint
UNUSED_RMSE = 999.0  # epochs no arm is scored at
VISIT_SPREAD = 5.0


def _rmse_at(trajectory_id: str, epoch: int, family: str, config_id: str, rmse_of: RmseOf) -> float:
    arms = [a for a in ARMS if a.trajectory_id == trajectory_id and a.epochs == epoch]
    return rmse_of(family, config_id, arms[0].name) if arms else UNUSED_RMSE


def _job_logs(split: PilotSplit, trajectory: PilotTrajectory, rmse_of: RmseOf) -> tuple[list, list]:
    epochs, visits = [], []
    for config_id in PILOT.configurations:
        for seed in PILOT.seeds:
            for epoch in range(1, trajectory.training.epochs + 1):  # decreasing loss: e* = E
                ids = {"config_id": config_id, "seed": seed, "epoch": epoch}
                epochs.append({**ids, "train_loss": 1.0 / epoch, "is_best": True, "val_mse_scaled": 0.5})
                rmse = _rmse_at(trajectory.trajectory_id, epoch, split.family, config_id, rmse_of)
                first, *rest = split.validation  # first participant: two visits averaging to `rmse`
                visits += [{**ids, "participant_id": first, "visit_id": f"{first}_{v}", "n_s": 100,
                            "rmse": rmse + d} for v, d in (("Ramp", VISIT_SPREAD), ("Step", -VISIT_SPREAD))]
                visits += [{**ids, "participant_id": p, "visit_id": f"{p}_Ramp", "n_s": 100, "rmse": rmse}
                           for p in rest]
    return epochs, visits


def _write_logs(root: Path, rmse_of: RmseOf) -> None:
    for split in PILOT.splits:
        for trajectory in PILOT.trajectories:
            folder = root / PILOT.pilot_id / split.split_id / trajectory.trajectory_id
            folder.mkdir(parents=True)
            epochs, visits = _job_logs(split, trajectory, rmse_of)
            pd.DataFrame(epochs, columns=PILOT_EPOCH_COLUMNS).to_csv(folder / EPOCHS_NAME, index=False)
            pd.DataFrame(visits, columns=VAL_VISIT_COLUMNS).to_csv(folder / VAL_VISITS_NAME, index=False)
            manifest = {"K": 10, "epochs": trajectory.training.epochs, "wall_s": 12.0, "train_wall_s": 10.0,
                        "val_wall_s": 2.0}
            (folder / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def _levels(levels: dict[str, float], exceptions: dict[tuple[str, str, str], float] | None = None) -> RmseOf:
    exceptions = exceptions or {}
    return lambda family, config_id, arm: exceptions.get((family, config_id, arm), levels[arm])


def test_decide_picks_the_fewest_epoch_arm_that_qualifies_everywhere(tmp_path: Path) -> None:
    _write_logs(tmp_path, _levels({"C": 200.0, "D": 149.0, "A": 142.0, "B": 141.0, "R": 140.0}))
    decision = decide(PILOT, output_root=tmp_path)
    assert decision.arm == "D" and decision.cells.shape == (15, 5)
    assert decision.qualifies_everywhere == {"C": False, "D": True, "A": True, "B": True, "R": True}
    np.testing.assert_allclose(decision.cells["D"], 149.0)
    written = json.loads((tmp_path / PILOT.pilot_id / DECISION_JSON).read_text(encoding="utf-8"))
    assert written["chosen"] == {"arm": "D", "learning_rate": 5e-4, "epochs": 3, "batch_size": 32}
    assert len(written["wall"]["jobs"]) == len(PILOT.splits) * len(PILOT.trajectories)
    assert written["wall"]["per_model_epoch_s"] == pytest.approx(21 * 10.0 / (10 * 7 * (30 + 10 + 3)))
    assert "Chosen arm: D" in (tmp_path / PILOT.pilot_id / DECISION_MD).read_text(encoding="utf-8")


def test_decide_breaks_an_a_b_tie_by_mean_rmse(tmp_path: Path) -> None:
    levels = {"C": 200.0, "D": 145.0, "A": 146.0, "B": 143.0, "R": 140.0}
    _write_logs(tmp_path, _levels(levels, {("Lucas", "WR", "D"): 170.0}))
    decision = decide(PILOT, output_root=tmp_path)
    assert decision.arm == "B" and "tie" in decision.reason
    assert decision.qualifies_everywhere["A"] and decision.qualifies_everywhere["B"]
    assert not decision.qualifies_everywhere["D"]


def test_decide_falls_back_to_r(tmp_path: Path) -> None:
    levels = {"C": 200.0, "D": 141.0, "A": 141.0, "B": 141.0, "R": 140.0}
    failures = {("pooled", "WR", "D"): 160.0, ("Gies", "WR", "A"): 160.0,
                ("Lucas", "HR+VENT+TSI+ANTHRO", "B"): 160.0}
    _write_logs(tmp_path, _levels(levels, failures))
    decision = decide(PILOT, output_root=tmp_path)
    assert decision.arm == "R" and not any(decision.qualifies_everywhere[a] for a in "CDAB")


def test_decide_raises_on_a_missing_job(tmp_path: Path) -> None:
    _write_logs(tmp_path, _levels({"C": 1.0, "D": 1.0, "A": 1.0, "B": 1.0, "R": 1.0}))
    (tmp_path / PILOT.pilot_id / "lucas_L14" / "lr2e-4_ep10" / MANIFEST_NAME).unlink()
    with pytest.raises(PilotLogError, match="lucas_L14/lr2e-4_ep10"):
        decide(PILOT, output_root=tmp_path)


def test_choose_arm_qualifies_at_exactly_the_margin_and_keeps_a_on_the_lower_mean() -> None:
    cells = pd.DataFrame({"C": [150.0, 300.0], "D": [150.0, 150.0], "A": [141.0, 141.0], "B": [142.0, 142.0],
                          "R": [140.0, 140.0]})
    assert choose_arm(cells).arm == "D"
    assert choose_arm(cells.assign(D=[151.0, 150.0])).arm == "A"


def test_arm_checkpoint_is_the_earliest_lowest_loss_up_to_e() -> None:
    losses = np.linspace(1.0, 0.5, 30)
    losses[3] = losses[6] = 0.1  # epochs 4 and 7 tie below everything up to epoch 10
    losses[19] = 0.01  # global minimum, only within R's 30 epochs
    frame = pd.DataFrame({"split_id": "s", "trajectory_id": ARM["R"].trajectory_id, "config_id": "WR", "seed": 0,
                          "epoch": np.arange(1, 31), "train_loss": losses})
    checkpoints = arm_checkpoints(frame.iloc[::-1], (ARM["A"], ARM["R"]))
    assert dict(zip(checkpoints["arm"], checkpoints["epoch"])) == {"A": 4, "R": 20}
    with pytest.raises(PilotLogError, match="lacks epochs"):
        arm_checkpoints(frame[frame["epoch"] != 5], (ARM["A"],))


def test_rmse_averages_visits_then_participants() -> None:
    checkpoints = pd.DataFrame({"split_id": ["s"], "trajectory_id": ["t"], "config_id": ["WR"], "seed": [0],
                                "epoch": [3], "arm": ["D"]})
    visits = pd.DataFrame({"split_id": "s", "trajectory_id": "t", "config_id": "WR", "seed": 0, "epoch": 3,
                           "participant_id": ["P1", "P1", "P2"], "visit_id": ["a", "b", "c"], "n_s": 10,
                           "rmse": [100.0, 300.0, 100.0]})
    [value] = participant_rmse(checkpoints, visits)["rmse"]
    assert value == pytest.approx(150.0)  # not the flat visit mean 166.7


def test_pilot_rejects_a_repeated_split() -> None:
    with pytest.raises(ValueError, match="repeated split"):
        replace(PILOT, splits=PILOT_SPLITS[:1] * 2)
