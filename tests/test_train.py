"""Fold outputs, resume, evaluate, and seed determinism (synthetic data; logic on CPU, determinism on XPU)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import train as train_module
from pipeline.evaluate import MissingFoldsError, evaluate
from pipeline.fit import configure_determinism, fit_packed, pack_layout, to_device
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask
from pipeline.metrics import VISIT_METRIC_COLUMNS
from pipeline.runs import TrainingSettings
from pipeline.train import (EPOCH_COLUMNS, MANIFEST_NAME, PREDICTION_COLUMNS, PREDICTIONS_NAME, RunData,
                            train_folds)
from tests.synthetic import PARTICIPANTS, SYNTHETIC_RUN, Synthetic, make_synthetic

CPU = torch.device("cpu")
TINY = TrainingSettings(epochs=2)
SUMMARY_FILES = ("summary.csv", "folds.csv", "targets.csv")


def run_data(data: Synthetic) -> RunData:
    return RunData(data.cache, data.inclusion, "synthetic-sha", shared_mask(data.cache, data.inclusion))


@pytest.fixture
def trained(tmp_path: Path) -> tuple[RunData, Path]:
    data = run_data(make_synthetic())
    train_folds(SYNTHETIC_RUN, data, output_root=tmp_path, device=CPU)
    return data, tmp_path


def test_predictions_have_exactly_the_section_3_3_columns(trained: tuple[RunData, Path]) -> None:
    data, root = trained
    for pid, _, _ in PARTICIPANTS:
        predictions = pd.read_parquet(root / SYNTHETIC_RUN.run_id / pid / PREDICTIONS_NAME)
        assert list(predictions.columns) == PREDICTION_COLUMNS
        assert set(predictions["participant_id"]) == {pid} and set(predictions["fold"]) == {pid}
        test_rows = np.flatnonzero(data.mask & (data.cache.row_ids("participant_id") == pid))
        assert sorted(predictions["t_s"]) == sorted(data.cache.column("t")[test_rows].astype(int))
        assert predictions["vo2_pred"].notna().all()


def test_fold_files_and_manifest(trained: tuple[RunData, Path]) -> None:
    _, root = trained
    fold_dir = root / SYNTHETIC_RUN.run_id / "G01"
    epochs = pd.read_csv(fold_dir / "epochs.csv")
    assert list(epochs.columns) == EPOCH_COLUMNS
    assert len(epochs) == SYNTHETIC_RUN.training.epochs * len(SYNTHETIC_RUN.seeds)
    best = epochs.loc[epochs["train_loss"].idxmin()]
    assert bool(best["is_best"])
    assert list(pd.read_csv(fold_dir / "visit_metrics.csv").columns) == VISIT_METRIC_COLUMNS
    manifest = json.loads((fold_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["n_train_participants"] == len(PARTICIPANTS) - 1
    assert manifest["checkpoint_epochs"]["WR"]["0"] == int(best["epoch"])
    assert not list((root / SYNTHETIC_RUN.run_id).glob("*.tmp"))


def test_resume_skips_a_finished_fold_and_wipes_a_leftover_tmp(trained: tuple[RunData, Path],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    data, root = trained
    run_dir = root / SYNTHETIC_RUN.run_id
    finished = (run_dir / "G01" / MANIFEST_NAME).read_text(encoding="utf-8")

    def must_not_train(*args: object, **kwargs: object) -> None:
        raise AssertionError("a finished fold was retrained")

    monkeypatch.setattr(train_module, "build_fold_data", must_not_train)
    train_folds(SYNTHETIC_RUN, data, folds=["G01"], output_root=root, device=CPU)
    assert (run_dir / "G01" / MANIFEST_NAME).read_text(encoding="utf-8") == finished
    monkeypatch.undo()

    leftover = run_dir / "G03.tmp"
    (run_dir / "G03").rename(leftover)
    (leftover / "partial.bin").write_bytes(b"junk")
    train_folds(SYNTHETIC_RUN, data, folds=["G03"], output_root=root, device=CPU)
    assert (run_dir / "G03" / MANIFEST_NAME).exists()
    assert not leftover.exists()
    assert not (run_dir / "G03" / "partial.bin").exists()


def test_evaluate_raises_listing_missing_folds(tmp_path: Path) -> None:
    data = run_data(make_synthetic())
    train_folds(SYNTHETIC_RUN, data, folds=["G01"], output_root=tmp_path, device=CPU)
    with pytest.raises(MissingFoldsError, match="G03, L01"):
        evaluate(SYNTHETIC_RUN, output_root=tmp_path, data=data)


def test_evaluate_writes_the_summary(trained: tuple[RunData, Path]) -> None:
    data, root = trained
    summary_dir = evaluate(SYNTHETIC_RUN, output_root=root, data=data)
    assert all((summary_dir / name).exists() for name in SUMMARY_FILES)
    summary = pd.read_csv(summary_dir / "summary.csv")
    assert set(summary["config_id"]) == {"WR"} and summary["group"].iloc[0] == "all"
    assert int(summary["n_participants"].iloc[0]) == len(PARTICIPANTS)
    targets = pd.read_csv(summary_dir / "targets.csv")
    assert int(targets["target_s"].sum()) == int(data.mask.sum())
    assert len(pd.read_csv(summary_dir / "folds.csv")) == len(PARTICIPANTS)
    assert not (root / SYNTHETIC_RUN.run_id / "summary.tmp").exists()


def _tiny_fit(device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    data = make_synthetic()
    mask = shared_mask(data.cache, data.inclusion)
    fold = run_folds(SYNTHETIC_RUN, data.cache)[0]
    fold_data = build_fold_data(data.cache, mask, fold, ("WR",))
    arrays = to_device(fold_data.features, fold_data.target, device)
    branches = pack_layout((("WR", (0,)),), (seed,))
    return fit_packed(arrays, fold_data.train_targets, branches, TINY, dropout_seed=11)[0].state_dict


@pytest.mark.xpu
def test_same_job_on_xpu_is_bit_identical_and_seeds_differ() -> None:
    xpu = torch.device("xpu")
    configure_determinism(xpu)
    first, again, other = _tiny_fit(xpu, 0), _tiny_fit(xpu, 0), _tiny_fit(xpu, 1)
    for name in first:
        assert torch.equal(first[name], again[name]), name
    assert any(not torch.equal(first[name], other[name]) for name in first)
