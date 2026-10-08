"""Full evaluation (slice 8): metric ports against legacy, seed averaging, VO₂peak windows, summary/ files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.evaluate import (SEEDAVG_COLUMNS, SUMMARY_COLUMNS, SUMMARY_FILES, VISIT_METRICS_SEEDAVG_COLUMNS,
                               VO2PEAK_COLUMNS, evaluate, seedavg_predictions, visit_metrics_seedavg)
from pipeline.inclusion import shared_mask
from pipeline.metrics import one_way_ms, rm_bland_altman, rm_corr_r, visit_metrics, vo2peak
from pipeline.runs import Run, TrainingSettings
from pipeline.train import PREDICTION_COLUMNS, PREDICTIONS_NAME, RunData, train_folds
from src2_training_ready.analysis import Analysis
from tests.synthetic import PARTICIPANTS, make_synthetic

# IDs that are not substrings of each other: legacy counts observations with `str.count`
SUBJECTS = ("P01", "P02", "P03", "P04", "P05")
OBSERVATIONS = (40, 55, 60, 75, 90)
EVAL_RUN = Run("synthetic_eval", ("Gies", "Lucas"), ("WR",), seeds=(0, 1), training=TrainingSettings(epochs=1))


def _legacy_frame(measured: np.ndarray, predicted: np.ndarray, subject: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Participant": subject, "Measured": measured, "Predicted": predicted})


def _subjects() -> np.ndarray:
    return np.repeat(SUBJECTS, OBSERVATIONS)


def test_rm_bland_altman_matches_legacy_when_ms_between_exceeds_ms_within() -> None:
    rng = np.random.default_rng(1)
    subject = _subjects()
    offsets = dict(zip(SUBJECTS, rng.normal(0, 150, len(SUBJECTS))))
    measured = rng.normal(3000, 500, len(subject))
    predicted = measured + np.array([offsets[s] for s in subject]) + rng.normal(0, 50, len(subject))
    ms_between, ms_within, _ = one_way_ms(predicted - measured, subject)
    assert ms_between > ms_within

    legacy = Analysis.bland_altman_global(_legacy_frame(measured, predicted, subject))
    ours = rm_bland_altman(measured, predicted, subject)
    assert ours.bias == pytest.approx(legacy.bias, rel=1e-9)
    assert ours.sd == pytest.approx(legacy.sd, rel=1e-9)
    assert ours.loa_hi == pytest.approx(ours.bias + 1.96 * ours.sd)
    assert ours.loa_lo == pytest.approx(ours.bias - 1.96 * ours.sd)


def test_rm_bland_altman_variance_is_ms_within_when_ms_between_is_smaller() -> None:
    rng = np.random.default_rng(2)
    subject = _subjects()
    noise = pd.Series(rng.normal(0, 80, len(subject)))
    diff = (noise - noise.groupby(subject).transform("mean")).to_numpy() + 5.0  # equal subject means
    measured = rng.normal(3000, 500, len(subject))
    predicted = measured + diff
    ms_between, ms_within, _ = one_way_ms(diff, subject)
    assert ms_between < ms_within

    ours = rm_bland_altman(measured, predicted, subject)
    assert ours.sd**2 == pytest.approx(ms_within, rel=1e-9)
    legacy = Analysis.bland_altman_global(_legacy_frame(measured, predicted, subject))
    assert legacy.sd < ours.sd  # legacy does not clamp the negative variance component


def test_rm_corr_matches_legacy() -> None:
    rng = np.random.default_rng(3)
    subject = _subjects()
    offsets = dict(zip(SUBJECTS, rng.normal(0, 400, len(SUBJECTS))))
    measured = rng.normal(2500, 600, len(subject)) + np.array([offsets[s] for s in subject])
    predicted = 0.8 * measured + rng.normal(0, 250, len(subject)) - np.array([offsets[s] for s in subject])
    legacy = Analysis.rm_corr(_legacy_frame(measured, predicted, subject))
    assert rm_corr_r(measured, predicted, subject) == pytest.approx(legacy.r, abs=1e-9)


def _toy_predictions() -> pd.DataFrame:
    true = np.array([1000.0, 1500.0, 2000.0, 2500.0])
    frames = [pd.DataFrame({"run_id": "toy", "config_id": "WR", "seed": seed, "fold": "G01",
                            "participant_id": "G01", "cohort": "Gies", "protocol": "Ramp", "visit_id": "G01_Ramp",
                            "t_s": np.arange(len(true)), "work_rate": 100.0, "vo2_true": true,
                            "vo2_pred": true + shift})
              for seed, shift in ((0, 10.0), (1, -10.0))]
    return pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]


def test_seed_averaged_rmse_is_not_the_ensemble_rmse() -> None:
    predictions = _toy_predictions()
    seed_averaged = visit_metrics_seedavg(visit_metrics(predictions))
    ensemble = visit_metrics(seedavg_predictions(predictions).assign(seed=-1))
    assert seed_averaged["rmse"].iloc[0] == pytest.approx(10.0)
    assert ensemble["rmse"].iloc[0] == pytest.approx(0.0)


def test_vo2peak_uses_only_full_windows() -> None:
    # 30 contiguous seconds at 100, then a 21-s run at 1000: only the first is a full window
    t = np.concatenate([np.arange(0, 30), np.arange(40, 61)])
    vo2 = np.where(t < 30, 100.0, 1000.0)
    assert vo2peak(t, vo2, 30) == pytest.approx(100.0)
    assert np.isnan(vo2peak(np.arange(29), np.ones(29), 30))
    # values = t on 0..59: trailing mean of the last 30 s; a missing second 45 voids every window holding it
    t_full = np.arange(60)
    assert vo2peak(t_full, t_full.astype(float), 30) == pytest.approx(np.arange(30, 60).mean())
    t_gap = t_full[t_full != 45]
    assert vo2peak(t_gap, t_gap.astype(float), 30) == pytest.approx(np.arange(15, 45).mean())


def test_seedavg_columns_are_the_prediction_columns_minus_seed() -> None:
    assert SEEDAVG_COLUMNS == [c for c in PREDICTION_COLUMNS if c != "seed"]


@pytest.fixture
def summary_dir(tmp_path: Path) -> Path:
    synthetic = make_synthetic()
    data = RunData(synthetic.cache, synthetic.inclusion, "synthetic-sha",
                   shared_mask(synthetic.cache, synthetic.inclusion))
    train_folds(EVAL_RUN, data, output_root=tmp_path, device=torch.device("cpu"))
    return evaluate(EVAL_RUN, output_root=tmp_path, data=data)


def test_evaluate_writes_every_summary_file_with_the_documented_columns(summary_dir: Path) -> None:
    assert sorted(p.name for p in summary_dir.iterdir()) == sorted(SUMMARY_FILES)
    assert not (summary_dir.parent / "summary.tmp").exists()

    summary = pd.read_csv(summary_dir / "summary.csv")
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary["group"]) == ["all", "cohort=Gies", "cohort=Lucas", "protocol=Ramp", "protocol=Step"]
    assert (summary["n_seeds"] == len(EVAL_RUN.seeds)).all()
    by_group = summary.set_index("group")
    assert by_group.loc["all", "n_participants"] == len(PARTICIPANTS)
    assert np.isfinite(by_group.loc["all", ["r_rm_mean", "r_p_mean", "ba_loa_hi_mean", "rmse_sd_seeds"]].astype(float)).all()
    assert np.isnan(by_group.loc["protocol=Step", "vo2peak_diff_mean"])
    assert np.isfinite(by_group.loc["protocol=Ramp", "vo2peak_abs_error_mean"])

    run_dir = summary_dir.parent
    per_seed = pd.concat([pd.read_parquet(run_dir / pid / PREDICTIONS_NAME) for pid, _, _ in PARTICIPANTS])
    seedavg = pd.read_parquet(summary_dir / "predictions_seedavg.parquet")
    assert list(seedavg.columns) == SEEDAVG_COLUMNS
    assert len(seedavg) * len(EVAL_RUN.seeds) == len(per_seed)

    visits = pd.read_csv(summary_dir / "visit_metrics_seedavg.csv")
    assert list(visits.columns) == VISIT_METRICS_SEEDAVG_COLUMNS
    assert len(visits) == len(PARTICIPANTS)

    peaks = pd.read_csv(summary_dir / "vo2peak.csv")
    assert list(peaks.columns) == VO2PEAK_COLUMNS
    ramp_visits = {f"{pid}_{protocol}" for pid, _, protocol in PARTICIPANTS if protocol == "Ramp"}
    assert set(peaks["visit_id"]) == ramp_visits
    assert (peaks["n_seeds"] == len(EVAL_RUN.seeds)).all()
