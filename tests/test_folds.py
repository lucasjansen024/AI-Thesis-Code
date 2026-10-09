"""LOSO integrity, scaler leakage and window causality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.cache import ProcessedCache
from pipeline.fit import gather_windows, to_device
from pipeline.folds import (ColumnScaler, Fold, FoldIntegrityError, WINDOW_OFFSETS, build_fold_data, fit_scalers,
                            check_fold_integrity, check_windows, loso_folds, run_folds, window_rows)
from pipeline.inclusion import WINDOW_S, shared_mask
from pipeline.prepare.visit import ID_FIELDS
from pipeline.runs import A1_POOLED
from tests.synthetic import PARTICIPANTS, SYNTHETIC_RUN, ColumnEdit, Synthetic, make_synthetic

EXTREME = 1e6
FEATURES = ("WR",)
N_POOLED_FOLDS = 27


def _fold_scalers(edit: ColumnEdit | None = None) -> dict[str, dict[str, ColumnScaler]]:
    data = make_synthetic(edit=edit)
    mask = shared_mask(data.cache, data.inclusion)
    return {fold.test_participant: build_fold_data(data.cache, mask, fold, FEATURES).scalers
            for fold in run_folds(SYNTHETIC_RUN, data.cache)}


@pytest.mark.parametrize("test_participant", [p for p, _, _ in PARTICIPANTS])
def test_extreme_test_participant_values_leave_fold_scalers_unchanged(test_participant: str) -> None:
    def extreme(pid: str, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if pid != test_participant:
            return columns
        n = len(columns["t"])
        return {**columns, "WR": np.full(n, EXTREME), "VO2": np.linspace(-EXTREME, EXTREME, n)}

    assert _fold_scalers(extreme)[test_participant] == _fold_scalers()[test_participant]


def test_scaler_kinds_follow_the_domain_table(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    fold = run_folds(SYNTHETIC_RUN, synthetic.cache)[0]
    scalers = build_fold_data(synthetic.cache, mask, fold, FEATURES).scalers
    assert (scalers["WR"].kind, scalers["VO2"].kind) == ("minmax", "zscore")


def test_loso_has_one_fold_per_participant_and_never_trains_on_the_test_participant(synthetic: Synthetic) -> None:
    folds = loso_folds(synthetic.cache.visits)
    assert [f.test_participant for f in folds] == sorted(p for p, _, _ in PARTICIPANTS)
    for fold in folds:
        assert not any(v.startswith(fold.test_participant) for v in fold.train_visits)


def test_integrity_check_rejects_duplicates_and_leaks(synthetic: Synthetic) -> None:
    visits = synthetic.cache.visits
    fold = Fold("G01", frozenset({"G03_Ramp"}))
    with pytest.raises(FoldIntegrityError, match="duplicate"):
        check_fold_integrity((fold, fold), visits)
    with pytest.raises(FoldIntegrityError, match="test participant in training"):
        check_fold_integrity((Fold("G01", frozenset({"G01_Ramp", "G03_Ramp"})),), visits)


def test_scaler_fit_rows_never_include_the_test_participant(synthetic: Synthetic) -> None:
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    leaky = Fold("G01", frozenset({"G01_Ramp", "G03_Ramp"}))
    with pytest.raises(FoldIntegrityError, match="scaler-fit rows"):
        build_fold_data(synthetic.cache, mask, leaky, FEATURES)


def test_window_for_target_t_is_exactly_t_minus_187_to_t_minus_1() -> None:
    n_rows = 1000
    index_signal = np.arange(n_rows, dtype=np.float32)[:, None]
    data = to_device(index_signal, np.zeros(n_rows, dtype=np.float32), torch.device("cpu"))
    targets = np.array([WINDOW_S, 500, n_rows - 1])
    windows = gather_windows(data, torch.as_tensor(targets)).squeeze(-1).numpy()
    for target, window in zip(targets, windows):
        np.testing.assert_array_equal(window, np.arange(target - WINDOW_S, target))
    check_windows(targets, np.zeros(n_rows, dtype=np.int64))
    assert WINDOW_OFFSETS[-1] == -1 and len(WINDOW_OFFSETS) == WINDOW_S


def test_window_leaving_its_visit_raises() -> None:
    visit_starts = np.zeros(1000, dtype=np.int64)
    visit_starts[600:] = 600
    with pytest.raises(FoldIntegrityError, match="start before their visit"):
        check_windows(np.array([700]), visit_starts)


@pytest.mark.real
def test_pooled_loso_folds(real_cache: ProcessedCache) -> None:
    folds = run_folds(A1_POOLED, real_cache)
    test_ids = [f.test_participant for f in folds]
    assert len(folds) == N_POOLED_FOLDS
    assert len(set(test_ids)) == N_POOLED_FOLDS
    participant_of = dict(zip(real_cache.visits["visit_id"], real_cache.visits["participant_id"]))
    for fold in folds:
        assert fold.test_participant not in {participant_of[v] for v in fold.train_visits}


# --- scaler fit rows: the inputs the training windows read ------------------------------------------------------

def test_window_rows_are_t_minus_187_to_t_minus_1_within_each_visit() -> None:
    visit_stops = np.repeat([500, 900], [500, 400])  # two visits: rows 0-499 and 500-899
    rows = window_rows(np.array([300, 750]), visit_stops)
    np.testing.assert_array_equal(rows, np.concatenate([np.arange(300 - WINDOW_S, 300),
                                                        np.arange(750 - WINDOW_S, 750)]))


def test_feature_scaler_ignores_a_nan_at_the_target_second_itself() -> None:
    """Only t-187 ... t-1 must be present; the target's own second may lack an input (e.g. a Tymewear gap)."""
    n = 400
    wr = np.linspace(10.0, 300.0, n)
    wr[350] = np.nan
    vo2 = np.linspace(500.0, 3000.0, n)
    matrix = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n), wr, vo2]).astype(np.float32)
    visits = pd.DataFrame({field: ["X"] for field in ID_FIELDS}).assign(start=[0], stop=[n])
    cache = ProcessedCache("k", ("t", "phase", "unit", "WR", "VO2"), matrix, visits)
    targets = np.array([250, 350])  # the NaN sits at the later target's own second, in no window
    scalers = fit_scalers(cache, targets, ("WR",))
    assert scalers["WR"].offset == pytest.approx(float(matrix[250 - WINDOW_S, 3]))  # min-max: earliest window row
    assert scalers["VO2"].offset == pytest.approx(float(matrix[targets, 4].astype(np.float64).mean()))
