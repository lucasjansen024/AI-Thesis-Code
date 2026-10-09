"""Folds, fold scalers and target indices. In memory only; nothing per fold goes to disk.

Runtime checks (explicit exceptions, never `assert`): unique test IDs; test participant not in training;
scaler-fit rows exclude the test participant; the window of every target is exactly t-187 ... t-1 inside its
visit; every gathered input and target is finite.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.cache import ProcessedCache
from pipeline.domains import SCALING, TARGET_COLUMN, Scaling
from pipeline.inclusion import WINDOW_S, context_present
from pipeline.runs import Run

WINDOW_OFFSETS = np.arange(-WINDOW_S, 0)  # rows gathered for target t: t-187 ... t-1
CHECK_CHUNK = 8192


class FoldIntegrityError(RuntimeError):
    """A fold violates LOSO integrity, leaks the test participant, or gathers the wrong window."""


@dataclass(frozen=True)
class Fold:
    test_participant: str
    train_visits: frozenset[str]  # visit_ids; Aim 2 may pass any explicit list


@dataclass(frozen=True)
class ColumnScaler:
    column: str
    kind: str  # Scaling value
    offset: float
    scale: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.offset) / self.scale

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=np.float64) * self.scale + self.offset


@dataclass(frozen=True)
class FoldData:
    fold: Fold | ValidationFold  # ValidationFold: pilot split, `test_targets` = validation participants' rows
    feature_columns: tuple[str, ...]
    features: np.ndarray  # float32 (n_rows, n_features), scaled, all cache rows
    target: np.ndarray  # float32 (n_rows,), scaled V̇O₂
    train_targets: np.ndarray  # int64 cache rows
    test_targets: np.ndarray  # int64 cache rows
    scalers: dict[str, ColumnScaler]  # feature columns + TARGET_COLUMN
    n_train_participants: int


def check_fold_integrity(folds: tuple[Fold, ...], visits: pd.DataFrame) -> None:
    test_ids = [f.test_participant for f in folds]
    duplicated = sorted({p for p in test_ids if test_ids.count(p) > 1})
    if duplicated:
        raise FoldIntegrityError(f"duplicate test participants {duplicated}")
    participant_of = dict(zip(visits["visit_id"], visits["participant_id"]))
    for fold in folds:
        unknown = sorted(fold.train_visits - set(participant_of))
        if unknown:
            raise FoldIntegrityError(f"fold {fold.test_participant}: unknown training visits {unknown}")
        leaked = sorted(v for v in fold.train_visits if participant_of[v] == fold.test_participant)
        if leaked:
            raise FoldIntegrityError(f"fold {fold.test_participant}: test participant in training via {leaked}")


def loso_folds(visits: pd.DataFrame) -> tuple[Fold, ...]:
    """One fold per participant in `visits`; the others' visits train it."""
    folds = tuple(
        Fold(p, frozenset(visits.loc[visits["participant_id"] != p, "visit_id"]))
        for p in sorted(visits["participant_id"].unique())
    )
    check_fold_integrity(folds, visits)
    return folds


def run_folds(run: Run, cache: ProcessedCache) -> tuple[Fold, ...]:
    """LOSO over the run's cohorts, limited to `run.fold_ids` when the run names its folds."""
    folds = loso_folds(cache.visits[cache.visits["cohort"].isin(run.cohorts)])
    if run.fold_ids is None:
        return folds
    by_id = {f.test_participant: f for f in folds}
    unknown = sorted(set(run.fold_ids) - set(by_id))
    if unknown:
        raise ValueError(f"{run.run_id}: fold_ids {unknown} not in its cohorts {run.cohorts}")
    return tuple(by_id[p] for p in run.fold_ids)


def fit_scaler(column: str, values: np.ndarray) -> ColumnScaler:
    kind = SCALING[column]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"scaler for {column}: non-finite values in the fit rows")
    if kind == Scaling.MINMAX:
        offset, scale = values.min(), values.max() - values.min()
    elif kind == Scaling.ZSCORE:
        offset, scale = values.mean(), values.std()
    else:
        offset, scale = 0.0, 1.0
    if not scale > 0:
        raise ValueError(f"scaler for {column} ({kind}): zero spread in the fit rows")
    return ColumnScaler(column, str(kind), float(offset), float(scale))


def window_rows(targets: np.ndarray, visit_stops: np.ndarray) -> np.ndarray:
    """Rows read by at least one target's window (t-187 ... t-1), never across a visit boundary.

    `visit_stops`: per row, the end (exclusive) of its visit. Row r is read iff some target t in r's visit has
    r + 1 <= t <= r + 187.
    """
    n_rows = len(visit_stops)
    target_counts = np.concatenate(([0], np.cumsum(np.isin(np.arange(n_rows), targets))))  # targets among rows < i
    rows = np.arange(n_rows)
    last = np.minimum(rows + WINDOW_S, visit_stops - 1)
    return np.flatnonzero(target_counts[last + 1] - target_counts[rows + 1] > 0)


def fit_scalers(cache: ProcessedCache, train_targets: np.ndarray,
                feature_columns: tuple[str, ...]) -> dict[str, ColumnScaler]:
    """Feature scalers on the rows the training windows read; the V̇O₂ scaler on the training targets.

    A target's own second is never a model input and may lack its inputs (only t-187 ... t-1 must be present), so
    fitting features on target rows would include values the model never sees, or NaN.
    """
    visit_stops = cache.visits["stop"].to_numpy()[cache.row_visit()]
    inputs = window_rows(train_targets, visit_stops)
    scalers = {c: fit_scaler(c, cache.column(c)[inputs]) for c in feature_columns}
    return {**scalers, TARGET_COLUMN: fit_scaler(TARGET_COLUMN, cache.column(TARGET_COLUMN)[train_targets])}


def check_windows(targets: np.ndarray, visit_starts: np.ndarray) -> None:
    """Every target's gathered rows end at t-1 and start inside the target's visit."""
    if WINDOW_OFFSETS[-1] != -1 or len(WINDOW_OFFSETS) != WINDOW_S:
        raise FoldIntegrityError(f"window offsets {WINDOW_OFFSETS[[0, -1]]} do not cover t-{WINDOW_S} ... t-1")
    for begin in range(0, len(targets), CHECK_CHUNK):
        chunk = targets[begin:begin + CHECK_CHUNK]
        rows = chunk[:, None] + WINDOW_OFFSETS
        if not (rows[:, -1] == chunk - 1).all():
            raise FoldIntegrityError("last gathered window row is not t-1")
        outside = rows[:, 0] < visit_starts[chunk]
        if outside.any():
            raise FoldIntegrityError(f"{int(outside.sum())} windows start before their visit")


def _check_finite(features: np.ndarray, target: np.ndarray, targets: np.ndarray, visit_starts: np.ndarray,
                  label: str) -> None:
    inputs_ok = context_present(np.isfinite(features).all(axis=1), visit_starts, WINDOW_S)[targets]
    target_ok = np.isfinite(target[targets])
    bad = ~(inputs_ok & target_ok)
    if bad.any():
        raise FoldIntegrityError(f"{label}: {int(bad.sum())} targets with non-finite inputs or target")


def build_fold_data(cache: ProcessedCache, mask: np.ndarray, fold: Fold,
                    feature_columns: tuple[str, ...]) -> FoldData:
    row_participant = cache.row_ids("participant_id")
    train_rows = np.flatnonzero(mask & np.isin(cache.row_ids("visit_id"), list(fold.train_visits)))
    test_rows = np.flatnonzero(mask & (row_participant == fold.test_participant))
    if len(train_rows) == 0 or len(test_rows) == 0:
        raise ValueError(f"fold {fold.test_participant}: {len(train_rows)} train / {len(test_rows)} test targets")
    fit_participants = set(row_participant[train_rows])
    if fold.test_participant in fit_participants:
        raise FoldIntegrityError(f"fold {fold.test_participant}: scaler-fit rows include the test participant")
    scalers = fit_scalers(cache, train_rows, feature_columns)
    features = np.column_stack([scalers[c].transform(cache.column(c)) for c in feature_columns]).astype(np.float32)
    target = scalers[TARGET_COLUMN].transform(cache.column(TARGET_COLUMN)).astype(np.float32)
    visit_starts = cache.visits["start"].to_numpy()[cache.row_visit()]
    for label, rows in (("train", train_rows), ("test", test_rows)):
        check_windows(rows, visit_starts)
        _check_finite(features, target, rows, visit_starts, f"fold {fold.test_participant} {label}")
    return FoldData(fold, feature_columns, features, target, train_rows, test_rows, scalers, len(fit_participants))


@dataclass(frozen=True)
class ValidationFold:
    """A pilot validation split: several held-out participants; the rest of the split's cohorts train it."""

    split_id: str
    validation_participants: frozenset[str]
    train_visits: frozenset[str]


def validation_fold(split_id: str, validation_participants: tuple[str, ...], cohorts: tuple[str, ...],
                    visits: pd.DataFrame) -> ValidationFold:
    """Held-out participants of `cohorts`; every other visit of those cohorts trains the split."""
    pool = visits[visits["cohort"].isin(cohorts)]
    held_out = frozenset(validation_participants)
    unknown = sorted(held_out - set(pool["participant_id"]))
    if unknown or len(held_out) != len(validation_participants):
        raise FoldIntegrityError(f"split {split_id}: validation participants {list(validation_participants)} "
                                 f"repeated or not in cohorts {cohorts} (unknown {unknown})")
    fold = ValidationFold(split_id, held_out,
                          frozenset(pool.loc[~pool["participant_id"].isin(held_out), "visit_id"]))
    participant_of = dict(zip(visits["visit_id"], visits["participant_id"]))
    leaked = sorted(v for v in fold.train_visits if participant_of[v] in held_out)
    if leaked or not fold.train_visits:
        raise FoldIntegrityError(f"split {split_id}: {len(fold.train_visits)} training visits, leaked {leaked}")
    return fold


def build_validation_data(cache: ProcessedCache, mask: np.ndarray, fold: ValidationFold,
                          feature_columns: tuple[str, ...]) -> FoldData:
    """`build_fold_data` for a validation split: `test_targets` are the validation participants' target rows.

    Same runtime checks; additionally every validation participant must have targets. `FoldData.fold` holds the
    `ValidationFold`.
    """
    row_participant = cache.row_ids("participant_id")
    train_rows = np.flatnonzero(mask & np.isin(cache.row_ids("visit_id"), list(fold.train_visits)))
    held_out_rows = mask & np.isin(row_participant, list(fold.validation_participants))
    val_rows = np.flatnonzero(held_out_rows)
    without_targets = sorted(fold.validation_participants - set(row_participant[val_rows]))
    if len(train_rows) == 0 or without_targets:
        raise ValueError(f"split {fold.split_id}: {len(train_rows)} train targets; validation participants "
                         f"without targets {without_targets}")
    fit_participants = set(row_participant[train_rows])
    if fit_participants & fold.validation_participants:
        raise FoldIntegrityError(f"split {fold.split_id}: scaler-fit rows include validation participants "
                                 f"{sorted(fit_participants & fold.validation_participants)}")
    scalers = fit_scalers(cache, train_rows, feature_columns)
    features = np.column_stack([scalers[c].transform(cache.column(c)) for c in feature_columns]).astype(np.float32)
    target = scalers[TARGET_COLUMN].transform(cache.column(TARGET_COLUMN)).astype(np.float32)
    visit_starts = cache.visits["start"].to_numpy()[cache.row_visit()]
    for label, rows in (("train", train_rows), ("validation", val_rows)):
        check_windows(rows, visit_starts)
        _check_finite(features, target, rows, visit_starts, f"split {fold.split_id} {label}")
    return FoldData(fold, feature_columns, features, target, train_rows, val_rows, scalers, len(fit_participants))
