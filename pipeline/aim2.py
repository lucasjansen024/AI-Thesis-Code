"""Aim 2 training (METHODOLOGY Appendix B): left-out-condition arms (2a/2b) and cohort transfer (2c).

Additive on top of `pipeline.train`; the Aim 1 code paths are unchanged. Every Aim 2 fold folder has the Aim 1
layout (`outputs/runs/<run_id>/<participant_id>/`), so `evaluate(spec.run)` summarises it unchanged.

- Left-out arms: one `train.train_fold` per test participant. The seen arm's subsample is a per-fold copy of the
  shared mask with the dropped training targets cleared, so `build_fold_data` fits the scalers on the subsample and
  the test participant's targets are untouched.
- Cohort transfer: one packed job on the training cohort. Its models predict every test participant, and each
  participant's outputs are written as a fold folder of its own (one shared training, recorded in the manifest).
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from pipeline.cache import ProcessedCache
from pipeline.domains import feature_union, parse_configuration
from pipeline.fit import configure_determinism, fit_packed, pack_layout, pick_device, to_device
from pipeline.folds import Fold, ValidationFold, build_validation_data, check_fold_integrity
from pipeline.provenance import derive_seed, utc_now
from pipeline.runs import CohortTransfer, LeftOutArm
from pipeline.train import (MANIFEST_NAME, MODELS_NAME, RUNS_ROOT, TMP_SUFFIX, JobResult, RunData, TrainedModel,
                            _manifest, _save_models, _standalone_model, _write_outputs, load_run_data, mask_groups,
                            select_folds, train_fold)

logger = logging.getLogger(__name__)


class UnscorableParticipantError(ValueError):
    """A spec's test participant has nothing to score (no left-out targets, or not in the test cohort)."""


class PartialTransferError(RuntimeError):
    """Some, but not all, test participants of a cohort transfer have finished folders."""


# --- 2a/2b: left-out arms ---------------------------------------------------------------------------------------

def train_visits(arm: LeftOutArm, visits: pd.DataFrame, test_participant: str, is_seen: bool) -> frozenset[str]:
    """The other participants' pool visits of the arm's cohort; without `left_out` unless `is_seen`."""
    protocols = arm.pool if is_seen else tuple(p for p in arm.pool if p != arm.left_out)
    chosen = ((visits["cohort"] == arm.run.cohorts[0]) & visits["protocol"].isin(protocols)
              & (visits["participant_id"] != test_participant))
    return frozenset(visits.loc[chosen, "visit_id"])


def left_out_folds(arm: LeftOutArm, visits: pd.DataFrame) -> tuple[Fold, ...]:
    """One fold per test participant (`run.fold_ids`), trained on the arm's visits."""
    folds = tuple(Fold(p, train_visits(arm, visits, p, arm.is_seen)) for p in arm.run.fold_ids)
    check_fold_integrity(folds, visits)
    return folds


def check_test_participants(arm: LeftOutArm, cache: ProcessedCache, mask: np.ndarray) -> None:
    """Every test participant has `left_out` targets in the arm's cohort, else the arm pair has nothing to score."""
    scored = mask & (cache.row_ids("protocol") == arm.left_out) & (cache.row_ids("cohort") == arm.run.cohorts[0])
    missing = sorted(set(arm.run.fold_ids) - set(cache.row_ids("participant_id")[scored]))
    if missing:
        raise UnscorableParticipantError(f"{arm.run.run_id}: no {arm.left_out} targets for test participants {missing}")


def subsampled_mask(arm: LeftOutArm, cache: ProcessedCache, mask: np.ndarray, fold: Fold) -> np.ndarray:
    """The seen arm's mask for one fold (Appendix B.4); a new array, `mask` is not changed.

    Keeps exactly the unseen arm's number of training targets, drawn uniformly without replacement from the
    fold's training targets with a seed from (run, fold). Rows outside the fold's training visits are unchanged.
    """
    visit_ids = cache.row_ids("visit_id")
    unseen = train_visits(arm, cache.visits, fold.test_participant, is_seen=False)
    n_keep = int((mask & np.isin(visit_ids, list(unseen))).sum())
    pool_rows = np.flatnonzero(mask & np.isin(visit_ids, list(fold.train_visits)))
    if not 0 < n_keep < len(pool_rows):
        raise ValueError(f"{arm.run.run_id} fold {fold.test_participant}: keep {n_keep} of {len(pool_rows)} "
                         f"training targets; the seen pool must add {arm.left_out} targets to a non-empty unseen set")
    rng = np.random.default_rng(derive_seed("subsample", arm.run.run_id, fold.test_participant))
    kept = rng.choice(pool_rows, size=n_keep, replace=False)
    subsampled = mask.copy()
    subsampled[pool_rows] = False
    subsampled[kept] = True
    return subsampled


def run_left_out(arm: LeftOutArm, folds: Sequence[str] | None = None, rebuild_cache: bool = False) -> list[Path]:
    """Train the arm's folds (all, or the named test participants); finished folds are skipped."""
    return train_left_out(arm, load_run_data(arm.run, rebuild_cache), folds)


def train_left_out(arm: LeftOutArm, data: RunData, folds: Sequence[str] | None = None,
                   output_root: Path = RUNS_ROOT, device: torch.device | None = None) -> list[Path]:
    device = pick_device() if device is None else device
    configure_determinism(device)
    check_test_participants(arm, data.cache, data.mask)
    selected = select_folds(left_out_folds(arm, data.cache.visits), folds)
    logger.info("%s: %d folds, %s left out, %s arm", arm.run.run_id, len(selected), arm.left_out,
                "seen" if arm.is_seen else "unseen")
    paths = []
    for fold in selected:
        fold_data = replace(data, mask=subsampled_mask(arm, data.cache, data.mask, fold)) if arm.is_seen else data
        paths.append(train_fold(arm.run, fold_data, fold, output_root / arm.run.run_id, device))
    return paths


# --- 2c: cohort transfer ----------------------------------------------------------------------------------------

def transfer_fold(spec: CohortTransfer, visits: pd.DataFrame) -> ValidationFold:
    """Every visit of the training cohort trains; the test participants must all belong to the test cohort."""
    tested = visits[visits["participant_id"].isin(spec.run.fold_ids)]
    outside = sorted(set(spec.run.fold_ids) - set(tested.loc[tested["cohort"] == spec.test_cohort, "participant_id"]))
    if outside:
        raise UnscorableParticipantError(f"{spec.run.run_id}: test participants {outside} not in {spec.test_cohort}")
    return ValidationFold(spec.run.run_id, frozenset(spec.run.fold_ids),
                          frozenset(visits.loc[visits["cohort"] == spec.train_cohort, "visit_id"]))


def run_transfer(spec: CohortTransfer, rebuild_cache: bool = False) -> list[Path]:
    """Train the transfer once and write one folder per test participant; skipped when all are finished."""
    return train_transfer(spec, load_run_data(spec.run, rebuild_cache))


def train_transfer(spec: CohortTransfer, data: RunData, output_root: Path = RUNS_ROOT,
                   device: torch.device | None = None) -> list[Path]:
    run_dir = output_root / spec.run.run_id
    finals = [run_dir / p for p in spec.run.fold_ids]
    finished = [f.name for f in finals if f.exists()]
    if len(finished) == len(finals):
        logger.info("%s: all %d test folders finished, skipped", spec.run.run_id, len(finals))
        return finals
    if finished:
        raise PartialTransferError(f"{spec.run.run_id}: only {finished} of {len(finals)} test folders finished; "
                                   f"they share one training, so delete them and rerun")
    device = pick_device() if device is None else device
    configure_determinism(device)
    started, clock = utc_now(), time.perf_counter()
    configurations = tuple(parse_configuration(c) for c in spec.run.configurations)
    (group,) = mask_groups(spec.run, configurations, data.mask)  # shared mask: one group
    fold = transfer_fold(spec, data.cache.visits)
    fold_data = build_validation_data(data.cache, group.mask, fold, feature_union(configurations))
    column_index = {c.config_id: tuple(fold_data.feature_columns.index(col) for col in c.columns)
                    for c in configurations}
    branches = pack_layout(tuple(column_index.items()), spec.run.seeds)
    dropout_seed = derive_seed("dropout", spec.run.run_id, spec.train_cohort, group.index)
    logger.info("%s: %s -> %s, packed job K=%d, %d training targets", spec.run.run_id, spec.train_cohort,
                spec.test_cohort, len(branches), len(fold_data.train_targets))
    fits = fit_packed(to_device(fold_data.features, fold_data.target, device), fold_data.train_targets, branches,
                      spec.run.training, dropout_seed)
    by_branch = {(b.config_id, b.seed): fit for b, fit in zip(branches, fits)}
    wall_s = time.perf_counter() - clock
    config_data = {config_id: to_device(fold_data.features[:, list(columns)], fold_data.target, device)
                   for config_id, columns in column_index.items()}
    row_participant = data.cache.row_ids("participant_id")
    paths = []
    for participant in spec.run.fold_ids:
        rows = fold_data.test_targets[row_participant[fold_data.test_targets] == participant]
        participant_data = replace(fold_data, fold=Fold(participant, fold.train_visits), test_targets=rows)
        models = tuple(
            _standalone_model(spec.run.run_id, data.cache, participant_data, config_data[config_id], config_id, seed,
                              by_branch[(config_id, seed)])
            for config_id in column_index for seed in spec.run.seeds
        )
        job = JobResult(group, participant_data, branches, dropout_seed, models)
        paths.append(_write_transfer_folder(spec, data, job, device, started, wall_s, run_dir))
    return paths


def _write_transfer_folder(spec: CohortTransfer, data: RunData, job: JobResult, device: torch.device, started: str,
                           wall_s: float, run_dir: Path) -> Path:
    """One test participant's fold folder, written via `.tmp` + rename like `train.train_fold`."""
    participant = job.fold_data.fold.test_participant
    tmp = run_dir / f"{participant}{TMP_SUFFIX}"
    if tmp.exists():
        logger.info("%s: wiping leftover %s", spec.run.run_id, tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    models: list[TrainedModel] = list(job.models)
    _write_outputs(tmp, models)
    _save_models(tmp / MODELS_NAME, models)
    manifest = {
        **_manifest(spec.run, data, [job], device, started, wall_s),
        "transfer": {"train_cohort": spec.train_cohort, "test_participants": list(spec.run.fold_ids),
                     "shared_training": "one packed job for every test participant; wall_s is that job's"},
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    final = run_dir / participant
    tmp.rename(final)
    return final
