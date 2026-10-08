"""`run(run, folds=...)`: cache -> inclusion -> shared mask -> LOSO folds -> train -> fold outputs, resumable.

Fold folder `outputs/runs/<run_id>/<participant_id>/` holds `predictions.parquet`, `visit_metrics.csv`,
`epochs.csv`, `models.pt` and `manifest.json`. It is written as `<participant_id>.tmp/` and renamed after the
manifest; a finished fold is skipped and a leftover `.tmp` is wiped.

Training unit = one packed job per (fold, mask group): all of the group's configurations x the run's seeds
(`fit.fit_packed`). Under the shared mask a fold has exactly one mask group. Under `per_config` the configurations
whose target sets within the fold are identical form one group (seed streams are then those of unpacked runs);
the groups train one after another into the same fold folder, so the fold stays the unit of resuming.
Predictions come from the extracted standalone models only.

Manifest: fold-level `n_train_participants`, `n_train_targets`, `n_test_targets` and `feature_columns` cover the
union of the fold's mask groups (= the single group under the shared mask); `scalers` is the fold's scalers when
it has one mask group and null otherwise. Each `packing.jobs` entry carries its own sizes and scalers.
"""
from __future__ import annotations

import json
import math
import logging
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from inventory.build_inclusion import INCLUSION_CSV
from pipeline.cache import ProcessedCache, load_or_build
from pipeline.domains import TARGET_COLUMN, Configuration, feature_union, parse_configuration
from pipeline.fit import (Branch, DeviceData, FitResult, configure_determinism, device_name, fit_packed,
                          pack_layout, pick_device, predict, to_device)
from pipeline.folds import Fold, FoldData, build_fold_data, run_folds
from pipeline.inclusion import per_config_masks, read_inclusion, shared_mask
from pipeline.metrics import visit_metrics
from pipeline.provenance import REPO_ROOT, derive_seed, git_state, package_versions, sha256_file, utc_now
from pipeline.runs import PER_CONFIG_MASK, SHARED_MASK, Run

logger = logging.getLogger(__name__)

RUNS_ROOT = REPO_ROOT / "outputs" / "runs"
TMP_SUFFIX = ".tmp"
MANIFEST_NAME = "manifest.json"
PREDICTIONS_NAME, VISIT_METRICS_NAME, EPOCHS_NAME = "predictions.parquet", "visit_metrics.csv", "epochs.csv"
PREDICTION_COLUMNS = ["run_id", "config_id", "seed", "fold", "participant_id", "cohort", "protocol", "visit_id",
                      "t_s", "work_rate", "vo2_true", "vo2_pred"]
EPOCH_COLUMNS = ["config_id", "seed", "epoch", "train_loss", "is_best"]
WORK_RATE_COLUMN = "WR"
MODELS_NAME = "models.pt"
BRANCH_LAYOUT = "seed-major: branch = seed_index * n_configurations + configuration_index"


@dataclass(frozen=True)
class RunData:
    cache: ProcessedCache
    inclusion: pd.DataFrame
    inclusion_sha256: str
    mask: np.ndarray  # bool per cache row: target second under the run's mask (per_config: of any configuration)
    config_masks: Mapping[str, np.ndarray] = field(default_factory=dict)  # per_config only: {config_id: mask}


@dataclass(frozen=True)
class TrainedModel:
    config_id: str
    seed: int
    fit: FitResult
    predictions: pd.DataFrame


@dataclass(frozen=True)
class MaskGroup:
    """Configurations of one fold whose target sets are identical; they train as one packed job."""

    index: int
    configurations: tuple[Configuration, ...]
    mask: np.ndarray  # bool per cache row: the group's target seconds


@dataclass(frozen=True)
class JobResult:
    group: MaskGroup
    fold_data: FoldData
    branches: tuple[Branch, ...]
    dropout_seed: int
    models: tuple[TrainedModel, ...]  # configuration order, then seed order


def load_run_data(run_spec: Run, rebuild_cache: bool = False, inclusion_path: Path = INCLUSION_CSV) -> RunData:
    cache = load_or_build(run_spec.prep, rebuild_cache, inclusion_path=inclusion_path)
    return build_run_data(run_spec, cache, read_inclusion(inclusion_path), sha256_file(inclusion_path))


def build_run_data(run_spec: Run, cache: ProcessedCache, inclusion: pd.DataFrame, inclusion_sha256: str) -> RunData:
    """The run's target mask(s): the shared mask, or one mask per configuration under `per_config`."""
    if run_spec.mask == SHARED_MASK:
        return RunData(cache, inclusion, inclusion_sha256, shared_mask(cache, inclusion))
    if run_spec.mask == PER_CONFIG_MASK:
        configurations = tuple(parse_configuration(c) for c in run_spec.configurations)
        masks = per_config_masks(cache, inclusion, configurations)
        return RunData(cache, inclusion, inclusion_sha256, np.logical_or.reduce(list(masks.values())), masks)
    raise ValueError(f"{run_spec.run_id}: mask {run_spec.mask!r} has no target-mask rule")


def run(run_spec: Run, folds: Sequence[str] | None = None, rebuild_cache: bool = False) -> None:
    """Train the run's folds (all, or the named test participants); finished folds are skipped."""
    train_folds(run_spec, load_run_data(run_spec, rebuild_cache), folds)


def select_folds(all_folds: tuple[Fold, ...], wanted: Sequence[str] | None) -> tuple[Fold, ...]:
    if wanted is None:
        return all_folds
    by_id = {f.test_participant: f for f in all_folds}
    unknown = sorted(set(wanted) - set(by_id))
    if unknown:
        raise ValueError(f"folds {unknown} are not folds of this run (have {sorted(by_id)})")
    return tuple(by_id[p] for p in wanted)


def train_folds(run_spec: Run, data: RunData, folds: Sequence[str] | None = None, output_root: Path = RUNS_ROOT,
                device: torch.device | None = None) -> list[Path]:
    device = pick_device() if device is None else device
    configure_determinism(device)
    selected = select_folds(run_folds(run_spec, data.cache), folds)
    logger.info("%s: %d folds on %s (%s)", run_spec.run_id, len(selected), device, device_name(device))
    return [train_fold(run_spec, data, fold, output_root / run_spec.run_id, device) for fold in selected]


def train_fold(run_spec: Run, data: RunData, fold: Fold, run_dir: Path, device: torch.device) -> Path:
    final = run_dir / fold.test_participant
    if final.exists():
        logger.info("%s: fold %s finished, skipped", run_spec.run_id, fold.test_participant)
        return final
    tmp = run_dir / f"{fold.test_participant}{TMP_SUFFIX}"
    if tmp.exists():
        logger.info("%s: wiping leftover %s", run_spec.run_id, tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    started, clock = utc_now(), time.perf_counter()
    configurations = tuple(parse_configuration(c) for c in run_spec.configurations)
    groups = mask_groups(run_spec, configurations, data.mask, config_masks=data.config_masks,
                         fold_rows=fold_row_mask(data.cache, fold))
    logger.info("%s: fold %s has %d mask groups", run_spec.run_id, fold.test_participant, len(groups))
    jobs = [_train_job(run_spec, data.cache, fold, group, device) for group in groups]
    models = [m for job in jobs for m in job.models]
    _write_outputs(tmp, models)
    _save_models(tmp / MODELS_NAME, models)
    manifest = _manifest(run_spec, data, jobs, device, started, time.perf_counter() - clock)
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(final)
    logger.info("%s: fold %s written to %s", run_spec.run_id, fold.test_participant, final)
    return final


def fold_row_mask(cache: ProcessedCache, fold: Fold) -> np.ndarray:
    """Bool per cache row: the row can be a target of this fold (a training visit or the test participant)."""
    return (np.isin(cache.row_ids("visit_id"), list(fold.train_visits))
            | (cache.row_ids("participant_id") == fold.test_participant))


def mask_groups(run_spec: Run, configurations: tuple[Configuration, ...], mask: np.ndarray, *,
                config_masks: Mapping[str, np.ndarray] | None = None,
                fold_rows: np.ndarray | None = None) -> tuple[MaskGroup, ...]:
    """The fold's packed jobs.

    Shared mask: exactly one group holding every configuration (`mask`). `per_config`: configurations whose
    masks agree on `fold_rows` (identical train and test targets) form one group, in order of first appearance;
    a group's `mask` is its first configuration's mask.
    """
    if run_spec.mask == SHARED_MASK:
        return (MaskGroup(0, configurations, mask),)
    if run_spec.mask != PER_CONFIG_MASK:
        raise ValueError(f"{run_spec.run_id}: mask {run_spec.mask!r} has no mask-group rule")
    if config_masks is None or fold_rows is None:
        raise ValueError(f"{run_spec.run_id}: per_config mask groups need config_masks and fold_rows")
    missing = [c.config_id for c in configurations if c.config_id not in config_masks]
    if missing:
        raise ValueError(f"{run_spec.run_id}: no per_config mask for {missing}")
    members: dict[bytes, list[Configuration]] = {}
    for config in configurations:
        key = np.packbits(config_masks[config.config_id] & fold_rows).tobytes()
        members.setdefault(key, []).append(config)
    return tuple(MaskGroup(index, tuple(group), config_masks[group[0].config_id])
                 for index, group in enumerate(members.values()))


def _train_job(run_spec: Run, cache: ProcessedCache, fold: Fold, group: MaskGroup,
               device: torch.device) -> JobResult:
    """One packed job: every configuration of the group x every seed; predictions from the extracted models."""
    fold_data = build_fold_data(cache, group.mask, fold, feature_union(group.configurations))
    union = fold_data.feature_columns
    column_index = {c.config_id: tuple(union.index(col) for col in c.columns) for c in group.configurations}
    branches = pack_layout(tuple(column_index.items()), run_spec.seeds)
    dropout_seed = derive_seed("dropout", run_spec.run_id, fold.test_participant, group.index)
    logger.info("%s: fold %s mask group %d: packed job K=%d (%d configurations x %d seeds)", run_spec.run_id,
                fold.test_participant, group.index, len(branches), len(column_index), len(run_spec.seeds))
    fits = fit_packed(to_device(fold_data.features, fold_data.target, device), fold_data.train_targets, branches,
                      run_spec.training, dropout_seed)
    by_branch = {(b.config_id, b.seed): fit for b, fit in zip(branches, fits)}
    models = []
    for config_id, columns in column_index.items():
        config_data = to_device(fold_data.features[:, list(columns)], fold_data.target, device)
        for seed in run_spec.seeds:
            models.append(_standalone_model(run_spec.run_id, cache, fold_data, config_data, config_id, seed,
                                            by_branch[(config_id, seed)]))
    return JobResult(group, fold_data, branches, dropout_seed, tuple(models))


def _standalone_model(run_id: str, cache: ProcessedCache, fold_data: FoldData, config_data: DeviceData,
                      config_id: str, seed: int, fit: FitResult) -> TrainedModel:
    scaled = predict(fit.state_dict, config_data, fold_data.test_targets)
    frame = prediction_frame(run_id, cache, fold_data, config_id, seed, scaled)
    return TrainedModel(config_id, seed, fit, frame)


def prediction_frame(run_id: str, cache: ProcessedCache, fold_data: FoldData, config_id: str, seed: int,
                     scaled_pred: np.ndarray) -> pd.DataFrame:
    """The §3.3 columns for the test participant's target seconds, in mL/min."""
    rows = fold_data.test_targets
    visits = cache.visits.iloc[cache.row_visit()[rows]]
    frame = pd.DataFrame({
        "run_id": run_id,
        "config_id": config_id,
        "seed": np.full(len(rows), seed, dtype=np.int64),
        "fold": fold_data.fold.test_participant,
        **{field: visits[field].to_numpy() for field in ("participant_id", "cohort", "protocol", "visit_id")},
        "t_s": cache.column("t")[rows].astype(np.int64),
        "work_rate": cache.column(WORK_RATE_COLUMN)[rows].astype(np.float64),
        "vo2_true": cache.column(TARGET_COLUMN)[rows].astype(np.float64),
        "vo2_pred": fold_data.scalers[TARGET_COLUMN].inverse(scaled_pred),
    })
    return frame[PREDICTION_COLUMNS]


def _epoch_rows(models: list[TrainedModel]) -> pd.DataFrame:
    rows = [
        {"config_id": m.config_id, "seed": m.seed, "epoch": epoch, "train_loss": loss, "is_best": best}
        for m in models
        for epoch, (loss, best) in enumerate(zip(m.fit.epoch_losses, m.fit.is_best), start=1)
    ]
    return pd.DataFrame(rows, columns=EPOCH_COLUMNS)


def _write_outputs(folder: Path, models: list[TrainedModel]) -> None:
    predictions = pd.concat([m.predictions for m in models], ignore_index=True)
    predictions.to_parquet(folder / PREDICTIONS_NAME, index=False)
    visit_metrics(predictions).to_csv(folder / VISIT_METRICS_NAME, index=False)
    _epoch_rows(models).to_csv(folder / EPOCHS_NAME, index=False)


def _per_model(models: list[TrainedModel], value: Callable[[TrainedModel], object]) -> dict:
    out: dict[str, dict[str, object]] = {}
    for m in models:
        out.setdefault(m.config_id, {})[str(m.seed)] = value(m)
    return out


def _save_models(path: Path, models: list[TrainedModel]) -> None:
    """`models.pt`: standalone TCN state dicts `{config_id: {seed: state_dict}}` plus their checkpoint epochs."""
    state_dicts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    checkpoint_epochs: dict[str, dict[int, int]] = {}
    for m in models:
        state_dicts.setdefault(m.config_id, {})[m.seed] = m.fit.state_dict
        checkpoint_epochs.setdefault(m.config_id, {})[m.seed] = m.fit.checkpoint_epoch
    torch.save({"state_dicts": state_dicts, "checkpoint_epochs": checkpoint_epochs}, path)


def _scalers(fold_data: FoldData) -> dict:
    return {c: asdict(s) for c, s in fold_data.scalers.items()}


def _packing(run_spec: Run, jobs: list[JobResult]) -> dict:
    return {
        "layout": BRANCH_LAYOUT,
        "jobs": [{
            "mask_group": job.group.index,
            "K": len(job.branches),
            "config_order": [c.config_id for c in job.group.configurations],
            "seeds": list(run_spec.seeds),
            "feature_columns": list(job.fold_data.feature_columns),
            "dropout_seed": job.dropout_seed,
            "n_train_participants": job.fold_data.n_train_participants,
            "n_train_targets": int(len(job.fold_data.train_targets)),
            "n_test_targets": int(len(job.fold_data.test_targets)),
            "scalers": _scalers(job.fold_data),
            "steps_per_epoch": math.ceil(len(job.fold_data.train_targets) / run_spec.training.batch_size),
            "epoch_wall_s": list(job.models[0].fit.epoch_wall_s),
        } for job in jobs],
    }


def _fold_level(cache: ProcessedCache, jobs: list[JobResult]) -> dict:
    """Fold-level sizes over the union of the mask groups' targets (one group: exactly that group's)."""
    train = np.unique(np.concatenate([job.fold_data.train_targets for job in jobs]))
    test = np.unique(np.concatenate([job.fold_data.test_targets for job in jobs]))
    configurations = tuple(c for job in jobs for c in job.group.configurations)
    return {
        "feature_columns": list(feature_union(configurations)),
        "scalers": _scalers(jobs[0].fold_data) if len(jobs) == 1 else None,  # per group: packing.jobs[i]
        "n_train_participants": int(len(np.unique(cache.row_ids("participant_id")[train]))),
        "n_train_targets": int(len(train)),
        "n_test_targets": int(len(test)),
    }


def _manifest(run_spec: Run, data: RunData, jobs: list[JobResult], device: torch.device, started: str,
              wall_s: float) -> dict:
    """Fold-level fields cover all of the fold's mask groups; per-group fields live in `packing.jobs`."""
    models = [m for job in jobs for m in job.models]
    return {
        "run_id": run_spec.run_id,
        "fold": jobs[0].fold_data.fold.test_participant,
        "git": git_state(),
        "run": asdict(run_spec),
        "prep_settings": asdict(run_spec.prep),
        "cache_key": data.cache.key,
        "inclusion_sha256": data.inclusion_sha256,
        "seeds": list(run_spec.seeds),
        "device": {"type": device.type, "name": device_name(device)},
        "versions": {**package_versions(), "python": platform.python_version()},
        "started_at": started,
        "finished_at": utc_now(),
        "wall_s": wall_s,
        "mask": run_spec.mask,
        "mask_groups": len(jobs),
        **_fold_level(data.cache, jobs),
        "checkpoint_epochs": _per_model(models, lambda m: m.fit.checkpoint_epoch),
        "epoch_wall_s": _per_model(models, lambda m: list(m.fit.epoch_wall_s)),
        "models_file": MODELS_NAME,
        "packing": _packing(run_spec, jobs),
    }
