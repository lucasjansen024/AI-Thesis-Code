"""Training-settings pilot (§3.0, tickets 09 and 10): `run_pilot(pilot)` trains, `decide(pilot)` picks the arm.

Each (validation split, trajectory) is one packed job: the pilot's configurations x seeds on the split's training
participants under the shared mask, scalers fit on those participants only. After every epoch each branch's
current weights are extracted and scored on the split's validation participants (`fit_packed(on_epoch=...)`).
Outputs per job, `outputs/<pilot_id>/<split_id>/<trajectory_id>/` (written as `.tmp/`, renamed after the
manifest, finished jobs skipped): `manifest.json`, `epochs.csv`, `val_visits.csv`. No predictions, no weights.

`decide()` reads only those logs: every arm is scored at its checkpoint e* = argmin_{e <= E} train_loss of its
trajectory (earliest epoch on a tie); RMSE is averaged over visits, then participants, then splits x seeds, giving
one value per (family, configuration, arm) cell. The arm with the fewest epochs within `QUALIFY_MARGIN` of each
cell's best arm in every cell wins; arms with equal epochs (A, B) go to the lower mean RMSE over the cells; if no
arm cheaper than R qualifies everywhere, R. Test participants of the A1 runs play no part; nothing reads
any run output folder, and nothing reads the decision at train time (the dev copies it into `runs.py`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from inventory.build_inclusion import INCLUSION_CSV
from pipeline.cache import ProcessedCache, load_or_build
from pipeline.domains import TARGET_COLUMN, feature_union, parse_configuration
from pipeline.fit import (Branch, DeviceData, FitResult, configure_determinism, device_name, fit_packed,
                          pack_layout, pick_device, predict, to_device)
from pipeline.folds import FoldData, build_validation_data, validation_fold
from pipeline.inclusion import read_inclusion
from pipeline.provenance import REPO_ROOT, derive_seed, git_state, package_versions, sha256_file, utc_now
from pipeline.runs import (COHORTS, PILOT, PILOT_SPLIT_SEED, PILOT_TRAJECTORY_B, PILOT_TRAJECTORY_D,
                           PILOT_TRAJECTORY_R, POOLED_FAMILY, SHARED_MASK, Pilot, PilotSplit, PilotTrajectory, Run)
from pipeline.train import BRANCH_LAYOUT, EPOCH_COLUMNS, MANIFEST_NAME, TMP_SUFFIX, RunData, build_run_data

logger = logging.getLogger(__name__)

OUTPUTS_ROOT = REPO_ROOT / "outputs"  # a pilot writes to OUTPUTS_ROOT / pilot.pilot_id
EPOCHS_NAME, VAL_VISITS_NAME = "epochs.csv", "val_visits.csv"
PILOT_EPOCH_COLUMNS = [*EPOCH_COLUMNS, "val_mse_scaled"]
VAL_VISIT_COLUMNS = ["config_id", "seed", "epoch", "participant_id", "visit_id", "n_s", "rmse"]
DECISION_JSON, DECISION_MD = "decision.json", "decision.md"
QUALIFY_MARGIN = 10.0  # mL/min above a cell's best arm
SHARED_MASK_GROUP = 0  # dropout seed part, as for a fold's single mask group
N_POOLED_SPLITS, POOLED_SIZE = 3, 3
N_COHORT_SPLITS, COHORT_SIZE = 2, 2
JOB_KEYS = ["split_id", "trajectory_id"]
MODEL_KEYS = ["split_id", "config_id", "seed"]


@dataclass(frozen=True)
class Arm:
    name: str
    trajectory_id: str
    epochs: int  # E: scored at e* = argmin_{e <= E} train_loss of its trajectory


ARMS = (Arm("C", PILOT_TRAJECTORY_D, 1), Arm("D", PILOT_TRAJECTORY_D, 3), Arm("A", PILOT_TRAJECTORY_R, 10),
        Arm("B", PILOT_TRAJECTORY_B, 10), Arm("R", PILOT_TRAJECTORY_R, 30))
FALLBACK_ARM = "R"


# --- splits -----------------------------------------------------------------------------------------------------

def _ranking(participants: Sequence[str], seed: int, family: str) -> list[str]:
    def key(pid: str) -> str:
        return hashlib.sha256(f"pilot_split|{seed}|{family}|{pid}".encode()).hexdigest()
    return sorted(participants, key=key)


def draw_splits(participants: Mapping[str, Sequence[str]], seed: int = PILOT_SPLIT_SEED) -> tuple[PilotSplit, ...]:
    """The documented draw behind `runs.PILOT_SPLITS` (participants per cohort, e.g. from the inclusion CSV)."""
    gies, lucas = (_ranking(participants[c], seed, POOLED_FAMILY) for c in COHORTS)
    interleaved = [p for pair in zip(gies, lucas) for p in pair]
    pooled = tuple(PilotSplit(f"{POOLED_FAMILY}_{i + 1}", POOLED_FAMILY,
                              tuple(sorted(interleaved[i * POOLED_SIZE:(i + 1) * POOLED_SIZE])))
                   for i in range(N_POOLED_SPLITS))
    within = tuple(PilotSplit(f"{cohort.lower()}_{i + 1}", cohort,
                              tuple(sorted(ranked[i * COHORT_SIZE:(i + 1) * COHORT_SIZE])))
                   for cohort in COHORTS for ranked in [_ranking(participants[cohort], seed, cohort)]
                   for i in range(N_COHORT_SPLITS))
    return pooled + within


def check_splits(splits: tuple[PilotSplit, ...], visits: pd.DataFrame) -> None:
    """Disjoint within a family, every ID in its family's cohorts, pooled splits mix both cohorts."""
    cohort_of = dict(zip(visits["participant_id"], visits["cohort"]))
    errors = []
    for family in dict.fromkeys(s.family for s in splits):
        ids = [p for s in splits if s.family == family for p in s.validation]
        if len(set(ids)) != len(ids):
            errors.append(f"{family}: overlapping validation participants {sorted(ids)}")
    for split in splits:
        cohorts = {cohort_of.get(p) for p in split.validation}
        if not cohorts <= set(split.cohorts):
            errors.append(f"{split.split_id}: participants {split.validation} outside {split.cohorts}")
        if split.family == POOLED_FAMILY and cohorts != set(COHORTS):
            errors.append(f"{split.split_id}: pooled split {split.validation} does not mix both cohorts")
    if errors:
        raise ValueError("pilot splits: " + " | ".join(errors))


# --- training ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationTargets:
    rows: np.ndarray  # int64 cache rows of the validation participants' target seconds
    target_scaled: np.ndarray  # float64
    vo2_true: np.ndarray  # float64, mL/min
    participant_id: np.ndarray
    visit_id: np.ndarray


@dataclass(frozen=True)
class PilotJob:
    split: PilotSplit
    trajectory: PilotTrajectory
    fold_data: FoldData
    branches: tuple[Branch, ...]
    dropout_seed: int
    fits: tuple[FitResult, ...]
    val_wall_s: float


@dataclass(frozen=True)
class EpochScores:
    epochs: pd.DataFrame  # config_id, seed, epoch, val_mse_scaled
    visits: pd.DataFrame  # VAL_VISIT_COLUMNS
    wall_s: float


def pilot_run_spec(pilot: Pilot) -> Run:
    """The pilot as a shared-mask `Run` over both cohorts: only used to build its target mask."""
    return Run(pilot.pilot_id, COHORTS, pilot.configurations, seeds=pilot.seeds, mask=SHARED_MASK, prep=pilot.prep)


def load_pilot_data(pilot: Pilot, rebuild_cache: bool = False, inclusion_path: Path = INCLUSION_CSV) -> RunData:
    cache = load_or_build(pilot.prep, rebuild_cache, inclusion_path=inclusion_path)
    inclusion = read_inclusion(inclusion_path)
    return build_run_data(pilot_run_spec(pilot), cache, inclusion, sha256_file(inclusion_path))


def run_pilot(pilot: Pilot = PILOT, rebuild_cache: bool = False) -> list[Path]:
    """Train every (split, trajectory) job of the pilot; finished jobs are skipped."""
    return train_pilot(pilot, load_pilot_data(pilot, rebuild_cache))


def train_pilot(pilot: Pilot, data: RunData, output_root: Path = OUTPUTS_ROOT,
                device: torch.device | None = None) -> list[Path]:
    device = pick_device() if device is None else device
    configure_determinism(device)
    if data.config_masks:
        raise ValueError(f"{pilot.pilot_id}: the pilot uses the shared mask, got per_config masks")
    check_splits(pilot.splits, data.cache.visits)
    logger.info("%s: %d splits x %d trajectories on %s (%s)", pilot.pilot_id, len(pilot.splits),
                len(pilot.trajectories), device, device_name(device))
    return [train_trajectory(pilot, data, split, trajectory, output_root / pilot.pilot_id, device)
            for split in pilot.splits for trajectory in pilot.trajectories]


def train_trajectory(pilot: Pilot, data: RunData, split: PilotSplit, trajectory: PilotTrajectory,
                     pilot_dir: Path, device: torch.device) -> Path:
    final = pilot_dir / split.split_id / trajectory.trajectory_id
    if final.exists():
        logger.info("%s: %s/%s finished, skipped", pilot.pilot_id, split.split_id, trajectory.trajectory_id)
        return final
    tmp = final.with_name(f"{final.name}{TMP_SUFFIX}")
    if tmp.exists():
        logger.info("%s: wiping leftover %s", pilot.pilot_id, tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    started, clock = utc_now(), time.perf_counter()
    fold = validation_fold(split.split_id, split.validation, split.cohorts, data.cache.visits)
    configurations = tuple(parse_configuration(c) for c in pilot.configurations)
    fold_data = build_validation_data(data.cache, data.mask, fold, feature_union(configurations))
    column_index = {c.config_id: tuple(fold_data.feature_columns.index(col) for col in c.columns)
                    for c in configurations}
    branches = pack_layout(tuple(column_index.items()), pilot.seeds)
    dropout_seed = derive_seed("dropout", pilot.pilot_id, split.split_id, SHARED_MASK_GROUP)
    fits, scores = _fit_and_score(data.cache, fold_data, column_index, branches, trajectory, dropout_seed, device)
    epoch_log(branches, fits, scores.epochs).to_csv(tmp / EPOCHS_NAME, index=False)
    scores.visits.to_csv(tmp / VAL_VISITS_NAME, index=False)
    job = PilotJob(split, trajectory, fold_data, branches, dropout_seed, fits, scores.wall_s)
    manifest = _manifest(pilot, data, job, device, started, time.perf_counter() - clock)
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(final)
    logger.info("%s: %s/%s written to %s", pilot.pilot_id, split.split_id, trajectory.trajectory_id, final)
    return final


def validation_targets(cache: ProcessedCache, fold_data: FoldData) -> ValidationTargets:
    rows = fold_data.test_targets
    visits = cache.visits.iloc[cache.row_visit()[rows]]
    return ValidationTargets(rows, fold_data.target[rows].astype(np.float64),
                             cache.column(TARGET_COLUMN)[rows].astype(np.float64),
                             visits["participant_id"].to_numpy(), visits["visit_id"].to_numpy())


def visit_rmse(targets: ValidationTargets, vo2_pred: np.ndarray) -> pd.DataFrame:
    """participant_id, visit_id, n_s, rmse (mL/min) over each validation visit's target seconds."""
    frame = pd.DataFrame({"participant_id": targets.participant_id, "visit_id": targets.visit_id,
                          "sq_error": (np.asarray(vo2_pred, np.float64) - targets.vo2_true) ** 2})
    out = frame.groupby(["participant_id", "visit_id"], sort=True).agg(
        n_s=("sq_error", "size"), mse=("sq_error", "mean")).reset_index()
    return out.assign(rmse=np.sqrt(out["mse"])).drop(columns="mse")


def score_epoch(epoch: int, branches: tuple[Branch, ...], states: tuple[dict[str, torch.Tensor], ...],
                config_data: Mapping[str, DeviceData], targets: ValidationTargets,
                target_scaler_inverse: Callable[[np.ndarray], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every branch's current standalone weights on the validation targets: (val_mse_scaled rows, visit rows)."""
    mse_rows, visit_frames = [], []
    for branch, state in zip(branches, states):
        scaled = predict(state, config_data[branch.config_id], targets.rows)
        ids = {"config_id": branch.config_id, "seed": branch.seed, "epoch": epoch}
        mse_rows.append({**ids, "val_mse_scaled": float(np.mean((scaled - targets.target_scaled) ** 2))})
        visit_frames.append(visit_rmse(targets, target_scaler_inverse(scaled)).assign(**ids))
    return pd.DataFrame(mse_rows), pd.concat(visit_frames, ignore_index=True)[VAL_VISIT_COLUMNS]


def _fit_and_score(cache: ProcessedCache, fold_data: FoldData, column_index: Mapping[str, tuple[int, ...]],
                   branches: tuple[Branch, ...], trajectory: PilotTrajectory, dropout_seed: int,
                   device: torch.device) -> tuple[tuple[FitResult, ...], EpochScores]:
    targets = validation_targets(cache, fold_data)
    config_data = {cid: to_device(fold_data.features[:, list(cols)], fold_data.target, device)
                   for cid, cols in column_index.items()}
    inverse = fold_data.scalers[TARGET_COLUMN].inverse
    mse_frames, visit_frames, walls = [], [], []

    def on_epoch(epoch: int, states: tuple[dict[str, torch.Tensor], ...]) -> None:
        start = time.perf_counter()
        mse, visits = score_epoch(epoch, branches, states, config_data, targets, inverse)
        mse_frames.append(mse)
        visit_frames.append(visits)
        walls.append(time.perf_counter() - start)

    fits = fit_packed(to_device(fold_data.features, fold_data.target, device), fold_data.train_targets, branches,
                      trajectory.training, dropout_seed, on_epoch=on_epoch)
    scores = EpochScores(pd.concat(mse_frames, ignore_index=True), pd.concat(visit_frames, ignore_index=True),
                         float(sum(walls)))
    return fits, scores


def epoch_log(branches: tuple[Branch, ...], fits: tuple[FitResult, ...], val_mse: pd.DataFrame) -> pd.DataFrame:
    """`epochs.csv`: the fold columns plus `val_mse_scaled`, one row per (model, epoch)."""
    rows = [{"config_id": b.config_id, "seed": b.seed, "epoch": epoch, "train_loss": loss, "is_best": best}
            for b, fit in zip(branches, fits)
            for epoch, (loss, best) in enumerate(zip(fit.epoch_losses, fit.is_best), start=1)]
    merged = pd.DataFrame(rows).merge(val_mse, on=["config_id", "seed", "epoch"], how="left", validate="1:1")
    if merged["val_mse_scaled"].isna().any():
        raise RuntimeError("pilot: an epoch has no validation score")
    return merged[PILOT_EPOCH_COLUMNS]


def _manifest(pilot: Pilot, data: RunData, job: PilotJob, device: torch.device, started: str, wall_s: float) -> dict:
    split, trajectory, fold_data, fits = job.split, job.trajectory, job.fold_data, job.fits
    epoch_wall_s = list(fits[0].epoch_wall_s)
    return {
        "pilot_id": pilot.pilot_id,
        "split_id": split.split_id,
        "family": split.family,
        "trajectory_id": trajectory.trajectory_id,
        "validation_participants": list(split.validation),
        "train_cohorts": list(split.cohorts),
        "git": git_state(),
        "pilot": asdict(pilot),
        "training": asdict(trajectory.training),
        "prep_settings": asdict(pilot.prep),
        "cache_key": data.cache.key,
        "inclusion_sha256": data.inclusion_sha256,
        "mask": SHARED_MASK,
        "configurations": list(pilot.configurations),
        "seeds": list(pilot.seeds),
        "device": {"type": device.type, "name": device_name(device)},
        "versions": {**package_versions(), "python": platform.python_version()},
        "started_at": started,
        "finished_at": utc_now(),
        "wall_s": wall_s,
        "train_wall_s": float(sum(epoch_wall_s)),
        "val_wall_s": job.val_wall_s,
        "epoch_wall_s": epoch_wall_s,
        "K": len(job.branches),
        "epochs": trajectory.training.epochs,
        "layout": BRANCH_LAYOUT,
        "dropout_seed": job.dropout_seed,
        "feature_columns": list(fold_data.feature_columns),
        "scalers": {c: asdict(s) for c, s in fold_data.scalers.items()},
        "n_train_participants": fold_data.n_train_participants,
        "n_train_targets": int(len(fold_data.train_targets)),
        "n_val_targets": int(len(fold_data.test_targets)),
        "checkpoint_epochs": _checkpoint_epochs(job.branches, fits),
    }


def _checkpoint_epochs(branches: tuple[Branch, ...], fits: tuple[FitResult, ...]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for b, fit in zip(branches, fits):
        out.setdefault(b.config_id, {})[str(b.seed)] = fit.checkpoint_epoch
    return out


# --- decision ---------------------------------------------------------------------------------------------------

class PilotLogError(RuntimeError):
    """The pilot logs are missing jobs, epochs or validation participants."""


@dataclass(frozen=True)
class PilotLogs:
    epochs: pd.DataFrame  # PILOT_EPOCH_COLUMNS + split_id, trajectory_id
    val_visits: pd.DataFrame  # VAL_VISIT_COLUMNS + split_id, trajectory_id
    jobs: pd.DataFrame  # split_id, trajectory_id, K, epochs, wall_s, train_wall_s, val_wall_s


@dataclass(frozen=True)
class Decision:
    arm: str
    reason: str
    cells: pd.DataFrame  # index (family, config_id), one RMSE column per arm (mL/min)
    qualifies_everywhere: dict[str, bool]
    mean_rmse: dict[str, float]


def load_logs(pilot: Pilot, pilot_dir: Path) -> PilotLogs:
    """Every (split, trajectory) job's logs; raises listing the missing job folders."""
    folders = {(s.split_id, t.trajectory_id): pilot_dir / s.split_id / t.trajectory_id
               for s in pilot.splits for t in pilot.trajectories}
    missing = sorted(f"{s}/{t}" for (s, t), folder in folders.items() if not (folder / MANIFEST_NAME).exists())
    if missing:
        raise PilotLogError(f"{pilot.pilot_id}: missing pilot jobs {', '.join(missing)} under {pilot_dir}")
    epochs, visits, jobs = [], [], []
    for (split_id, trajectory_id), folder in folders.items():
        ids = {"split_id": split_id, "trajectory_id": trajectory_id}
        epochs.append(pd.read_csv(folder / EPOCHS_NAME).assign(**ids))
        visits.append(pd.read_csv(folder / VAL_VISITS_NAME, dtype={"participant_id": str, "visit_id": str})
                      .assign(**ids))
        manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        jobs.append({**ids, **{k: manifest[k] for k in ("K", "epochs", "wall_s", "train_wall_s", "val_wall_s")}})
    return PilotLogs(pd.concat(epochs, ignore_index=True), pd.concat(visits, ignore_index=True), pd.DataFrame(jobs))


def arm_checkpoints(epochs: pd.DataFrame, arms: tuple[Arm, ...] = ARMS) -> pd.DataFrame:
    """split_id, trajectory_id, config_id, seed, arm, epoch: e* = argmin_{e <= E} train_loss (earliest on a tie)."""
    frames = []
    for arm in arms:
        sub = epochs[(epochs["trajectory_id"] == arm.trajectory_id) & (epochs["epoch"] <= arm.epochs)]
        sub = sub.sort_values([*MODEL_KEYS, "epoch"], kind="stable")
        counts = sub.groupby(MODEL_KEYS)["epoch"].nunique()
        if sub.empty or (counts != arm.epochs).any():
            raise PilotLogError(f"arm {arm.name}: trajectory {arm.trajectory_id} lacks epochs 1..{arm.epochs}")
        finite = sub[np.isfinite(sub["train_loss"])]
        if finite.groupby(MODEL_KEYS).ngroups != counts.size:
            raise PilotLogError(f"arm {arm.name}: a model has no finite training loss up to epoch {arm.epochs}")
        best = finite.loc[finite.groupby(MODEL_KEYS)["train_loss"].idxmin()]
        frames.append(best[[*MODEL_KEYS, "trajectory_id", "epoch"]].assign(arm=arm.name))
    return pd.concat(frames, ignore_index=True)


def participant_rmse(checkpoints: pd.DataFrame, val_visits: pd.DataFrame) -> pd.DataFrame:
    """split_id, config_id, seed, arm, rmse: mean over visits within a participant, then over participants."""
    merged = checkpoints.merge(val_visits, on=[*MODEL_KEYS, "trajectory_id", "epoch"], how="left")
    if merged["rmse"].isna().any():
        raise PilotLogError("validation RMSE missing at some arm checkpoints")
    keys = [*MODEL_KEYS, "arm"]
    per_participant = merged.groupby([*keys, "participant_id"], as_index=False)["rmse"].mean()
    return per_participant.groupby(keys, as_index=False)["rmse"].mean()


def check_participants(checkpoints: pd.DataFrame, val_visits: pd.DataFrame, splits: tuple[PilotSplit, ...]) -> None:
    """Every arm checkpoint is scored on exactly its split's validation participants."""
    merged = checkpoints.merge(val_visits, on=[*MODEL_KEYS, "trajectory_id", "epoch"], how="inner")
    seen = {key: frozenset(ids) for key, ids in merged.groupby([*MODEL_KEYS, "arm"])["participant_id"]}
    expected = {s.split_id: frozenset(s.validation) for s in splits}
    wrong = [key for key, ids in seen.items() if ids != expected[key[0]]]
    if wrong or len(seen) != len(checkpoints):
        raise PilotLogError(f"validation participants differ from the splits for {wrong[:5]} "
                            f"({len(seen)} of {len(checkpoints)} checkpoints scored)")


def cell_table(scores: pd.DataFrame, splits: tuple[PilotSplit, ...], arms: tuple[Arm, ...] = ARMS) -> pd.DataFrame:
    """Mean over splits x seeds per (family, config_id) and arm; columns in `arms` order."""
    family_of = {s.split_id: s.family for s in splits}
    with_family = scores.assign(family=scores["split_id"].map(family_of))
    cells = with_family.pivot_table(index=["family", "config_id"], columns="arm", values="rmse", aggfunc="mean")
    return cells.reindex(columns=[a.name for a in arms])  # a missing arm becomes NaN: choose_arm raises


def choose_arm(cells: pd.DataFrame, arms: tuple[Arm, ...] = ARMS, margin: float = QUALIFY_MARGIN) -> Decision:
    """Fewest-epoch arm within `margin` of each cell's best in every cell; equal epochs -> lower mean RMSE; else R."""
    if cells.isna().any().any() or cells.empty:
        raise PilotLogError("incomplete cell table:\n" + cells.to_string())
    qualifies = cells.le(cells.min(axis=1) + margin, axis=0)
    everywhere = {a.name: bool(qualifies[a.name].all()) for a in arms}
    mean_rmse = {a.name: float(cells[a.name].mean()) for a in arms}
    fallback = next(a for a in arms if a.name == FALLBACK_ARM)
    candidates = [a for a in arms if everywhere[a.name] and a.epochs < fallback.epochs]
    if not candidates:
        return Decision(FALLBACK_ARM, f"no arm cheaper than {FALLBACK_ARM} qualifies in every cell",
                        cells, everywhere, mean_rmse)
    fewest = min(a.epochs for a in candidates)
    tied = [a for a in candidates if a.epochs == fewest]
    chosen = min(tied, key=lambda a: mean_rmse[a.name])  # stable: `arms` order on an exact tie
    reason = (f"fewest epochs ({fewest}) qualifying in every cell" if len(tied) == 1 else
              f"{'/'.join(a.name for a in tied)} tie at {fewest} epochs: lower mean RMSE over the cells")
    return Decision(chosen.name, reason, cells, everywhere, mean_rmse)


def wall_table(jobs: pd.DataFrame) -> pd.DataFrame:
    """Per job: wall s, training wall s, validation wall s and training s per model-epoch."""
    return jobs.assign(per_model_epoch_s=jobs["train_wall_s"] / (jobs["K"] * jobs["epochs"]))


def decide(pilot: Pilot = PILOT, output_root: Path = OUTPUTS_ROOT) -> Decision:
    """Apply the decision rule to the pilot logs; writes `decision.json` and `decision.md` next to the splits."""
    pilot_dir = output_root / pilot.pilot_id
    logs = load_logs(pilot, pilot_dir)
    checkpoints = arm_checkpoints(logs.epochs)
    check_participants(checkpoints, logs.val_visits, pilot.splits)
    cells = cell_table(participant_rmse(checkpoints, logs.val_visits), pilot.splits)
    decision = choose_arm(cells)
    walls = wall_table(logs.jobs)
    (pilot_dir / DECISION_JSON).write_text(json.dumps(_decision_json(pilot, decision, walls), indent=2),
                                           encoding="utf-8")
    (pilot_dir / DECISION_MD).write_text(_decision_md(pilot, decision, walls), encoding="utf-8")
    logger.info("%s: chosen arm %s (%s)", pilot.pilot_id, decision.arm, decision.reason)
    return decision


def _arm_setting(pilot: Pilot, arm_name: str) -> dict:
    arm = next(a for a in ARMS if a.name == arm_name)
    trajectory = next(t for t in pilot.trajectories if t.trajectory_id == arm.trajectory_id)
    return {"arm": arm.name, "learning_rate": trajectory.training.learning_rate, "epochs": arm.epochs,
            "batch_size": trajectory.training.batch_size}


def _decision_json(pilot: Pilot, decision: Decision, walls: pd.DataFrame) -> dict:
    total_model_epochs = float((walls["K"] * walls["epochs"]).sum())
    return {
        "pilot_id": pilot.pilot_id,
        "chosen": _arm_setting(pilot, decision.arm),
        "reason": decision.reason,
        "qualify_margin_ml_min": QUALIFY_MARGIN,
        "arms": [asdict(a) for a in ARMS],
        "qualifies_everywhere": decision.qualifies_everywhere,
        "mean_rmse": decision.mean_rmse,
        "cells": decision.cells.reset_index().to_dict(orient="records"),
        "wall": {
            "jobs": walls.to_dict(orient="records"),
            "per_model_epoch_s": float(walls["train_wall_s"].sum()) / total_model_epochs,
            "total_s": float(walls["wall_s"].sum()),
        },
        "decided_at": utc_now(),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        return f"{value:.1f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def _decision_md(pilot: Pilot, decision: Decision, walls: pd.DataFrame) -> str:
    chosen = _arm_setting(pilot, decision.arm)
    cells = decision.cells.assign(best=decision.cells.min(axis=1)).reset_index()
    summary = pd.DataFrame({"arm": [a.name for a in ARMS], "epochs": [a.epochs for a in ARMS],
                            "trajectory": [a.trajectory_id for a in ARMS],
                            "mean_rmse": [decision.mean_rmse[a.name] for a in ARMS],
                            "qualifies_everywhere": [decision.qualifies_everywhere[a.name] for a in ARMS]})
    wall_columns = ["split_id", "trajectory_id", "K", "epochs", "wall_s", "train_wall_s", "val_wall_s",
                    "per_model_epoch_s"]
    per_model_epoch = float(walls["train_wall_s"].sum()) / float((walls["K"] * walls["epochs"]).sum())
    return "\n\n".join([
        f"# Pilot decision ({pilot.pilot_id})",
        f"**Chosen arm: {chosen['arm']}** = lr {chosen['learning_rate']:g} x {chosen['epochs']} epochs "
        f"(batch {chosen['batch_size']}). Reason: {decision.reason}. Copy it into `runs.py` (`A1_TRAINING`).",
        f"Validation RMSE (mL/min) per cell: visits -> participants -> splits x seeds, each arm at its "
        f"checkpoint. Qualifies: within {QUALIFY_MARGIN:g} of the cell's best.",
        _markdown_table(cells),
        _markdown_table(summary),
        f"Wall time: {float(walls['wall_s'].sum()) / 3600:.2f} h over {len(walls)} jobs; "
        f"{per_model_epoch * 1000:.1f} ms of training per model-epoch.",
        _markdown_table(walls[wall_columns]),
    ]) + "\n"
