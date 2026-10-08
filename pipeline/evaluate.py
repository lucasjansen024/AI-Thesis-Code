"""`evaluate(run)`: check every fold is finished and trained with the run's training settings, then write
`<run_id>/summary/` (via `summary.tmp/` + rename).

Nothing here selects a winner. Seed-averaged metric = mean of the per-seed metrics (never the metric of the
seed-averaged prediction); a participant's value = mean of its visits. Groups: all, per cohort, per protocol
(`cohort=<x>`, `protocol=<x>`), taken from the run's own prediction columns.

`summary/` files:
- `predictions_seedavg.parquet`: §3.3 columns minus `seed`, `vo2_pred` = mean over seeds.
- `visit_metrics_seedavg.csv`: config x visit, rmse/mae/bias averaged over seeds (input for LMM, Holm, TOST).
- `summary.csv`: one row per (config, group): RMSE/MAE/bias (mean, sd_participants, sd_seeds); r_rm, r_p and
  rm Bland-Altman bias/LoA (mean, sd_seeds; computed per seed over all test seconds of the group); VO₂peak
  diff and abs_error (mean, sd_participants).
- `vo2peak.csv`: config x ramp visit: measured peak, predicted peak (mean, sd over seeds), diff, abs_error.
- `folds.csv`: training-set sizes and wall time; `targets.csv`: exercise and target seconds per visit x unit.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.folds import run_folds
from pipeline.metrics import VISIT_KEYS, VO2PEAK_WINDOW_S, pearson_r, rm_bland_altman, rm_corr_r, vo2peak
from pipeline.prepare.grid import EXERCISE, UNITS
from pipeline.runs import Run
from pipeline.train import (MANIFEST_NAME, PREDICTIONS_NAME, RUNS_ROOT, TMP_SUFFIX, VISIT_METRICS_NAME, RunData,
                            load_run_data)

logger = logging.getLogger(__name__)

SUMMARY_NAME = "summary"
SEEDAVG_PREDICTIONS_NAME = "predictions_seedavg.parquet"
VISIT_METRICS_SEEDAVG_NAME = "visit_metrics_seedavg.csv"
SUMMARY_TABLE_NAME, VO2PEAK_NAME, FOLDS_NAME, TARGETS_NAME = "summary.csv", "vo2peak.csv", "folds.csv", "targets.csv"
SUMMARY_FILES = (SEEDAVG_PREDICTIONS_NAME, VISIT_METRICS_SEEDAVG_NAME, SUMMARY_TABLE_NAME, VO2PEAK_NAME,
                 FOLDS_NAME, TARGETS_NAME)

METRICS = ("rmse", "mae", "bias")
SECONDARY_METRICS = ("r_rm", "r_p", "ba_bias", "ba_loa_lo", "ba_loa_hi")
VO2PEAK_METRICS = ("diff", "abs_error")
ALL_GROUP = "all"
GROUP_COLUMNS = ("cohort", "protocol")
RAMP_PROTOCOL = "Ramp"
FOLD_COLUMNS = ["fold", "n_train_participants", "n_train_targets", "n_test_targets", "wall_s"]
TARGET_ID_COLUMNS = ["participant_id", "cohort", "protocol", "visit_id"]
SEEDAVG_VISIT_KEYS = [k for k in VISIT_KEYS if k != "seed"]
VISIT_METRICS_SEEDAVG_COLUMNS = [*SEEDAVG_VISIT_KEYS, "n_s", *METRICS]
VO2PEAK_COLUMNS = [*SEEDAVG_VISIT_KEYS, "n_seeds", "vo2peak_measured", "vo2peak_pred_mean", "vo2peak_pred_sd_seeds",
                   *VO2PEAK_METRICS]
SECONDARY_SEED_COLUMNS = ["config_id", "seed", "group", *SECONDARY_METRICS]
SUMMARY_COLUMNS = ["config_id", "group", "n_participants", "n_visits", "n_seeds",
                   *[f"{m}_{s}" for m in METRICS for s in ("mean", "sd_participants", "sd_seeds")],
                   *[f"{m}_{s}" for m in SECONDARY_METRICS for s in ("mean", "sd_seeds")],
                   *[f"vo2peak_{m}_{s}" for m in VO2PEAK_METRICS for s in ("mean", "sd_participants")]]
SEEDAVG_SCHEMA = pa.schema([("run_id", pa.string()), ("config_id", pa.string()), ("fold", pa.string()),
                            ("participant_id", pa.string()), ("cohort", pa.string()), ("protocol", pa.string()),
                            ("visit_id", pa.string()), ("t_s", pa.int64()), ("work_rate", pa.float64()),
                            ("vo2_true", pa.float64()), ("vo2_pred", pa.float64())])
SEEDAVG_COLUMNS = SEEDAVG_SCHEMA.names  # train.PREDICTION_COLUMNS minus `seed` (checked in the tests)


class MissingFoldsError(RuntimeError):
    """`evaluate` was called before every fold of the run finished."""


class TrainingSettingsMismatchError(RuntimeError):
    """A fold manifest's resolved training settings differ from the run's (one procedure per run, §5)."""


@dataclass(frozen=True)
class Group:
    label: str
    column: str | None = None
    value: str | None = None

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame if self.column is None else frame[frame[self.column] == self.value]


def run_groups(frame: pd.DataFrame) -> tuple[Group, ...]:
    """all, then every cohort and every protocol present in `frame`."""
    per_value = (Group(f"{c}={v}", c, v) for c in GROUP_COLUMNS for v in sorted(frame[c].unique()))
    return (Group(ALL_GROUP), *per_value)


# --- primary metrics (from the per-visit metrics) ---------------------------------------------------------------

def _primary(visit_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per configuration: mean/SD across participants of seed-averaged participant values; SD across seeds."""
    per_seed_participant = visit_metrics.groupby(["config_id", "seed", "participant_id"])[list(METRICS)].mean()
    seedavg_participant = per_seed_participant.groupby(level=["config_id", "participant_id"]).mean()
    per_seed_config = per_seed_participant.groupby(level=["config_id", "seed"]).mean()
    across_participants = seedavg_participant.groupby(level="config_id").agg(["mean", "std"])
    across_seeds = per_seed_config.groupby(level="config_id").std()
    counts = visit_metrics.groupby("config_id").agg(n_participants=("participant_id", "nunique"),
                                                    n_visits=("visit_id", "nunique"), n_seeds=("seed", "nunique"))
    columns = {}
    for metric in METRICS:
        columns[f"{metric}_mean"] = across_participants[(metric, "mean")]
        columns[f"{metric}_sd_participants"] = across_participants[(metric, "std")]
        columns[f"{metric}_sd_seeds"] = across_seeds[metric]
    return counts.join(pd.DataFrame(columns)).reset_index()


def visit_metrics_seedavg(visit_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = visit_metrics.groupby(SEEDAVG_VISIT_KEYS, sort=True)
    out = grouped.agg(n_s=("n_s", "first"), **{m: (m, "mean") for m in METRICS}).reset_index()
    return out[VISIT_METRICS_SEEDAVG_COLUMNS]


# --- secondary metrics and VO₂peak (from the predictions) -------------------------------------------------------

def seedavg_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged prediction per (config, visit, second); raises if a second lacks any seed."""
    keys = ["config_id", "visit_id", "t_s"]
    others = [c for c in SEEDAVG_COLUMNS if c not in (*keys, "vo2_pred")]
    out = (predictions.groupby(keys, sort=True)
           .agg(**{c: (c, "first") for c in others}, vo2_pred=("vo2_pred", "mean"), n_seeds=("seed", "nunique"))
           .reset_index())
    expected = out["config_id"].map(predictions.groupby("config_id")["seed"].nunique())
    short = out["n_seeds"] != expected
    if short.any():
        raise ValueError(f"{int(short.sum())} predicted seconds lack a prediction from every seed")
    return out[SEEDAVG_COLUMNS]


def _secondary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    measured, predicted = frame["vo2_true"].to_numpy(), frame["vo2_pred"].to_numpy()
    subject = frame["participant_id"].to_numpy()
    agreement = rm_bland_altman(measured, predicted, subject)
    return {"r_rm": rm_corr_r(measured, predicted, subject), "r_p": pearson_r(measured, predicted),
            "ba_bias": agreement.bias, "ba_loa_lo": agreement.loa_lo, "ba_loa_hi": agreement.loa_hi}


def secondary_per_seed(predictions: pd.DataFrame, groups: tuple[Group, ...]) -> pd.DataFrame:
    """One row per (config, seed, group) over all test seconds of the group."""
    rows = [{"config_id": config_id, "seed": seed, "group": group.label, **_secondary_metrics(group.select(frame))}
            for (config_id, seed), frame in predictions.groupby(["config_id", "seed"], sort=True)
            for group in groups]
    return pd.DataFrame(rows, columns=SECONDARY_SEED_COLUMNS)


def vo2peak_per_seed(predictions: pd.DataFrame, window_s: int = VO2PEAK_WINDOW_S) -> pd.DataFrame:
    """Measured and predicted VO₂peak per (config, seed, ramp visit)."""
    ramps = predictions[predictions["protocol"] == RAMP_PROTOCOL]
    rows = [{**dict(zip(VISIT_KEYS, key)), "measured": vo2peak(visit["t_s"], visit["vo2_true"], window_s),
             "predicted": vo2peak(visit["t_s"], visit["vo2_pred"], window_s)}
            for key, visit in ramps.groupby(VISIT_KEYS, sort=True)]
    return pd.DataFrame(rows, columns=[*VISIT_KEYS, "measured", "predicted"])


def vo2peak_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Config x ramp visit; diff and abs_error are per-seed errors averaged over seeds."""
    if per_seed.empty:
        return pd.DataFrame(columns=VO2PEAK_COLUMNS)
    error = per_seed["predicted"] - per_seed["measured"]
    grouped = per_seed.assign(diff=error, abs_error=error.abs()).groupby(SEEDAVG_VISIT_KEYS, sort=True)
    out = grouped.agg(n_seeds=("seed", "nunique"), vo2peak_measured=("measured", "first"),
                      vo2peak_pred_mean=("predicted", "mean"), vo2peak_pred_sd_seeds=("predicted", "std"),
                      diff=("diff", "mean"), abs_error=("abs_error", "mean")).reset_index()
    return out[VO2PEAK_COLUMNS]


# --- summary.csv ------------------------------------------------------------------------------------------------

def _secondary_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = per_seed.groupby(["config_id", "group"])[list(SECONDARY_METRICS)]
    means, sds = grouped.mean(), grouped.std()
    columns = {}
    for metric in SECONDARY_METRICS:
        columns[f"{metric}_mean"] = means[metric]
        columns[f"{metric}_sd_seeds"] = sds[metric]
    return pd.DataFrame(columns).reset_index()


def _vo2peak_stats(peaks: pd.DataFrame, group: Group) -> pd.DataFrame:
    participant = (group.select(peaks).astype({m: np.float64 for m in VO2PEAK_METRICS})
                   .groupby(["config_id", "participant_id"])[list(VO2PEAK_METRICS)].mean())
    rows = [{"config_id": config_id, "group": group.label,
             **{f"vo2peak_{m}_mean": values[m].mean() for m in VO2PEAK_METRICS},
             **{f"vo2peak_{m}_sd_participants": values[m].std() for m in VO2PEAK_METRICS}}
            for config_id, values in participant.groupby(level="config_id")]
    return pd.DataFrame(rows, columns=["config_id", "group", *SUMMARY_COLUMNS[-2 * len(VO2PEAK_METRICS):]])


def summary_table(visit_metrics: pd.DataFrame, secondary: pd.DataFrame, peaks: pd.DataFrame,
                  groups: tuple[Group, ...]) -> pd.DataFrame:
    """One row per (config, group); groups without ramp visits get NaN VO₂peak columns."""
    keys = ["config_id", "group"]
    primary = pd.concat([_primary(g.select(visit_metrics)).assign(group=g.label) for g in groups], ignore_index=True)
    peak_stats = pd.concat([_vo2peak_stats(peaks, g) for g in groups], ignore_index=True)
    table = (primary.merge(_secondary_stats(secondary), on=keys, how="left")
             .merge(peak_stats, on=keys, how="left"))
    return table.sort_values("config_id", kind="stable").reset_index(drop=True)[SUMMARY_COLUMNS]


# --- folds.csv, targets.csv -------------------------------------------------------------------------------------

def folds_table(manifests: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{c: m[c] for c in FOLD_COLUMNS} for m in manifests], columns=FOLD_COLUMNS)


def targets_table(data: RunData, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Every visit x unit of the run's cohorts: exercise seconds and target seconds after the mask."""
    cache = data.cache
    exercise = cache.column("phase").astype(np.int64) == EXERCISE
    counts = (pd.DataFrame({"visit": cache.row_visit()[exercise],
                            "unit": cache.column("unit").astype(np.int64)[exercise],
                            "target": data.mask[exercise]})
              .groupby(["visit", "unit"])
              .agg(exercise_s=("target", "size"), target_s=("target", "sum"))
              .reset_index())
    ids = cache.visits[TARGET_ID_COLUMNS].iloc[counts["visit"]].reset_index(drop=True)
    table = pd.concat([ids, counts.drop(columns="visit")], axis=1)
    table = table.assign(unit=[UNITS[u] for u in table["unit"]])
    return table[table["cohort"].isin(cohorts)].reset_index(drop=True)


# --- evaluate ---------------------------------------------------------------------------------------------------

def _finished_folds(run_spec: Run, data: RunData, run_dir: Path) -> list[str]:
    expected = [f.test_participant for f in run_folds(run_spec, data.cache)]
    missing = [p for p in expected if not (run_dir / p / MANIFEST_NAME).exists()]
    if missing:
        raise MissingFoldsError(f"{run_spec.run_id}: {len(missing)} of {len(expected)} folds missing in "
                                f"{run_dir}: {', '.join(missing)}")
    return expected


def check_training_settings(run_spec: Run, run_dir: Path, folds: list[str]) -> None:
    """Every fold manifest records the run's resolved training settings, so the folds agree with each other."""
    expected = asdict(run_spec.training)
    recorded = {p: json.loads((run_dir / p / MANIFEST_NAME).read_text(encoding="utf-8"))["run"]["training"]
                for p in folds}
    differing = {p: settings for p, settings in recorded.items() if settings != expected}
    if differing:
        raise TrainingSettingsMismatchError(f"{run_spec.run_id}: run trains with {expected}, but folds were trained "
                                            f"with other settings: {differing}")


def _prediction_tables(run_dir: Path, folds: list[str], config_ids: list[str], groups: tuple[Group, ...],
                       seedavg_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One configuration at a time (all configs x seeds x ~150k s would not fit comfortably in memory).

    Streams the seed-averaged predictions to `seedavg_path`; returns (secondary per seed, VO₂peak per seed).
    """
    secondary, peaks = [], []
    with pq.ParquetWriter(seedavg_path, SEEDAVG_SCHEMA) as writer:
        for config_id in config_ids:
            predictions = pd.concat([pd.read_parquet(run_dir / p / PREDICTIONS_NAME,
                                                     filters=[("config_id", "==", config_id)]) for p in folds],
                                    ignore_index=True)
            seedavg = seedavg_predictions(predictions)
            writer.write_table(pa.Table.from_pandas(seedavg, schema=SEEDAVG_SCHEMA, preserve_index=False))
            secondary.append(secondary_per_seed(predictions, groups))
            peaks.append(vo2peak_per_seed(predictions))
    return pd.concat(secondary, ignore_index=True), pd.concat(peaks, ignore_index=True)


def _write_summary(folder: Path, run_spec: Run, data: RunData, run_dir: Path, folds: list[str]) -> None:
    visit_metrics = pd.concat([pd.read_csv(run_dir / p / VISIT_METRICS_NAME) for p in folds], ignore_index=True)
    manifests = [json.loads((run_dir / p / MANIFEST_NAME).read_text(encoding="utf-8")) for p in folds]
    groups = run_groups(visit_metrics)
    config_ids = sorted(visit_metrics["config_id"].unique())
    secondary, peaks_per_seed = _prediction_tables(run_dir, folds, config_ids, groups,
                                                   folder / SEEDAVG_PREDICTIONS_NAME)
    peaks = vo2peak_table(peaks_per_seed)
    visit_metrics_seedavg(visit_metrics).to_csv(folder / VISIT_METRICS_SEEDAVG_NAME, index=False)
    summary_table(visit_metrics, secondary, peaks, groups).to_csv(folder / SUMMARY_TABLE_NAME, index=False)
    peaks.to_csv(folder / VO2PEAK_NAME, index=False)
    folds_table(manifests).to_csv(folder / FOLDS_NAME, index=False)
    targets_table(data, run_spec.cohorts).to_csv(folder / TARGETS_NAME, index=False)


def evaluate(run_spec: Run, output_root: Path = RUNS_ROOT, data: RunData | None = None) -> Path:
    """Write `<output_root>/<run_id>/summary/`; raises `MissingFoldsError` listing unfinished folds and
    `TrainingSettingsMismatchError` listing folds trained with other settings."""
    data = data if data is not None else load_run_data(run_spec)
    run_dir = output_root / run_spec.run_id
    folds = _finished_folds(run_spec, data, run_dir)
    check_training_settings(run_spec, run_dir, folds)
    tmp = run_dir / f"{SUMMARY_NAME}{TMP_SUFFIX}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    _write_summary(tmp, run_spec, data, run_dir, folds)
    final = run_dir / SUMMARY_NAME
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    logger.info("%s: summary written to %s", run_spec.run_id, final)
    return final
