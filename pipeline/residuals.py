"""Residual analyses (METHODOLOGY §11.6) on a run's seed-averaged predictions. Residual = predicted - measured
(mL/min). Descriptives only; the runs-test z and p are the only test statistics.

Phase events come from the WR column of the run's own processed cache (found by the fold manifests' `cache_key`,
never rebuilt): the predictions hold target seconds only, so the 330-s onset change and the Step's first-block clock
are invisible there. The cache WR must equal the predictions' `work_rate` on every target second. Every visit's
target seconds must be contiguous (per configuration), else nothing is written.

Two labellings: `primary` (PRBS switch windows 0-30 s) and `prbs_whole_run` (each PRBS switch to the next change);
phases 3-8 are identical in both. Target seconds outside every window are phase 0, "other".

Files in `outputs/residuals/<run_id>/` (or `out_dir`):
- `events.csv`: events per labelling x phase x cohort x protocol, with and without target seconds (the count check).
- `runs_test.csv`: Wald-Wolfowitz runs test joined across participants per protocol x configuration (output 1).
- `visit_runs.csv`, `run_length.csv`: run length per visit, and per protocol x configuration over participants
  (output 2).
- `phases.csv`: bias (95 % t-CI, participant as the unit) and error share / time share per labelling x group x
  phase x configuration; groups pooled, Gies, Lucas (output 3).
- `traces.csv`: measured, predicted and residual per offset from the phase anchor, per participant first, then
  mean and SD across participants; pooled only (output 4).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.cache import CACHE_ROOT, load
from pipeline.evaluate import RAMP_PROTOCOL, SEEDAVG_COLUMNS, SEEDAVG_PREDICTIONS_NAME, SUMMARY_NAME
from pipeline.prepare.grid import EXERCISE
from pipeline.provenance import REPO_ROOT
from pipeline.runs import A1_POOLED, A2_CONFIGURATIONS
from pipeline.train import MANIFEST_NAME, RUNS_ROOT, WORK_RATE_COLUMN

logger = logging.getLogger(__name__)

RESIDUALS_ROOT = REPO_ROOT / "outputs" / "residuals"
GET_POWER_CSV = REPO_ROOT / "inventory" / "reference" / "get_power.csv"
EVENTS_NAME, RUNS_TEST_NAME, VISIT_RUNS_NAME = "events.csv", "runs_test.csv", "visit_runs.csv"
RUN_LENGTH_NAME, PHASES_NAME, TRACES_NAME = "run_length.csv", "phases.csv", "traces.csv"

OTHER, PRBS_UP, PRBS_DOWN, INCREASE, DECREASE, SUSTAINED_BELOW, SUSTAINED_ABOVE, RAMP_GET, RAMP_END = range(9)
PHASE_NAMES = {OTHER: "other", PRBS_UP: "PRBS switch up", PRBS_DOWN: "PRBS switch down", INCREASE: "increase",
               DECREASE: "decrease", SUSTAINED_BELOW: "sustained below GET", SUSTAINED_ABOVE: "sustained above GET",
               RAMP_GET: "ramp through GET", RAMP_END: "ramp end"}
PRBS_PROTOCOLS = ("Easy", "Medium", "Hard", "PRBS")
CHANGE_PROTOCOLS = ("Interval", "Step")
SUSTAINED_PROTOCOL = "Step"
PRBS_WINDOW_S = 30
CHANGE_WINDOW_S = 180  # also the sustained threshold: more than 180 s since the last WR change
RAMP_GET_HALF_WINDOW_S = 120
RAMP_END_WINDOW_S = 120
LONG_RUN_S = 60
PRIMARY, WHOLE_RUN = "primary", "prbs_whole_run"
LABELLINGS = (PRIMARY, WHOLE_RUN)
POOLED = "pooled"
CI_LEVEL = 0.95
WR_TOLERANCE_W = 1e-3

VISIT_COLUMNS = ["participant_id", "cohort", "protocol", "visit_id"]
SECOND_KEYS = ["visit_id", "t_s"]
EVENT_COLUMNS = ["phase", "anchor_s", "start_s", "stop_s"]
VISIT_RUN_KEYS = ["config_id", *VISIT_COLUMNS]


class NonContiguousTargetsError(ValueError):
    """A visit's target seconds (for one configuration) do not form one contiguous block."""


class WorkRateMismatchError(ValueError):
    """The cache WR differs from the predictions' `work_rate` on some target second."""


# --- inputs -----------------------------------------------------------------------------------------------------

def load_predictions(run_id: str, configurations: Sequence[str], runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    """The run's seed-averaged predictions for `configurations`, plus `residual` = predicted - measured."""
    path = runs_root / run_id / SUMMARY_NAME / SEEDAVG_PREDICTIONS_NAME
    predictions = pd.read_parquet(path, columns=SEEDAVG_COLUMNS, filters=[("config_id", "in", list(configurations))])
    missing = sorted(set(configurations) - set(predictions["config_id"]))
    if missing:
        raise ValueError(f"{run_id}: configurations {missing} not in {path}")
    return predictions.assign(residual=predictions["vo2_pred"] - predictions["vo2_true"])


def check_contiguous(predictions: pd.DataFrame) -> None:
    per_visit = predictions.groupby(["config_id", "visit_id"])["t_s"].agg(["size", "nunique", "min", "max"])
    broken = per_visit[(per_visit["size"] != per_visit["nunique"])
                       | (per_visit["max"] - per_visit["min"] + 1 != per_visit["size"])]
    if len(broken):
        raise NonContiguousTargetsError(f"{len(broken)} (configuration, visit) target blocks are not contiguous, "
                                        f"e.g. {broken.head().reset_index().to_dict('records')}")


def cache_work_rate(run_id: str, predictions: pd.DataFrame, runs_root: Path = RUNS_ROOT,
                    cache_root: Path = CACHE_ROOT) -> pd.DataFrame:
    """`visit_id, t_s, phase_code, wr` for every grid second of the predicted visits, from the run's own cache."""
    keys = {json.loads((runs_root / run_id / fold / MANIFEST_NAME).read_text(encoding="utf-8"))["cache_key"]
            for fold in predictions["fold"].unique()}
    if len(keys) != 1:
        raise ValueError(f"{run_id}: folds were trained on {len(keys)} caches ({sorted(keys)}); expected one")
    key = keys.pop()
    folder = cache_root / key
    if not folder.exists():
        raise FileNotFoundError(f"{run_id}: training cache {folder} is missing; it is not rebuilt here")
    cache = load(folder, key)
    visit_ids = predictions["visit_id"].unique()
    unknown = sorted(set(visit_ids) - set(cache.visits["visit_id"]))
    if unknown:
        raise ValueError(f"{run_id}: visits {unknown} are not in cache {key[:12]}")
    row_visit_ids = cache.row_ids("visit_id")
    rows = np.isin(row_visit_ids, visit_ids)
    return pd.DataFrame({"visit_id": row_visit_ids[rows],
                         "t_s": cache.column("t")[rows].astype(np.int64),
                         "phase_code": cache.column("phase")[rows].astype(np.int64),
                         "wr": cache.column(WORK_RATE_COLUMN)[rows].astype(np.float64)})


def check_work_rate(predictions: pd.DataFrame, work_rate: pd.DataFrame) -> None:
    merged = predictions[[*SECOND_KEYS, "work_rate"]].merge(work_rate, on=SECOND_KEYS, how="left")
    off = merged[~(np.abs(merged["work_rate"] - merged["wr"]) <= WR_TOLERANCE_W)]  # NaN wr (missing second) fails
    if len(off):
        raise WorkRateMismatchError(f"{len(off)} target seconds where cache WR != predictions work_rate, e.g. "
                                    f"{off.head().to_dict('records')}")


def read_get_power(path: Path = GET_POWER_CSV) -> pd.DataFrame:
    return pd.read_csv(path).set_index("participant_id")


# --- phase events -----------------------------------------------------------------------------------------------

def wr_changes(t: np.ndarray, wr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seconds at which WR differs from the second before, the new level, and the sign (+1 up, -1 down)."""
    index = np.flatnonzero(np.diff(wr) != 0) + 1
    return t[index], wr[index], np.sign(wr[index] - wr[index - 1])


def events_frame(phase: np.ndarray, anchor: np.ndarray, start: np.ndarray, stop: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"phase": phase, "anchor_s": anchor, "start_s": start, "stop_s": stop}, columns=EVENT_COLUMNS)


def change_events(changes: tuple[np.ndarray, np.ndarray, np.ndarray], exercise_start: int, end: int,
                  window_s: int | None, up_phase: int, down_phase: int) -> pd.DataFrame:
    """One event per exercise-phase change: [change, change + window) capped at the next change (or the end)."""
    times, _, signs = changes
    is_exercise = times >= exercise_start
    starts = times[is_exercise]
    nexts = np.append(times, end)[1:][is_exercise]
    stops = nexts if window_s is None else np.minimum(starts + window_s, nexts)
    return events_frame(np.where(signs[is_exercise] > 0, up_phase, down_phase), starts, starts, stops)


def sustained_events(changes: tuple[np.ndarray, np.ndarray, np.ndarray], exercise_start: int, end: int,
                     get_w: float) -> pd.DataFrame:
    """One block per level held > 180 s, clocked from its change (warm-up changes count); anchor = the change."""
    times, levels, _ = changes
    starts = np.maximum(times + CHANGE_WINDOW_S, exercise_start)
    stops = np.append(times, end)[1:]
    held = starts < stops
    phase = np.where(levels[held] > get_w, SUSTAINED_ABOVE, SUSTAINED_BELOW)
    return events_frame(phase, times[held], starts[held], stops[held])


def ramp_events(visit_id: str, exercise_start: int, end: int, get_time_s: int) -> pd.DataFrame:
    """GET crossing +/- 120 s and the last 120 s before task failure (= the visit's last grid second + 1)."""
    get_start, get_stop = get_time_s - RAMP_GET_HALF_WINDOW_S, get_time_s + RAMP_GET_HALF_WINDOW_S
    end_start = end - RAMP_END_WINDOW_S
    if get_start < exercise_start or get_stop > end_start:
        raise ValueError(f"{visit_id}: GET window [{get_start}, {get_stop}) leaves the exercise phase "
                         f"[{exercise_start}, ...) or overlaps the ramp-end window [{end_start}, {end})")
    return events_frame(np.array([RAMP_GET, RAMP_END]), np.array([get_time_s, end]),
                        np.array([get_start, end_start]), np.array([get_stop, end]))


def get_power_row(get_power: pd.DataFrame, participant_id: str) -> pd.Series:
    if participant_id not in get_power.index:
        raise KeyError(f"{participant_id}: no row in the GET power table")
    return get_power.loc[participant_id]


def visit_events(visit: pd.DataFrame, visit_id: str, participant_id: str, protocol: str,
                 get_power: pd.DataFrame) -> pd.DataFrame:
    """Both labellings' events of one visit; `visit` = its grid seconds in time order."""
    t, wr = visit["t_s"].to_numpy(), visit["wr"].to_numpy()
    exercise_start, end = int(t[visit["phase_code"].to_numpy() == EXERCISE][0]), int(t[-1]) + 1
    changes = wr_changes(t, wr)
    if protocol in PRBS_PROTOCOLS:
        return pd.concat([
            change_events(changes, exercise_start, end, PRBS_WINDOW_S, PRBS_UP, PRBS_DOWN).assign(labelling=PRIMARY),
            change_events(changes, exercise_start, end, None, PRBS_UP, PRBS_DOWN).assign(labelling=WHOLE_RUN)])
    if protocol in CHANGE_PROTOCOLS:
        events = [change_events(changes, exercise_start, end, CHANGE_WINDOW_S, INCREASE, DECREASE)]
        if protocol == SUSTAINED_PROTOCOL:
            get_w = float(get_power_row(get_power, participant_id)["get_w"])
            events.append(sustained_events(changes, exercise_start, end, get_w))
        shared = pd.concat(events)
    elif protocol == RAMP_PROTOCOL:
        row = get_power_row(get_power, participant_id)
        if not bool(row["ramp_included"]):
            raise ValueError(f"{visit_id}: predicted, but its ramp is not included in the GET power table")
        shared = ramp_events(visit_id, exercise_start, end, int(row["ramp_get_time_s"]))
    else:
        raise ValueError(f"{visit_id}: no phase rule for protocol {protocol!r}")
    return pd.concat([shared.assign(labelling=labelling) for labelling in LABELLINGS])


def all_events(work_rate: pd.DataFrame, visits: pd.DataFrame, get_power: pd.DataFrame) -> pd.DataFrame:
    """Events of every visit in `visits` (VISIT_COLUMNS); `event` is unique across the table."""
    by_visit = dict(tuple(work_rate.sort_values(SECOND_KEYS).groupby("visit_id")))
    tables = [visit_events(by_visit[v.visit_id], v.visit_id, v.participant_id, v.protocol, get_power)
              .assign(participant_id=v.participant_id, cohort=v.cohort, protocol=v.protocol, visit_id=v.visit_id)
              for v in visits.itertuples(index=False)]
    events = pd.concat(tables, ignore_index=True)
    return events.assign(event=np.arange(len(events)))


def label_seconds(events: pd.DataFrame) -> pd.DataFrame:
    """`labelling, visit_id, t_s, phase, event, offset_s` for every second of every event window."""
    lengths = (events["stop_s"] - events["start_s"]).to_numpy()
    repeated = events.loc[events.index.repeat(lengths)]
    t_s = repeated["start_s"] + repeated.groupby(level=0).cumcount()
    labels = pd.DataFrame({"labelling": repeated["labelling"], "visit_id": repeated["visit_id"], "t_s": t_s,
                           "phase": repeated["phase"], "event": repeated["event"],
                           "offset_s": t_s - repeated["anchor_s"]}).reset_index(drop=True)
    overlaps = labels.duplicated(["labelling", *SECOND_KEYS])
    if overlaps.any():
        raise ValueError(f"{int(overlaps.sum())} seconds lie in two phase windows, e.g. "
                         f"{labels[overlaps].head().to_dict('records')}")
    return labels


def label_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per labelling x prediction row; unlabelled seconds get phase OTHER."""
    frames = [predictions.merge(labels[labels["labelling"] == labelling].drop(columns="labelling"),
                                on=SECOND_KEYS, how="left").assign(labelling=labelling)
              for labelling in LABELLINGS]
    labelled = pd.concat(frames, ignore_index=True)
    return labelled.assign(phase=labelled["phase"].fillna(OTHER).astype(np.int64))


def event_counts(events: pd.DataFrame, labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    targets = predictions[SECOND_KEYS].drop_duplicates()
    with_targets = labels.merge(targets, on=SECOND_KEYS)["event"].unique()
    counted = events.assign(has_targets=events["event"].isin(with_targets))
    table = (counted.groupby(["labelling", "phase", "cohort", "protocol"])
             .agg(n_visits=("visit_id", "nunique"), n_events=("event", "size"),
                  n_events_with_targets=("has_targets", "sum"))
             .reset_index())
    return table.assign(phase_name=table["phase"].map(PHASE_NAMES))


# --- outputs ----------------------------------------------------------------------------------------------------

VISIT_RUN_COLUMNS = [*VISIT_RUN_KEYS, "n_s", "n_pos", "n_neg", "n_zero", "runs", "mean_run_length_s", "share_long"]
RUNS_TEST_COLUMNS = ["protocol", "config_id", "n_participants", "n_visits", "n", "n_pos", "n_neg", "n_zero", "runs",
                     "expected_runs", "z", "p", "mean_run_length_s"]
PHASE_KEYS = ["labelling", "group", "config_id", "phase"]
PHASE_COLUMNS = [*PHASE_KEYS, "phase_name", "n_participants", "n_events", "n_seconds", "bias_mean", "bias_sd",
                 "bias_ci_lo", "bias_ci_hi", "time_share", "error_share", "error_ratio"]
TRACE_KEYS = ["labelling", "phase", "config_id", "offset_s"]
TRACE_SIGNALS = {"measured": "vo2_true", "predicted": "vo2_pred", "residual": "residual"}


def t_interval(mean: np.ndarray, sd: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided t-CI of a mean across participants (NaN below 2)."""
    df = np.where(n > 1, n - 1, np.nan)
    half_width = stats.t.ppf(0.5 + CI_LEVEL / 2, df) * sd / np.sqrt(n)
    return mean - half_width, mean + half_width


def visit_runs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per configuration x visit: sign runs of the residual. Zero residuals are dropped first (their neighbours
    join); n_s counts the non-zero seconds, `share_long` the share of them in runs longer than 60 s."""
    ordered = predictions.sort_values([*VISIT_RUN_KEYS, "t_s"])
    n_zero = (ordered.assign(n_zero=ordered["residual"] == 0).groupby(VISIT_RUN_KEYS)["n_zero"].sum()
              .reset_index())
    nonzero = ordered[ordered["residual"] != 0]
    keys = nonzero[VISIT_RUN_KEYS]
    is_positive = nonzero["residual"].to_numpy() > 0
    is_new_visit = (keys != keys.shift()).any(axis=1).to_numpy()
    is_new_run = is_new_visit | np.append(True, is_positive[1:] != is_positive[:-1])
    run_number = np.cumsum(is_new_run)
    run_length = np.bincount(run_number)[run_number]
    per_second = keys.assign(is_positive=is_positive, is_new_run=is_new_run, is_long=run_length > LONG_RUN_S)
    table = (per_second.groupby(VISIT_RUN_KEYS)
             .agg(n_s=("is_new_run", "size"), n_pos=("is_positive", "sum"), runs=("is_new_run", "sum"),
                  long_s=("is_long", "sum"))
             .reset_index()
             .merge(n_zero, on=VISIT_RUN_KEYS, how="left"))
    return table.assign(n_neg=table["n_s"] - table["n_pos"], mean_run_length_s=table["n_s"] / table["runs"],
                        share_long=table["long_s"] / table["n_s"])[VISIT_RUN_COLUMNS]


def runs_test(visit_table: pd.DataFrame) -> pd.DataFrame:
    """Wald-Wolfowitz per protocol x configuration: runs summed over visits, E and Var from the pooled signs."""
    table = (visit_table.groupby(["protocol", "config_id"])
             .agg(n_participants=("participant_id", "nunique"), n_visits=("visit_id", "size"), n=("n_s", "sum"),
                  n_pos=("n_pos", "sum"), n_neg=("n_neg", "sum"), n_zero=("n_zero", "sum"), runs=("runs", "sum"))
             .reset_index())
    n, pos, neg = (table[c].to_numpy(dtype=np.float64) for c in ("n", "n_pos", "n_neg"))
    expected = 2 * pos * neg / n + 1
    variance = 2 * pos * neg * (2 * pos * neg - n) / (n ** 2 * (n - 1))
    z = (table["runs"].to_numpy() - expected) / np.sqrt(variance)
    return table.assign(expected_runs=expected, z=z, p=2 * stats.norm.sf(np.abs(z)),
                        mean_run_length_s=n / table["runs"].to_numpy())[RUNS_TEST_COLUMNS]


def run_length_summary(visit_table: pd.DataFrame) -> pd.DataFrame:
    """Participant value = mean over its visits of the protocol; then mean and SD across participants."""
    participants = (visit_table.groupby(["protocol", "config_id", "participant_id"])
                    [["mean_run_length_s", "share_long"]].mean().reset_index())
    return (participants.groupby(["protocol", "config_id"])
            .agg(n_participants=("participant_id", "size"),
                 mean_run_length_mean=("mean_run_length_s", "mean"), mean_run_length_sd=("mean_run_length_s", "std"),
                 share_long_mean=("share_long", "mean"), share_long_sd=("share_long", "std"))
            .reset_index())


def phase_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Bias across participants and squared-error share / time share pooled over each group's target seconds."""
    grouped = pd.concat([labelled.assign(group=POOLED), labelled.assign(group=labelled["cohort"])],
                        ignore_index=True)
    participant_bias = grouped.groupby([*PHASE_KEYS, "participant_id"])["residual"].mean()
    bias = participant_bias.groupby(level=PHASE_KEYS).agg(n_participants="size", bias_mean="mean", bias_sd="std")
    seconds = (grouped.assign(squared=grouped["residual"] ** 2).groupby(PHASE_KEYS)
               .agg(n_events=("event", "nunique"), n_seconds=("residual", "size"), sse=("squared", "sum")))
    table = seconds.join(bias).reset_index()
    totals = table.groupby(["labelling", "group", "config_id"])[["n_seconds", "sse"]].transform("sum")
    lo, hi = t_interval(table["bias_mean"].to_numpy(), table["bias_sd"].to_numpy(),
                        table["n_participants"].to_numpy())
    time_share, error_share = table["n_seconds"] / totals["n_seconds"], table["sse"] / totals["sse"]
    return table.assign(phase_name=table["phase"].map(PHASE_NAMES), bias_ci_lo=lo, bias_ci_hi=hi,
                        time_share=time_share, error_share=error_share,
                        error_ratio=error_share / time_share)[PHASE_COLUMNS]


def trace_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Per offset from the anchor: participant means over its events, then mean and SD across participants."""
    windows = labelled[labelled["phase"] != OTHER]
    windows = windows.assign(offset_s=windows["offset_s"].astype(np.int64))
    per_participant = (windows.groupby([*TRACE_KEYS, "participant_id"])
                       .agg(n_events=("event", "size"),
                            **{name: (column, "mean") for name, column in TRACE_SIGNALS.items()})
                       .reset_index())
    spreads = {f"{name}_{label}": (name, stat)
               for name in TRACE_SIGNALS for stat, label in (("mean", "mean"), ("std", "sd"))}
    return (per_participant.groupby(TRACE_KEYS)
            .agg(n_participants=("participant_id", "size"), n_events=("n_events", "sum"), **spreads)
            .reset_index())


def write_residuals(run_id: str = A1_POOLED.run_id, configurations: Sequence[str] = A2_CONFIGURATIONS,
                    runs_root: Path = RUNS_ROOT, cache_root: Path = CACHE_ROOT, get_power_path: Path = GET_POWER_CSV,
                    out_dir: Path | None = None) -> Path:
    """All §11.6 outputs for one evaluated run; returns the output folder."""
    predictions = load_predictions(run_id, configurations, runs_root)
    check_contiguous(predictions)
    work_rate = cache_work_rate(run_id, predictions, runs_root, cache_root)
    check_work_rate(predictions, work_rate)
    visits = predictions[VISIT_COLUMNS].drop_duplicates()
    events = all_events(work_rate, visits, read_get_power(get_power_path))
    labels = label_seconds(events)
    labelled = label_predictions(predictions, labels)
    visit_table = visit_runs(predictions)
    tables = {EVENTS_NAME: event_counts(events, labels, predictions), RUNS_TEST_NAME: runs_test(visit_table),
              VISIT_RUNS_NAME: visit_table, RUN_LENGTH_NAME: run_length_summary(visit_table),
              PHASES_NAME: phase_table(labelled), TRACES_NAME: trace_table(labelled)}
    folder = out_dir if out_dir is not None else RESIDUALS_ROOT / run_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / name, index=False)
    primary = events[events["labelling"] == PRIMARY].groupby("phase").size()
    logger.info("Residual analyses of %s written to %s; primary events per phase: %s", run_id, folder,
                primary.to_dict())
    return folder
