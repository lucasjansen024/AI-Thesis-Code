"""Aim 2 outcome (METHODOLOGY Appendix B.6): per participant, seed-averaged RMSE without vs with the condition in
training, and the mean difference (unseen - seen, mL/min) with a 95 % t-CI across participants.

Reads only each run's `summary/visit_metrics_seedavg.csv` (so `evaluate` must have run). Pairs:
- 2a/2b: a condition's unseen and seen arms, on the test participants' `left_out` visits (scope = that protocol).
- 2c: the transfer run and its within-cohort A1 reference, restricted to the transfer's configurations and test
  participants; scope `all` (participant mean over visits) and one scope per protocol.
Both sides of a pair must hold exactly the same (configuration, visit, n target seconds) rows, else nothing is
written; matching `n_s` guards against a reference scored on other seconds (e.g. after an inclusion change).

Files in `outputs/aim2/` (or `out_dir`): `participants.csv` (one row per study x condition x scope x configuration
x participant) and `summary.csv` (one row per study x condition x scope x configuration).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.evaluate import SUMMARY_NAME, VISIT_METRICS_SEEDAVG_NAME
from pipeline.provenance import REPO_ROOT
from pipeline.runs import A2C_STUDY, CohortTransfer, LeftOutArm
from pipeline.train import RUNS_ROOT

logger = logging.getLogger(__name__)

OUTCOME_ROOT = REPO_ROOT / "outputs" / "aim2"
PARTICIPANTS_NAME, SUMMARY_TABLE_NAME = "participants.csv", "summary.csv"
CI_LEVEL = 0.95
ALL_PROTOCOLS = "all"
PAIR_KEYS = ["config_id", "participant_id", "protocol", "visit_id", "n_s"]
LABEL_COLUMNS = ["study", "condition", "scope", "config_id"]
PARTICIPANT_COLUMNS = [*LABEL_COLUMNS, "participant_id", "n_visits", "rmse_seen", "rmse_unseen", "diff"]
SUMMARY_COLUMNS = [*LABEL_COLUMNS, "n_participants", "rmse_seen_mean", "rmse_unseen_mean", "diff_mean", "diff_sd",
                   "diff_ci_lo", "diff_ci_hi"]


class UnpairedVisitsError(ValueError):
    """The two sides of a seen/unseen pair do not hold the same (configuration, visit) rows."""


def visit_rmse(run_id: str, runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    path = runs_root / run_id / SUMMARY_NAME / VISIT_METRICS_SEEDAVG_NAME
    return pd.read_csv(path)[[*PAIR_KEYS, "rmse"]]


def paired_visits(seen: pd.DataFrame, unseen: pd.DataFrame, label: str) -> pd.DataFrame:
    """`rmse_seen`, `rmse_unseen` per (configuration, visit); raises unless both sides hold the same rows."""
    merged = seen.merge(unseen, on=PAIR_KEYS, how="outer", suffixes=("_seen", "_unseen"), indicator=True)
    unpaired = merged[merged["_merge"] != "both"]
    if len(unpaired):
        sample = unpaired[["config_id", "visit_id", "_merge"]].head().to_dict("records")
        raise UnpairedVisitsError(f"{label}: {len(unpaired)} (configuration, visit) rows on one side only, e.g. "
                                  f"{sample}")
    return merged.drop(columns="_merge")


def participant_rows(pairs: pd.DataFrame, study: str, condition: str, scope: str) -> pd.DataFrame:
    """Participant value = mean over its paired visits (Appendix B.6); diff = unseen - seen."""
    means = (pairs.groupby(["config_id", "participant_id"])
             .agg(n_visits=("visit_id", "size"), rmse_seen=("rmse_seen", "mean"),
                  rmse_unseen=("rmse_unseen", "mean"))
             .reset_index())
    labelled = means.assign(study=study, condition=condition, scope=scope,
                            diff=means["rmse_unseen"] - means["rmse_seen"])
    return labelled[PARTICIPANT_COLUMNS]


def summary_rows(participants: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD (ddof 1) of the differences across participants, with a two-sided t-CI (NaN below 2)."""
    grouped = (participants.groupby(LABEL_COLUMNS, sort=False)
               .agg(n_participants=("participant_id", "size"), rmse_seen_mean=("rmse_seen", "mean"),
                    rmse_unseen_mean=("rmse_unseen", "mean"), diff_mean=("diff", "mean"), diff_sd=("diff", "std"))
               .reset_index())
    n = grouped["n_participants"].to_numpy()
    df = np.where(n > 1, n - 1, np.nan)
    half_width = stats.t.ppf(0.5 + CI_LEVEL / 2, df) * grouped["diff_sd"].to_numpy() / np.sqrt(n)
    return grouped.assign(diff_ci_lo=grouped["diff_mean"] - half_width,
                          diff_ci_hi=grouped["diff_mean"] + half_width)[SUMMARY_COLUMNS]


def left_out_participants(arms: tuple[LeftOutArm, ...], runs_root: Path = RUNS_ROOT) -> list[pd.DataFrame]:
    """One table per (study, left-out condition); each needs exactly one unseen and one seen arm."""
    by_condition: dict[tuple[str, str], dict[bool, LeftOutArm]] = {}
    for arm in arms:
        pair = by_condition.setdefault((arm.study, arm.left_out), {})
        if arm.is_seen in pair:
            raise ValueError(f"{arm.study} {arm.left_out}: two {'seen' if arm.is_seen else 'unseen'} arms "
                             f"({pair[arm.is_seen].run.run_id}, {arm.run.run_id}); pass one of each")
        pair[arm.is_seen] = arm
    tables = []
    for (study, condition), pair in by_condition.items():
        if set(pair) != {False, True}:
            raise ValueError(f"{study} {condition}: needs one unseen and one seen arm, got {sorted(pair)}")
        seen, unseen = (visit_rmse(pair[s].run.run_id, runs_root) for s in (True, False))
        pairs = paired_visits(seen[seen["protocol"] == condition], unseen[unseen["protocol"] == condition],
                              f"{study} {condition}")
        tables.append(participant_rows(pairs, study, condition, condition))
    return tables


def transfer_participants(spec: CohortTransfer, runs_root: Path = RUNS_ROOT) -> list[pd.DataFrame]:
    """Scope `all`, then one scope per protocol, against the test cohort's within-cohort A1 run."""
    reference = visit_rmse(spec.reference_run_id, runs_root)
    reference = reference[reference["config_id"].isin(spec.run.configurations)
                          & reference["participant_id"].isin(spec.run.fold_ids)]
    pairs = paired_visits(reference, visit_rmse(spec.run.run_id, runs_root),
                          f"{spec.run.run_id} vs {spec.reference_run_id}")
    condition = f"{spec.train_cohort}_to_{spec.test_cohort}"
    scoped = [participant_rows(pairs, A2C_STUDY, condition, ALL_PROTOCOLS)]
    return scoped + [participant_rows(pairs[pairs["protocol"] == p], A2C_STUDY, condition, p)
                     for p in sorted(pairs["protocol"].unique())]


def write_outcome(arms: tuple[LeftOutArm, ...] = (), transfers: tuple[CohortTransfer, ...] = (),
                  runs_root: Path = RUNS_ROOT, out_dir: Path = OUTCOME_ROOT) -> Path:
    """Write `participants.csv` and `summary.csv` for the given arms and transfers; returns `out_dir`."""
    transfer_tables = [t for spec in transfers for t in transfer_participants(spec, runs_root)]
    tables = left_out_participants(arms, runs_root) + transfer_tables
    if not tables:
        raise ValueError("write_outcome: no arms and no transfers given")
    participants = pd.concat(tables, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(out_dir / PARTICIPANTS_NAME, index=False)
    summary_rows(participants).to_csv(out_dir / SUMMARY_TABLE_NAME, index=False)
    logger.info("Aim 2 outcome: %d participant rows written to %s", len(participants), out_dir)
    return out_dir
