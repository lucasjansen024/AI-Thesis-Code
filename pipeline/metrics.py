"""Metrics in float64 on mL/min: per-visit errors, rm_corr, Pearson r, repeated-measures Bland-Altman, VO₂peak."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pingouin as pg

logger = logging.getLogger(__name__)

VISIT_KEYS = ["config_id", "seed", "participant_id", "cohort", "protocol", "visit_id"]
VISIT_METRIC_COLUMNS = [*VISIT_KEYS, "n_s", "rmse", "mae", "bias"]
VO2PEAK_WINDOW_S = 30  # dev flag `vo2peak_window_s` (legacy used a 1-s max)
LOA_Z = 1.96
MIN_RM_CORR_SUBJECTS = 3  # pingouin.rm_corr refuses fewer subjects
MIN_BA_SUBJECTS = 2  # m-tilde divides by (n - 1)


@dataclass(frozen=True)
class RmBlandAltman:
    bias: float
    sd: float
    loa_lo: float
    loa_hi: float


def visit_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, seed, visit): n_s, RMSE, MAE, bias = mean(pred - true)."""
    error = predictions["vo2_pred"].astype(np.float64) - predictions["vo2_true"].astype(np.float64)
    grouped = predictions.assign(error=error, abs_error=error.abs(), sq_error=error**2).groupby(VISIT_KEYS, sort=True)
    out = grouped.agg(n_s=("error", "size"), mse=("sq_error", "mean"), mae=("abs_error", "mean"),
                      bias=("error", "mean")).reset_index()
    return out.assign(rmse=np.sqrt(out["mse"]))[VISIT_METRIC_COLUMNS]


def pearson_r(measured: np.ndarray, predicted: np.ndarray) -> float:
    measured, predicted = np.asarray(measured, np.float64), np.asarray(predicted, np.float64)
    if len(measured) < 2:
        return float("nan")
    return float(np.corrcoef(measured, predicted)[0, 1])


def rm_corr_r(measured: np.ndarray, predicted: np.ndarray, subject: np.ndarray) -> float:
    """`pingouin.rm_corr` r (subject = participant); NaN, logged, below 3 subjects."""
    frame = pd.DataFrame({"measured": np.asarray(measured, np.float64),
                          "predicted": np.asarray(predicted, np.float64), "subject": np.asarray(subject)})
    n_subjects = frame["subject"].nunique()
    if n_subjects < MIN_RM_CORR_SUBJECTS:
        logger.warning("rm_corr needs >= %d subjects, got %d: r_rm = NaN", MIN_RM_CORR_SUBJECTS, n_subjects)
        return float("nan")
    return float(pg.rm_corr(data=frame, x="measured", y="predicted", subject="subject")["r"].iloc[0])


def one_way_ms(values: np.ndarray, subject: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(MS_between, MS_within, m_i) of a one-way ANOVA of `values` on `subject`, grouped by exact ID."""
    frame = pd.DataFrame({"v": np.asarray(values, np.float64), "s": np.asarray(subject)})
    grouped = frame.groupby("s", sort=False)["v"]
    counts = grouped.size().to_numpy()
    n_subjects, total = len(counts), counts.sum()
    ss_between = float((counts * (grouped.mean().to_numpy() - frame["v"].mean()) ** 2).sum())
    ss_within = float(((frame["v"] - grouped.transform("mean")) ** 2).sum())
    return ss_between / (n_subjects - 1), ss_within / (total - n_subjects), counts


def rm_bland_altman(measured: np.ndarray, predicted: np.ndarray, subject: np.ndarray) -> RmBlandAltman:
    """Bland & Altman (2007), true value varies: bias = mean(pred - true); LoA = bias +- 1.96 sigma.

    sigma^2 = max(0, (MS_b - MS_w) / m~) + MS_w, m~ = ((sum m_i)^2 - sum m_i^2) / ((n - 1) sum m_i).
    Port of legacy `Analysis.bland_altman_global` with the variance component clamped at 0. Below two subjects
    sigma is undefined: NaN, logged.
    """
    diff = np.asarray(predicted, np.float64) - np.asarray(measured, np.float64)
    bias = float(diff.mean()) if len(diff) else float("nan")
    n_subjects = pd.Series(np.asarray(subject)).nunique()
    if n_subjects < MIN_BA_SUBJECTS or len(diff) <= n_subjects:
        logger.warning("rm Bland-Altman needs >= %d subjects with repeats, got %d subjects / %d values: "
                       "sigma = NaN", MIN_BA_SUBJECTS, n_subjects, len(diff))
        return RmBlandAltman(bias, float("nan"), float("nan"), float("nan"))
    ms_between, ms_within, counts = one_way_ms(diff, subject)
    total = counts.sum()
    m_tilde = (total**2 - (counts**2).sum()) / ((n_subjects - 1) * total)
    sd = float(np.sqrt(max(0.0, (ms_between - ms_within) / m_tilde) + ms_within))
    return RmBlandAltman(bias, sd, bias - LOA_Z * sd, bias + LOA_Z * sd)


def vo2peak(t_s: np.ndarray, vo2: np.ndarray, window_s: int = VO2PEAK_WINDOW_S) -> float:
    """Highest trailing `window_s` mean over full windows only: a window needs every second t-window+1..t.

    NaN when no full window exists.
    """
    series = pd.Series(np.asarray(vo2, np.float64), index=np.asarray(t_s, np.int64))
    if series.index.has_duplicates:
        raise ValueError("vo2peak: duplicate seconds in one visit")
    if series.empty:
        return float("nan")
    series = series.sort_index()
    full_grid = series.reindex(np.arange(series.index[0], series.index[-1] + 1))
    return float(full_grid.rolling(window_s, min_periods=window_s).mean().max())
