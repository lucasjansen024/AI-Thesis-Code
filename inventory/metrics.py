"""Per-signal and per-visit inventory rows: coverage, gaps, fit factor, metadata, flags.

Rows are plain dicts because they go straight into CSV files.
Flags describe what is missing or suspect; they are not inclusion decisions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from inventory.annotations import expected_exclusions
from inventory.sensors import FIT_FACTOR, NIRS_CHANNELS, TSI, Loaded, VisitData
from src2_training_ready.preparation.data_preparer import VISIT_ORDER

BREATH_GAP_S = 15  # a breath-to-breath gap longer than this counts as a gap
SERIES_GAP_S = 3  # consecutive empty seconds in a ~1 Hz series that count as a gap
COVERAGE_FLAG = 0.95
REST_HR_COVERAGE_FLAG = 0.8
CPET_HR_SOURCE = "VO2"
FIT_FACTOR_SHARE_FLAG = 0.95
FIT_FACTOR_THRESHOLDS = {"ge99": 99.0, "ge999": 99.9}
TEN_SECOND_SPACING_FLAG_S = 6.0
RAMP_FIT_SKIP_START_S = 60  # skip the first minute after warm-up when fitting the ramp slope
RAMP_FIT_SKIP_END_S = 30
MIN_POINTS_FOR_FIT = 10
WARMUP_EDGE_SKIP_S = 15  # skip the load transitions at both ends of the ramp warm-up
SECONDS_PER_MINUTE = 60
# HR plausibility from rest to exercise end; starting values, tuned in the data-validity ticket.
HR_MIN_SPAN_BPM = 20.0  # p95 - p5; a working sensor spans far more than this in every protocol
HR_MIN_VO2_R = 0.5  # Pearson r of 1-s HR vs smoothed CPET VO2
HR_SPAN_PERCENTILES = (5, 95)
VO2_SMOOTH_S = 30
FITDAYS_COLUMNS = {
    "skeletal_muscle": ("Skeletal muscle", "Skeletspier"),
    "muscle_mass": ("Muscle mass", "Spiermassa"),
    "weight": ("Weight", "Gewicht"),
}


@dataclass(frozen=True)
class Coverage:
    coverage: float
    max_gap_s: float
    n_gaps: int


NO_COVERAGE = Coverage(math.nan, math.nan, 0)


def _empty_runs(covered: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate([[1], covered.astype(int), [1]]))
    return np.flatnonzero(edges == 1) - np.flatnonzero(edges == -1)


def series_coverage(t: pd.Series, values: pd.Series, start: float, end: float | None) -> Coverage:
    """Share of whole seconds in [start, end) holding at least one finite sample."""
    if end is None or end <= start:
        return NO_COVERAGE
    t_arr, v_arr = t.to_numpy(dtype=float), values.to_numpy(dtype=float)
    seconds = np.floor(t_arr[np.isfinite(v_arr) & np.isfinite(t_arr)]).astype(int)
    seconds = seconds[(seconds >= start) & (seconds < end)] - int(start)
    covered = np.zeros(int(end - start), dtype=bool)
    covered[seconds] = True
    runs = _empty_runs(covered)
    return Coverage(float(covered.mean()), float(runs.max(initial=0)), int((runs >= SERIES_GAP_S).sum()))


def breath_coverage(t: pd.Series, values: pd.Series, start: float, end: float | None) -> Coverage:
    """1 - (time inside breath gaps > BREATH_GAP_S) / window length."""
    if end is None or end <= start:
        return NO_COVERAGE
    t_arr, v_arr = t.to_numpy(dtype=float), values.to_numpy(dtype=float)
    breaths = np.sort(t_arr[np.isfinite(v_arr) & (t_arr >= start) & (t_arr < end)])
    gaps = np.diff(np.concatenate([[start], breaths, [end]]))
    long_gaps = gaps[gaps > BREATH_GAP_S]
    return Coverage(float(1 - long_gaps.sum() / (end - start)), float(gaps.max()), int(len(long_gaps)))


def _per_second(t: pd.Series, values: pd.Series, start: float, end: float) -> pd.Series:
    """Mean per whole second in [start, end); NaN where a second has no finite sample."""
    t_arr, v_arr = t.to_numpy(dtype=float), values.to_numpy(dtype=float)
    keep = np.isfinite(t_arr) & np.isfinite(v_arr) & (t_arr >= start) & (t_arr < end)
    means = pd.Series(v_arr[keep]).groupby(np.floor(t_arr[keep]).astype(int)).mean()
    return means.reindex(range(int(start), int(end)))


def hr_quality(visit: VisitData, t: pd.Series, hr: pd.Series) -> dict:
    """HR span and its correlation with VO2 from rest to exercise end: catches stuck HR that coverage misses.

    Rest is included because the rest-to-exercise rise separates a stuck sensor from a real one; within a
    Hard PRBS alone, a real HR can span < 20 bpm and decouple from VO2 (cardiac drift).
    """
    end = visit.timeline.exercise_end_s
    empty = {"hr_span_bpm": math.nan, "hr_vo2_r": math.nan}
    if end is None:
        return empty
    hr_s = _per_second(t, hr, 0, end)
    if hr_s.notna().sum() < MIN_POINTS_FOR_FIT:
        return empty
    low, high = np.nanpercentile(hr_s, HR_SPAN_PERCENTILES)
    r = math.nan
    if visit.cpet.ok:
        cpet = visit.cpet.data.df
        vo2 = (_per_second(cpet["t"], cpet["VO2"], 0, end).interpolate(limit_area="inside")
               .rolling(VO2_SMOOTH_S, center=True, min_periods=1).mean())
        if hr_s.std() > 0 and vo2.std() > 0:  # r is undefined for a constant series
            r = float(hr_s.corr(vo2))
    return {"hr_span_bpm": float(high - low), "hr_vo2_r": r}


def _file_fields(sensor: str, loaded: Loaded) -> dict:
    return {
        "sensor": sensor,
        "file_selected": loaded.choice.selected.name if loaded.choice.selected else "",
        "alternatives_ignored": "; ".join(loaded.choice.ignored),
        "selection_problem": loaded.choice.problem or "",
        "loaded": loaded.ok,
        "error": loaded.error or "",
    }


def _coverage_fields(cov: Coverage) -> dict:
    return {"exercise_coverage": cov.coverage, "max_gap_s": cov.max_gap_s, "n_gaps": cov.n_gaps}


def _rep_coverage_fields(
    visit: VisitData, cover: Callable[[pd.Series, pd.Series, float, float | None], Coverage],
    t: pd.Series, values: pd.Series,
) -> dict:
    """Coverage and longest gap per PRBS repetition: validity is decided per repetition, not per visit."""
    reps = visit.timeline.repetition_starts_s
    bounds = zip(reps, [*reps[1:], visit.timeline.exercise_end_s])
    fields = {}
    for number, (start, end) in enumerate(bounds, start=1):
        cov = cover(t, values, start, end)
        fields.update({f"exercise_coverage_rep{number}": cov.coverage, f"max_gap_s_rep{number}": cov.max_gap_s})
    return fields


def _window(visit: VisitData) -> tuple[float, float | None]:
    return visit.timeline.warmup_end_s, visit.timeline.exercise_end_s


def _cpet_rows(visit: VisitData) -> list[dict]:
    base = _file_fields("cpet", visit.cpet)
    if not visit.cpet.ok:
        return [{**base, "signal": "cpet_vo2"}]
    df = visit.cpet.data.df
    start, end = _window(visit)
    shared = {
        "duration_s": float(df["t"].max() - df["t"].min()),
        "mean_breath_spacing_s": float(df["t"].diff().mean()),
    }
    rows = []
    for column, signal in (("VO2", "cpet_vo2"), ("HR", "cpet_hr"), ("Power", "cpet_power")):
        values = df[column] if column in df.columns else pd.Series(np.nan, index=df.index)
        cov = breath_coverage(df["t"], values, start, end)
        rest = breath_coverage(df["t"], values, 0, visit.timeline.rest_end_s)
        quality = hr_quality(visit, df["t"], values) if column == "HR" else {}
        rows.append({**base, "signal": signal, **shared, **_coverage_fields(cov), "rest_coverage": rest.coverage,
                     **quality, **_rep_coverage_fields(visit, breath_coverage, df["t"], values)})
    return rows


def _rest_fields(series: pd.DataFrame, column: str, rest_end_s: int) -> dict:
    rest = series_coverage(series["t"], series[column], 0, rest_end_s)
    in_rest = series.loc[(series["t"] >= 0) & (series["t"] < rest_end_s), column]
    return {"rest_coverage": rest.coverage, "rest_mean": float(in_rest.mean())}


def _tymewear_rows(visit: VisitData) -> list[dict]:
    base = _file_fields("tymewear", visit.tymewear)
    if not visit.tymewear.ok:
        return [{**base, "signal": "tymewear_hr"}]
    data = visit.tymewear.data
    start, end = _window(visit)
    shared = {
        "duration_s": float(data.series["t"].max() - data.series["t"].min()),
        "sample_rate_hz": float(1 / data.series["t"].diff().median()),
        "sync": data.lag_note,
    }
    rows = []
    for column in ("HR", "VE", "BR"):
        cov = series_coverage(data.series["t"], data.series[column], start, end)
        rest = _rest_fields(data.series, column, visit.timeline.rest_end_s)
        quality = hr_quality(visit, data.series["t"], data.series[column]) if column == "HR" else {}
        rows.append({**base, "signal": f"tymewear_{column.lower()}", **shared, **_coverage_fields(cov), **rest,
                     **quality, **_rep_coverage_fields(visit, series_coverage, data.series["t"], data.series[column])})
    breath_cov = breath_coverage(data.breaths["t"], data.breaths["VE"], start, end)
    rows.append({
        **base, "signal": "tymewear_ve_bxb", **shared, **_coverage_fields(breath_cov),
        "mean_breath_spacing_s": float(data.breaths["t"].diff().mean()),
    })
    return rows


def _fit_factor_share(df: pd.DataFrame, start: float, end: float | None) -> dict:
    if end is None:
        return {key: math.nan for key in FIT_FACTOR_THRESHOLDS}
    in_window = df.loc[(df["t"] >= start) & (df["t"] < end), FIT_FACTOR].dropna()
    if in_window.empty:
        return {key: math.nan for key in FIT_FACTOR_THRESHOLDS}
    return {key: float((in_window >= value).mean()) for key, value in FIT_FACTOR_THRESHOLDS.items()}


def _fit_factor_fields(visit: VisitData) -> dict:
    df = visit.nirs.data.df
    start, end = _window(visit)
    fields = {f"ff_{k}_exercise": v for k, v in _fit_factor_share(df, start, end).items()}
    reps = visit.timeline.repetition_starts_s
    bounds = list(zip(reps, [*reps[1:], end]))
    for number, (rep_start, rep_end) in enumerate(bounds, start=1):
        shares = _fit_factor_share(df, rep_start, rep_end)
        fields.update({f"ff_{k}_rep{number}": v for k, v in shares.items()})
    return fields


def _nirs_rows(visit: VisitData) -> list[dict]:
    base = _file_fields("nirs", visit.nirs)
    if not visit.nirs.ok:
        return [{**base, "signal": "nirs_tsi"}]
    data = visit.nirs.data
    df = data.df
    start, end = _window(visit)
    hbdiff = pd.concat([df[f"O2Hb{ch}"] - df[f"HHb{ch}"] for ch in NIRS_CHANNELS], axis=1).mean(axis=1)
    shared = {
        "duration_s": float(df["t"].max() - df["t"].min()),
        "sample_rate_hz": data.frequency_hz,
        "header_sample_rate": data.header_frequency,
        "sync": data.sync,
        "n_events": len(data.events),
        "events": " | ".join(data.events[:5]),
        "n_all_zero_rows": data.n_all_zero_rows,
    }
    rows = []
    for signal, values in (("nirs_tsi", df[TSI]), ("nirs_hbdiff", hbdiff), ("nirs_fit_factor", df[FIT_FACTOR])):
        cov = series_coverage(df["t"], values, start, end)
        rows.append({**base, "signal": signal, **shared, **_coverage_fields(cov),
                     **_rep_coverage_fields(visit, series_coverage, df["t"], values)})
    rows[-1].update(_fit_factor_fields(visit))
    return rows


def _measured_power(visit: VisitData) -> pd.DataFrame | None:
    if not visit.cpet.ok or "Power" not in visit.cpet.data.df.columns:
        return None
    return visit.cpet.data.df[["t", "Power"]].dropna()


def _wr_rows(visit: VisitData) -> list[dict]:
    rows = []
    measured = _measured_power(visit)
    start, end = _window(visit)
    for sensor, loaded in (("protocol_wr", visit.protocol), ("reference_wr", visit.reference_wr)):
        row = {**_file_fields(sensor, loaded), "signal": f"{sensor}_w"}
        if loaded.ok:
            ideal = loaded.data
            row["duration_s"] = float(len(ideal))
            if measured is not None and end is not None:
                window = measured[(measured["t"] >= start) & (measured["t"] < end)]
                idx = np.clip(window["t"].to_numpy(dtype=int), 0, len(ideal) - 1)
                diff = np.abs(window["Power"].to_numpy(dtype=float) - ideal["W"].to_numpy()[idx])
                row["median_abs_diff_vs_measured_w"] = float(np.median(diff)) if len(diff) else math.nan
        rows.append(row)
    return rows


def signal_rows(visit: VisitData) -> list[dict]:
    rows = [*_cpet_rows(visit), *_tymewear_rows(visit), *_nirs_rows(visit), *_wr_rows(visit)]
    return [{"visit_id": visit.ref.visit_id, **row} for row in rows]


def _fitdays_fields(visit: VisitData) -> dict:
    fitdays, ref = visit.fitdays, visit.ref
    if not fitdays.ok:
        return {"fitdays_match": "no data"}
    df = fitdays.data
    if "Visit" in df.columns:
        hits = df[df["Visit"].astype(str).str.strip() == ref.source_visit]
        row, method = (hits.iloc[0] if len(hits) else None), "visit label"
    elif ref.protocol in VISIT_ORDER.get(ref.source_participant, []):
        position = VISIT_ORDER[ref.source_participant].index(ref.protocol)
        row, method = (df.iloc[position] if position < len(df) else None), "visit order"
    else:
        row, method = None, "no rule"
    fields = {"fitdays_rows": len(df), "fitdays_match": method if row is not None else f"{method}: no row"}
    for key, names in FITDAYS_COLUMNS.items():
        name = next((n for n in names if n in df.columns), None)
        fields[key] = row[name] if row is not None and name is not None else None
    return fields


def _ramp_rate_fit(visit: VisitData) -> float:
    measured = _measured_power(visit)
    end = visit.timeline.exercise_end_s
    if visit.ref.protocol != "Ramp" or measured is None or end is None:
        return math.nan
    start = visit.timeline.warmup_end_s + RAMP_FIT_SKIP_START_S
    window = measured[(measured["t"] >= start) & (measured["t"] < end - RAMP_FIT_SKIP_END_S)]
    if len(window) < MIN_POINTS_FOR_FIT:
        return math.nan
    slope_w_per_s = np.polyfit(window["t"].to_numpy(dtype=float), window["Power"].to_numpy(dtype=float), 1)[0]
    return float(slope_w_per_s * SECONDS_PER_MINUTE)


def _ramp_warmup_power(visit: VisitData) -> float:
    """Median measured power inside the nominal warm-up, edges skipped: checks the warm-up load per cohort."""
    measured = _measured_power(visit)
    if visit.ref.protocol != "Ramp" or measured is None:
        return math.nan
    start = visit.timeline.rest_end_s + WARMUP_EDGE_SKIP_S
    end = visit.timeline.warmup_end_s - WARMUP_EDGE_SKIP_S
    window = measured[(measured["t"] >= start) & (measured["t"] < end)]
    return float(window["Power"].median()) if len(window) else math.nan


def _header(visit: VisitData) -> dict:
    header = visit.cpet.data.header if visit.cpet.ok else {}
    return {"sex": header.get("sex"), "age": header.get("age"), "height": header.get("height"),
            "cpet_protocol": header.get("protocol")}


def _ramp_step_w(visit: VisitData) -> float:
    """Most common positive load increment of measured power during the ramp: the ergometer's step size."""
    measured = _measured_power(visit)
    end = visit.timeline.exercise_end_s
    if visit.ref.protocol != "Ramp" or measured is None or end is None:
        return math.nan
    in_ramp = measured[(measured["t"] >= visit.timeline.warmup_end_s) & (measured["t"] < end)]
    increments = in_ramp["Power"].diff()
    increments = increments[increments > 0].round(1)
    return float(increments.mode().iloc[0]) if len(increments) else math.nan


def visit_row(visit: VisitData, signals: list[dict]) -> dict:
    ref, annotation, timeline = visit.ref, visit.annotation, visit.timeline
    end = timeline.exercise_end_s
    return {
        "participant_id": ref.participant_id, "cohort": ref.cohort, "protocol": ref.protocol,
        "visit_id": ref.visit_id, "role": ref.role,
        "source_participant": ref.source_participant, "source_visit": ref.source_visit,
        **_header(visit),
        "annotation_present": annotation.present, "ve_lag_s": annotation.tymewear_lag_s,
        "hr_source": annotation.hr_source, "nirs_repetition": annotation.nirs_repetition,
        "annotation_notes": annotation.notes,
        "expected_exclusions_handoff": "; ".join(expected_exclusions(ref)),
        "annotation_exclusions": "; ".join(annotation.predicted_exclusions()),
        "rest_end_s": timeline.rest_end_s, "warmup_end_s": timeline.warmup_end_s, "exercise_end_s": end,
        "exercise_duration_s": end - timeline.warmup_end_s if end is not None else None,
        "cpet_warmup_start_s": visit.cpet_phase_starts["warmup"],
        "cpet_exercise_start_s": visit.cpet_phase_starts["exercise"],
        "ramp_warmup_power_w": _ramp_warmup_power(visit),
        "ramp_rate_annotated_w_per_min": annotation.ramp_rate_w_per_min,
        "ramp_rate_fit_w_per_min": _ramp_rate_fit(visit),
        "ramp_step_w": _ramp_step_w(visit),
        **_fitdays_fields(visit),
        "flags": "; ".join(visit_flags(visit, signals)),
    }


def _sensor_flags(signals: list[dict]) -> list[str]:
    flags, seen = [], set()
    for row in signals:
        sensor = row["sensor"]
        if sensor in seen:
            continue
        seen.add(sensor)
        if row["selection_problem"]:
            flags.append(f"{sensor}: {row['selection_problem']}")
        if row["error"]:
            flags.append(f"{sensor}: load error")
    return flags


def hr_is_plausible(row: dict) -> bool:
    """False only when a measured check fails; a missing check (no HR, no VO2) is a coverage issue, not this one."""
    span, r = row.get("hr_span_bpm", math.nan), row.get("hr_vo2_r", math.nan)
    return not (span < HR_MIN_SPAN_BPM or r < HR_MIN_VO2_R)


def _signal_flags(row: dict) -> list[str]:
    signal, flags = row["signal"], []
    coverage = row.get("exercise_coverage", math.nan)
    if coverage < COVERAGE_FLAG:
        flags.append(f"{signal}: {coverage:.0%} of exercise, max gap {row['max_gap_s']:.0f} s")
    if signal == "cpet_vo2" and row.get("mean_breath_spacing_s", 0) > TEN_SECOND_SPACING_FLAG_S:
        flags.append(f"cpet: mean breath spacing {row['mean_breath_spacing_s']:.1f} s (10-s export?)")
    if signal == "tymewear_hr" and row.get("rest_coverage", 1) < REST_HR_COVERAGE_FLAG:
        flags.append(f"tymewear_hr: {row['rest_coverage']:.0%} of rest")
    if not hr_is_plausible(row):
        flags.append(f"{signal}: not representative (span {row['hr_span_bpm']:.0f} bpm, r vs VO2 {row['hr_vo2_r']:.2f})")
    if signal == "nirs_fit_factor" and row.get("ff_ge99_exercise", 1) < FIT_FACTOR_SHARE_FLAG:
        flags.append(f"nirs: fit factor >= 99 in {row['ff_ge99_exercise']:.0%} of exercise")
    if str(row.get("sync", "")).startswith("unsynced"):
        flags.append(f"{row['sensor']}: {row['sync']}")
    return flags


def _is_expected_absence(visit: VisitData, row: dict) -> bool:
    """CPET HR is only used where annotated as the HR source (HR_source='VO2')."""
    return row["signal"] == "cpet_hr" and visit.annotation.hr_source != CPET_HR_SOURCE


def visit_flags(visit: VisitData, signals: list[dict]) -> list[str]:
    relevant = [row for row in signals if not _is_expected_absence(visit, row)]
    flags = _sensor_flags(relevant)
    for row in relevant:
        flags.extend(f for f in _signal_flags(row) if f not in flags)
    if visit.timeline.exercise_end_s is None:
        flags.append("exercise end unknown")
    if not visit.annotation.present:
        flags.append("no annotation")
    flags.extend(f"annotated: {reason}" for reason in visit.annotation.predicted_exclusions())
    return flags
