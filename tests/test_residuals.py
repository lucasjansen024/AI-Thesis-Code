"""Residual analyses (METHODOLOGY §11.6): phase events and windows on the real ergometer timelines (figure-draft
counts per visit), sign runs and the runs test, phase shares, the input checks, and one end-to-end run on a
synthetic run folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.cache import from_visits, save
from pipeline.evaluate import SEEDAVG_COLUMNS, SEEDAVG_PREDICTIONS_NAME, SUMMARY_NAME
from pipeline.prepare.grid import EXERCISE, NO_UNIT, REST, UNITS, WARMUP
from pipeline.prepare.settings import PrepSettings
from pipeline.prepare.visit import ProcessedVisit
from pipeline.residuals import (DECREASE, EVENTS_NAME, INCREASE, OTHER, PHASES_NAME, PRBS_DOWN, PRBS_UP, PRIMARY,
                                RAMP_END, RAMP_GET, RUNS_TEST_NAME, SUSTAINED_ABOVE, SUSTAINED_BELOW, TRACES_NAME,
                                WHOLE_RUN, NonContiguousTargetsError, WorkRateMismatchError, all_events,
                                check_contiguous, check_work_rate, label_predictions, label_seconds, phase_table,
                                runs_test, visit_runs, write_residuals)
from pipeline.train import MANIFEST_NAME

REST_END_S, SUBMAX_WARMUP_END_S, RAMP_WARMUP_END_S = 120, 330, 360
LUCAS_END_S, GIES_END_S, RAMP_END_S = 2190, 2250, 1000
L01_GET_W, G01_GET_W = 173.75, 179.0
G01_RAMP_GET_S = 720  # 360 + ceil((179 - 30) / 5) * 12 at 25 W/min (5 W / 12 s)
GIES_LOW_W, GIES_HIGH_W = 25.0, 161.0
# Ergometer programs (start s: load W) from the figure-drafts artifact; Lucas = L01, Gies Easy = G01.
STEP = {0: 0, 120: 139, 690: 228, 1050: 139, 1410: 282, 1590: 25, 1890: 104}
INTERVAL = {0: 0, 120: 139, 330: 25, **{450 + 180 * k: 301 for k in range(10)},
            **{510 + 180 * k: 25 for k in range(10)}}
LUCAS_PRBS_HIGH_S = (450, 510, 570, 690, 840, 1050, 1140, 1230, 1380, 1440, 1500, 1620, 1770, 1980, 2070, 2160)
LUCAS_PRBS_LOW_S = (330, 480, 540, 660, 750, 990, 1110, 1170, 1260, 1410, 1470, 1590, 1680, 1920, 2040, 2100)
LUCAS_PRBS = {0: 0, 120: 139, **{t: 246 for t in LUCAS_PRBS_HIGH_S}, **{t: 25 for t in LUCAS_PRBS_LOW_S}}
GIES_HIGH_S = (150, 210, 360, 420, 540, 630, 750, 840, 930, 1020, 1200, 1260, 1320, 1380, 1500, 1590, 1710, 1800,
               1890, 1980, 2160, 2220)
GIES_LOW_S = (120, 180, 270, 390, 480, 600, 720, 810, 870, 960, 1110, 1230, 1290, 1350, 1440, 1560, 1680, 1770,
              1830, 1920, 2070, 2190)
GIES_EASY = {0: 0, **{t: GIES_HIGH_W for t in GIES_HIGH_S}, **{t: GIES_LOW_W for t in GIES_LOW_S}}
VISITS = (("L01", "Lucas", "Step"), ("L01", "Lucas", "Interval"), ("L01", "Lucas", "PRBS"),
          ("G01", "Gies", "Easy"), ("G01", "Gies", "Ramp"))
PRIMARY_COUNTS = {PRBS_UP: 16 + 20, PRBS_DOWN: 16 + 19, INCREASE: 3 + 10, DECREASE: 2 + 11, SUSTAINED_BELOW: 4,
                  SUSTAINED_ABOVE: 1, RAMP_GET: 1, RAMP_END: 1}
CONFIGS = ("WR", "WR+VENT")
RUN_ID = "synthetic_residuals"


def _step_function(program: dict[int, float], end_s: int) -> np.ndarray:
    t = np.arange(end_s)
    starts = np.array(sorted(program))
    return np.array([program[s] for s in starts], dtype=np.float64)[np.searchsorted(starts, t, side="right") - 1]


def _ramp(end_s: int) -> np.ndarray:
    t = np.arange(end_s)
    return np.where(t < REST_END_S, 0.0, np.where(t < RAMP_WARMUP_END_S, 30.0, 30.0 + 5 * ((t - 360) // 12)))


def _work_rate(protocol: str) -> np.ndarray:
    programs = {"Step": (STEP, LUCAS_END_S), "Interval": (INTERVAL, LUCAS_END_S), "PRBS": (LUCAS_PRBS, LUCAS_END_S),
                "Easy": (GIES_EASY, GIES_END_S)}
    return _ramp(RAMP_END_S) if protocol == "Ramp" else _step_function(*programs[protocol])


def _phase_codes(protocol: str, n: int) -> np.ndarray:
    t = np.arange(n)
    warmup_end = RAMP_WARMUP_END_S if protocol == "Ramp" else SUBMAX_WARMUP_END_S
    return np.where(t < REST_END_S, REST, np.where(t < warmup_end, WARMUP, EXERCISE))


def _work_rate_frame() -> pd.DataFrame:
    frames = []
    for pid, _, protocol in VISITS:
        wr = _work_rate(protocol)
        frames.append(pd.DataFrame({"visit_id": f"{pid}_{protocol}", "t_s": np.arange(len(wr)),
                                    "phase_code": _phase_codes(protocol, len(wr)), "wr": wr}))
    return pd.concat(frames, ignore_index=True)


def _visits() -> pd.DataFrame:
    return pd.DataFrame([{"participant_id": p, "cohort": c, "protocol": pr, "visit_id": f"{p}_{pr}"}
                         for p, c, pr in VISITS])


def _get_power() -> pd.DataFrame:
    return pd.DataFrame({"participant_id": ["G01", "L01"], "cohort": ["Gies", "Lucas"], "get_w": [G01_GET_W, L01_GET_W],
                         "ramp_get_time_s": [G01_RAMP_GET_S, 560], "ramp_included": [True, True]}
                        ).set_index("participant_id")


def _events(get_power: pd.DataFrame | None = None) -> pd.DataFrame:
    return all_events(_work_rate_frame(), _visits(), _get_power() if get_power is None else get_power)


def _label_at(labels: pd.DataFrame, visit_id: str, t_s: int, labelling: str = PRIMARY) -> tuple[int, int]:
    row = labels[(labels["labelling"] == labelling) & (labels["visit_id"] == visit_id) & (labels["t_s"] == t_s)]
    return (OTHER, 0) if row.empty else (int(row["phase"].iloc[0]), int(row["offset_s"].iloc[0]))


def test_event_counts_match_the_figure_draft_per_visit() -> None:
    events = _events()
    for labelling in (PRIMARY, WHOLE_RUN):
        counts = events[events["labelling"] == labelling].groupby("phase").size().to_dict()
        assert counts == PRIMARY_COUNTS


def test_step_windows_and_the_sustained_clock_across_the_warm_up() -> None:
    labels = label_seconds(_events())
    expected = {330: (SUSTAINED_BELOW, 210), 689: (SUSTAINED_BELOW, 569), 690: (INCREASE, 0), 869: (INCREASE, 179),
                870: (SUSTAINED_ABOVE, 180), 1049: (SUSTAINED_ABOVE, 359), 1229: (DECREASE, 179),
                1230: (SUSTAINED_BELOW, 180), 1589: (INCREASE, 179), 1770: (SUSTAINED_BELOW, 180),
                2189: (SUSTAINED_BELOW, 299)}
    assert {t: _label_at(labels, "L01_Step", t) for t in expected} == expected


def test_interval_windows_are_capped_at_the_next_change_and_the_onset_is_a_decrease() -> None:
    labels = label_seconds(_events())
    expected = {330: (DECREASE, 0), 449: (DECREASE, 119), 450: (INCREASE, 0), 509: (INCREASE, 59),
                510: (DECREASE, 0), 629: (DECREASE, 119), 2189: (DECREASE, 59)}
    assert {t: _label_at(labels, "L01_Interval", t) for t in expected} == expected


def test_prbs_primary_and_whole_run_windows() -> None:
    labels = label_seconds(_events())
    assert _label_at(labels, "L01_PRBS", 330) == (PRBS_DOWN, 0)
    assert _label_at(labels, "L01_PRBS", 359) == (PRBS_DOWN, 29)
    assert _label_at(labels, "L01_PRBS", 360) == (OTHER, 0)
    assert _label_at(labels, "L01_PRBS", 449, WHOLE_RUN) == (PRBS_DOWN, 119)
    assert _label_at(labels, "G01_Easy", 345, WHOLE_RUN) == (OTHER, 0)  # before the first exercise switch
    assert _label_at(labels, "G01_Easy", 360, WHOLE_RUN) == (PRBS_UP, 0)
    assert _label_at(labels, "G01_Easy", 2249, WHOLE_RUN) == (PRBS_UP, 29)


def test_ramp_windows_and_their_checks() -> None:
    labels = label_seconds(_events())
    assert _label_at(labels, "G01_Ramp", G01_RAMP_GET_S - 120) == (RAMP_GET, -120)
    assert _label_at(labels, "G01_Ramp", G01_RAMP_GET_S + 119) == (RAMP_GET, 119)
    assert _label_at(labels, "G01_Ramp", RAMP_END_S - 121) == (OTHER, 0)
    assert _label_at(labels, "G01_Ramp", RAMP_END_S - 1) == (RAMP_END, -1)
    too_late = _get_power().assign(ramp_get_time_s=[RAMP_END_S - 200, 560])
    with pytest.raises(ValueError, match="overlaps the ramp-end window"):
        _events(too_late)
    with pytest.raises(ValueError, match="not included"):
        _events(_get_power().assign(ramp_included=[False, True]))


def _predictions(residuals: dict[str, list[float]]) -> pd.DataFrame:
    rows = [{"config_id": "WR", "participant_id": v[:3], "cohort": "Lucas", "protocol": "Step", "visit_id": v,
             "t_s": t, "residual": r} for v, values in residuals.items() for t, r in enumerate(values)]
    return pd.DataFrame(rows)


def test_visit_runs_drop_zeros_and_measure_long_runs() -> None:
    table = visit_runs(_predictions({"L01_Step": [1, 2, 3, -1, -2, 0, -1, 4], "L02_Step": [1.0] * 70 + [-1.0] * 10}))
    first, second = table.iloc[0], table.iloc[1]
    assert (first["n_s"], first["n_zero"], first["runs"], first["n_pos"]) == (7, 1, 3, 4)
    assert first["mean_run_length_s"] == pytest.approx(7 / 3)
    assert (second["runs"], second["share_long"]) == (2, pytest.approx(70 / 80))


def test_runs_test_matches_the_wald_wolfowitz_reference() -> None:
    visit_table = pd.DataFrame({"protocol": ["Step"] * 2, "config_id": ["WR"] * 2, "participant_id": ["L01", "L02"],
                                "visit_id": ["L01_Step", "L02_Step"], "n_s": [10, 10], "n_pos": [5, 5],
                                "n_neg": [5, 5], "n_zero": [0, 0], "runs": [3, 3]})
    row = runs_test(visit_table).iloc[0]
    assert row["expected_runs"] == pytest.approx(11.0)
    assert row["z"] == pytest.approx((6 - 11) / np.sqrt(36000 / 7600))
    assert row["mean_run_length_s"] == pytest.approx(20 / 6)


def test_phase_shares_sum_to_one_and_the_ratio_is_the_mse_ratio() -> None:
    labelled = pd.DataFrame({"labelling": PRIMARY, "config_id": "WR", "cohort": ["Lucas"] * 4 + ["Gies"] * 2,
                             "participant_id": ["L01", "L01", "L02", "L02", "G01", "G01"],
                             "phase": [INCREASE, OTHER, INCREASE, OTHER, INCREASE, OTHER],
                             "event": [0, np.nan, 1, np.nan, 2, np.nan], "residual": [4.0, 1.0, 2.0, 1.0, 6.0, -1.0]})
    table = phase_table(labelled)
    pooled = table[table["group"] == "pooled"].set_index("phase")
    assert pooled["error_share"].sum() == pytest.approx(1.0)
    assert pooled.loc[INCREASE, "error_ratio"] == pytest.approx((56 / 3) / (59 / 6))
    assert pooled.loc[INCREASE, "bias_mean"] == pytest.approx(4.0)
    assert pooled.loc[INCREASE, "n_events"] == 3
    assert set(table["group"]) == {"pooled", "Gies", "Lucas"}


def test_input_checks_raise() -> None:
    gap = _predictions({"L01_Step": [1.0, 2.0, 3.0]})
    with pytest.raises(NonContiguousTargetsError):
        check_contiguous(gap[gap["t_s"] != 1])
    targets = pd.DataFrame({"visit_id": "L01_Step", "t_s": [700], "work_rate": [139.0]})
    with pytest.raises(WorkRateMismatchError):
        check_work_rate(targets, _work_rate_frame())


def _processed_visit(work_rate: pd.DataFrame, pid: str, cohort: str, protocol: str) -> ProcessedVisit:
    visit = work_rate[work_rate["visit_id"] == f"{pid}_{protocol}"]
    ids = {"participant_id": pid, "cohort": cohort, "protocol": protocol, "visit_id": f"{pid}_{protocol}",
           "source_visit": protocol}
    columns = {"t": visit["t_s"].to_numpy(), "phase": visit["phase_code"].to_numpy(),
               "unit": np.where(visit["phase_code"] == EXERCISE, UNITS.index("visit"), NO_UNIT),
               "WR": visit["wr"].to_numpy()}
    return ProcessedVisit(ids, columns)


def _write_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Cache, fold manifests, seed-averaged predictions (exercise seconds, the last 5 of the ramp lost) and GET CSV."""
    work_rate, key = _work_rate_frame(), "0" * 64
    cache_root, runs_root = tmp_path / "cache", tmp_path / "runs"
    processed = [_processed_visit(work_rate, *visit) for visit in VISITS]
    save(from_visits(processed, key), cache_root / key, PrepSettings())
    targets = work_rate[(work_rate["phase_code"] == EXERCISE)
                        & ~((work_rate["visit_id"] == "G01_Ramp") & (work_rate["t_s"] >= RAMP_END_S - 5))]
    targets = targets.merge(_visits(), on="visit_id")
    rng = np.random.default_rng(0)
    predictions = pd.concat([targets.assign(run_id=RUN_ID, config_id=c, fold=targets["participant_id"],
                                            work_rate=targets["wr"], vo2_true=1000 + 5 * targets["wr"],
                                            vo2_pred=1000 + 5 * targets["wr"] + rng.normal(0, 50, len(targets)))
                             for c in CONFIGS], ignore_index=True)[SEEDAVG_COLUMNS]
    (runs_root / RUN_ID / SUMMARY_NAME).mkdir(parents=True)
    predictions.to_parquet(runs_root / RUN_ID / SUMMARY_NAME / SEEDAVG_PREDICTIONS_NAME, index=False)
    for fold in ("G01", "L01"):
        (runs_root / RUN_ID / fold).mkdir()
        (runs_root / RUN_ID / fold / MANIFEST_NAME).write_text(json.dumps({"cache_key": key}), encoding="utf-8")
    get_power_path = tmp_path / "get_power.csv"
    _get_power().reset_index().to_csv(get_power_path, index=False)
    return runs_root, cache_root, get_power_path


def test_write_residuals_end_to_end(tmp_path: Path) -> None:
    runs_root, cache_root, get_power_path = _write_run(tmp_path)
    folder = write_residuals(RUN_ID, CONFIGS, runs_root, cache_root, get_power_path, tmp_path / "out")
    events = pd.read_csv(folder / EVENTS_NAME)
    primary = events[events["labelling"] == PRIMARY].groupby("phase")["n_events"].sum().to_dict()
    assert primary == PRIMARY_COUNTS
    assert (events["n_events_with_targets"] == events["n_events"]).all()
    assert len(pd.read_csv(folder / RUNS_TEST_NAME)) == len(VISITS) * len(CONFIGS)
    phases = pd.read_csv(folder / PHASES_NAME)
    shares = phases.groupby(["labelling", "group", "config_id"])["error_share"].sum()
    assert np.allclose(shares, 1.0)
    traces = pd.read_csv(folder / TRACES_NAME)
    ramp_end = traces[(traces["phase"] == RAMP_END) & (traces["labelling"] == PRIMARY)]
    assert ramp_end["offset_s"].max() == -6  # the last 5 target seconds are lost; task failure stays the anchor


def test_label_predictions_marks_unlabelled_seconds_other() -> None:
    labels = label_seconds(_events())
    predictions = pd.DataFrame({"visit_id": "G01_Ramp", "t_s": [400, G01_RAMP_GET_S], "residual": [1.0, 2.0]})
    labelled = label_predictions(predictions, labels)
    primary = labelled[labelled["labelling"] == PRIMARY].set_index("t_s")
    assert primary.loc[400, "phase"] == OTHER
    assert primary.loc[G01_RAMP_GET_S, "phase"] == RAMP_GET
