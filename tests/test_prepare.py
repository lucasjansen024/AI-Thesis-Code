"""Grid, phase/unit labels and the rebuilt PRBS work rate (synthetic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory.catalog import NOMINAL_TIMELINES, Timeline
from pipeline.prepare.grid import EXERCISE, NO_UNIT, REST, UNITS, WARMUP, build_grid
from pipeline.prepare.wr import LUCAS_PRBS_LOW_W, rebuild_prbs_wr

RAMP_END_S = 1000
REST_END_S, WARMUP_END_S, END_S = 120, 330, 2190
HIGH_W, WARMUP_W, TEMPLATE_HIGH_W, TEMPLATE_WARMUP_W = 180.0, 90.0, 202.0, 118.0


def test_gies_prbs_grid() -> None:
    grid = build_grid(NOMINAL_TIMELINES[("Gies", "Medium")])
    assert len(grid.t) == 2250 and grid.t[0] == 0
    assert (grid.phase[:120] == REST).all() and (grid.phase[120:330] == WARMUP).all()
    assert (grid.phase[330:] == EXERCISE).all()
    assert (grid.unit[:330] == NO_UNIT).all()
    assert (grid.unit[330:1290] == UNITS.index("rep1")).all()
    assert (grid.unit[1290:] == UNITS.index("rep2")).all()


def test_ramp_grid_ends_at_the_ramp_end() -> None:
    timeline = Timeline(120, 360, RAMP_END_S)
    grid = build_grid(timeline)
    assert len(grid.t) == RAMP_END_S
    assert (grid.unit[360:] == UNITS.index("visit")).all()


def test_grid_needs_an_exercise_end() -> None:
    with pytest.raises(ValueError, match="exercise end"):
        build_grid(NOMINAL_TIMELINES[("Gies", "Ramp")])


def _template() -> pd.DataFrame:
    t = np.arange(END_S)
    high = ((t - WARMUP_END_S) // 30) % 2 == 1
    w = np.where(t < REST_END_S, 0.0, np.where(t < WARMUP_END_S, TEMPLATE_WARMUP_W,
                                                 np.where(high, TEMPLATE_HIGH_W, LUCAS_PRBS_LOW_W)))
    return pd.DataFrame({"t": t, "W": w})


def _measured(template: pd.DataFrame) -> pd.DataFrame:
    """Breaths every 3 s with the visit's own levels plus noise."""
    t = np.arange(0, END_S, 3.0)
    ideal = template["W"].to_numpy()[t.astype(int)]
    rng = np.random.default_rng(0)
    power = np.select([ideal == TEMPLATE_HIGH_W, ideal == TEMPLATE_WARMUP_W], [HIGH_W, WARMUP_W], ideal)
    return pd.DataFrame({"t": t, "Power": power + rng.normal(0, 1, len(t))})


def test_rebuilt_prbs_keeps_the_timing_and_takes_levels_from_measured_power() -> None:
    template = _template()
    rebuilt = rebuild_prbs_wr(template, _measured(template), REST_END_S, WARMUP_END_S)
    template_w, w = template["W"].to_numpy(), rebuilt["W"].to_numpy()
    assert (w[:REST_END_S] == 0).all()
    np.testing.assert_allclose(w[REST_END_S:WARMUP_END_S], WARMUP_W, atol=1.0)
    assert (w[template_w == LUCAS_PRBS_LOW_W] == LUCAS_PRBS_LOW_W).all()
    np.testing.assert_allclose(w[template_w == TEMPLATE_HIGH_W], HIGH_W, atol=1.0)


def test_rebuilt_prbs_rejects_an_unexpected_template() -> None:
    template = _template().assign(W=lambda df: df["W"].replace(LUCAS_PRBS_LOW_W, 30.0))
    with pytest.raises(ValueError, match="low level"):
        rebuild_prbs_wr(template, _measured(_template()), REST_END_S, WARMUP_END_S)
