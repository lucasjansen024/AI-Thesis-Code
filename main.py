"""Entry point. Pick what runs by commenting/uncommenting calls in the main block."""
from __future__ import annotations

import logging

from inventory.__main__ import main as inventory_main
from inventory.build_inclusion import main as inclusion_main
from pipeline.evaluate import evaluate
from pipeline.pilot import decide, run_pilot
from pipeline.runs import A1_GIES, A1_LUCAS, A1_POOLED, PILOT, PILOT_SMOKE, SMOKE
from pipeline.train import run

ROUGH_INVENTORY_VISITS = ["L03_Ramp", "L12_PRBS", "G11_Hard"]
PIPELINE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def run_inventory(visits: list[str] | None = None, is_plotting: bool = True) -> int:
    """Data inventory of both cohorts → outputs/inventory/. `visits=None` runs every visit."""
    argv = (["--visits", *visits] if visits else []) + ([] if is_plotting else ["--no-plots"])
    return inventory_main(argv)


def build_inclusion() -> int:
    """Inclusion CSV from a full inventory run → inventory/inclusion/inclusion.csv (committed)."""
    return inclusion_main()


if __name__ == "__main__":
    # run_inventory(ROUGH_INVENTORY_VISITS)
    # run_inventory()
    build_inclusion()

    # Aim 1 pipeline → outputs/runs/<run_id>/. Pass folds=["G07", "L04"] to train a subset of folds.
    # logging.basicConfig(level=logging.INFO, format=PIPELINE_LOG_FORMAT)
    # Smoke run first (2 configurations x folds G07, L04 x seed 0, 3 epochs):
    # run(SMOKE)
    # evaluate(SMOKE)
    # run(A1_POOLED)
    # evaluate(A1_POOLED)
    # run(A1_GIES)
    # evaluate(A1_GIES)
    # run(A1_LUCAS)
    # evaluate(A1_LUCAS)

    # Training-settings pilot → outputs/pilot/<split>/<trajectory>/ (validation participants only).
    # Pilot smoke first (pooled_1 x lr 5e-4 x 3 epochs x WR, WR+HR+VENT+TSI+ANTHRO x seed 0) → outputs/pilot_smoke/:
    # run_pilot(PILOT_SMOKE)
    # run_pilot(PILOT)
    # decide(PILOT)  # → outputs/pilot/decision.json, decision.md; copy the chosen lr x epochs into runs.A1_TRAINING
