"""Entry point. Pick what runs by commenting/uncommenting calls in the main block."""
from __future__ import annotations

import logging

from figures.study_design import write_study_design
from figures.table1 import write_table1
from inventory.__main__ import main as inventory_main
from inventory.build_inclusion import main as inclusion_main
from pipeline.aim2 import run_left_out, run_transfer
from pipeline.aim2_outcome import OUTCOME_ROOT, write_outcome
from pipeline.evaluate import evaluate
from pipeline.pilot import decide, run_pilot
from pipeline.residuals import write_residuals
from pipeline.runs import (A1_GIES, A1_GIES_SHORT, A1_LUCAS, A1_LUCAS_SHORT, A1_POOLED, A1_POOLED_SHORT,
                           A2_SMOKE_ARMS, A2_SMOKE_TRANSFER, A2A_ARMS, A2B_ARMS, A2C_TRANSFERS, PILOT, PILOT_SMOKE,
                           SMOKE)
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
    logging.basicConfig(level=logging.INFO, format=PIPELINE_LOG_FORMAT)
    # run_inventory(ROUGH_INVENTORY_VISITS)
    # run_inventory()
    # build_inclusion()

    # Aim 1 pipeline → outputs/runs/<run_id>/. Pass folds=["G07", "L04"] to train a subset of folds.
    # Smoke run first (2 configurations x folds G07, L04 x seed 0, 3 epochs):
    # run(SMOKE)
    # evaluate(SMOKE)
    # Training-setting sensitivity first (lr 5e-4 x 1 epoch, ~1.5 h for all three):
    # run(A1_POOLED_SHORT)
    # evaluate(A1_POOLED_SHORT)
    # run(A1_GIES_SHORT)
    # evaluate(A1_GIES_SHORT)
    # run(A1_LUCAS_SHORT)
    # evaluate(A1_LUCAS_SHORT)
    # Primary (lr 5e-5 x 30 epochs, ≤ ~1.7 days for all three):
    # run(A1_POOLED)
    # evaluate(A1_POOLED)
    # run(A1_GIES)
    # evaluate(A1_GIES)
    # run(A1_LUCAS)
    # evaluate(A1_LUCAS)

    # Aim 2 (METHODOLOGY Appendix B) → outputs/runs/<run_id>/; paired outcome → outputs/aim2/. Train Aim 2 specs
    # only with run_left_out / run_transfer, never run(spec.run). run_left_out(arm, folds=["G01"]) trains a subset;
    # evaluate(arm.run) needs every fold of the arm.
    # Smoke first (2a Hard seen + unseen on G01, 2c Gies -> Lucas; 3 configurations x seed 0, 3 epochs):
    # for arm in A2_SMOKE_ARMS:
    #     run_left_out(arm)
    #     evaluate(arm.run)
    # run_transfer(A2_SMOKE_TRANSFER)
    # evaluate(A2_SMOKE_TRANSFER.run)
    # write_outcome(A2_SMOKE_ARMS, (A2_SMOKE_TRANSFER,), out_dir=OUTCOME_ROOT / "smoke")
    # 2c, both directions (one training each, ~1 h in total); needs A1_gies and A1_lucas evaluated:
    # for transfer in A2C_TRANSFERS:
    #     run_transfer(transfer)
    #     evaluate(transfer.run)
    # write_outcome(transfers=A2C_TRANSFERS)
    # 2a (76 fold jobs) and 2b (104 fold jobs), after the map:
    for arm in (*A2A_ARMS, *A2B_ARMS):
        run_left_out(arm)
        evaluate(arm.run)
    write_outcome(A2A_ARMS + A2B_ARMS, A2C_TRANSFERS)

    # Residual analyses (METHODOLOGY §11.6) → outputs/residuals/<run_id>/; needs the run evaluated and its cache:
    # write_residuals()  # A1_pooled x the three carry-forward configurations

    # Display items (paper) → outputs/paper/. Table 1 reads the inventory CSV and the ramp MRT sheets; the study
    # design figure also needs A1_pooled's cache:
    # write_table1()
    # write_study_design()

    # Training-settings pilot, LOSO (54 splits x 3 trajectories, ~27 h) → outputs/pilot/<split>/<trajectory>/.
    # Stopped 9 Oct 2026 after pooled G01, G03-G09 (dev): replaced by the sensitivity runs above.
    # Pilot smoke first (pooled_G01 x lr 5e-4 x 3 epochs x WR, WR+HR+VENT+TSI+ANTHRO x seed 0) → outputs/pilot_smoke/:
    # run_pilot(PILOT_SMOKE)
    # run_pilot(PILOT)
    # decide(PILOT)  # → outputs/pilot/decision.json, decision.md; copy the chosen lr x epochs into runs.A1_TRAINING
