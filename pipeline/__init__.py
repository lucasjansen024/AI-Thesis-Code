"""Aim 1 harmonised pipeline (PyTorch): raw -> processed cache -> shared mask -> LOSO folds -> TCN -> outputs.

Loads raw data only through `inventory.catalog`/`inventory.sensors`; legacy `src2_training_ready/` stays untouched.
Entry points live in the repo-root `main.py`: `run(...)` (pipeline.train) and `evaluate(...)` (pipeline.evaluate).
"""
