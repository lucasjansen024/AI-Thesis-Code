"""Configuration list, A1 run registry, SMOKE, and the one-procedure / identical-data checks (§5)."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from pipeline.cache import ProcessedCache
from pipeline.domains import INPUTS, STATIC_DOMAINS, Domain, generate_configurations, parse_configuration
from pipeline.evaluate import TrainingSettingsMismatchError, evaluate
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import shared_mask
from pipeline.runs import (A1_CONFIGURATIONS, A1_GIES, A1_LUCAS, A1_POOLED, A1_RUNS, A1_SHORT_RUNS,
                           A1_SHORT_TRAINING, A1_TRAINING, SMOKE, Run, TrainingSettings)
from pipeline.train import MANIFEST_NAME, PREDICTIONS_NAME, RunData, mask_groups, train_folds
from tests.synthetic import PARTICIPANTS, SYNTHETIC_RUN, make_synthetic

N_EXCLUSIVE, N_COMBINABLE = 46, 62
FOLD_COUNTS = {A1_POOLED.run_id: 27, A1_GIES.run_id: 14, A1_LUCAS.run_id: 13}
IDENTICAL_DATA_FOLD = "G07"
MULTI_RUN = replace(SYNTHETIC_RUN, run_id="synthetic_multi", configurations=("WR", "WR+HR+VENT+TSI+ANTHRO"))
CPU = torch.device("cpu")


# --- configuration list -----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("nirs_exclusive", "expected"), [(True, N_EXCLUSIVE), (False, N_COMBINABLE)])
def test_configuration_count_and_uniqueness(nirs_exclusive: bool, expected: int) -> None:
    ids = generate_configurations(nirs_exclusive)
    assert len(ids) == expected
    assert len(set(ids)) == expected


def test_default_configurations_never_combine_tsi_and_hbdiff() -> None:
    assert all(not {"TSI", "HBDIFF"} <= set(c.split("+")) for c in generate_configurations())
    assert any({"TSI", "HBDIFF"} <= set(c.split("+")) for c in generate_configurations(nirs_exclusive=False))


@pytest.mark.parametrize("nirs_exclusive", [True, False])
def test_configurations_parse_and_never_hold_anthro_alone(nirs_exclusive: bool) -> None:
    ids = generate_configurations(nirs_exclusive)
    assert "ANTHRO" not in ids
    for config_id in ids:
        config = parse_configuration(config_id)  # raises on unknown, repeated, static-only or non-canonical
        assert config.config_id == config_id
        assert any(INPUTS[t].domain not in STATIC_DOMAINS for t in config.tokens)


@pytest.mark.parametrize("nirs_exclusive", [True, False])
def test_each_domain_brings_all_of_its_features(nirs_exclusive: bool) -> None:
    for config_id in generate_configurations(nirs_exclusive):
        columns = set(parse_configuration(config_id).columns)
        for spec in INPUTS.values():
            used = columns & set(spec.columns)
            assert used in (set(), set(spec.columns)), (config_id, spec.columns)


def test_every_configuration_comes_with_and_without_anthro() -> None:
    ids = set(generate_configurations())
    with_anthro = {c for c in ids if c.endswith("+ANTHRO")}
    assert {c.removesuffix("+ANTHRO") for c in with_anthro} == ids - with_anthro
    assert len(with_anthro) == N_EXCLUSIVE // 2


def test_configurations_are_ordered_by_size_then_canonical_order() -> None:
    ids = generate_configurations()
    canonical = list(INPUTS)
    keys = [(len(c.split("+")), [canonical.index(t) for t in c.split("+")]) for c in ids]
    assert keys == sorted(keys)
    assert ids[:4] == ("WR", "HR", "VENT", "TSI")
    assert ids[-2:] == ("WR+HR+VENT+TSI+ANTHRO", "WR+HR+VENT+HBDIFF+ANTHRO")
    assert {INPUTS[t].domain for t in ("TSI", "HBDIFF")} == {Domain.NIRS}


# --- run registry -----------------------------------------------------------------------------------------------

def test_a1_runs_train_every_configuration_with_five_seeds_and_the_shared_mask() -> None:
    for run in A1_RUNS:
        assert run.configurations == A1_CONFIGURATIONS == generate_configurations()
        assert run.seeds == (0, 1, 2, 3, 4)
        assert run.mask == "shared" and run.fold_ids is None and run.nirs_exclusive
    assert [r.cohorts for r in A1_RUNS] == [("Gies", "Lucas"), ("Gies",), ("Lucas",)]


def test_resolved_training_settings_are_identical_across_a1_runs() -> None:
    """The manifest stores `asdict(run)`, so equal resolved settings here mean equal settings in every manifest."""
    resolved = [asdict(r)["training"] for r in A1_RUNS]
    assert resolved == [asdict(A1_TRAINING)] * len(A1_RUNS)
    assert asdict(A1_TRAINING) == {"learning_rate": 5e-5, "epochs": 30, "batch_size": 32}


def test_short_runs_differ_from_a1_runs_only_in_run_id_and_training() -> None:
    assert asdict(A1_SHORT_TRAINING) == {"learning_rate": 5e-4, "epochs": 1, "batch_size": 32}
    assert len({r.run_id for r in (*A1_RUNS, *A1_SHORT_RUNS)}) == 2 * len(A1_RUNS)
    for primary, short in zip(A1_RUNS, A1_SHORT_RUNS):
        assert short == replace(primary, run_id=short.run_id, training=A1_SHORT_TRAINING)


def test_smoke_run_definition() -> None:
    assert SMOKE.cohorts == ("Gies", "Lucas") and SMOKE.mask == "shared" and SMOKE.seeds == (0,)
    assert SMOKE.configurations == ("WR", "WR+HR+VENT+TSI+ANTHRO")
    assert set(SMOKE.configurations) <= set(A1_CONFIGURATIONS)
    assert SMOKE.fold_ids == ("G07", "L04")
    assert SMOKE.training == replace(A1_TRAINING, epochs=3)


@pytest.mark.parametrize("configurations", [("WR+TSI+HBDIFF",), ("ANTHRO",), ("WR", "WR"), ("HR+WR",), ("WR+FOO",)])
def test_run_rejects_invalid_configurations(configurations: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        Run("bad", ("Gies",), configurations)


def test_nirs_flag_off_allows_tsi_with_hbdiff() -> None:
    run = Run("combinable", ("Gies",), generate_configurations(nirs_exclusive=False), nirs_exclusive=False)
    assert "WR+TSI+HBDIFF" in run.configurations


# --- one procedure, identical data (synthetic outputs) ----------------------------------------------------------

@pytest.fixture
def multi_trained(tmp_path: Path) -> tuple[RunData, Path]:
    synthetic = make_synthetic()
    data = RunData(synthetic.cache, synthetic.inclusion, "synthetic-sha",
                   shared_mask(synthetic.cache, synthetic.inclusion))
    train_folds(MULTI_RUN, data, output_root=tmp_path, device=CPU)
    return data, tmp_path / MULTI_RUN.run_id


def test_every_configuration_is_evaluated_on_the_same_seconds(multi_trained: tuple[RunData, Path]) -> None:
    _, run_dir = multi_trained
    for pid, _, _ in PARTICIPANTS:
        manifest = json.loads((run_dir / pid / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["mask_groups"] == 1
        predictions = pd.read_parquet(run_dir / pid / PREDICTIONS_NAME)
        seconds = {c: set(zip(f["visit_id"], f["t_s"])) for c, f in predictions.groupby("config_id")}
        assert set(seconds) == set(MULTI_RUN.configurations)
        assert seconds["WR"] == seconds["WR+HR+VENT+TSI+ANTHRO"]


def test_evaluate_rejects_folds_trained_with_other_settings(multi_trained: tuple[RunData, Path]) -> None:
    data, run_dir = multi_trained
    path = run_dir / "G03" / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    altered = {**manifest, "run": {**manifest["run"], "training": {**manifest["run"]["training"],
                                                                   "learning_rate": 5e-4}}}
    path.write_text(json.dumps(altered, indent=2), encoding="utf-8")
    with pytest.raises(TrainingSettingsMismatchError, match="G03"):
        evaluate(MULTI_RUN, output_root=run_dir.parent, data=data)
    assert not (run_dir / "summary.tmp").exists()


def test_evaluate_rejects_a_run_whose_settings_changed_after_training(multi_trained: tuple[RunData, Path]) -> None:
    data, run_dir = multi_trained
    changed = replace(MULTI_RUN, training=TrainingSettings(epochs=MULTI_RUN.training.epochs + 1))
    with pytest.raises(TrainingSettingsMismatchError, match="G01"):
        evaluate(changed, output_root=run_dir.parent, data=data)


# --- real data --------------------------------------------------------------------------------------------------

@pytest.mark.real
@pytest.mark.parametrize("run", A1_RUNS, ids=lambda r: r.run_id)
def test_a1_fold_counts_with_unique_test_ids(real_cache: ProcessedCache, run: Run) -> None:
    test_ids = [f.test_participant for f in run_folds(run, real_cache)]
    assert len(test_ids) == FOLD_COUNTS[run.run_id]
    assert len(set(test_ids)) == len(test_ids)


@pytest.mark.real
def test_smoke_folds_exist(real_cache: ProcessedCache) -> None:
    assert [f.test_participant for f in run_folds(SMOKE, real_cache)] == list(SMOKE.fold_ids)


@pytest.mark.real
def test_shared_mask_gives_identical_seconds_for_every_configuration(real_cache: ProcessedCache,
                                                                     committed_inclusion: pd.DataFrame) -> None:
    """One fold, every A1 configuration built on its own: identical test seconds, and one mask group."""
    mask = shared_mask(real_cache, committed_inclusion)
    configurations = tuple(parse_configuration(c) for c in A1_CONFIGURATIONS)
    groups = mask_groups(A1_POOLED, configurations, mask)
    assert len(groups) == 1 and groups[0].configurations == configurations
    fold = next(f for f in run_folds(A1_POOLED, real_cache) if f.test_participant == IDENTICAL_DATA_FOLD)
    visit_ids, t = real_cache.row_ids("visit_id"), real_cache.column("t").astype(int)
    seconds = {}
    for config in configurations:
        rows = build_fold_data(real_cache, mask, fold, config.columns).test_targets
        seconds[config.config_id] = set(zip(visit_ids[rows], t[rows]))
    reference = seconds[A1_CONFIGURATIONS[0]]
    assert reference
    assert all(s == reference for s in seconds.values())
