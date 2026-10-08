"""Packed training (slice 6): packed <-> standalone and last-only <-> full parity, packing-independent init,
channel masks, per-branch checkpoints, seed reproducibility, fold outputs with `models.pt`, and K=230 step time."""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.domains import feature_union, parse_configuration
from pipeline.fit import Branch, configure_determinism, fit_packed, pack_layout, predict, stream_seeds, to_device
from pipeline.folds import build_fold_data, run_folds
from pipeline.inclusion import WINDOW_S, shared_mask
from pipeline.runs import TrainingSettings
from pipeline.TCN import FILTERS, PackedTCN, TCN, build_packed_tcn, build_tcn, extract_branch
from pipeline.train import EPOCHS_NAME, MANIFEST_NAME, MODELS_NAME, PREDICTIONS_NAME, RunData, train_folds
from tests.synthetic import SYNTHETIC_RUN, make_synthetic

DEVICES = [pytest.param("cpu"), pytest.param("xpu", marks=pytest.mark.xpu)]
N_FEATURES = 3
COLUMN_SETS = ((0,), (0, 2), (0, 1, 2))
N_STREAMS = 2
BATCH = 32
OUTPUT_TOLERANCE = 1e-5  # prototype on B580: max |Δoutput| ~1e-6
GRAD_TOLERANCE = 1e-4  # prototype on B580: max |Δgrad| ~4e-5
FIRST_CONV = "blocks.0.convs.0.conv.weight"
PROJECTION = "blocks.0.projection.conv.weight"


def _packed(device: str, column_sets: tuple[tuple[int, ...], ...] = COLUMN_SETS * N_STREAMS,
            n_streams: int = N_STREAMS) -> PackedTCN:
    seeds = tuple(range(len(column_sets)))
    return build_packed_tcn(column_sets, N_FEATURES, n_streams, seeds).to(device)


def _inputs(device: str, n_streams: int = N_STREAMS) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    windows = torch.randn(n_streams, BATCH, WINDOW_S, N_FEATURES, generator=generator)
    truth = torch.randn(n_streams, BATCH, generator=generator)
    return windows.to(device), truth.to(device)


def _packed_output_and_grads(model: PackedTCN, windows: torch.Tensor, truth: torch.Tensor,
                             last_only: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(windows, last_only=last_only)
    loss = torch.mean((out - truth[model.branch_stream].T) ** 2, dim=0).sum()
    names, params = zip(*model.named_parameters())
    return out.detach(), dict(zip(names, torch.autograd.grad(loss, params)))


def _max_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return max((a[n].cpu() - b[n].cpu()).abs().max().item() for n in a)


@pytest.mark.parametrize("device", DEVICES)
def test_packed_matches_standalone_outputs_and_gradients(device: str) -> None:
    configure_determinism(torch.device(device))
    column_sets = COLUMN_SETS * N_STREAMS
    packed = _packed(device).eval()
    windows, truth = _inputs(device)
    out, packed_grads = _packed_output_and_grads(packed, windows, truth, last_only=False)
    per_stream = len(column_sets) // N_STREAMS
    for k, columns in enumerate(column_sets):
        stream = k // per_stream
        model = build_tcn(len(columns), k).to(device).eval()
        single = model(windows[stream][:, :, list(columns)])
        names, params = zip(*model.named_parameters())
        grads = torch.autograd.grad(torch.mean((single - truth[stream]) ** 2), params)
        assert (out[:, k] - single.detach()).abs().max().item() < OUTPUT_TOLERANCE, k
        branch_grads = extract_branch(packed_grads, k, columns, len(column_sets), N_FEATURES)
        assert _max_diff(branch_grads, dict(zip(names, grads))) < GRAD_TOLERANCE, k


@pytest.mark.xpu
def test_single_input_channel_gradients_match_cpu_on_xpu() -> None:
    """B580 regression (9 Oct 2026): Conv1d's backward was wrong for a 1x1 conv with one input channel.

    Covers a one-feature standalone TCN and a one-feature packed job (e.g. a WR-only run) against the CPU.
    """
    generator = torch.Generator().manual_seed(11)
    windows = torch.randn(1, BATCH, WINDOW_S, 1, generator=generator)
    truth = torch.randn(1, BATCH, generator=generator)

    def gradients(device: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        standalone = build_tcn(1, 0).to(device).eval()
        names, params = zip(*standalone.named_parameters())
        out = standalone(windows[0].to(device))
        single = torch.autograd.grad(torch.mean((out - truth[0].to(device)) ** 2), params)
        packed = build_packed_tcn(((0,), (0,)), 1, 1, (0, 1)).to(device).eval()
        _, packed_grads = _packed_output_and_grads(packed, windows.to(device), truth.to(device), last_only=False)
        return {n: g.cpu() for n, g in zip(names, single)}, {n: g.cpu() for n, g in packed_grads.items()}

    xpu_single, xpu_packed = gradients("xpu")
    cpu_single, cpu_packed = gradients("cpu")
    assert _max_diff(xpu_single, cpu_single) < GRAD_TOLERANCE
    assert _max_diff(xpu_packed, cpu_packed) < GRAD_TOLERANCE


@pytest.mark.parametrize("device", DEVICES)
def test_last_only_matches_full_forward(device: str) -> None:
    configure_determinism(torch.device(device))
    packed = _packed(device).eval()
    windows, truth = _inputs(device)
    full_out, full_grads = _packed_output_and_grads(packed, windows, truth, last_only=False)
    last_out, last_grads = _packed_output_and_grads(packed, windows, truth, last_only=True)
    assert (full_out - last_out).abs().max().item() < OUTPUT_TOLERANCE
    assert _max_diff(full_grads, last_grads) < GRAD_TOLERANCE


def test_branch_init_does_not_depend_on_packing() -> None:
    alone = build_packed_tcn(((0, 2),), N_FEATURES, 1, (42,))
    crowded = build_packed_tcn(((1,), (0, 1, 2), (0, 2), (2,)), N_FEATURES, 2, (1, 2, 42, 3))
    expected = build_tcn(2, 42).state_dict()
    from_alone = extract_branch(alone.state_dict(), 0, (0, 2), 1, N_FEATURES)
    from_crowded = extract_branch(crowded.state_dict(), 2, (0, 2), 4, N_FEATURES)
    for name, tensor in expected.items():
        assert torch.equal(from_alone[name], tensor), name
        assert torch.equal(from_crowded[name], tensor), name


def test_unused_feature_weights_start_and_stay_zero() -> None:
    column_sets = ((0,), (1, 2))
    model = _packed("cpu", column_sets, n_streams=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    windows, truth = _inputs("cpu", n_streams=1)
    dropout = torch.Generator().manual_seed(0)
    for _ in range(3):
        loss = torch.mean((model(windows, dropout, last_only=True) - truth[0][:, None]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for name in (FIRST_CONV, PROJECTION):
        weight = model.state_dict()[name].view(len(column_sets), FILTERS, N_FEATURES, -1)
        assert torch.count_nonzero(weight[0][:, [1, 2]]) == 0, name
        assert torch.count_nonzero(weight[1][:, [0]]) == 0, name
        assert torch.count_nonzero(weight[0][:, [0]]) > 0, name
    model.eval()
    perturbed = windows.clone()
    perturbed[..., 1:] += 1e3
    assert torch.equal(model(windows)[:, 0], model(perturbed)[:, 0])


def test_layout_is_seed_major_and_validated() -> None:
    branches = pack_layout((("a", (0,)), ("b", (1, 2))), (3, 5))
    assert [(b.config_id, b.seed) for b in branches] == [("a", 3), ("b", 3), ("a", 5), ("b", 5)]
    assert stream_seeds(branches) == (3, 5)
    interleaved = (branches[0], branches[2], branches[1], branches[3])
    with pytest.raises(ValueError, match="seed-major"):
        stream_seeds(interleaved)
    with pytest.raises(ValueError, match="duplicate"):
        stream_seeds((Branch("a", 0, (0,)), Branch("a", 0, (1,))))


def _tiny_job(device: str, seeds: tuple[int, ...] = (0, 1), epochs: int = 3) -> tuple:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, N_FEATURES)).astype(np.float32)
    target = rng.normal(size=400).astype(np.float32)
    data = to_device(features, target, torch.device(device))
    branches = pack_layout((("a", (0,)), ("b", (1, 2))), seeds)
    settings = TrainingSettings(learning_rate=1e-2, epochs=epochs, batch_size=BATCH)
    return branches, fit_packed(data, np.arange(WINDOW_S, WINDOW_S + 100), branches, settings, dropout_seed=9)


def test_each_branch_keeps_its_own_best_epoch() -> None:
    configure_determinism(torch.device("cpu"))
    branches, fits = _tiny_job("cpu", epochs=5)
    for branch, fit in zip(branches, fits):
        losses = np.asarray(fit.epoch_losses)
        running = [losses[e] < losses[:e].min(initial=np.inf) for e in range(len(losses))]
        assert list(fit.is_best) == running
        assert fit.checkpoint_epoch == int(np.argmin(losses)) + 1
        reference = TCN(len(branch.columns)).state_dict()
        assert {n: t.shape for n, t in fit.state_dict.items()} == {n: t.shape for n, t in reference.items()}
        assert all(t.device.type == "cpu" for t in fit.state_dict.values())


@pytest.mark.parametrize("device", DEVICES)
def test_packed_job_is_reproducible_and_seeds_differ(device: str) -> None:
    configure_determinism(torch.device(device))
    branches, first = _tiny_job(device)
    _, again = _tiny_job(device)
    for a, b in zip(first, again):
        assert a.epoch_losses == b.epoch_losses
        for name in a.state_dict:
            assert torch.equal(a.state_dict[name], b.state_dict[name]), name
    by_branch = {(b.config_id, b.seed): fit.state_dict for b, fit in zip(branches, first)}
    seed0, seed1 = by_branch[("a", 0)], by_branch[("a", 1)]
    assert any(not torch.equal(seed0[name], seed1[name]) for name in seed0)


PACKED_RUN = replace(SYNTHETIC_RUN, run_id="synthetic_packed", configurations=("WR", "HR", "WR+HR"), seeds=(0, 1))


def test_packed_fold_writes_slice_1_files_plus_models(tmp_path: Path) -> None:
    synthetic = make_synthetic()
    mask = shared_mask(synthetic.cache, synthetic.inclusion)
    data = RunData(synthetic.cache, synthetic.inclusion, "synthetic-sha", mask)
    [fold_dir] = train_folds(PACKED_RUN, data, folds=["G01"], output_root=tmp_path, device=torch.device("cpu"))
    manifest = json.loads((fold_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    [job] = manifest["packing"]["jobs"]
    assert manifest["mask_groups"] == 1 and manifest["models_file"] == MODELS_NAME
    assert job["K"] == 6 and job["config_order"] == ["WR", "HR", "WR+HR"] and job["seeds"] == [0, 1]
    epochs = pd.read_csv(fold_dir / EPOCHS_NAME)
    assert len(epochs) == 3 * 2 * PACKED_RUN.training.epochs

    saved = torch.load(fold_dir / MODELS_NAME, weights_only=True)
    configurations = tuple(parse_configuration(c) for c in PACKED_RUN.configurations)
    fold = next(f for f in run_folds(PACKED_RUN, synthetic.cache) if f.test_participant == "G01")
    fold_data = build_fold_data(synthetic.cache, mask, fold, feature_union(configurations))
    predictions = pd.read_parquet(fold_dir / PREDICTIONS_NAME)
    for config in configurations:
        columns = [fold_data.feature_columns.index(c) for c in config.columns]
        config_data = to_device(fold_data.features[:, columns], fold_data.target, torch.device("cpu"))
        for seed in PACKED_RUN.seeds:
            assert saved["checkpoint_epochs"][config.config_id][seed] == \
                manifest["checkpoint_epochs"][config.config_id][str(seed)]
            scaled = predict(saved["state_dicts"][config.config_id][seed], config_data, fold_data.test_targets)
            stored = predictions[(predictions["config_id"] == config.config_id) & (predictions["seed"] == seed)]
            expected = fold_data.scalers["VO2"].inverse(scaled)
            np.testing.assert_allclose(stored["vo2_pred"].to_numpy(), expected, rtol=1e-6)


K230_FEATURES = 9
K230_CONFIGURATIONS = 46
K230_SEEDS = (0, 1, 2, 3, 4)
K230_STEPS = 150
REFERENCE_MS_PER_STEP = 28.0  # prototype, K=310 last-only deterministic on the B580
STEP_TIME_FACTOR = 2.0


@pytest.mark.xpu
def test_k230_step_time_is_within_twice_the_prototype() -> None:
    device = torch.device("xpu")
    configure_determinism(device)
    subsets = [c for r in range(1, K230_FEATURES + 1) for c in itertools.combinations(range(K230_FEATURES), r)]
    configurations = tuple((f"c{i}", subset) for i, subset in enumerate(subsets[:K230_CONFIGURATIONS]))
    rng = np.random.default_rng(0)
    n_targets = K230_STEPS * BATCH
    features = rng.normal(size=(WINDOW_S + n_targets, K230_FEATURES)).astype(np.float32)
    data = to_device(features, rng.normal(size=len(features)).astype(np.float32), device)
    branches = pack_layout(configurations, K230_SEEDS)
    fits = fit_packed(data, np.arange(WINDOW_S, WINDOW_S + n_targets), branches, TrainingSettings(epochs=2),
                      dropout_seed=0)
    ms_per_step = 1000 * fits[0].epoch_wall_s[1] / K230_STEPS  # epoch 2: warm-up excluded
    assert len(branches) == 230
    assert ms_per_step < STEP_TIME_FACTOR * REFERENCE_MS_PER_STEP, f"{ms_per_step:.1f} ms/step"
