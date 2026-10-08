"""Train one packed job (K = configurations x seeds of one fold and mask group) and predict with standalone TCNs.

Procedure (§2.3, ticket 09): Adam (lr from the run, eps 1e-7), batch 32 with the last partial batch kept, loss =
sum of per-branch MSEs (each branch gets exactly its own MSE gradient), last-only forward. Each seed is one input
stream with its own permutation per epoch (generator from `("shuffle", seed)`), shared by that seed's
configurations. Branch init from `("init", seed, config_id)`; one dropout RNG per job. Checkpoint per branch =
lowest Keras-style epoch training loss (batch-size weighted, dropout active, strict <), held in memory. After
training every branch is extracted into a standalone `TCN` state dict; prediction uses only those.
Windows are gathered on the device from target rows.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from pipeline.folds import WINDOW_OFFSETS
from pipeline.provenance import derive_seed
from pipeline.runs import TrainingSettings
from pipeline.TCN import TCN, PackedTCN, build_packed_tcn, extract_branch

logger = logging.getLogger(__name__)

KERAS_ADAM_EPS = 1e-7
PREDICT_BATCH = 4096
CUBLAS_WORKSPACE = ":4096:8"


@dataclass(frozen=True)
class DeviceData:
    """One fold's arrays on the device, for a set of feature columns."""

    features: torch.Tensor  # float32 (n_rows, n_features)
    target: torch.Tensor  # float32 (n_rows,)
    offsets: torch.Tensor  # int64 WINDOW_OFFSETS


@dataclass(frozen=True)
class Branch:
    """One model of a packed job."""

    config_id: str
    seed: int
    columns: tuple[int, ...]  # the configuration's feature columns, as indices into the job's feature columns


@dataclass(frozen=True)
class FitResult:
    state_dict: dict[str, torch.Tensor]  # standalone TCN checkpoint weights, on the CPU
    epoch_losses: tuple[float, ...]
    is_best: tuple[bool, ...]  # best so far (strict <); the last True epoch is the checkpoint
    checkpoint_epoch: int  # 1-based
    epoch_wall_s: tuple[float, ...]  # the packed job's epochs (shared by all its branches)


EpochCallback = Callable[[int, tuple[dict[str, torch.Tensor], ...]], None]


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cpu":
        return "cpu"
    return getattr(torch, device.type).get_device_name(device)


def configure_determinism(device: torch.device) -> None:
    """Deterministic algorithms on (process-wide, as decided in ticket 09)."""
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    torch.use_deterministic_algorithms(True)


def synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def to_device(features: np.ndarray, target: np.ndarray, device: torch.device) -> DeviceData:
    return DeviceData(
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
        torch.as_tensor(WINDOW_OFFSETS, dtype=torch.int64, device=device),
    )


def gather_windows(data: DeviceData, targets: torch.Tensor) -> torch.Tensor:
    """(..., n) target rows -> (..., n, 187, n_features): rows t-187 ... t-1, never t or later."""
    return data.features[targets[..., None] + data.offsets]


def pack_layout(configurations: Sequence[tuple[str, tuple[int, ...]]], seeds: Sequence[int]) -> tuple[Branch, ...]:
    """Seed-major branches: branch k = seed_index * n_configurations + configuration_index."""
    return tuple(Branch(config_id, seed, tuple(columns)) for seed in seeds for config_id, columns in configurations)


def stream_seeds(branches: tuple[Branch, ...]) -> tuple[int, ...]:
    """The job's input streams (one per seed, in branch order); raises unless the layout is seed-major and even."""
    seeds = tuple(dict.fromkeys(b.seed for b in branches))
    if not branches or len(branches) % len(seeds) != 0:
        raise ValueError(f"{len(branches)} branches do not split evenly over seeds {seeds}")
    per_stream = len(branches) // len(seeds)
    if tuple(b.seed for b in branches) != tuple(seeds[k // per_stream] for k in range(len(branches))):
        raise ValueError("branches must be seed-major with the same number of configurations per seed")
    if len({(b.config_id, b.seed) for b in branches}) != len(branches):
        raise ValueError("duplicate (config_id, seed) branch in the packed job")
    return seeds


def _train_epoch(model: PackedTCN, optimizer: torch.optim.Optimizer, data: DeviceData, orders: torch.Tensor,
                 batch_size: int, dropout_generator: torch.Generator) -> np.ndarray:
    """One epoch over (n_streams, n_targets) shuffled target rows; returns each branch's epoch loss."""
    model.train()
    n_targets = orders.shape[1]
    loss_sum = torch.zeros(model.n_branches, device=orders.device)
    for begin in range(0, n_targets, batch_size):
        idx = orders[:, begin:begin + batch_size]
        pred = model(gather_windows(data, idx), dropout_generator, last_only=True)  # (b, K)
        truth = data.target[idx][model.branch_stream].T  # (b, K): each branch's stream targets
        branch_loss = torch.mean((pred - truth) ** 2, dim=0)
        optimizer.zero_grad(set_to_none=True)
        branch_loss.sum().backward()
        optimizer.step()
        loss_sum += branch_loss.detach() * idx.shape[1]
    return (loss_sum / n_targets).cpu().numpy().astype(np.float64)


def _keep_best(best: dict[str, torch.Tensor], current: dict[str, torch.Tensor], improved: torch.Tensor,
               n_branches: int) -> dict[str, torch.Tensor]:
    """New best-state dict: improved branches take their current weights, the others keep theirs."""
    rows = improved[:, None]
    return {name: torch.where(rows, tensor.detach().reshape(n_branches, -1), best[name].reshape(n_branches, -1))
            .reshape(tensor.shape) for name, tensor in current.items()}


def _branch_results(branches: tuple[Branch, ...], best_state: dict[str, torch.Tensor], losses: np.ndarray,
                    is_best: np.ndarray, best_epoch: np.ndarray, walls: list[float],
                    n_features: int) -> tuple[FitResult, ...]:
    failed = [f"{b.config_id}/seed {b.seed}" for b, e in zip(branches, best_epoch) if e == 0]
    if failed:
        raise RuntimeError(f"no finite epoch loss, nothing to checkpoint for {failed}")
    cpu_state = {name: tensor.to("cpu") for name, tensor in best_state.items()}
    return tuple(
        FitResult(extract_branch(cpu_state, k, b.columns, len(branches), n_features), tuple(losses[:, k].tolist()),
                  tuple(bool(x) for x in is_best[:, k]), int(best_epoch[k]), tuple(walls))
        for k, b in enumerate(branches)
    )


def current_branch_states(model: PackedTCN, branches: tuple[Branch, ...]) -> tuple[dict[str, torch.Tensor], ...]:
    """Every branch's current (not checkpoint) weights as standalone `TCN` state dicts on the CPU."""
    state = model.state_dict()
    return tuple(extract_branch(state, k, b.columns, len(branches), model.n_features) for k, b in enumerate(branches))


def fit_packed(data: DeviceData, train_targets: np.ndarray, branches: tuple[Branch, ...],
               settings: TrainingSettings, dropout_seed: int,
               on_epoch: EpochCallback | None = None) -> tuple[FitResult, ...]:
    """Train all branches as one packed job; results align with `branches` (see `pack_layout`).

    `on_epoch(epoch, states)` (the pilot's validation pass) is called after each epoch's wall time is taken, with
    the 1-based epoch and every branch's current standalone weights. It reads copies only, so training, its RNGs
    and its results are the same with or without it.
    """
    device, n_branches, n_features = data.features.device, len(branches), data.features.shape[1]
    seeds = stream_seeds(branches)
    init_seeds = tuple(derive_seed("init", b.seed, b.config_id) for b in branches)
    model = build_packed_tcn(tuple(b.columns for b in branches), n_features, len(seeds), init_seeds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, eps=KERAS_ADAM_EPS)
    shuffles = [torch.Generator().manual_seed(derive_seed("shuffle", s)) for s in seeds]
    dropout_generator = torch.Generator(device=device).manual_seed(dropout_seed)
    targets = torch.as_tensor(train_targets, dtype=torch.int64, device=device)
    losses, is_best = np.empty((settings.epochs, n_branches)), np.zeros((settings.epochs, n_branches), dtype=bool)
    best_loss, best_epoch = np.full(n_branches, np.inf), np.zeros(n_branches, dtype=np.int64)
    best_state, walls = {k: v.detach().clone() for k, v in model.state_dict().items()}, []
    for epoch in range(settings.epochs):
        start = time.perf_counter()
        orders = torch.stack([targets[torch.randperm(len(targets), generator=g).to(device)] for g in shuffles])
        losses[epoch] = _train_epoch(model, optimizer, data, orders, settings.batch_size, dropout_generator)
        is_best[epoch] = losses[epoch] < best_loss
        best_loss = np.where(is_best[epoch], losses[epoch], best_loss)
        best_epoch = np.where(is_best[epoch], epoch + 1, best_epoch)
        improved = torch.as_tensor(is_best[epoch], device=device)
        best_state = _keep_best(best_state, model.state_dict(), improved, n_branches)
        synchronize(device)
        walls.append(time.perf_counter() - start)
        logger.info("packed K=%d epoch %d/%d: mean loss %.5f, %d branches improved (%.1f s)", n_branches,
                    epoch + 1, settings.epochs, float(np.nanmean(losses[epoch])), int(is_best[epoch].sum()),
                    walls[-1])
        if on_epoch is not None:
            on_epoch(epoch + 1, current_branch_states(model, branches))
    return _branch_results(branches, best_state, losses, is_best, best_epoch, walls, n_features)


def predict(state_dict: dict[str, torch.Tensor], data: DeviceData, targets: np.ndarray) -> np.ndarray:
    """Scaled predictions (float64, CPU) of a standalone checkpoint model at the target rows, dropout off."""
    model = TCN(data.features.shape[1]).to(data.features.device)
    model.load_state_dict(state_dict)
    model.eval()
    rows = torch.as_tensor(targets, dtype=torch.int64, device=data.features.device)
    outputs = []
    with torch.no_grad():
        for begin in range(0, len(rows), PREDICT_BATCH):
            outputs.append(model(gather_windows(data, rows[begin:begin + PREDICT_BATCH])).cpu())
    return torch.cat(outputs).numpy().astype(np.float64)
