"""TCN port: Keras-default init, receptive field, dropout RNG, and causality of the prediction at t."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipeline.fit import gather_windows, to_device
from pipeline.inclusion import WINDOW_S
from pipeline.TCN import FILTERS, KERNEL, RECEPTIVE_FIELD, build_tcn

N_FEATURES = 3
N_ROWS = 1000
TARGET = 600
PERTURBATION = 1e3


def test_receptive_field_is_the_window() -> None:
    assert RECEPTIVE_FIELD == WINDOW_S == 187


def test_keras_default_init() -> None:
    model = build_tcn(N_FEATURES, init_seed=1)
    first_conv = model.blocks[0].convs[0].conv
    limit = math.sqrt(6.0 / (N_FEATURES * KERNEL + FILTERS * KERNEL))
    assert first_conv.weight.abs().max() <= limit
    assert first_conv.weight.abs().max() > 0.5 * limit
    head_limit = math.sqrt(6.0 / (FILTERS + 1))
    assert model.head.weight.abs().max() <= head_limit
    for name, parameter in model.named_parameters():
        if name.endswith("conv.bias") or name == "head.bias":
            assert torch.count_nonzero(parameter) == 0, name
        if ".norms." in name:
            expected = torch.ones_like(parameter) if name.endswith("weight") else torch.zeros_like(parameter)
            assert torch.equal(parameter, expected), name


def test_init_depends_only_on_the_seed() -> None:
    a, b, c = build_tcn(N_FEATURES, 5), build_tcn(N_FEATURES, 5), build_tcn(N_FEATURES, 6)
    for (name, pa), pb, pc in zip(a.named_parameters(), b.parameters(), c.parameters()):
        assert torch.equal(pa, pb), name
    assert not torch.equal(a.blocks[0].convs[0].conv.weight, c.blocks[0].convs[0].conv.weight)


def test_training_forward_needs_a_dropout_generator() -> None:
    model = build_tcn(N_FEATURES, 0).train()
    with pytest.raises(ValueError, match="dropout generator"):
        model(torch.zeros(2, WINDOW_S, N_FEATURES))


def _features(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)


def _output_at_target(features: np.ndarray, train_mode: bool) -> torch.Tensor:
    model = build_tcn(N_FEATURES, 0).train(train_mode)
    data = to_device(features, np.zeros(N_ROWS, dtype=np.float32), torch.device("cpu"))
    generator = torch.Generator().manual_seed(3) if train_mode else None
    with torch.no_grad():
        return model(gather_windows(data, torch.tensor([TARGET])), generator)


@pytest.mark.parametrize("train_mode", [False, True], ids=["eval", "train"])
def test_output_at_t_ignores_inputs_at_or_after_t(train_mode: bool) -> None:
    features = _features(0)
    perturbed = features.copy()
    perturbed[TARGET:] = PERTURBATION * _features(1)[TARGET:]
    assert torch.equal(_output_at_target(features, train_mode), _output_at_target(perturbed, train_mode))


def test_output_at_t_depends_on_input_t_minus_1() -> None:
    features = _features(0)
    perturbed = features.copy()
    perturbed[TARGET - 1] += PERTURBATION
    assert not torch.equal(_output_at_target(features, False), _output_at_target(perturbed, False))
