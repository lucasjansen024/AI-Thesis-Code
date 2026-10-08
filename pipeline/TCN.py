"""§2.3 TCN in PyTorch, Keras defaults pinned (port of legacy `src2_training_ready/models/TCN.py`).

Two residual blocks: causal dilated Conv1D (k 7, 16 filters) at dilations 1/2/4 and 8/16, each followed by
LayerNorm (eps 1e-3), ReLU and dropout 0.2, plus a 1x1 projection of the block input; then Dense(1) on the last
time step. Receptive field 1 + 6 x 31 = 187 s. Glorot-uniform kernels, zero biases (Keras Conv1D/Dense defaults).

`TCN` is the standalone (unpacked, full-forward) model: every prediction and evaluation uses it. `PackedTCN` holds
K of them for training (grouped convs, per-branch LayerNorm, channel masks, last-only forward); `build_packed_tcn`
packs freshly initialised standalone TCNs and `extract_branch` turns a trained branch back into a `TCN` state dict.
Dropout draws come from an explicit generator so a job owns its dropout RNG. Initialisation draws from a CPU
generator in `INIT_ORDER`, so weights are device-independent.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pipeline.inclusion import WINDOW_S

FILTERS = 16
KERNEL = 7
DROPOUT = 0.2
BLOCK_DILATIONS = ((1, 2, 4), (8, 16))
KERAS_LAYER_NORM_EPS = 1e-3
RECEPTIVE_FIELD = 1 + (KERNEL - 1) * sum(d for block in BLOCK_DILATIONS for d in block)
if RECEPTIVE_FIELD != WINDOW_S:
    raise ImportError(f"TCN receptive field {RECEPTIVE_FIELD} != window {WINDOW_S}")


def glorot_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int, generator: torch.Generator) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-limit, limit, generator=generator)


def pointwise_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """A kernel-1 (grouped) Conv1d computed as a matmul with the same parameters.

    On the B580 (torch 2.14 XPU), Conv1d's backward returns wrong weight and bias gradients for a 1x1 conv with
    one input channel (diagnosed 9 Oct 2026; forward is correct, CPU is correct, determinism makes no difference).
    The matmul form is the same function on every device, so all 1x1 projections use it.
    """
    b, c_in, t = x.shape
    groups = conv.groups
    c_out = conv.weight.shape[0]
    weight = conv.weight.view(groups, c_out // groups, c_in // groups)
    y = torch.einsum("goc,bgct->bgot", weight, x.reshape(b, groups, c_in // groups, t))
    return y.reshape(b, c_out, t) + conv.bias[None, :, None]


class CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def keras_init_(self, generator: torch.Generator) -> None:
        c_out, c_in, kernel = self.conv.weight.shape
        glorot_uniform_(self.conv.weight, c_in * kernel, c_out * kernel, generator)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv.kernel_size[0] == 1:
            return pointwise_conv(x, self.conv)
        return self.conv(F.pad(x, (self.pad, 0)))


class ChannelNorm(nn.Module):
    """Keras LayerNormalization(axis=-1) on a (batch, channels, time) tensor: normalise over channels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + KERAS_LAYER_NORM_EPS) * self.weight[:, None] + self.bias[:, None]


def dropout(x: torch.Tensor, rate: float, generator: torch.Generator | None) -> torch.Tensor:
    """Keras/TF inverted dropout with an explicit RNG."""
    if generator is None:
        raise ValueError("training-mode forward needs a dropout generator")
    keep = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) >= rate
    return x * keep / (1.0 - rate)


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        channels = (c_in,) + (FILTERS,) * (len(dilations) - 1)
        self.convs = nn.ModuleList(CausalConv(c, FILTERS, KERNEL, d) for c, d in zip(channels, dilations))
        self.norms = nn.ModuleList(ChannelNorm(FILTERS) for _ in dilations)
        self.projection = CausalConv(c_in, FILTERS, 1, 1)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h)))
            if self.training:
                h = dropout(h, DROPOUT, generator)
        return h + self.projection(x)


class TCN(nn.Module):
    """(batch, time, features) -> (batch,): V̇O₂ (scaled) at the second after the window's last input."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        c_ins = (n_features,) + (FILTERS,) * (len(BLOCK_DILATIONS) - 1)
        self.blocks = nn.ModuleList(ResidualBlock(c, d) for c, d in zip(c_ins, BLOCK_DILATIONS))
        self.head = nn.Linear(FILTERS, 1)

    def keras_init_(self, generator: torch.Generator) -> None:
        """INIT_ORDER: per block, its convs in order then its projection; then the head. LayerNorm 1/0."""
        for block in self.blocks:
            for conv in block.convs:
                conv.keras_init_(generator)
            block.projection.keras_init_(generator)
        glorot_uniform_(self.head.weight, FILTERS, 1, generator)
        nn.init.zeros_(self.head.bias)

    def forward(self, windows: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        x = windows.transpose(1, 2)
        for block in self.blocks:
            x = block(x, generator)
        return self.head(x[:, :, -1]).squeeze(-1)


def build_tcn(n_features: int, init_seed: int) -> TCN:
    """A freshly initialised TCN on the CPU; the same seed gives the same weights on any device."""
    model = TCN(n_features)
    model.keras_init_(torch.Generator().manual_seed(init_seed))
    return model


# --- Packed form: K branches of the TCN above in one module (training only; evaluation uses extracted TCNs) ---

INPUT_WEIGHTS = ("blocks.0.convs.0.conv.weight", "blocks.0.projection.conv.weight")  # read the feature columns


class PackedConv(nn.Module):
    """K independent `CausalConv`s as one grouped Conv1d (branch-major channels). Weights start at 0."""

    def __init__(self, branches: int, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(branches * c_in, branches * c_out, kernel, dilation=dilation, groups=branches)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv.kernel_size[0] == 1:
            return pointwise_conv(x, self.conv)
        return self.conv(F.pad(x, (self.pad, 0)))

    def strided(self, x: torch.Tensor) -> torch.Tensor:
        """The same kernel on a sequence already thinned to this layer's dilation: stride 2, no padding."""
        return F.conv1d(x, self.conv.weight, self.conv.bias, stride=2, groups=self.conv.groups)


class PackedNorm(nn.Module):
    """`ChannelNorm` per branch: each branch's 16 channels are normalised on their own."""

    def __init__(self, branches: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(branches, FILTERS))
        self.bias = nn.Parameter(torch.zeros(branches, FILTERS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape
        x = x.view(b, self.weight.shape[0], FILTERS, t)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + KERAS_LAYER_NORM_EPS) * self.weight[:, :, None] + self.bias[:, :, None]
        return x.reshape(b, -1, t)


class PackedBlock(nn.Module):
    def __init__(self, branches: int, c_in: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        channels = (c_in,) + (FILTERS,) * (len(dilations) - 1)
        self.convs = nn.ModuleList(PackedConv(branches, c, FILTERS, KERNEL, d) for c, d in zip(channels, dilations))
        self.norms = nn.ModuleList(PackedNorm(branches) for _ in dilations)
        self.projection = PackedConv(branches, c_in, FILTERS, 1, 1)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None, last_only: bool) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv.strided(h) if last_only else conv(h)))
            if self.training:
                h = dropout(h, DROPOUT, generator)
        if not last_only:
            return h + self.projection(x)
        stride = 2 ** len(self.convs)
        start = x.shape[-1] - 1 - stride * (h.shape[-1] - 1)
        return h + self.projection(x[:, :, start::stride])


class PackedHead(nn.Module):
    """K `nn.Linear(16, 1)` heads: weight (K, 16), bias (K,)."""

    def __init__(self, branches: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(branches, FILTERS))
        self.bias = nn.Parameter(torch.zeros(branches))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, K*16) -> (B, K)
        return (h.view(h.shape[0], -1, FILTERS) * self.weight).sum(dim=-1) + self.bias


class PackedTCN(nn.Module):
    """K standalone TCNs packed as grouped convolutions with a per-branch LayerNorm.

    Every branch sees all `n_features` columns of the job; a fixed channel mask zeroes the columns its
    configuration doesn't use, so their weights get zero gradient and stay at their initial 0. Branches are
    stream-major: the job has `n_streams` input streams (one per seed: its own batch), each read by
    K / n_streams consecutive branches; `branch_stream[k]` is branch k's stream. State-dict keys and per-branch
    layout match `TCN`, so `pack_branch_`/`extract_branch` move weights between the two forms.

    `last_only` computes just the positions that reach the last output: each dilation-d layer becomes a stride-2
    conv on the sequence thinned by d (187 -> 91 -> 43 -> 19 -> 7 -> 1, no padding). Same weights and function.
    """

    def __init__(self, column_sets: tuple[tuple[int, ...], ...], n_features: int, n_streams: int) -> None:
        super().__init__()
        branches = len(column_sets)
        if branches == 0 or n_streams < 1 or branches % n_streams != 0:
            raise ValueError(f"{branches} branches do not split into {n_streams} equal streams")
        mask = torch.zeros(branches, n_features)
        for k, columns in enumerate(column_sets):
            if not columns or len(set(columns)) != len(columns) or not all(0 <= c < n_features for c in columns):
                raise ValueError(f"branch {k}: columns {columns} invalid for {n_features} features")
            mask[k, list(columns)] = 1.0
        self.column_sets = tuple(tuple(c) for c in column_sets)
        self.n_features = n_features
        self.n_branches = branches
        self.n_streams = n_streams
        self.register_buffer("channel_mask", mask.reshape(-1), persistent=False)
        self.register_buffer("branch_stream", torch.arange(branches) // (branches // n_streams), persistent=False)
        c_ins = (n_features,) + (FILTERS,) * (len(BLOCK_DILATIONS) - 1)
        self.blocks = nn.ModuleList(PackedBlock(branches, c, d) for c, d in zip(c_ins, BLOCK_DILATIONS))
        self.head = PackedHead(branches)

    def forward(self, windows: torch.Tensor, generator: torch.Generator | None = None,
                last_only: bool = False) -> torch.Tensor:
        """(n_streams, batch, time, n_features) -> (batch, K): branch k reads stream `branch_stream[k]`."""
        streams, b, t, n_features = windows.shape
        if streams != self.n_streams or n_features != self.n_features:
            raise ValueError(f"windows {tuple(windows.shape)}: expected {self.n_streams} streams x "
                             f"{self.n_features} features")
        if last_only and t != RECEPTIVE_FIELD:
            raise ValueError(f"last-only forward needs exactly {RECEPTIVE_FIELD} time steps, got {t}")
        per_stream = self.n_branches // streams
        x = windows.permute(1, 0, 3, 2)[:, :, None].expand(b, streams, per_stream, n_features, t)
        x = x.reshape(b, -1, t) * self.channel_mask[:, None]
        for block in self.blocks:
            x = block(x, generator, last_only)
        return self.head(x[:, :, -1])


def _branch_view(name: str, tensor: torch.Tensor, n_branches: int, n_features: int) -> torch.Tensor:
    """Branch-major view (K, ...) of a packed tensor; input-feature weights as (K, 16, n_features, kernel)."""
    if name in INPUT_WEIGHTS:
        return tensor.view(n_branches, FILTERS, n_features, -1)
    return tensor.view(n_branches, -1)


def pack_branch_(packed: PackedTCN, branch: int, state: dict[str, torch.Tensor]) -> None:
    """Copy a standalone `TCN(len(columns))` state dict into one branch of `packed` (in place)."""
    columns = list(packed.column_sets[branch])
    with torch.no_grad():
        for name, tensor in packed.state_dict(keep_vars=True).items():
            target = _branch_view(name, tensor, packed.n_branches, packed.n_features)[branch]
            source = state[name].to(tensor.device, tensor.dtype)
            if name in INPUT_WEIGHTS:
                target[:, columns] = source
            else:
                target.copy_(source.reshape(-1))


def extract_branch(packed_state: dict[str, torch.Tensor], branch: int, columns: tuple[int, ...],
                   n_branches: int, n_features: int) -> dict[str, torch.Tensor]:
    """One branch of a packed state dict as a standalone `TCN(len(columns))` state dict (new CPU tensors)."""
    template = TCN(len(columns)).state_dict()
    if set(template) != set(packed_state):
        raise ValueError(f"packed keys {sorted(packed_state)} do not match the TCN's {sorted(template)}")
    out = {}
    for name, reference in template.items():
        part = _branch_view(name, packed_state[name], n_branches, n_features)[branch]
        if name in INPUT_WEIGHTS:
            part = part[:, list(columns)]
        out[name] = part.detach().to("cpu").reshape(reference.shape).clone()
    return out


def build_packed_tcn(column_sets: tuple[tuple[int, ...], ...], n_features: int, n_streams: int,
                     init_seeds: tuple[int, ...]) -> PackedTCN:
    """Branch k starts as `build_tcn(len(column_sets[k]), init_seeds[k])`: the same weights however it is
    packed, Glorot on the true fan-in, unused feature weights 0. On the CPU."""
    if len(init_seeds) != len(column_sets):
        raise ValueError(f"{len(init_seeds)} init seeds for {len(column_sets)} branches")
    packed = PackedTCN(column_sets, n_features, n_streams)
    for k, (columns, seed) in enumerate(zip(column_sets, init_seeds)):
        pack_branch_(packed, k, build_tcn(len(columns), seed).state_dict())
    return packed
