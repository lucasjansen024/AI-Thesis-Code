"""Model inputs: domains, the input tokens of configuration IDs, their cache columns, sensor groups and scaling.

A configuration ID joins input tokens with "+" (`WR`, later `WR+HR+TSI+ANTHRO`). Each token maps to one
`InputSpec`. To add a domain: add a `Domain` member, an `INPUTS` row (NIRS registers two tokens, TSI and HBDIFF),
and a `SCALING` entry per feature column; the cache must carry those columns and `present_<sensor_group>`.
`generate_configurations()` lists every configuration (46 by default, 62 with `nirs_exclusive=False`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

TARGET_COLUMN = "VO2"
TARGET_GROUP = "VO2"
CONFIG_SEPARATOR = "+"


class Domain(StrEnum):
    WR = "WR"
    HR = "HR"
    VENT = "VENT"
    NIRS = "NIRS"
    ANTHRO = "ANTHRO"  # static (sex, SMM): never a configuration on its own


STATIC_DOMAINS = frozenset({Domain.ANTHRO})  # constant within a visit; only ever added to other domains


class Scaling(StrEnum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    NONE = "none"


@dataclass(frozen=True)
class InputSpec:
    domain: Domain
    sensor_group: str  # inclusion.csv sensor group whose presence/validity gates this input
    columns: tuple[str, ...]  # processed-cache feature columns, always used together


# Token -> input, in canonical order (configuration IDs and feature columns follow this order).
INPUTS: dict[str, InputSpec] = {
    "WR": InputSpec(Domain.WR, "WR", ("WR",)),
    "HR": InputSpec(Domain.HR, "HR", ("HR", "HRR")),
    "VENT": InputSpec(Domain.VENT, "VENT", ("VE", "BR")),
    "TSI": InputSpec(Domain.NIRS, "NIRS", ("TSI",)),  # NIRS variables: TSI and HBDIFF gate together
    "HBDIFF": InputSpec(Domain.NIRS, "NIRS", ("HBDIFF",)),
    "ANTHRO": InputSpec(Domain.ANTHRO, "static", ("sex", "SMM")),
}

# Fold scaler per feature column (§2.1): WR min-max, other continuous z-score, sex unscaled.
SCALING: dict[str, Scaling] = {
    "WR": Scaling.MINMAX,
    "HR": Scaling.ZSCORE,
    "HRR": Scaling.ZSCORE,
    "VE": Scaling.ZSCORE,
    "BR": Scaling.ZSCORE,
    "TSI": Scaling.ZSCORE,
    "HBDIFF": Scaling.ZSCORE,
    "sex": Scaling.NONE,
    "SMM": Scaling.ZSCORE,
    TARGET_COLUMN: Scaling.ZSCORE,
}


@dataclass(frozen=True)
class Configuration:
    tokens: tuple[str, ...]  # canonical order

    @property
    def config_id(self) -> str:
        return CONFIG_SEPARATOR.join(self.tokens)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(c for token in self.tokens for c in INPUTS[token].columns)

    @property
    def sensor_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(INPUTS[token].sensor_group for token in self.tokens))

    @property
    def nirs_variables(self) -> tuple[str, ...]:
        return tuple(t for t in self.tokens if INPUTS[t].domain == Domain.NIRS)

    @property
    def has_time_varying(self) -> bool:
        return any(INPUTS[t].domain not in STATIC_DOMAINS for t in self.tokens)


def parse_configuration(config_id: str) -> Configuration:
    tokens = config_id.split(CONFIG_SEPARATOR)
    unknown = [t for t in tokens if t not in INPUTS]
    if unknown:
        raise ValueError(f"{config_id!r}: unknown input tokens {unknown}; known {list(INPUTS)}")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{config_id!r}: repeated input token")
    if all(INPUTS[t].domain in STATIC_DOMAINS for t in tokens):
        raise ValueError(f"{config_id!r}: static inputs need at least one time-varying domain")
    canonical = tuple(t for t in INPUTS if t in tokens)
    if canonical != tuple(tokens):
        raise ValueError(f"{config_id!r}: not in canonical order, expected {CONFIG_SEPARATOR.join(canonical)!r}")
    return Configuration(canonical)


def check_nirs_exclusive(config: Configuration) -> None:
    """Under `nirs_exclusive` a configuration uses at most one NIRS variable (TSI or HBDIFF)."""
    if len(config.nirs_variables) > 1:
        raise ValueError(f"{config.config_id!r}: NIRS variables {config.nirs_variables} are mutually exclusive "
                         f"(nirs_exclusive=True)")


def generate_configurations(nirs_exclusive: bool = True) -> tuple[str, ...]:
    """Every configuration ID: each non-empty set of time-varying inputs, with and without the static ones.

    `nirs_exclusive=True` keeps at most one NIRS variable (46 IDs); `False` lets TSI and HBDIFF co-occur (62).
    Ordered by size (number of tokens), then canonical order: `combinations` keeps `INPUTS` order.
    """
    tokens = tuple(INPUTS)
    candidates = (Configuration(combo) for size in range(1, len(tokens) + 1)
                  for combo in combinations(tokens, size))
    return tuple(c.config_id for c in candidates
                 if c.has_time_varying and (not nirs_exclusive or len(c.nirs_variables) <= 1))


def feature_union(configurations: tuple[Configuration, ...]) -> tuple[str, ...]:
    """All feature columns the configurations use, in canonical order."""
    used = {c for config in configurations for c in config.columns}
    return tuple(c for spec in INPUTS.values() for c in spec.columns if c in used)


def input_sensor_groups() -> tuple[str, ...]:
    """Sensor groups of every registered input: the shared mask requires all of them present in the context."""
    return tuple(dict.fromkeys(spec.sensor_group for spec in INPUTS.values()))
