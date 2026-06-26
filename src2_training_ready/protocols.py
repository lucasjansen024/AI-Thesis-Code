"""Protocol helpers for Lucas.

These helpers create *ideal* 1 Hz work-rate sequences (WR) for the three submax protocols.
They mirror the protocol definitions used when the tests were performed.

All sequences include:
- 2:00 rest (0 W)
- 3:30 warm-up (usually 80% GET)
- 31:00 protocol-specific segment

The values (powers) are passed in as integers (Watts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


REST_S = 120
WARMUP_S = 210


def interval_sequence(power_warmup: int, power_delta70: int) -> List[int]:
    """Interval protocol:

    - 10x (2:00 25 W / 1:00 Δ60%)
    - final 1:00 25 W
    """
    seq: List[int] = []

    # rest
    seq += [0] * REST_S
    # warm-up
    seq += [int(power_warmup)] * WARMUP_S

    # main
    for _ in range(10):
        seq += [25] * 120
        seq += [int(power_delta70)] * 60

    seq += [25] * 60
    return seq


def step_sequence(power_80pct: int, power_60pct: int, power_delta30: int, power_delta60: int) -> List[int]:
    """Step protocol:

    - 6:00 80% GET
    - 6:00 Δ25%
    - 6:00 80% GET
    - 3:00 Δ50%
    - 5:00 25 W
    - 5:00 60% GET
    """
    seq: List[int] = []

    seq += [0] * REST_S
    seq += [int(power_80pct)] * WARMUP_S

    seq += [int(power_80pct)] * 360
    seq += [int(power_delta30)] * 360
    seq += [int(power_80pct)] * 360
    seq += [int(power_delta60)] * 180
    seq += [25] * 300
    seq += [int(power_60pct)] * 300

    return seq
