"""Preparation settings. Every field is part of the cache key; later domains add theirs here."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_INTERP_GAP_S = 20  # same level as inventory/build_inclusion.py MAX_INTERP_GAP_S
DEFAULT_HRREST_WINDOW_S = 120  # handoff §2.5: HRrest = mean HR over the visit's first 2 min (rest)
HBDIFF_TX3, HBDIFF_MEAN3 = "tx3", "mean3"  # Hb-diff from channel Tx3, or the mean over channels 1-3
HBDIFF_CHANNELS = (HBDIFF_TX3, HBDIFF_MEAN3)
HBDIFF_RAW, HBDIFF_REST_MEAN = "none", "rest_mean"  # raw as exported, or minus the visit's 0-120 s mean
HBDIFF_BASELINES = (HBDIFF_RAW, HBDIFF_REST_MEAN)
SMM_PER_VISIT, SMM_PARTICIPANT_MEAN = "per_visit", "participant_mean"  # the visit's SMM, or the participant's mean
SMM_MODES = (SMM_PER_VISIT, SMM_PARTICIPANT_MEAN)


@dataclass(frozen=True)
class PrepSettings:
    max_interp_gap_s: int = DEFAULT_MAX_INTERP_GAP_S  # a longer raw gap makes its seconds "not present"
    hrrest_window_s: int = DEFAULT_HRREST_WINDOW_S  # HRrest = mean of the visit's HR source over [0, this)
    hbdiff_channel: str = HBDIFF_TX3  # MAPPING §7: Tx3 (departs from the handoff's mean of 3)
    hbdiff_baseline: str = HBDIFF_RAW  # MAPPING §7: raw as exported
    smm: str = SMM_PER_VISIT  # MAPPING §7: per visit

    def __post_init__(self) -> None:
        if self.smm not in SMM_MODES:
            raise ValueError(f"smm {self.smm!r} not in {SMM_MODES}")
        if self.hbdiff_channel not in HBDIFF_CHANNELS:
            raise ValueError(f"hbdiff_channel {self.hbdiff_channel!r} not in {HBDIFF_CHANNELS}")
        if self.hbdiff_baseline not in HBDIFF_BASELINES:
            raise ValueError(f"hbdiff_baseline {self.hbdiff_baseline!r} not in {HBDIFF_BASELINES}")
