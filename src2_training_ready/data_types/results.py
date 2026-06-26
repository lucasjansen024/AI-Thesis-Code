from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import Participant, Visit


@dataclass
class BlandAltman:
    mean: np.ndarray
    diff: np.ndarray
    md: float
    sd: float

@dataclass
class BlandAltmanResult:
    bias: float
    sd: float

    def ULoA(self):
        return self.bias + 1.96 * self.sd

    def LLoA(self):
        return self.bias - 1.96 * self.sd


@dataclass
class RMCorrResult:
    r: float
    p: float

@dataclass
class PearsonResult:
    r: float
    p: float
    
    
@dataclass
class ResidualRunsTestResult:
    n: int
    n_positive: int
    n_negative: int
    runs: Optional[int]
    expected_runs: Optional[float]
    std_runs: Optional[float]
    z: Optional[float]
    p: Optional[float]
    systematic: Optional[bool]
    status: str


@dataclass
class ResidualRunsTestCollection:
    overall: ResidualRunsTestResult
    by_protocol: Dict[str, ResidualRunsTestResult]
    by_participant_protocol: Dict[str, ResidualRunsTestResult]

@dataclass
class SetOfModelsResult:
    VO2_bland_altman_global: BlandAltmanResult
    VO2_rm_corr: RMCorrResult
    VO2_pearson: PearsonResult
    MNG_bland_altman_global: BlandAltmanResult
    MNG_rm_corr: RMCorrResult
    MNG_bland_altman_global_averaged: Optional[BlandAltmanResult]
    MNG_rm_corr_averaged: Optional[RMCorrResult]
    mse: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    VO2_residual_runs_test: Optional[ResidualRunsTestCollection] = None
    
    
@dataclass
class ParticipantModelResult:
    rmse: float
    mse: float
    mae: float
    pearson: PearsonResult
    bland_altman: BlandAltmanResult

@dataclass
class ShapResult:
    values: np.ndarray
    features_mean: np.ndarray

@dataclass
class ParticipantModelMetric:
    metric: float
    participant_folder_path: Path

@dataclass
class MNG:
    participant: Participant
    visit: Visit
    cut_off: float

    both: float
    first_half: float
    second_half: float

    averaged: float
    averaged_first_half: float
    averaged_second_half: float

    complete: float
    first_600: float
    averaged_600: float


@dataclass
class ArrayResult:
    mean: float
    std: float
    min: float
    max: float

    @staticmethod
    def from_array(array: np.ndarray):
        return ArrayResult(
            mean=np.mean(array),
            std=np.std(array),
            min=np.min(array),
            max=np.max(array)
        )

@dataclass
class MNG_Window:
    MNG: ArrayResult
    window_length: int
    step: int