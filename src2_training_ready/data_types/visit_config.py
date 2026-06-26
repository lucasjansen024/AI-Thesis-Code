from enum import Enum
from typing import TypedDict, Dict
from .base import Participant, Visit


class HRSource(Enum):
    HR = 'HR'
    VO2 = 'VO2'
    None_ = None


class NIRSRepetition(Enum):
    One = 'One'
    Two = 'Two'
    Both = 'Both'
    None_ = None


class Occlusion(TypedDict):
    start: int
    end: int


class VisitConfig(TypedDict):
    VE_lag: int
    HR_source: HRSource
    notes: str | None
    Occlusion: Occlusion | None
    NIRS_repetition: NIRSRepetition | None


Config = Dict[Participant, Dict[Visit, VisitConfig]]