from dataclasses import dataclass
from enum import StrEnum, IntEnum, Enum
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


class Dataset(StrEnum):
    Lucas = "Lucas"
    Gies = "Gies"


class Domain(StrEnum):
    VE = 'VE'
    HR = 'HR'
    NIRS = 'NIRS'
    WR = 'WR'
    Fit = 'Fit'


class File(StrEnum):
    VO2BxB = 'VO2'
    HR = 'HR'
    WR = 'WR'
    VE = 'VE.csv'
    VEBxB = 'VEBxB'
    Oxy = 'Oxy'
    Occlusion = 'Occlusion'
    Fit = 'Fit'


class Participant(StrEnum):
    # Lucas study naming (current dataset)
    P01 = 'P01'
    P02 = 'P02'
    P03 = 'P03'
    P04 = 'P04'
    P05 = 'P05'
    P06 = 'P06'
    P07 = 'P07'
    P08 = 'P08'
    P09 = 'P09'
    P10 = 'P10'
    P12 = 'P12'
    P13 = 'P13'
    P14 = 'P14'

    # Legacy / Gies naming kept for compatibility with old scripts
    #P1 = 'P1'
    #P3 = 'P3'
    #P4 = 'P4'
    #P5 = 'P5'
    #P6 = 'P6'
    #P7 = 'P7'
    #P8 = 'P8'
    #P9 = 'P9'
    #P11 = 'P11'
    #P12 = 'P12'
    #P13 = 'P13'
    #P14 = 'P14'
    #P15 = 'P15'

    @property
    def number(self):
        return int(self.value[1:])


class Visit(StrEnum):
    # Lucas submax protocols
    Interval = 'Interval'
    Step = 'Step'
    PRBS = 'PRBS'
    Ramp = 'Ramp'

    # Legacy / Gies visits kept for compatibility
    #Hard = 'Hard'
    #Hard2 = 'Hard2'
    #FailedRamp = 'FailedRamp'
    #Easy = 'Easy'
    #Medium = 'Medium'
    #Medium2 = 'Medium2'

    def coerce_to_normal(self):
        return self

    @staticmethod
    def from_str(visit: str):
        visit = str(visit)
        for candidate in [Visit.Interval, Visit.Step, Visit.PRBS, Visit.Ramp,
                          ]:
            if candidate.value in visit:
                return candidate
        return None

    def is_ramp(self):
        return self == Visit.Ramp
    def is_prbs(self):
        return self == Visit.PRBS

    def is_interval(self):
        return self == Visit.Interval

    def is_step(self):
        return self == Visit.Step

    def is_submax(self):
        return self in {Visit.PRBS, Visit.Interval, Visit.Step}

    @property
    def lower_wr(self):
        match self.coerce_to_normal():
            case Visit.Ramp | Visit.Interval | Visit.Step | Visit.PRBS:
                return None            
        return None

    @property
    def upper_wr(self):
        match self.coerce_to_normal():
            case Visit.Ramp | Visit.Interval | Visit.Step | Visit.PRBS:
                return None
        return None


class WorkRate(StrEnum):
    W25 = '25W'
    GET90 = '90% GET'
    GET = 'GET'
    D30 = '$\Delta$ 30%'


InputDataName = str


class ExercisePhase(IntEnum):
    REST = 1
    WARMUP = 2
    EXERCISE = 3
    RECOVERY = 4


Boundaries = List[Tuple[int, ExercisePhase]]


@dataclass
class PreparedVisitData:
    participant: Participant
    visit: Visit
    data: Dict[File, pd.DataFrame]


@dataclass
class ProcessedVisitData:
    participant: Participant
    visit: Visit
    data: pd.DataFrame


@dataclass
class TrainingVisitData:
    participant: Participant
    visit: Visit
    X: np.ndarray
    y: np.ndarray


class NIRSRepetition(Enum):
    One = 'One'
    Two = 'Two'
    Both = 'Both'
    None_ = None


PreparedData = Dict[Participant, Dict[Visit, PreparedVisitData]]
ProcessedData = Dict[Participant, Dict[Visit, ProcessedVisitData]]
ResultsVisitData = ProcessedVisitData
ResultsData = Dict[Participant, Dict[Visit, ResultsVisitData]]
TrainingData = Dict[Participant, Dict[Visit, TrainingVisitData]]


@dataclass
class ParticipantDetail:
    name: Participant
    weight: float
    height: float
    age: float

    MRT: float
    m: float
    c: float
    ramp_rate: int

    VO2Max: float
    VO2GET: float
    WRMax: float
    WRGET: float

    def calculate_delta_wr(self, percentage):
        D40mL = ((self.VO2Max - self.VO2GET) * (percentage / 100)) + self.VO2GET
        D40xs = (D40mL - self.c) / self.m
        D40T = D40xs - self.MRT
        return ((D40T / 60) * self.ramp_rate) + 30


ParticipantDetails = Dict[Participant, ParticipantDetail]
