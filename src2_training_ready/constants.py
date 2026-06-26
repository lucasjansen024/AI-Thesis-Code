from enum import Enum
from pathlib import Path

from .data_types import Participant, Visit

# -----------------------------------------------------------------------------
# Participants
# -----------------------------------------------------------------------------
LUCAS_PARTICIPANTS_ALL = [
    Participant.P01, Participant.P02, Participant.P03, Participant.P04, Participant.P05,
    Participant.P06, Participant.P07, Participant.P08, Participant.P09, Participant.P10,
    Participant.P12, Participant.P13, Participant.P14,
]

# Participants that currently have all 4 protocols available and are intended
# for the thesis experiments
COMPLETE_4_PROTOCOLS_LIST = [
    Participant.P01,
    Participant.P02,
    Participant.P03,
    Participant.P04,
    Participant.P05,
    Participant.P06,
    Participant.P07,
    Participant.P08,
    Participant.P09,
    Participant.P10,
    Participant.P12,
    Participant.P13,
    Participant.P14,
]

# Backward-compatible aliases / generic participant collections
PARTICIPANTS_ALL = LUCAS_PARTICIPANTS_ALL


class ParticipantSelection(Enum):
    # Main thesis selection
    COMPLETE_4_PROTOCOLS = COMPLETE_4_PROTOCOLS_LIST

    # Backward-compatible aliases expected elsewhere in the codebase
    ALL_PARTICIPANTS = COMPLETE_4_PROTOCOLS_LIST
    LUCAS_ALL = COMPLETE_4_PROTOCOLS_LIST
    LUCAS_COMPLETE = COMPLETE_4_PROTOCOLS_LIST

    # Per single protocol
    RAMP = COMPLETE_4_PROTOCOLS_LIST
    PRBS = COMPLETE_4_PROTOCOLS_LIST
    INTERVAL = COMPLETE_4_PROTOCOLS_LIST
    STEP = COMPLETE_4_PROTOCOLS_LIST

    # Per pair for cross-protocol testing
    RAMP_PRBS = COMPLETE_4_PROTOCOLS_LIST
    RAMP_INTERVAL = COMPLETE_4_PROTOCOLS_LIST
    RAMP_STEP = COMPLETE_4_PROTOCOLS_LIST
    PRBS_INTERVAL = COMPLETE_4_PROTOCOLS_LIST
    PRBS_STEP = COMPLETE_4_PROTOCOLS_LIST
    INTERVAL_STEP = COMPLETE_4_PROTOCOLS_LIST

    # For LOPO / 3-protocol training
    PRBS_INTERVAL_STEP = COMPLETE_4_PROTOCOLS_LIST

    # Extra aliases in case older scripts still use *_LIST names
    RAMP_LIST = COMPLETE_4_PROTOCOLS_LIST
    PRBS_LIST = COMPLETE_4_PROTOCOLS_LIST
    INTERVAL_LIST = COMPLETE_4_PROTOCOLS_LIST
    STEP_LIST = COMPLETE_4_PROTOCOLS_LIST
    RAMP_PRBS_LIST = COMPLETE_4_PROTOCOLS_LIST
    RAMP_INTERVAL_LIST = COMPLETE_4_PROTOCOLS_LIST
    RAMP_STEP_LIST = COMPLETE_4_PROTOCOLS_LIST
    PRBS_INTERVAL_LIST = COMPLETE_4_PROTOCOLS_LIST
    PRBS_STEP_LIST = COMPLETE_4_PROTOCOLS_LIST
    INTERVAL_STEP_LIST = COMPLETE_4_PROTOCOLS_LIST
    PRBS_INTERVAL_STEP_LIST = COMPLETE_4_PROTOCOLS_LIST


# -----------------------------------------------------------------------------
# Visits / protocols
# -----------------------------------------------------------------------------
VISITS_SUBMAX_ALL = [Visit.Interval, Visit.Step, Visit.PRBS]
VISITS_ALL = [Visit.Ramp, *VISITS_SUBMAX_ALL]

# Backward-compatible visit aliases
VISITS_ALL_NORMAL = VISITS_ALL
VISITS_PRBS_ALL = VISITS_SUBMAX_ALL
VISITS_PRBS_NORMAL = VISITS_SUBMAX_ALL
VISITS_ALL_EXCL_HARD = VISITS_ALL
VISITS_PRBS_EXCL_HARD = VISITS_SUBMAX_ALL

class VisitSelection(Enum):
    # Single protocols
    ONLY_RAMP = [Visit.Ramp]
    ONLY_INTERVAL = [Visit.Interval]
    ONLY_STEP = [Visit.Step]
    ONLY_PRBS = [Visit.PRBS]

    # Pairs
    PRBS_INTERVAL = [Visit.PRBS, Visit.Interval]
    PRBS_STEP = [Visit.PRBS, Visit.Step]
    INTERVAL_STEP = [Visit.Interval, Visit.Step]

    RAMP_PRBS = [Visit.Ramp, Visit.PRBS]
    RAMP_INTERVAL = [Visit.Ramp, Visit.Interval]
    RAMP_STEP = [Visit.Ramp, Visit.Step]

    # Triples
    RAMP_PRBS_INTERVAL = [Visit.Ramp, Visit.PRBS, Visit.Interval]
    RAMP_PRBS_STEP = [Visit.Ramp, Visit.PRBS, Visit.Step]
    RAMP_INTERVAL_STEP = [Visit.Ramp, Visit.Interval, Visit.Step]

    # All protocols
    SUBMAX_ALL = VISITS_SUBMAX_ALL
    SUBMAX_PLUS_RAMP = VISITS_ALL
    ALL_PROTOCOLS = VISITS_ALL
    ALL_VISITS = VISITS_ALL

    # LOPO selections
    ALL_EXCEPT_RAMP = [Visit.PRBS, Visit.Interval, Visit.Step]
    ALL_EXCEPT_PRBS = [Visit.Ramp, Visit.Interval, Visit.Step]
    ALL_EXCEPT_INTERVAL = [Visit.Ramp, Visit.PRBS, Visit.Step]
    ALL_EXCEPT_STEP = [Visit.Ramp, Visit.PRBS, Visit.Interval]

    ALL_EXCL_HARD = VISITS_ALL
    PRBS_EXCL_HARD = VISITS_SUBMAX_ALL
    ONLY_EASY = [Visit.Interval]

    # Legacy aliases used elsewhere in the codebase


# -----------------------------------------------------------------------------
# Protocol timing constants
# -----------------------------------------------------------------------------
PRBS_UNIT_LENGTH = 30

REST_UNITS = 4
BASELINE_UNITS = 8
STEP_UNITS = 12

PRBS_WARMUP_UNITS = 7
RAMP_WARMUP_UNITS = 8
BASELINE_TIME = BASELINE_UNITS * PRBS_UNIT_LENGTH

SEQUENCE_UNITS = 32
SEQUENCEV2_UNITS = 31

REST_TIME = REST_UNITS * PRBS_UNIT_LENGTH
WARMUP_START = REST_TIME
PRBS_WARMUP_TIME = PRBS_WARMUP_UNITS * PRBS_UNIT_LENGTH
RAMP_WARMUP_TIME = RAMP_WARMUP_UNITS * PRBS_UNIT_LENGTH
SEQUENCE_START = WARMUP_START + PRBS_WARMUP_TIME

SEQUENCE_TIME = SEQUENCE_UNITS * PRBS_UNIT_LENGTH
SEQUENCEV2_TIME = SEQUENCEV2_UNITS * PRBS_UNIT_LENGTH

TOTAL_STEP_TIME = (REST_UNITS + BASELINE_UNITS + 2 * STEP_UNITS) * PRBS_UNIT_LENGTH
TOTAL_TIME = (REST_UNITS + PRBS_WARMUP_UNITS + 2 * SEQUENCE_UNITS) * PRBS_UNIT_LENGTH
TOTAL_V2_TIME = (PRBS_WARMUP_UNITS + 2 * SEQUENCEV2_UNITS) * PRBS_UNIT_LENGTH

# Lucas submax timing helpers
WARMUP_TIME = PRBS_WARMUP_TIME
SUBMAX_SEGMENT_TIME = SEQUENCEV2_TIME * 2  # 31:00
SUBMAX_TOTAL_TIME = REST_TIME + WARMUP_TIME + SUBMAX_SEGMENT_TIME  # 36:30

INTERVAL_CYCLES = 10
INTERVAL_LOW_S = 120
INTERVAL_HIGH_S = 60
INTERVAL_FINAL_LOW_S = 60
INTERVAL_SEGMENT_TIME = INTERVAL_CYCLES * (INTERVAL_LOW_S + INTERVAL_HIGH_S) + INTERVAL_FINAL_LOW_S

STEP_SEGMENT_TIME = 31 * 60

NIRS_HZ = 10

CUT_OFF_FREQUENCY = 0.01
WINDOW_LENGTH = 600

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
MODEL_PATH = Path('./m/')
DATA_PATH = Path('./data/')
RAW_PATH = Path('./data/raw/')
PREPARED_PATH = Path('./data/prepared/')
IMAGE_PATH = Path('./images/')
PROTOCOL_PATH = Path('./data/protocols/')

EXCEL_RESULTS_PATH = Path('./generated/excel/')
GRID_SEARCH_RESULTS_PATH = Path('./generated/grid_search/')
MNG_RESULTS_PATH = Path('./generated/mng/')

BEST_FOLDER = 'best'
COMMON_FOLDER = 'common'
RESULTS_FOLDER = 'results'