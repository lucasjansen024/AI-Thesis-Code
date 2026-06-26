from typing import Tuple, List

import numpy as np
from .constants import *
from .data_types.base import ExercisePhase


def sum_string(string):
    sum = 0
    for i in range(len(string)):
        sum += int(string[i])
    return sum


def prbs(
        input='101111',
        length=SEQUENCE_UNITS,
        work_rates: Tuple[int, int]=None,
):
    output = []
    for i in range(length):
        sum = int(input[0]) + int(input[len(input) - 1])
        val = sum % 2
        output.append(val)
        input = str(val) + input[0:len(input) - 1]

    if work_rates is not None:
        output = [work_rates[0] if i == 0 else work_rates[1] for i in output]

    return output


def prbsv2(
        input='10000',
        length=31,
        taps=None
):
    if taps is None:
        taps = [4, 1]

    input = [int(x) for x in input]
    output = []

    for _ in range(length):
        output.append(input[-1])
        # Feedback bit (XOR of tap bits)
        feedback = 0
        for t in taps:
            feedback ^= input[t]
        input = [feedback] + input[:-1]

    return output


def prts(
        input='001',
        length=26,
        work_rates: Tuple[int, int, int]=None,
):
    output = []
    for i in range(length):
        sum = int(input[len(input) - 2]) - int(input[len(input) - 1])
        val = sum % 3
        output.append(val)
        input = str(val) + input[0:len(input) - 1]

    if work_rates is not None:
        output = [work_rates[0] if i == 0 else work_rates[1] if i == 1 else work_rates[2] for i in output]

    return output

def prbs_hedge(work_rates: Tuple[int, int]=None):
    return prbs(input='1101', length=SEQUENCE_UNITS // 2 - 1, work_rates=work_rates)

def step_protocol(GET90):
    output = [0] * REST_UNITS + [25] * BASELINE_UNITS + [GET90] * STEP_UNITS + [25] * STEP_UNITS
    return output

def get_sequence(lower, upper):
    seq = prbs()
    sequence = seq[0:PRBS_WARMUP_UNITS] + seq + seq
    sequence = [lower if i == 0 else upper for i in sequence]
    sequence = [0] * REST_UNITS + sequence

    return np.repeat(sequence, PRBS_UNIT_LENGTH)

def get_sequencev2(lower, upper, warmup_power=None):
    """PRBSv2 protocol used in Lucas.

    Structure (30 s units):
    - 2:00 rest (0 W)
    - 3:30 warm-up (constant warmup_power, typically 80% GET) if provided
      (otherwise: legacy PRBS warm-up based on the tail of the PRBS sequence)
    - 31:00 PRBS segment (2 * SEQUENCEV2_UNITS units)
    """
    seq = prbsv2()

    if warmup_power is None:
        warmup_bits = seq[-PRBS_WARMUP_UNITS:]
        warmup_units = [lower if i == 0 else upper for i in warmup_bits]
    else:
        warmup_units = [warmup_power] * PRBS_WARMUP_UNITS

    body_bits = seq + seq
    body_units = [lower if i == 0 else upper for i in body_bits]

    protocol_units = [0] * REST_UNITS + warmup_units + body_units
    return np.repeat(protocol_units, PRBS_UNIT_LENGTH)

def units_to_time(units):
    return units * PRBS_UNIT_LENGTH

def get_sequencev2_boundaries():
    boundaries = []
    boundaries.append((0, ExercisePhase.REST))
    boundaries.append((units_to_time(REST_UNITS), ExercisePhase.WARMUP))
    boundaries.append((units_to_time(REST_UNITS + PRBS_WARMUP_UNITS), ExercisePhase.EXERCISE))
    boundaries.append((units_to_time(REST_UNITS + PRBS_WARMUP_UNITS + 2 * SEQUENCEV2_UNITS), ExercisePhase.RECOVERY))
    return boundaries

def get_sequence_with_step(lower, upper, GET90):
    step_sequence = step_protocol(GET90)
    prbs_sequence = prbs()
    sequence = prbs_sequence[0:PRBS_WARMUP_UNITS] + prbs_sequence + prbs_sequence
    sequence = [lower if i == 0 else upper for i in sequence]

    protocol = step_sequence + sequence
    return np.repeat(protocol, PRBS_UNIT_LENGTH)

def get_sequencev2_with_step(lower, upper, GET90, warmup_power=None):
    """Legacy: step protocol followed by PRBSv2.

    Kept for backwards compatibility. If warmup_power is provided, we enforce a constant warm-up
    during the PRBS warm-up segment.
    """
    step_sequence = step_protocol(GET90)
    prbs_sequence = prbsv2()

    if warmup_power is None:
        warmup_bits = prbs_sequence[-PRBS_WARMUP_UNITS:]
        warmup_units = [lower if i == 0 else upper for i in warmup_bits]
    else:
        warmup_units = [warmup_power] * PRBS_WARMUP_UNITS

    body_bits = prbs_sequence + prbs_sequence
    body_units = [lower if i == 0 else upper for i in body_bits]

    protocol = step_sequence + warmup_units + body_units
    return np.repeat(protocol, PRBS_UNIT_LENGTH)

def get_sequencev2_with_step_boundaries():
    boundaries = []
    step_end_units = REST_UNITS + BASELINE_UNITS + 2 * STEP_UNITS
    boundaries.append((0, ExercisePhase.REST))
    boundaries.append((units_to_time(REST_UNITS), ExercisePhase.WARMUP))
    boundaries.append((units_to_time(REST_UNITS + BASELINE_UNITS), ExercisePhase.EXERCISE))
    boundaries.append((units_to_time(step_end_units), ExercisePhase.WARMUP))

    boundaries.append((units_to_time(step_end_units + PRBS_WARMUP_UNITS), ExercisePhase.EXERCISE))
    boundaries.append((units_to_time(step_end_units + PRBS_WARMUP_UNITS + 2 * SEQUENCEV2_UNITS), ExercisePhase.RECOVERY))
    return boundaries

def get_ramp_sequence(ramp_rate, warmup=30):
    sequence = [0] * REST_TIME
    sequence += [warmup] * WARMUP_TIME

    time_interval = 60 / ramp_rate
    watt_step = 1

    if ramp_rate == 25:
        watt_step = 5
        time_interval = 12
    elif ramp_rate == 35:
        watt_step = 7
        time_interval = 12

    curr = warmup
    while len(sequence) < SEQUENCE_TIME * 3:
        sequence += [curr] * int(time_interval)
        curr += watt_step

    return sequence

