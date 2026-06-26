from tkinter.filedialog import askopenfilename

import re
import pandas as pd
import numpy as np


file = askopenfilename()
filename = file.split('/')[-1]
folder = '/'.join(file.split('/')[:-1])
participant_regex = r'P\d+'
# participant_name = re.search(r'\d+(?=(\s)?[wW])', filename).group(0)

df = pd.read_excel(file)

participant_name = df.iat[1,1]

start_rest = -1
start_warmup = -1
start_exercise = -1
length = len(df['Phase'])

for i in range(1, length):
    phase = df['Phase'][i]
    print(i, phase)
    if phase == 'REST' and start_rest == -1:
        start_rest = i
    elif phase == 'WARMUP' and start_warmup == -1:
        start_warmup = i
    elif phase == 'EXERCISE' and start_exercise == -1:
        start_exercise = i

ALL_SLICE = slice(start_rest, length)
REST_SLICE = slice(start_rest, start_warmup)
WARMUP_SLICE = slice(start_warmup, start_exercise)
EXERCISE_SLICE = slice(start_exercise, length)

VE = df['VE']
VO2 = df['VO2']
VCO2 = df['VCO2']

VEVO2 = df['VE/VO2']
VEVCO2 = df['VE/VCO2']

PetO2 = df['PetO2']
PetCO2 = df['PetCO2']

watt = df['Power']

VE_EX = VE[EXERCISE_SLICE]
VO2_EX = VO2[EXERCISE_SLICE]
VCO2_EX = VCO2[EXERCISE_SLICE]

VEVO2_EX = VEVO2[EXERCISE_SLICE]
VEVCO2_EX = VEVCO2[EXERCISE_SLICE]

PetO2_EX = PetO2[EXERCISE_SLICE]
PetCO2_EX = PetCO2[EXERCISE_SLICE]

def calculate_VO2Max(VO2):
    rolling = []
    max = 0
    for index, val in enumerate(VO2):
        if len(rolling) <= 3:
            rolling.append(val)
        if len(rolling) == 4:
            rolling.pop(0)
            val = np.mean(rolling)
            if val > max:
                max = val
    return max


VO2Max = calculate_VO2Max(VO2)
max_watt = watt[ALL_SLICE].max()