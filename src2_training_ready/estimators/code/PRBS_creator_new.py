import math
import pandas as pd
from lxml import etree
import uuid
import openpyxl
from pathlib import Path
import numpy as np

from tkinter.filedialog import askopenfilename

from src2_training_ready.data_types.base import Boundaries
from src2_training_ready.prbs import get_sequencev2_with_step, get_sequencev2, get_sequencev2_with_step_boundaries, get_sequencev2_boundaries

def last_edit(parent):
    etree.SubElement(parent, 'LastEdit').text = '635621472000000000'


def get_protocol(sequence, name, boundaries: Boundaries):

    root = etree.Element('COSMED_OMNIA_EXPORT')
    information = etree.SubElement(root, 'Information')
    export = etree.SubElement(information, 'Export')
    export.set('Version', '2.3')
    export.set('Type', 'ErgoProtocol')
    export.set('S_Data', '281474976776197')
    export.text = 'cLWiPw==$NKDvFZox/85BnzcEwLxxMA==$sXnx0yFFXsKUUGSnFGU6rA=='

    protocol = etree.SubElement(root, 'ErgoProtocol')
    protocol_id = str(uuid.uuid4())
    etree.SubElement(protocol, 'RecordID').text = protocol_id
    last_edit(protocol)
    etree.SubElement(protocol, 'Protected').text = 'False'
    etree.SubElement(protocol, 'Name').text = name
    etree.SubElement(protocol, 'ProtocolType').text = '0'
    etree.SubElement(protocol, 'ErgometerType').text = '1'
    etree.SubElement(protocol, 'RMR').text = 'False'
    etree.SubElement(protocol, 'EOT').text = 'False'

    boundary_times = [x[0] for x in boundaries]
    exercise_types = [x[1] for x in boundaries]

    current_exercise_type = None
    prev_step = None
    for i, val in enumerate(sequence):
        if val != prev_step or i in boundary_times or i == len(sequence) - 1:
            if i in boundary_times:
                boundary_index = boundary_times.index(i)
                current_exercise_type = exercise_types[boundary_index]

            prev_step = val
            step = etree.SubElement(protocol, 'ErgoProtocolStep')
            etree.SubElement(step, 'RecordID').text = str(uuid.uuid4())
            last_edit(step)
            etree.SubElement(step, 'protocolID').text = protocol_id
            etree.SubElement(step, 'Time').text = str(i) if i != len(sequence) - 1 else str(i + 1)
            etree.SubElement(step, 'Load1').text = str(val)
            etree.SubElement(step, 'Load2').text = '0'
            etree.SubElement(step, 'Load3').text = '0'

            etree.SubElement(step, 'Phase').text = str(current_exercise_type)

            etree.SubElement(step, 'BP').text = 'False'
            etree.SubElement(step, 'FV').text = 'False'
            etree.SubElement(step, 'ECG').text = 'False'
            etree.SubElement(step, 'ABG').text = 'False'
            etree.SubElement(step, 'SPO2').text = 'False'

    tree = etree.ElementTree(root)
    return tree


def modify_warmup_in_sequence(sequence, power_80pct):
    """
    Modify the warm-up period in an existing PRBS sequence:
    - Keeps 0-120s (2 min) unchanged
    - Changes 120-330s (3.5 min) to 80% GET
    - Keeps everything after 330s unchanged
    """
    # Make a copy to avoid modifying the original
    modified_sequence = sequence.copy()
    
    # Replace values from second 120 to 330 (indices 120:330) with 80% GET
    modified_sequence[120:330] = power_80pct
    
    return modified_sequence


def run():
    file = askopenfilename()
    filename = file.split('/')[-1]
    participant_folder = '/'.join(file.split('/')[:-2])
    participant_name = filename.split(' ')[-1].split('.')[0]

    df = pd.read_excel(file)
    
    # Read values directly from Excel
    # Row 33, Col 4: 40%Δ (D40)
    # Row 36, Col 4: 80% GET
    D40 = round(df.iat[33, 4])
    power_80pct = round(df.iat[36, 4])

    # Create PRBS Protocol (the "Medium" protocol from original code)
    # Get the original PRBS sequences for 25W to D40
    sequence_with_step = get_sequencev2_with_step(25, D40, D40)
    sequence = get_sequencev2(25, D40)
    
    # Modify only the warm-up period (120-330s) to be 80% GET
    sequence_with_step_modified = modify_warmup_in_sequence(sequence_with_step, power_80pct)
    sequence_modified = modify_warmup_in_sequence(sequence, power_80pct)
    
    # Use the original boundaries (no changes needed)
    name_with_step = f'PRBSV2 STEP {participant_name}'
    protocol_with_step = get_protocol(sequence_with_step_modified, name_with_step, get_sequencev2_with_step_boundaries())

    name = f'PRBSV2 {participant_name}'
    protocol = get_protocol(sequence_modified, name, get_sequencev2_boundaries())

    # Save protocols
    folder = f'{participant_folder}/PRBS/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    protocol_with_step.write(f'{folder}{name_with_step}.xml', pretty_print=True, encoding='utf-8',
                             xml_declaration=True)
    protocol.write(f'{folder}{name}.xml', pretty_print=True, encoding='utf-8', xml_declaration=True)
    
    print(f"PRBS Protocol created successfully for {participant_name}")
    print(f"Protocol length: {len(sequence_modified)} seconds ({len(sequence_modified)/60:.1f} minutes)")
    print(f"Power zones used:")
    print(f"  Warm-up (2:00-5:30): {power_80pct}W (80% GET)")
    print(f"  PRBS range: 25W to {D40}W (40%Δ)")
    print(f"Files saved to: {folder}")