import math
import pandas as pd
from lxml import etree
import uuid
from pathlib import Path
from tkinter.filedialog import askopenfilename

def last_edit(parent):
    etree.SubElement(parent, 'LastEdit').text = '635621472000000000'


def get_protocol(sequence, name, boundaries):
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


# --- Keep all imports, last_edit, and get_protocol as is ---

# Replace the old create_interval_protocol_sequence with this:

def create_interval_protocol_sequence(power_80pct, power_delta70):
    """
    Create the Interval Protocol sequence with corrected phase codes:
    - 2 min rest (0 Watt) → REST
    - 3.5 min warm up (80% GET) → WARMING-UP
    - 10x (2:00 25 Watt / 1:00 Δ70%) → RECOVERY / EXERCISE
    - 1:00 25 Watt → RECOVERY
    """
    sequence = []
    boundaries = []
    
    time = 0
    
    # 2 min rest (0 Watt) - Phase 0 (REST)
    boundaries.append((time, 0))
    for _ in range(120):
        sequence.append(0)
        time += 1
    
    # 3.5 min warm up (80% GET) - Phase 1 (WARMING-UP)
    boundaries.append((time, 1))
    for _ in range(210):
        sequence.append(power_80pct)
        time += 1
    
    # 10x intervals (2:00 25W / 1:00 Δ70%)
    for interval in range(10):
        # 2:00 at 25W - Phase 5 (RECOVERY)
        boundaries.append((time, 5))
        for _ in range(120):
            sequence.append(25)
            time += 1
        
        # 1:00 at Δ70% - Phase 4 (EXERCISE)
        boundaries.append((time, 4))
        for _ in range(60):
            sequence.append(power_delta70)
            time += 1
    
    # Final 1:00 at 25W - Phase 5 (RECOVERY)
    boundaries.append((time, 5))
    for _ in range(60):
        sequence.append(25)
        time += 1
    
    return sequence, boundaries

# --- Keep your run() function as is ---


def run():
    file = askopenfilename()
    filename = file.split('/')[-1]
    participant_folder = '/'.join(file.split('/')[:-2])
    participant_name = filename.split(' ')[-1].split('.')[0]

    df = pd.read_excel(file)
    
    # Read power values directly from Excel file
    # Row 36, Col 4: 80% GET
    # Row 49, Col 4: 60%Δ
    power_80pct = round(df.iat[36, 4])
    power_delta70 = round(df.iat[27, 4])
    
    # Create Interval Protocol
    sequence, boundaries = create_interval_protocol_sequence(power_80pct, power_delta70)
    name = f'Interval Protocol {participant_name}'
    protocol = get_protocol(sequence, name, boundaries)
    
    folder = f'{participant_folder}/Interval/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    protocol.write(f'{folder}{name}.xml', pretty_print=True, encoding='utf-8', xml_declaration=True)
    
    print(f"Interval Protocol created successfully for {participant_name}")
    print(f"Power zones used:")
    print(f"  80% GET: {power_80pct}W")
    print(f"  Δ70%: {power_delta70}W")
    print(f"File saved to: {folder}{name}.xml")