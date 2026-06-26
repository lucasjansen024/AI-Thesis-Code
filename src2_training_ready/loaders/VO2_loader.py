import dataclasses
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List
from enum import StrEnum
from xml.etree import ElementTree as ET

from src2_training_ready.data_types.base import ExercisePhase
from src2_training_ready.utils import Utils


class VO2Keys(StrEnum):
    TIME = 't'
    #WR = 'WR'
    VO2 = 'VO2'
    VO2kg = 'VO2/kg'
    VCO2 = 'VCO2'
    VE = 'VE'
    VEVO2 = 'VEVO2'
    VEVCO2 = 'VEVCO2'
    PetO2 = 'PetO2'
    PetCO2 = 'PetCO2'
    Rf = 'Rf'
    HR = 'HR'
    Phase = 'Phase'
    WR = 'Power'

class HHbKeys(StrEnum):
    """Keys for HHb NIRS data"""
    HHb = 'HHb'  # Column M (3rd channel)
    EVENT = 'Event'  # Column AB for synchronization
    TIME = 'Time'  # If time column exists

class HHbLoader:
    """Loader for HHb NIRS data from Excel files"""
    
    def __init__(self, file_path, sample_rate_hz=100):
        """
        Initialize HHb loader
        
        Args:
            file_path: Path to the NIRS Excel file
            sample_rate_hz: Sample rate of HHb data (default 100Hz)
        """
        self.file_path = Path(file_path)
        self.sample_rate_hz = sample_rate_hz
        self.df = None
        self.event_index = None
        self.synchronized_data = None
        
        # Load and process the data
        self._load_data()
        self._find_event()
        
    def _load_data(self):
        """Load HHb data from Excel file"""
        try:
            # Load the Excel file
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded HHb data with shape: {self.df.shape}")
            print(f"Columns available: {list(self.df.columns)}")
            
            # Get column indices (Excel columns start from A=0, so M=12, AB=27)
            hhb_col_index = 12  # Column M (0-indexed)
            event_col_index = 27  # Column AB (0-indexed)
            
            # Extract HHb data (Column M)
            if self.df.shape[1] > hhb_col_index:
                self.df[HHbKeys.HHb] = self.df.iloc[:, hhb_col_index]
                print(f"HHb data extracted from column {hhb_col_index+1} (Column M)")
            else:
                raise ValueError(f"File doesn't have enough columns. Expected at least {hhb_col_index+1}, got {self.df.shape[1]}")
            
            # Extract Event data (Column AB)
            if self.df.shape[1] > event_col_index:
                self.df[HHbKeys.EVENT] = self.df.iloc[:, event_col_index]
                print(f"Event data extracted from column {event_col_index+1} (Column AB)")
            else:
                raise ValueError(f"File doesn't have enough columns for events. Expected at least {event_col_index+1}, got {self.df.shape[1]}")
                
        except Exception as e:
            print(f"Error loading HHb data: {e}")
            raise
    
    def _find_event(self):
        """Find the A1 event in the event column"""
        try:
            # Look for 'A1' event in the event column
            event_column = self.df[HHbKeys.EVENT]
            event_column_stripped = event_column.astype(str).str.strip()
            a1_indices = event_column_stripped[event_column_stripped == 'A1'].index
            # Find where A1 event occurs
            
            if len(a1_indices) > 0:
                self.event_index = a1_indices[0]  # Take the first occurrence
                print(f"Found A1 event at index: {self.event_index}")
            else:
                # Try different variations of the event name
                variations = ['A1', 'a1', 'A 1', 'a 1', 'A1 ', ' A1', ' A1 ']
                for variation in variations:
                    indices = event_column[event_column == variation].index
                    if len(indices) > 0:
                        self.event_index = indices[0]
                        print(f"Found event '{variation}' at index: {self.event_index}")
                        break
                
                if self.event_index is None:
                    # Print available events for debugging
                    unique_events = event_column.dropna().unique()
                    print(f"Available events in column: {unique_events}")
                    raise ValueError("A1 event not found in event column")
                    
        except Exception as e:
            print(f"Error finding A1 event: {e}")
            raise
    
    def get_synchronized_data(self, sequence_duration_seconds):
        """
        Get HHb data synchronized from the A1 event
        
        Args:
            sequence_duration_seconds: Duration of the sequence to extract in seconds
            
        Returns:
            numpy array of HHb data for the specified duration
        """
        if self.event_index is None:
            raise ValueError("A1 event not found, cannot synchronize data")
        
        # Calculate number of samples needed
        num_samples = int(sequence_duration_seconds * self.sample_rate_hz)
        
        # Extract data starting from the event
        start_index = self.event_index
        end_index = start_index + num_samples
        
        if end_index > len(self.df):
            print(f"Warning: Requested sequence extends beyond available data")
            print(f"Available data: {len(self.df)} samples, requested: {end_index}")
            end_index = len(self.df)
        
        synchronized_data = self.df[HHbKeys.HHb].iloc[start_index:end_index].values
        
        print(f"Extracted {len(synchronized_data)} HHb samples starting from A1 event")
        print(f"Sequence duration: {len(synchronized_data) / self.sample_rate_hz:.1f} seconds")
        
        self.synchronized_data = synchronized_data
        return synchronized_data

    def get_data_by_rows(self, start_row, end_row):
        """
        Get HHb data between specific row indices
        
        Args:
            start_row: Starting row index (0-based)
            end_row: Ending row index (0-based, exclusive)
            
        Returns:
            numpy array of HHb data for the specified row range
        """
        if start_row < 0 or end_row > len(self.df):
            raise ValueError(f"Row indices out of bounds. Data has {len(self.df)} rows.")
        
        if start_row >= end_row:
            raise ValueError("Start row must be less than end row")
        
        hhb_data = self.df[HHbKeys.HHb].iloc[start_row:end_row].values
        
        print(f"Extracted {len(hhb_data)} HHb samples from rows {start_row} to {end_row-1}")
        
        return hhb_data


@dataclasses.dataclass
class ParticipantDetails:
    first_name: str
    last_name: str


PhaseSlices = Dict[ExercisePhase, List[slice]]

class VO2Loader:

    def __init__(
            self,
            file_path: Path,
            key_map: Dict[VO2Keys, str] = None
    ):
        self.file_path = file_path
        self.key_map = key_map
        self.is_xml = file_path.suffix == '.xml'
        self.complete_df = self._load_file(file_path)
        self.df, self.df_units = self._extract_df(self.complete_df.copy())
        self.details: ParticipantDetails = self._extract_details(self.complete_df.copy())
        self._phase_synonyms = {
            ExercisePhase.REST: ['None', 'Ruhe'],
            ExercisePhase.WARMUP: ['Erwärmung', 'Unloaded Pedalling'],
            ExercisePhase.EXERCISE: ['Belastung'],
            ExercisePhase.RECOVERY: ['Erholung']
        }
        self.slices: PhaseSlices = self._extract_slices(self.df)

    def _load_file(self, file_path: Path):
        if self.is_xml:
            tree = ET.parse(file_path)

            ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}

            # Find all tables within the XML
            tables = tree.findall(".//ss:Table", namespaces=ns)

            # Process each table
            dfs = []
            for table in tables:
                rows = []
                for row in table.findall("ss:Row", namespaces=ns):
                    cells = [cell.text if cell.text is not None else "" for cell in
                             row.findall("ss:Cell/ss:Data", namespaces=ns)]
                    rows.append(cells)

                df = pd.DataFrame(rows)
                dfs.append(df)

            # Print the first extracted DataFrame
            if not dfs:
                return None

            df = dfs[0]
            return df
        else:
            return pd.read_excel(file_path)

    def _rename_columns(self, df):
        synonyms = {
            VO2Keys.TIME: ['time'],
            VO2Keys.WR: ['Power', 'P'],
            VO2Keys.VO2: ["V'O2"],
            VO2Keys.VCO2: ["V'CO2"],
            VO2Keys.VE: ["V'E"],
            VO2Keys.VEVO2: ["VE/VO2", "V'E/V'O2"],
            VO2Keys.VEVCO2: ["VE/VCO2", "V'E/V'CO2"],
            VO2Keys.Rf: ["BF", "AF"],
            VO2Keys.HR: ['HF'],
            VO2Keys.VO2kg: ["V'O2/kg"]
        }

        df.rename(
            columns={
                synonym: key.value
                for key, synonyms in synonyms.items()
                for synonym in synonyms
            },
            inplace=True
        )

    def _extract_df(self, df) -> Tuple[pd.DataFrame, Dict[VO2Keys, str]]:
        df_units = None

        if self.is_xml:
            header_row = df[df.iloc[:, 1].eq('Phase')].index[0]
            df_columns = df.iloc[header_row]
            df_units = df.iloc[header_row + 1].to_dict()
            df = df.iloc[header_row + 2:]
            df.columns = df_columns  # set the header row as the df header

        self._rename_columns(df)

        if self.is_xml:
            df_units = { k: v for k, v in zip(df.columns, df_units.values()) }
            df[VO2Keys.TIME] = df[VO2Keys.TIME].apply(Utils.VO2_string_to_sec)

        else:
            cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            df.drop(df.columns[cols], axis=1, inplace=True)
            df_units = df.iloc[0].to_dict()
            df = df.iloc[2:]
            df[VO2Keys.TIME] = df[VO2Keys.TIME].apply(Utils.VO2_time_to_sec)

        df.reset_index(drop=True, inplace=True)
        df.dropna(axis=1, inplace=True, how='all')
        for col in df.columns:
            if col != VO2Keys.Phase:
                print(f"Col: {col}, {df[col]}")
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if VO2Keys.VCO2 not in df:
            df[VO2Keys.VCO2] = df[VO2Keys.VE] / df[VO2Keys.VEVCO2]

        if self.key_map is not None:
            df.rename(
                columns={
                    key: self.key_map[key]
                    for key in VO2Keys
                    if key in df.columns
                },
                inplace=True
            )

        return df, df_units

    def _extract_details(self, df) -> ParticipantDetails:
        if self.is_xml:
            first_name_row = df[df.iloc[:, 0].isin(['First Name', 'Vorname'])].index[0]
            first_name = df.iat[first_name_row, 1]
            last_name_row = df[df.iloc[:, 0].isin(['Last Name', 'Nachname'])].index[0]
            last_name = df.iat[last_name_row, 1]

        else:
            first_name = df.iat[1, 1]
            last_name = df.iat[0, 1]

        return ParticipantDetails(first_name, last_name)

    def _extract_slices(self, df) -> PhaseSlices:
        phase = df[VO2Keys.Phase]
        slices = {}

        current = None
        start = 0
        for i, p in enumerate(phase):
            for (k, v) in self._phase_synonyms.items():
                if p == k.name or p in v:
                    p = k

            if current is None:
                current = p

            if current != p or i == len(phase) - 1:
                if current not in slices:
                    slices[current] = []

                slices[current].append(slice(start, i))
                current = p
                start = i

        return slices

        # for i in range(1, length):
        #     phase = df['Phase'][i]
        #     print(i, phase)
        #     if phase == 'REST' and start_rest == -1:
        #         start_rest = i
        #     elif phase == 'WARMUP' and start_warmup == -1:
        #         start_warmup = i
        #     elif phase == 'EXERCISE' and start_exercise == -1:
        #         start_exercise = i
        #
        # ALL_SLICE = slice(start_rest, length)
        # REST_SLICE = slice(start_rest, start_warmup)
        # WARMUP_SLICE = slice(start_warmup, start_exercise)
        # EXERCISE_SLICE = slice(start_exercise, length)






