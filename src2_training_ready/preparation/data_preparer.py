import dataclasses
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import re
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src2_training_ready.data_types.base import Dataset
from src2_training_ready.data_types.file_keys import NIRSFileKeys, VO2Keys
from src2_training_ready.prbs import get_sequencev2, get_sequencev2_with_step, get_sequence, get_ramp_sequence
from src2_training_ready.preparation.config import config_lucas
from src2_training_ready.data_types.visit_config import HRSource, NIRSRepetition
from src2_training_ready.utils import Utils
from src2_training_ready.constants import *
from src2_training_ready.loaders.VO2_loader import VO2Loader
from src2_training_ready.loaders.NIRS_loader import NIRSLoader
from src2_training_ready.loaders.VE_loader import VELoader
from src2_training_ready.protocols import interval_sequence, step_sequence

VISIT_ORDER = {
        "P01": ["Ramp", "Interval", "Step", "PRBS"],
        "P02": ["Ramp", "PRBS", "Interval", "Step"],
        "P03": ["Ramp", "Interval", "Step", "PRBS"],
        "P04": ["Ramp", "Step", "Interval", "PRBS"],
        "P05": ["Ramp", "PRBS", "Interval", "Step"],
        "P06": ["Ramp", "Interval", "PRBS", "Step"],
        "P07": ["Ramp", "Step", "Interval", "PRBS"],
        "P08": ["Ramp", "Step", "Interval", "PRBS"],
        "P09": ["Ramp", "Interval", "PRBS", "Step"],
        "P10": ["Ramp", "PRBS", "Step", "Interval"],
        "P12": ["Ramp", "Step", "PRBS", "Interval"],
        "P13": ["Ramp", "PRBS", "Step", "Interval"],
        "P14": ["Ramp", "Step", "PRBS", "Interval"],
    }
@dataclasses.dataclass
class HR:
    rest: float
    peak: float

class Preparer:

    def __init__(self):
        self.HR_values: Dict[str, Dict[str, Dict[str, HR]]] = {}

        self.current_dataset: Dataset = ''
        self.current_participant = ''
        self.current_visit = ''

    @property
    def current_prepared_participant(self):
        return self.current_participant

    @property
    def current_raw_path(self):
        return RAW_PATH / self.current_dataset / self.current_participant / self.current_visit

    @property
    def current_prepared_p_folder(self):
        return PREPARED_PATH / self.current_prepared_participant

    def _NIRS_event_to_sec(self, event_string):
        split = event_string.split('min')
        minutes = int(split[0])
        if len(split) > 1 and len(split[1]) > 0:
            seconds = int(re.search(r'\d+', split[1]).group(0))
        else:
            seconds = 0
        time = minutes * 60 + seconds
        return time

    def _get_NIRS_df_using_event(self, NIRS_loader: NIRSLoader):
        NIRS_df = NIRS_loader.df
        event_row_index = np.where(NIRS_df[NIRSFileKeys.Event].notnull())[0][0]
        event_text = NIRS_df.at[event_row_index, NIRSFileKeys.Event_time]
        event_time = self._NIRS_event_to_sec(event_text)
        NIRS_start = event_row_index - event_time * NIRS_loader.details.frequency
        
        if NIRS_start < 0:
            empty_rows = pd.DataFrame([[np.nan]*len(NIRS_df.columns)]*abs(NIRS_start), columns=NIRS_df.columns)
            NIRS_df=pd.concat([empty_rows, NIRS_df],ignore_index=True)
            NIRS_start = 0

        NIRS_df = NIRS_df[NIRS_start:]
        NIRS_df.reset_index(inplace=True, drop=True)
        return NIRS_df, event_time

    def _truncate_ramp_df(self, df):
        if 'RPM' not in df.columns:
            print(f"ERROR: RPM column missing in {self.current_participant}/{self.current_visit}")
            return df
        
        last_valid_index = df[df['RPM'] >= 60].index.max() + 1
        df = df[:last_valid_index]
        return df

    def _split_df_with_step(self, df, breath_by_breath=False, frequency=1):
        if breath_by_breath:
            last_step_index = df[df['t'] <= TOTAL_STEP_TIME * frequency].index.max()
            last_prbs_index = df[df['t'] <= TOTAL_V2_TIME * frequency].index.max()
            step_df = df[:last_step_index + 2]
            prbs_df = df[last_step_index:last_prbs_index + 2]
            prbs_df['t'] = prbs_df['t'] - TOTAL_STEP_TIME
        else:
            step_df = df[:TOTAL_STEP_TIME * frequency]
            prbs_df = df[TOTAL_STEP_TIME * frequency:TOTAL_V2_TIME * frequency]
        prbs_df.reset_index(inplace=True, drop=True)
        return step_df, prbs_df

    def prepare_data_Lucas(self):
        print(f"Prepare Lucas' Data")
        self.current_dataset = Dataset.Lucas
        self.HR_values[self.current_dataset] = {}
        lucas_dir = RAW_PATH / self.current_dataset
        
        # Filter to only process directories (participant folders)
        for participant in os.listdir(lucas_dir):
            participant_path = lucas_dir / participant

            # Skip if not a directory
            if not participant_path.is_dir():
                continue

            # Only process participant folders like P01, P02, ...
            if not Utils.is_participant(participant):
                continue
            
            # NOTE: We do NOT hard-skip participants anymore.
            # Skipping is handled per-visit based on whether prepared output already exists.

                
            self.current_participant = participant
            self.HR_values[self.current_dataset][participant] = {}
            self._create_fit_files(participant_path)


            self._create_files(participant_path)


        self._create_HR_files("Lucas")

    def plot_prepared_data(self):
        print(f"Plotting prepared data")


    def _get_WR_df(self, VO2_loader: VO2Loader, visit) -> pd.DataFrame:
        """Create a clean 1 Hz WR signal.

        For Lucas we prefer *ideal* protocol WR sequences (rest/warm-up/steps exactly on whole seconds),
        with the actual participant-specific powers inferred from the measured WR column.

        If inference fails, we fall back to the measured WR (interpolated to 1 Hz).
        """
        # Interpolate measured WR to a 1 Hz grid (0..max_t)
        if VO2Keys.WR not in VO2_loader.df.columns:
            print(f"WARNING: WR column not found for {self.current_participant}/{self.current_visit}")
            print(f"Available columns: {VO2_loader.df.columns.tolist()}")
            return pd.DataFrame([0] * len(VO2_loader.df), columns=['WR'])

        VO2_WR = VO2_loader.df[[VO2Keys.WR, 't']].copy()
        # Interpolate WR to a clean 1 Hz grid, but *keep* a usable time base.
        # NOTE: Utils.interpolate_dataframe() drops the time column, so we don't use it here.
        VO2_WR['t'] = pd.to_numeric(VO2_WR['t'], errors='coerce')
        VO2_WR[VO2Keys.WR] = pd.to_numeric(VO2_WR[VO2Keys.WR], errors='coerce')
        VO2_WR = VO2_WR.dropna(subset=['t']).sort_values('t')
        # Drop duplicate time stamps
        VO2_WR = VO2_WR.drop_duplicates(subset='t', keep='first')
        t_vals = VO2_WR['t'].to_numpy(dtype=float)
        p_vals = VO2_WR[VO2Keys.WR].to_numpy(dtype=float)
        if t_vals.size == 0:
            max_t = 0
            measured = pd.Series([0.0])
        else:
            max_t = int(np.nanmax(t_vals))
            grid = np.arange(0, max_t + 1, 1, dtype=float)
            # Fill missing powers with nearest valid points for interpolation
            if np.isnan(p_vals).all():
                measured = pd.Series([0.0] * (max_t + 1))
            else:
                # Replace NaNs by forward/back fill before interpolation
                p_series = pd.Series(p_vals).interpolate(limit_direction='both').to_numpy(dtype=float)
                measured = pd.Series(np.interp(grid, t_vals, p_series))
        measured = measured.iloc[:max_t + 1].reset_index(drop=True)


        visit_str = str(visit)

        # Standardize length for submax protocols (no cool-down): 36:30 = SUBMAX_TOTAL_TIME seconds.
        # We keep Ramp length as measured/truncated ramp length.
        is_submax = any(k in visit_str for k in ("Interval", "Step", "PRBS"))
        if is_submax:
            target_len = int(SUBMAX_TOTAL_TIME)
            # measured currently spans 0..max_t; truncate/pad later to match the fixed protocol length
            measured = measured.iloc[:min(len(measured), target_len)].reset_index(drop=True)
            max_t = target_len - 1

        # --- Ramp: use configured ramp rate (or parse from header) to build the *ideal* ramp sequence.
        if 'Ramp' in visit_str:
            VO2_df = self._truncate_ramp_df(VO2_loader.df)
            ramp_len = int(np.nanmax(VO2_df['t'].values))

            ramp_rate = None
            try:
                ramp_rate = config_lucas.get(self.current_participant, {}).get('Ramp', {}).get('ramp_rate', None)
            except Exception:
                ramp_rate = None

            if ramp_rate is None:
                # Fallback: try extracting from protocol string in the BxB export
                try:
                    protocol = str(VO2_loader.complete_df.iloc[11, 4])
                    m = re.search(r'(\d{2})', protocol)
                    if m:
                        ramp_rate = int(m.group(1))
                except Exception:
                    ramp_rate = None

            if ramp_rate is None:
                print(f"WARNING: Ramp rate missing for {self.current_participant}. Falling back to measured WR.")
                WR = measured.iloc[:ramp_len + 1].to_numpy()
            else:
                WR = get_ramp_sequence(int(ramp_rate))[:ramp_len + 1]

            return pd.DataFrame(WR, columns=['WR'])

        # Helper: robust median for a window
        def _median_window(t0: int, t1: int):
            if t1 <= t0:
                return None
            if t0 >= len(measured):
                return None
            t1_ = min(t1, len(measured))
            vals = measured.iloc[t0:t1_].dropna().to_numpy()
            if vals.size == 0:
                return None
            return int(round(float(np.nanmedian(vals))))

        # Common warm-up power (80% GET) is the same for all three submax protocols
        warmup_power = _median_window(REST_TIME, REST_TIME + PRBS_WARMUP_TIME)

        def _pad_to_len(seq, target_len: int):
            """Pad or truncate a python list/np array to target_len using edge padding."""
            if seq is None:
                return None
            if hasattr(seq, 'tolist'):
                seq = seq.tolist()
            if len(seq) >= target_len:
                return seq[:target_len]
            if len(seq) == 0:
                return [0] * target_len
            return seq + [seq[-1]] * (target_len - len(seq))

        # --- Interval
        if 'Interval' in visit_str:
            # Estimate the Δ60% power from the expected 1:00 high segments.
            start = REST_TIME + PRBS_WARMUP_TIME
            highs = []
            for i in range(10):
                # Each cycle: 120s low + 60s high
                hi0 = start + i * 180 + 120
                hi1 = hi0 + 60
                v = _median_window(hi0 + 5, hi1 - 5)  # trim edges for transitions
                if v is not None:
                    highs.append(v)
            delta70 = int(round(float(np.nanmedian(highs)))) if len(highs) else None

            if warmup_power is None or delta70 is None:
                WR = _pad_to_len(measured.to_numpy(), max_t + 1)
            else:
                WR = _pad_to_len(interval_sequence(warmup_power, delta70), max_t + 1)

            return pd.DataFrame(WR, columns=['WR'])

        # --- Step
        if 'Step' in visit_str:
            start = REST_TIME + PRBS_WARMUP_TIME

            p80 = warmup_power or _median_window(start, start + 360)
            p_d30 = _median_window(start + 360, start + 720)
            p80_2 = _median_window(start + 720, start + 1080)
            p_d60 = _median_window(start + 1080, start + 1260)
            p25 = _median_window(start + 1260, start + 1560)
            p60 = _median_window(start + 1560, start + 1860)

            # Prefer p80 measured in-protocol if warmup was noisy
            if p80 is None and p80_2 is not None:
                p80 = p80_2

            # p25 should be exactly 25 W; if the estimate looks off, clamp.
            if p25 is not None and abs(p25 - 25) <= 5:
                p25 = 25

            if None in (p80, p60, p_d30, p_d60):
                WR = _pad_to_len(measured.to_numpy(), max_t + 1)
            else:
                WR = _pad_to_len(step_sequence(int(p80), int(p60), int(p_d30), int(p_d60)), max_t + 1)

            return pd.DataFrame(WR, columns=['WR'])

        # --- PRBS
        if 'PRBS' in visit_str:
            # Estimate upper plateau (40%Δ) as the median of values above ~30 W after warm-up.
            ex_start = REST_TIME + PRBS_WARMUP_TIME
            ex_vals = measured.iloc[ex_start:].dropna().to_numpy()
            upper = None
            if ex_vals.size:
                high = ex_vals[ex_vals > 30]
                if high.size:
                    upper = int(round(float(np.nanmedian(high))))

            if warmup_power is None or upper is None:
                WR = _pad_to_len(measured.to_numpy(), max_t + 1)
            else:
                WR = _pad_to_len(get_sequencev2(25, int(upper), warmup_power=int(warmup_power)), max_t + 1)

            return pd.DataFrame(WR, columns=['WR'])

        # Fallback: measured
        WR = measured.to_numpy()
        if 'is_submax' in locals() and is_submax:
            WR = _pad_to_len(WR, max_t + 1)
        return pd.DataFrame(WR, columns=['WR'])

    @staticmethod
    def _get_VE_file(visit_path: Path):
            """Find the VE(+HR) export file.
    
            We only consider actual data files (csv/xls/xlsx), because the visit folder may also
            contain plots (png) with 'VE' in the filename.
            """
            exts = {'.csv', '.xls', '.xlsx', '.xlsm'}
            candidates = []
            for f in os.listdir(visit_path):
                p = visit_path / f
                if not p.is_file():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                if 'VE' in f and ('BxB' not in f) and ('_NIRS' not in f) and ('Oxy' not in f) and ('10 sec' not in f):
                    candidates.append(p)
    
            if candidates:
                # Prefer CSV if available
                candidates_sorted = sorted(candidates, key=lambda x: (0 if x.suffix.lower()=='.csv' else 1, x.name.lower()))
                return candidates_sorted[0]
    
            # fallback: any VE-like data file
            for f in os.listdir(visit_path):
                p = visit_path / f
                if p.is_file() and p.suffix.lower() in exts and 'VE' in f:
                    return p
            return None
    def _create_files(self, participant_path):
        participant = self.current_participant
        for visit in os.listdir(participant_path):
            self.current_visit = visit
            visit_path = participant_path / self.current_visit

            if not visit_path.is_dir():
                continue

            # Skip visits that are already prepared (so you can re-run without re-processing old data)
            #output_dir = self.current_prepared_p_folder / visit
            #expected_out = output_dir / f"{participant}{visit}VO2BxB.csv"
            #if expected_out.exists():
            #    print(f"Skipping {participant}/{visit} (already prepared)")
            #    continue

            # if Visit.Medium not in visit:
            #     continue

            VO2_file = Utils.get_file(visit_path, 'BxB')
            if VO2_file is None:
                print(f"Skipping Fit for {visit}: no BxB file found in {visit_path}")
                continue
            VO2_loader = VO2Loader(VO2_file)

            NIRS_file = Utils.get_file(visit_path, "_NIRS")
            if NIRS_file is None:
                print(f"Missing NIRS file for {participant} in {visit}")
                continue

            VE_file = self._get_VE_file(visit_path)
            if VE_file is None:
                print(f"WARNING: Missing VE file for {participant} in {visit} (HR/VE will be missing)")

            VO2_df = VO2_loader.df

            # Load VE(+HR) from the dedicated VE export file (NOT from BxB).
            VE_df = None          # TymeWear breath-by-breath -> prepared VEBxB.csv
            VE_time_df = None     # TymeWear regular time-series -> prepared VE.csv
            if VE_file is not None:
                try:
                    VE_loader = VELoader(VE_file)
                    # VEBxB should come from the TymeWear breath-by-breath export columns.
                    VE_df = VE_loader.bxb_df.copy()
                    # VE.csv should come from the regular TymeWear time-series columns.
                    VE_time_df = VE_loader.ve_prepared_df.copy()
                    # HR stats still come from the regular time-based VE export (1 Hz-ish 'Time' column).
                    VE_hr_df = VE_loader.df
                    if 'HR' in VE_hr_df.columns:
                        rest_df = VE_hr_df[VE_hr_df['t'].le(REST_TIME)]
                        rest_HR = float(rest_df['HR'].mean())
                        peak_HR = float(VE_hr_df['HR'].max())
                        self.HR_values[self.current_dataset][participant][visit] = HR(rest=rest_HR, peak=peak_HR)
                except Exception as e:
                    print(f"WARNING: Failed to load VE file for {participant}/{visit}: {e}")
            output_dir = self.current_prepared_p_folder / visit
            expected_files = [
                output_dir / f"{participant}{visit}VO2BxB.csv",
                output_dir / f"{participant}{visit}Oxy.csv",
                output_dir / f"{participant}{visit}WR.csv",
                output_dir / f"{participant}{visit}VEBxB.csv",
                output_dir / f"{participant}{visit}VE.csv",
            ]

            if all(p.exists() for p in expected_files):
                print(f"Skipping writing files for {participant}/{visit} (already fully prepared)")
                continue
            WR_df = self._get_WR_df(VO2_loader, visit)

            NIRS_loader = NIRSLoader(NIRS_file)
            NIRS_df, event_time = self._get_NIRS_df_using_event(NIRS_loader)
           
            # Config is only used for occlusion slicing + repetition meta.
            # If a participant/visit isn't in config_lucas yet, we fall back to safe defaults.
            visit_config = config_lucas.get(self.current_participant, {}).get(
                self.current_visit,
                {
                    'HR_source': HRSource.HR,
                    'Occlusion': None,
                    'NIRS_repetition': NIRSRepetition.Both,
                }
            )
            occlusion = visit_config.get('Occlusion')
            Occlusion_df = None
            if occlusion is not None:
                Occlusion_df = NIRS_df[occlusion['start']:occlusion['end']]
                Occlusion_df.reset_index(inplace=True, drop=True)

            output_dir.mkdir(parents=True, exist_ok=True)

            # If your raw folders are separate protocols (Step / Interval / PRBS), we don't want the old
            # "split step+PRBS from one long file" logic.
            separate_protocol = any(k in str(visit) for k in ("Step", "Interval", "PRBS"))

            if Visit.Ramp in visit:
                VO2_df = self._truncate_ramp_df(VO2_df)
                duration_s = int(VO2_df['t'].max())
                self._plot_VO2_file(VO2_df, ramp_duration=duration_s, v2=True)
                self._plot_NIRS_file(
                    NIRS_df, event_time, NIRS_loader.details.frequency,
                    WR_df=WR_df, ramp_duration=duration_s, v2=True
                )
                NIRS_df = NIRS_df[:duration_s * NIRS_loader.details.frequency + 1]

                if VE_df is not None:
                    VE_df = VE_df[VE_df['t'].le(duration_s)].reset_index(drop=True)
                if VE_time_df is not None:
                    VE_time_df = VE_time_df[VE_time_df['Time'].le(duration_s)].reset_index(drop=True)

            else:
                if separate_protocol:
                    # New-style: each visit folder is already a single protocol
                    self._plot_VO2_file(VO2_df, v2=False)
                    self._plot_NIRS_file(
                        NIRS_df, event_time, NIRS_loader.details.frequency,
                        WR_df=WR_df, v2=False
                    )
                    # Truncate to the fixed submax protocol length (no cool-down): 36:30
                    duration_s = int(SUBMAX_TOTAL_TIME) - 1  # expected last second index
                    VO2_df = VO2_df[VO2_df['t'].le(duration_s)].reset_index(drop=True)
                    if VE_df is not None:
                        # Keep the final protocol second in VEBxB (inclusive upper bound, e.g. t=2190).
                        VE_df = VE_df[VE_df['t'].le(SUBMAX_TOTAL_TIME)].reset_index(drop=True)
                    if VE_time_df is not None:
                        # Same convention for VE.csv (time-based TymeWear export): keep Time=2190 if present.
                        VE_time_df = VE_time_df[VE_time_df['Time'].le(SUBMAX_TOTAL_TIME)].reset_index(drop=True)
                    # NIRS has no seconds column here; slice by samples (frequency Hz)
                    NIRS_df = NIRS_df[:duration_s * NIRS_loader.details.frequency + 1]
                    # WR is already built at 1 Hz; ensure consistent length
                    if len(WR_df) > int(SUBMAX_TOTAL_TIME):
                        WR_df = WR_df.iloc[:int(SUBMAX_TOTAL_TIME)].reset_index(drop=True)
                    elif len(WR_df) < int(SUBMAX_TOTAL_TIME):
                        pad_n = int(SUBMAX_TOTAL_TIME) - len(WR_df)
                        last_wr = float(WR_df['WR'].iloc[-1]) if len(WR_df) else 0.0
                        WR_df = pd.concat([WR_df, pd.DataFrame({'WR': [last_wr] * pad_n})], ignore_index=True)
                else:
                    # Legacy: one file contains STEP + PRBS, split them
                    self._plot_VO2_file(VO2_df, v2=True)
                    self._plot_NIRS_file(
                        NIRS_df, event_time, NIRS_loader.details.frequency,
                        WR_df=WR_df, v2=True
                    )
                    VO2_STEP_df, VO2_df = self._split_df_with_step(VO2_df, breath_by_breath=True)
                    NIRS_STEP_df, NIRS_df = self._split_df_with_step(NIRS_df, frequency=NIRS_loader.details.frequency)
                    WR_STEP_df, WR_df = self._split_df_with_step(WR_df)

                    VE_STEP_df = None
                    VE_time_STEP_df = None
                    if VE_df is not None:
                        VE_STEP_df = VE_df[VE_df['t'].le(TOTAL_STEP_TIME)].copy()
                        VE_df = VE_df[VE_df['t'].ge(TOTAL_STEP_TIME)].copy()
                        VE_df['t'] = VE_df['t'] - TOTAL_STEP_TIME
                        VE_df.reset_index(drop=True, inplace=True)
                    if VE_time_df is not None:
                        VE_time_STEP_df = VE_time_df[VE_time_df['Time'].le(TOTAL_STEP_TIME)].copy()
                        VE_time_df = VE_time_df[VE_time_df['Time'].ge(TOTAL_STEP_TIME)].copy()
                        VE_time_df['Time'] = VE_time_df['Time'] - TOTAL_STEP_TIME
                        VE_time_df.reset_index(drop=True, inplace=True)

                    step_output_dir = self.current_prepared_p_folder / f"step_{visit}"
                    step_output_dir.mkdir(parents=True, exist_ok=True)
                    # Drop unused/empty columns from VO2 STEP segment
                    drop_cols = ['Estimated METS', 'HR', 'VO2/HR', 'HRR']
                    VO2_STEP_df = VO2_STEP_df.drop(columns=drop_cols, errors='ignore')
                    VO2_STEP_df.to_csv(step_output_dir / f'{participant}{visit}StepVO2BxB.csv', index=False)
                    NIRS_STEP_df.to_csv(step_output_dir / f"{participant}{visit}StepOxy.csv", index=False)
                    WR_STEP_df.to_csv(step_output_dir / f"{participant}{visit}StepWR.csv", index=False)

                    if VE_STEP_df is not None:
                        VE_STEP_df.to_csv(step_output_dir / f"{participant}{visit}StepVEBxB.csv", index=False)
                    if VE_time_STEP_df is not None:
                        VE_time_STEP_df.to_csv(step_output_dir / f"{participant}{visit}StepVE.csv", index=False)

                    if Occlusion_df is not None:
                        Occlusion_df.to_csv(step_output_dir / f'{participant}{visit}Occlusion.csv', index=False)
            # Drop unused/empty columns from VO2 BxB
            drop_cols = ['Estimated METS', 'HR', 'VO2/HR', 'HRR']
            VO2_df = VO2_df.drop(columns=drop_cols, errors='ignore')
            VO2_df.to_csv(output_dir / f'{participant}{visit}VO2BxB.csv', index=False)
            NIRS_df.to_csv(output_dir / f"{participant}{visit}Oxy.csv", index=False)
            WR_df.to_csv(output_dir / f"{participant}{visit}WR.csv", index=False)

            if VE_df is not None:
                VE_df.to_csv(output_dir / f"{participant}{visit}VEBxB.csv", index=False)
            if VE_time_df is not None:
                VE_time_df.to_csv(output_dir / f"{participant}{visit}VE.csv", index=False)

            if Occlusion_df is not None:
                Occlusion_df.to_csv(output_dir / f'{participant}{visit}Occlusion.csv', index=False)

    
    def _create_fit_files(self, participant_path):
        # Debug: Print what files are available
        print(f"Processing participant: {participant_path}")
        #print(f"Available files/folders: {list(os.listdir(participant_path))}")
        
        fit_file_path = Utils.get_file(participant_path, 'Fit')
        if fit_file_path is None:
            print(f"No 'Fit' file found for {participant_path}")
            return
        
        print(f"Found fit file: {fit_file_path}")
        
        try:
            fit_df = None
            read_errors = []

            # 1) Probeer eerst als Excel (want jouw "csv" is vaak eigenlijk .xls)
            for reader in [
                lambda p: pd.read_excel(p, engine='xlrd'),   # old Excel .xls (ook als extensie fout is)
                lambda p: pd.read_excel(p),                  # xlsx/xlsm
                lambda p: pd.read_csv(p, sep=None, engine='python', encoding='utf-8-sig'),
                lambda p: pd.read_csv(p, sep=';', encoding='utf-8-sig'),
                lambda p: pd.read_csv(p, sep=';', encoding='latin1'),
                lambda p: pd.read_csv(p, sep=',', encoding='utf-8-sig'),
            ]:
                try:
                    tmp = reader(fit_file_path)
                    # simpele validatie: moet minstens iets van bekende kolommen bevatten
                    cols = {str(c).strip() for c in tmp.columns}
                    if any(c in cols for c in ['Date', 'Datum', 'Weight', 'Gewicht', 'Body Fat', 'Lichaamsvet']):
                        fit_df = tmp
                        break
                except Exception as e:
                    read_errors.append(str(e))

            if fit_df is None:
                print(f"Error reading fit file {fit_file_path}. Tried Excel/CSV readers. Errors: {read_errors}")
                return

            fit_df.columns = fit_df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)
            if 'Visit' in fit_df.columns:
                fit_df['Visit'] = fit_df['Visit'].astype(str).str.strip()

            print(f"Fit columns detected: {fit_df.columns.tolist()}")

        except Exception as e:
            print(f"Error reading fit file {fit_file_path}: {e}")
            return        # ... rest of your code


        if 'Visit' not in fit_df.columns:
            # Gebruik de visit-foldernamen direct (nieuwe Lucas-structuur)
            visit_folders = [
                v for v in os.listdir(participant_path)
                if (participant_path / v).is_dir() and Utils.get_file(participant_path / v, 'BxB') is not None
            ]
            # Handige volgorde: Ramp eerst, daarna Interval, Step, PRBS
            participant_id = str(self.current_participant)

            if participant_id in VISIT_ORDER:
                protocol_order = VISIT_ORDER[participant_id]

                def visit_sort_key(v):
                    for i, protocol in enumerate(protocol_order):
                        if protocol in str(v):
                            return i
                    return 99

                visit_folders = sorted(visit_folders, key=visit_sort_key)

            else:
                print(f"WARNING: No visit order defined for {participant_id}")

            # Koppel rijen in FitDays op volgorde aan je visit folders
            fit_df = fit_df.copy()
            fit_df['Visit'] = ''
            n = min(len(fit_df), len(visit_folders))
            fit_df.loc[:n-1, 'Visit'] = visit_folders[:n]

            if len(fit_df) != len(visit_folders):
                print(f"WARNING: aantal FitDays rijen ({len(fit_df)}) != aantal visit folders ({len(visit_folders)}) voor {participant_path}")

        for visit in os.listdir(participant_path):
            visit_path = participant_path / visit
            if not visit_path.is_dir():
                continue
            self.current_visit = visit
            VO2_file = Utils.get_file(visit_path, 'BxB')
            if VO2_file is None:
                print(f"Skipping Fit for {visit}: no BxB file found in {visit_path}")
                continue
            VO2_loader = VO2Loader(VO2_file)
            print("---- DEBUG BxB metadata ----")
            print(VO2_loader.complete_df.head(10))
            print("----------------------------")

            gender = VO2_loader.complete_df.iloc[2, 1]
            age = VO2_loader.complete_df.iloc[3, 1]
            height = VO2_loader.complete_df.iloc[4, 1]
            dob = VO2_loader.complete_df.iloc[6, 1]
            fit_file = fit_df[fit_df['Visit'].astype(str).str.strip() == str(visit).strip()].copy()
            if fit_file.empty:
                print(f"No Fit row matched for visit '{visit}' in {fit_file_path}. Available Visit values: {fit_df['Visit'].unique().tolist() if 'Visit' in fit_df.columns else 'NO Visit col'}")
                continue

            # GEEN merge op index -> gewoon metadata kolommen toevoegen
            fit_file = fit_file.reset_index(drop=True)
            fit_file['gender'] = gender
            fit_file['age'] = age
            fit_file['height'] = height
            fit_file['dob'] = dob

            def remove_percentage(x):
                return x.replace('%', '')

            def remove_kg(x):
                return x.replace('kg', '')


            fit_file.rename(columns={
                'Date': 'date',
                'Weight': 'weight',
                'Body Fat': 'body fat',
                'Subcutaneous fat': 'subcutaneous fat',
                'Visceral Fat': 'visceral fat',
                'Body Water': 'body water',
                'Skeletal muscle': 'skeletal muscle',
                'Muscle mass': 'muscle mass',
                'Bone Mass': 'bone mass',
                'Protein': 'protein',
                'Body age': 'body age',
                'Visit': 'visit',
                'BMR': 'bmr',
                'BMI': 'bmi',
                # Dutch
                'Datum': 'date',
                'Gewicht': 'weight',
                'Lichaamsvet': 'body fat',
                'Onderhuids vet': 'subcutaneous fat',
                'Visceraal Fat': 'visceral fat',
                'Lichaamswater': 'body water',
                'Skeletspier': 'skeletal muscle',
                'Spiermassa': 'muscle mass',
                'Botmassa': 'bone mass',
                'Eiwit': 'protein',
                'Lichaamsleeftijd': 'body age',
                'Hartslag': 'Heart rate',
                'Hart index': 'Cardiac Index'
            }, inplace=True)
            fit_file.drop(columns=['Heart rate', 'Cardiac Index'], inplace=True, errors='ignore')

            fit_file['gender'] = fit_file['gender'].apply(lambda x: 1 if x == 'Male' else 0)
            fit_file['weight'] = fit_file['weight'].apply(remove_kg).astype(float)
            fit_file['body fat'] = fit_file['body fat'].apply(remove_percentage).astype(float)
            fit_file['subcutaneous fat'] = fit_file['subcutaneous fat'].apply(remove_percentage).astype(float)
            fit_file['body water'] = fit_file['body water'].apply(remove_percentage).astype(float)
            fit_file['skeletal muscle'] = fit_file['skeletal muscle'].apply(remove_percentage).astype(float)
            fit_file['muscle mass'] = fit_file['muscle mass'].apply(remove_kg).astype(float)
            fit_file['bone mass'] = fit_file['bone mass'].apply(remove_kg).astype(float)
            fit_file['bmr'] = fit_file['bmr'].apply(lambda x: x.replace('kcal', '')).astype(float)

            output_dir = self.current_prepared_p_folder / visit
            output_dir.mkdir(parents=True, exist_ok=True)
            fit_file.to_csv(output_dir / f"{self.current_participant}{visit}Fit.csv", index=False)
    def _create_HR_files(self, dataset):
        print(F"Calculating HR files for {dataset}")
        self.current_dataset = dataset
        dataset_HRs = self.HR_values[dataset]
        HRs = {}
        ratios = []

        for participant in dataset_HRs:
            p_HRs = dataset_HRs[participant]
            p_HR = HR(200, 0)
            for visit in p_HRs:
                v_HR = p_HRs[visit]
                if v_HR.rest < p_HR.rest:
                    p_HR.rest = v_HR.rest
                if v_HR.peak > p_HR.peak:
                    p_HR.peak = v_HR.peak
            HRs[participant] = p_HR

            #if Visit.Hard in p_HRs and Visit.Ramp in p_HRs:
            #    HR_Hard = p_HRs[Visit.Hard]
            #    HR_Ramp = p_HRs[Visit.Ramp]
            #    ratios.append(HR_Ramp.peak / HR_Hard.peak)

        # Ramp/Hard peak HR ratio (legacy). If we don't have Hard/Ramp pairs (e.g. new protocols), keep it neutral.
        average_ratio = float(np.nanmean(ratios)) if len(ratios) > 0 else 1.0
        for participant in dataset_HRs:
            self.current_participant = participant
            p_HRs = dataset_HRs[participant]
            p_HR = HRs[participant]
            participant_path = RAW_PATH / dataset / participant

            if Visit.Ramp not in p_HRs:
                #If ramp is missing, estimate a peak HR.
                any_visit = None
                for v in os.listdir(participant_path):
                    if (participant_path / v).is_dir():
                        any_visit = v
                        break
                if any_visit is not None:
                    VO2_file_any = Utils.get_file(participant_path / any_visit, 'BxB')
                    if VO2_file_any is not None:
                        VO2_loader = VO2Loader(VO2_file_any)
                        age = VO2_loader.complete_df.iloc[3, 1]
                        peak_HR = 220 - float(age)
                    else:
                        peak_HR = p_HR.peak
                else:
                    peak_HR = p_HR.peak

                if peak_HR > p_HR.peak:
                    p_HR.peak = float(peak_HR)

            # Actually create HR files
            for visit in os.listdir(participant_path):
                self.current_visit = visit
                visit_path = participant_path / visit
                if not visit_path.is_dir():
                    continue
                VE_file_path = self._get_VE_file(visit_path)
                if VE_file_path is None:
                    continue

                try:
                    VE_loader = VELoader(VE_file_path)
                    VE_df = VE_loader.df
                except Exception:
                    continue

                if 'HR' not in VE_df.columns:
                    continue

                # Interpolate HR to 1 Hz on a 0..max_t grid
                max_t = int(np.nanmax(VE_df['t'].values))
                full_t = pd.DataFrame({'t': np.arange(0, max_t + 1, 1)})
                HR_source = VE_df[['t', 'HR']].dropna().drop_duplicates(subset='t').sort_values('t')
                HR_1hz = full_t.merge(HR_source, on='t', how='left')
                HR_1hz['HR'] = HR_1hz['HR'].interpolate(method='linear', limit_direction='both')

                HR_series = HR_1hz['HR']
                denom = (p_HR.peak - p_HR.rest) if (p_HR.peak - p_HR.rest) != 0 else np.nan
                HRR_series = (HR_series - p_HR.rest) / denom
                HR_df = pd.DataFrame({'HR': HR_series, 'HRR': HRR_series})

                if Visit.Ramp in visit:
                    # Truncate to ramp duration based on BxB file (more reliable for end-of-test)
                    VO2_file_path = Utils.get_file(visit_path, 'BxB')
                    if VO2_file_path is not None:
                        VO2_loader = VO2Loader(VO2_file_path)
                        Ramp_df = self._truncate_ramp_df(VO2_loader.df)
                        HR_df = HR_df[:int(Ramp_df['t'].max())]
                else:
                    separate_protocol = any(k in str(visit) for k in ("Step", "Interval", "PRBS"))
                    if (not separate_protocol) and len(HR_df) >= TOTAL_V2_TIME:  # Legacy combined step+PRBS file
                        HR_STEP_df, HR_df = self._split_df_with_step(HR_df)
                        step_output_dir = self.current_prepared_p_folder / f"step_{visit}"
                        HR_STEP_df.to_csv(step_output_dir / f"{participant}{visit}StepHR.csv", index=False)
                    else:
                        # For separate protocols, keep the full HR series length (no forced TOTAL_TIME truncation).
                        # If you *do* want truncation, you can add protocol-specific rules here.
                        pass

                output_dir = self.current_prepared_p_folder / visit
                HR_df.to_csv(output_dir / f"{participant}{visit}HR.csv", index=False)

        print(participant, p_HR.rest, p_HR.peak, p_HR.peak - p_HR.rest)


    def _plot_VO2_file(self, VO2_df, ramp_duration=-1, v2=False):
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, label='VO2')
        ax.plot(VO2_df['t'], VO2_df['VO2'], color='blue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('VO2 (mL/min)')

        if 'HR' in VO2_df.columns:
            ax2 = fig.add_subplot(111, label='HR', frame_on=False)
            ax2.yaxis.tick_right()
            ax2.plot(VO2_df['t'], VO2_df['HR'], color='red')
            ax2.yaxis.set_label_position('right')
            ax2.set_ylabel('HR (BPM)')
            ax2.set_ylim(50, 210)
            ax2.set_xlim(ax.get_xlim())

        title = f"VO2 for {self.current_participant} {self.current_visit} from {self.current_dataset}"
        plt.title(title)

        self._add_vlines(ax, 0, ramp_duration=ramp_duration, v2=v2, visit_name=self.current_visit)
        plt.savefig(f"{self.current_raw_path}/{title}.png")

        plt.show()

    def _plot_NIRS_file(self, NIRS_df, event_time, frequency, WR_df=None, ramp_duration=-1, v2=False):
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=2)
        title = f'NIRS for {self.current_participant} {self.current_visit} from {self.current_dataset}'
        fig.suptitle(title)

        axs[0, 0].plot(NIRS_df[NIRSFileKeys.TSI], color='green')  # TSI %
        axs[0, 0].plot(NIRS_df[NIRSFileKeys.fit_factor], color='blue')  # TSI Fit Factor
        axs[0, 0].set_ylim(0, 100.3)
        axs[0, 0].set_title('TSI')

        visit_config = config_lucas.get(self.current_participant, {}).get(
            self.current_visit,
            {
                'HR_source': HRSource.HR,
                'Occlusion': None,
                'NIRS_repetition': NIRSRepetition.Both,
            }
        )
        occlusion = visit_config.get('Occlusion')

        # ---- build WR time base in NIRS samples ----
        # ---- build WR time base in NIRS samples ----
        wr_samples = None
        if WR_df is not None and 'WR' in WR_df.columns and len(WR_df) > 0:
            wr = pd.to_numeric(WR_df['WR'], errors='coerce').ffill().fillna(0).to_numpy()
            t_wr = np.arange(len(wr)) * frequency  # seconds -> samples
            wr_samples = (t_wr, wr)
        for i, ax in enumerate(fig.axes):
            self._add_vlines(ax, 0, ramp_duration=ramp_duration, v2=v2, frequency=frequency, visit_name=self.current_visit)
            ax.axvline(event_time * frequency, color='g', linestyle='--')

            if occlusion is not None:
                ax.axvline(occlusion['start'], color='b', linestyle='--')
                ax.axvline(occlusion['end'], color='b', linestyle='--')

            if i != 0:
                ax.plot(NIRS_df[f'tHb{i}'], color='green')  # THb
                ax.plot(NIRS_df[f'HHb{i}'], color='blue')   # HHb
                ax.plot(NIRS_df[f'O2Hb{i}'], color='red')   # O2Hb
                ax.set_title(f'Tx{i}')

            # ---- overlay WR as secondary axis ----
            if wr_samples is not None:
                ax2 = ax.twinx()
                ax2.plot(wr_samples[0], wr_samples[1], color='black', alpha=0.35, linewidth=1.5)
                ax2.set_ylabel('WR (W)')
                ax2.grid(False)

        plt.savefig(f"{self.current_raw_path}/{title}.png")
        plt.show()
    def _add_vlines(self, ax, start, ramp_duration=-1, v2=False, frequency=1, visit_name: str | None = None):
        ax.axvline(start, color='r', linestyle='--')
        ax.axvline(start + REST_TIME * frequency, color='r', linestyle='--')
        if ramp_duration != -1:
            ax.axvline(start + (REST_TIME + BASELINE_TIME) * frequency, color='r', linestyle='--')
            ax.axvline(start + ramp_duration * frequency, color='r', linestyle='--')
        else:
            # For separate Interval/Step protocols we only want: rest end, warm-up end, protocol end.
            vname = (visit_name or self.current_visit or '').lower()
            if ('interval' in vname) or ('step' in vname):
                ax.axvline(start + REST_TIME * frequency, color='r', linestyle='--')
                ax.axvline(start + (REST_TIME + PRBS_WARMUP_TIME) * frequency, color='r', linestyle='--')
                ax.axvline(start + (REST_TIME + PRBS_WARMUP_TIME + SEQUENCEV2_TIME * 2) * frequency, color='r', linestyle='--')
                return

            prbs_start = start + REST_TIME * frequency
            if v2:
                prbs_start = start + (REST_TIME + BASELINE_TIME + STEP_TIME * 2) * frequency
                ax.axvline(start + (REST_TIME + BASELINE_TIME) * frequency, color='r', linestyle='--')
                ax.axvline(start + (REST_TIME + BASELINE_TIME + STEP_TIME) * frequency, color='r', linestyle='--')
                ax.axvline(prbs_start, color='r', linestyle='--')

            ax.axvline(prbs_start + (PRBS_WARMUP_TIME) * frequency, color='r', linestyle='--')
            ax.axvline(prbs_start + (PRBS_WARMUP_TIME + SEQUENCEV2_TIME) * frequency, color='r', linestyle='--')
            ax.axvline(prbs_start + (PRBS_WARMUP_TIME + SEQUENCEV2_TIME * 2) * frequency, color='r', linestyle='--')


if __name__ == "__main__":
    preparer = Preparer()

    preparer.prepare_data_Lucas()