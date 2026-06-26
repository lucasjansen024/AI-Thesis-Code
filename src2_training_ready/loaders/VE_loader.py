from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import numpy as np
import pandas as pd

from src2_training_ready.utils import Utils
from src2_training_ready.data_types.file_keys import VEBxBFileKeys


@dataclass
class VEDetails:
    """Minimal details for VE/HR exports."""
    frequency: Optional[float] = None


class VELoader:
    """Tolerant loader for VE (+ optional HR/VT/BR) export files.

    This export format often contains a short metadata block at the top (few columns),
    followed by the actual table with many columns. Pandas' default CSV reader will
    crash on that. We therefore:
    - detect the true header row in the raw text ("Time,...")
    - load from there
    - drop the unit row ("sec,br/min,...") if present
    - rename a small set of columns to canonical names:
        - t
        - HR
        - VE breath by breath (we use the plain "VE" column if present)
        - BR breath by breath (we use the plain "BR" column if present)
        - VT breath by breath (optional)

    Downstream processing expects VEBxBFileKeys.* names, so we map to those.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.complete_df = self._load_file(self.file_path)
        # self.df is the regular time-based VE/HR view (used for HR.csv interpolation)
        self.df = self._extract_df(self.complete_df.copy())
        # self.bxb_df is the TymeWear breath-by-breath export view (used for *VEBxB.csv)
        self.bxb_df = self._extract_bxb_df(self.complete_df.copy())
        # self.ve_prepared_df is the regular TymeWear time-series export used for prepared VE.csv
        self.ve_prepared_df = self._extract_prepared_ve_df(self.complete_df.copy())
        self.details = VEDetails()

    # -------------------------
    # Loading
    # -------------------------

    @staticmethod
    def _find_csv_header_row(lines: Iterable[str]) -> Optional[int]:
        # Match e.g. "Time,BR,VT,VE,HR" or localized "Tijd" / "Zeit"
        pat = re.compile(r'^\s*(time|tijd|zeit)\s*[,;]', re.IGNORECASE)
        for i, line in enumerate(lines):
            if pat.match(line):
                return i
        return None

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix != '.csv':
            try:
                return pd.read_excel(file_path)
            except Exception:
                # Some exports are CSVs without a .csv extension, or we accidentally picked a non-excel file.
                try:
                    return pd.read_csv(file_path, engine='python', on_bad_lines='skip')
                except Exception:
                    raise

        # First try the simple way.
        try:
            return pd.read_csv(file_path)
        except Exception:
            pass

        # Fallback for "metadata block" CSVs.
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        header_row = self._find_csv_header_row(lines)
        if header_row is None:
            # Last resort: let python engine swallow bad rows.
            return pd.read_csv(file_path, engine='python', on_bad_lines='skip')

        # Try delimiter sniffing (comma vs semicolon).
        header_line = lines[header_row]
        sep = ';' if header_line.count(';') > header_line.count(',') else ','

        return pd.read_csv(
            file_path,
            skiprows=header_row,
            header=0,
            sep=sep,
            engine='python',
        )

    # -------------------------
    # Normalization
    # -------------------------

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> None:
        """Rename the handful of columns we actually use.

        We deliberately avoid broad "contains 've'" matching, because the export contains
        many VE-related columns ("VE calibrated", "VE 15 sec sum", etc.), which would
        otherwise create duplicate column names.
        """
        colmap: Dict[Any, Any] = {}

        for c in df.columns:
            c_str = str(c).strip()
            c_low = c_str.lower()

            # time
            if c_low in ('t', 'time', 'tijd', 'zeit') or c_low.startswith('time'):
                colmap[c] = 't'
                continue

            # HR
            if c_low in ('hr', 'heart rate', 'heartrate', 'pulse', 'hf', 'external heart rate'):
                colmap[c] = 'HR'
                continue

            # Plain columns (preferred)
            if c_low == 've':
                colmap[c] = VEBxBFileKeys.VEBxB
                continue
            if c_low == 'br':
                colmap[c] = VEBxBFileKeys.BRBxB
                continue
            if c_low == 'vt':
                colmap[c] = VEBxBFileKeys.VTBxB
                continue
            # Breath-by-breath columns (keep as extra, *distinct* names to avoid duplicates)
            if c_low == 've breath by breath':
                colmap[c] = VEBxBFileKeys.VEBxB
                continue
            if c_low == 've breath by breath smoothed':
                colmap[c] = VEBxBFileKeys.VEBxB_smoothed
                continue

            if c_low == 'br breath by breath':
                colmap[c] = VEBxBFileKeys.BRBxB
                continue
            if c_low == 'br breath by breath smoothed':
                colmap[c] = VEBxBFileKeys.BRBxB_smoothed
                continue
            if c_low == 'br breath by breath outlier rejected average':
                colmap[c] = VEBxBFileKeys.BRBxB_smoothed
                continue

            if c_low == 'vt breath by breath':
                colmap[c] = VEBxBFileKeys.VTBxB
                continue
            if c_low == 'vt breath by breath smoothed':
                colmap[c] = VEBxBFileKeys.VTBxB_smoothed
                continue

        if colmap:
            df.rename(columns=colmap, inplace=True)

    @staticmethod
    def _coerce_time_to_seconds(series: pd.Series) -> pd.Series:
        if series.dtype.kind in ('i', 'u', 'f'):
            return pd.to_numeric(series, errors='coerce')

        # datetime.time objects
        try:
            if len(series) > 0 and hasattr(series.iloc[0], 'hour'):
                return series.apply(Utils.VO2_time_to_sec)
        except Exception:
            pass

        def _parse(x: Any):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            # plain seconds (e.g. '12.3')
            try:
                return float(s)
            except Exception:
                pass

            # mm:ss or hh:mm:ss
            if re.match(r'^\d{1,2}:\d{2}([\.,]\d+)?$', s):
                s = '00:' + s
            try:
                return Utils.VO2_string_to_sec(s)
            except Exception:
                return np.nan

        return series.apply(_parse)

    def _drop_unit_row_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # If the first row is units (e.g. Time == 'sec'), drop it.
        first_t = df.iloc[0].get('t', None)
        if isinstance(first_t, str) and re.search(r'^(sec|s|tijd|time)$', first_t.strip(), re.IGNORECASE):
            return df.iloc[1:].copy()

        # Sometimes 't' isn't renamed yet.
        if 't' not in df.columns:
            c0 = df.columns[0]
            v0 = df.iloc[0].get(c0, None)
            if isinstance(v0, str) and re.search(r'^sec$', v0.strip(), re.IGNORECASE):
                return df.iloc[1:].copy()

        return df

    def _extract_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(axis=1, how='all', inplace=True)

        self._rename_columns(df)

        # If the export contains multiple VE/BR/VT variants, renaming can create duplicates.
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Ensure time column
        if 't' not in df.columns:
            df.rename(columns={df.columns[0]: 't'}, inplace=True)

        df = self._drop_unit_row_if_present(df)

        # Convert types
        df['t'] = self._coerce_time_to_seconds(df['t'])

        for col in df.columns:
            if col == 't':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only what we need (plus HR if present)
        keep = ['t']
        if 'HR' in df.columns:
            keep.append('HR')
        for k in (VEBxBFileKeys.VEBxB, VEBxBFileKeys.VEBxB_smoothed, VEBxBFileKeys.BRBxB, VEBxBFileKeys.BRBxB_smoothed, VEBxBFileKeys.VTBxB, VEBxBFileKeys.VTBxB_smoothed):
            if k in df.columns:
                keep.append(k)

        df = df[keep].copy()
        df = df.dropna(subset=['t']).sort_values('t').reset_index(drop=True)
        return df

    def _extract_bxb_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the TymeWear breath-by-breath table in the same style as old prepared VEBxB files."""
        if df is None or df.empty:
            return pd.DataFrame(columns=['t'])

        df = df.dropna(axis=1, how='all').copy()
        name_map = {str(c).strip().lower(): c for c in df.columns}

        time_col = name_map.get('breath by breath time')
        if time_col is None:
            return pd.DataFrame(columns=['t'])

        ordered_cols = [
            'BR breath by breath',
            'BR breath by breath outlier rejected average',
            'VT breath by breath',
            'VT breath by breath smoothed',
            'VE breath by breath',
            'VE breath by breath smoothed',
            'Inhale:Exhale Ratio breath by breath',
            'Inhale:Exhale Ratio breath by breath smoothed',
        ]
        present_cols = [name_map[c.lower()] for c in ordered_cols if c.lower() in name_map]

        out = df[present_cols + [time_col]].copy()
        if not out.empty:
            first_t = out.iloc[0][time_col]
            if isinstance(first_t, str) and str(first_t).strip().lower() in {'sec', 's', 'time', 'tijd'}:
                out = out.iloc[1:].copy()

        out[time_col] = pd.to_numeric(out[time_col], errors='coerce')
        for c in present_cols:
            out[c] = pd.to_numeric(out[c], errors='coerce')

        # Old prepared-file convention: shift by 60 s and round to integer seconds.
        out['t'] = np.round(out[time_col])

        out = out.dropna(subset=['t'])
        out = out[out['t'] >= 0].copy()

        out = out[present_cols + ['t']].reset_index(drop=True)
        return out


    def _extract_prepared_ve_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the regular TymeWear time-series VE table for prepared VE.csv.

        Expected output columns (if present in the raw export) are kept in this exact order:
        Time, BR, VT, VE, HR, External Heart Rate, VT calibrated, VT calibrated inst,
        VE calibrated, VE calibrated inst, VE 15 sec sum

        Historical prepared-file convention: subtract 60 seconds so raw Time=60.0 becomes Time=0.0.
        """
        desired_cols = [
            'Time',
            'BR',
            'VT',
            'VE',
            'HR',
            'External Heart Rate',
            'VT calibrated',
            'VT calibrated inst',
            'VE calibrated',
            'VE calibrated inst',
            'VE 15 sec sum',
        ]

        if df is None or df.empty:
            return pd.DataFrame(columns=desired_cols)

        df = df.dropna(axis=1, how='all').copy()
        name_map = {str(c).strip().lower(): c for c in df.columns}

        out = pd.DataFrame()
        for c in desired_cols:
            raw_c = name_map.get(c.lower())
            if raw_c is None:
                out[c] = np.nan
            else:
                out[c] = pd.to_numeric(df[raw_c], errors='coerce')

        # Drop metadata/unit rows (Time becomes NaN there after numeric coercion)
        out = out.dropna(subset=['Time']).copy()

        # Historical convention: align the TymeWear time base to 0 s by subtracting 60 s
        out = out[out['Time'] >= 0].copy()

        return out.reset_index(drop=True)
