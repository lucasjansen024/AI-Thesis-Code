import dataclasses
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclasses.dataclass
class NIRSdetails:
    device_id: int
    device_template: str
    frequency: int

class NIRSLoader:

    def __init__(
            self,
            file_path: Path,
    ):
        self.file_path = file_path
        self.complete_df = self._load_file(file_path)
        self.df = self._extract_df(self.complete_df.copy())
        self.details: NIRSdetails = self._extract_details(self.complete_df.copy())
        # Some OxySoft exports keep the *original* device sample-rate in the header (e.g., 100 Hz)
        # even if the exported data table has been downsampled (e.g., 10 Hz). Infer the effective
        # export frequency from (rows / duration) when possible.
        inferred = self._infer_export_frequency(self.complete_df.copy(), self.df)
        if inferred is not None:
            self.details.frequency = inferred

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_excel(file_path)

    def _infer_export_frequency(self, complete_df: pd.DataFrame, extracted_df: pd.DataFrame) -> int | None:
        """Infer the effective sampling frequency of the exported data table.

        OxySoft headers can report the *device* sample rate, not the export/downsample rate.
        We infer the effective rate from the exported table length and the "Data file duration".
        """
        try:
            col0 = complete_df.iloc[:, 0].astype(str)
            idx = col0[col0.str.contains('Data file duration', na=False)].index
            if len(idx) == 0:
                return None
            dur_val = complete_df.iloc[idx[0], 1]
            duration_s = float(str(dur_val).replace(',', '.'))
            if duration_s <= 0:
                return None
            hz = int(round(len(extracted_df) / duration_s))
            # sanity bounds
            if 1 <= hz <= 250:
                return hz
            return None
        except Exception:
            return None

    def _extract_df(self, df: pd.DataFrame) -> pd.DataFrame:
        legend_header_row = df[df.iloc[:, 0].str.contains('Column', na=False)].index[0]
        next_empty_row = df.isnull().all(axis=1)[legend_header_row:].idxmax()
        # Create a dictionary between these rows, with column 0 as key, column 1 as value
        rename_df = df.iloc[legend_header_row + 1:next_empty_row, :2]
        values = rename_df.iloc[:, 1].values
        rename_dict = dict(zip(rename_df.iloc[:, 0], values))

        df = df.iloc[next_empty_row + 1:, :]  # Remove the legend and empty rows
        df.columns = df.iloc[0]
        df = df.iloc[2:]
        df.rename(columns=rename_dict, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)  # Remove the first column which is not needed

        renamed = []
        for col in df.columns:
            if not isinstance(col, str) or "Unnamed" in col or "text" in col:
                renamed.append("Event Time")

            elif "TSI%" in col:
                renamed.append("TSI%")

            elif "Fit Factor" in col:
                renamed.append("TSI Fit Factor")

            elif "Event" in col:
                renamed.append("Event")

            else:
                split = col.split("-")[1].split("(")[0].strip().split(" ")
                channel = split[0][-1]
                data_type = split[-1]
                new_name = f"{data_type}{channel}"
                if "Rx2" in col:
                    new_name = f"{new_name}-2"
                renamed.append(new_name)

        df.rename(columns=dict(zip(df.columns, renamed)), inplace=True)

        return df

    def _extract_details(self, df: pd.DataFrame) -> NIRSdetails:
        device_id = df.iloc[13, 1]
        device_template = df.iloc[8, 1]
        frequency = int(df.iloc[3, 1])
        return NIRSdetails(device_id=device_id, device_template=device_template, frequency=frequency)
