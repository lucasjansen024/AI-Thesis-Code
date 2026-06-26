# Class to plot data and results
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from ..analysis import Analysis
from .data_manager import DataManager
from ..data_types.base import Participant, Visit, ProcessedVisitData, File, ResultsVisitData, WorkRate
import matplotlib as mpl
from matplotlib import pyplot as plt

from ..constants import ParticipantSelection, SEQUENCE_TIME, PARTICIPANTS_ALL, VISITS_ALL, VISITS_PRBS_NORMAL, \
    VisitSelection, CUT_OFF_FREQUENCY, SEQUENCE_START
from ..data_types.results import RMCorrResult, PearsonResult, BlandAltman, BlandAltmanResult, MNG, MNG_Window
from ..processing.processors import VO2_WR
from ..utils import Utils


class Plotter:

    def __init__(
            self,
            participant_selection: ParticipantSelection = ParticipantSelection.ALL_PARTICIPANTS,
            fontsize: int = 17,
            figsize=(12, 7),
    ):
        self.participant_selection = participant_selection

        cm = plt.get_cmap('gist_rainbow')
        num_participants = len(PARTICIPANTS_ALL)
        self.colors = [cm(1 * i / num_participants) for i in range(num_participants)]
        self.scatter_kwargs = lambda p : { 'edgecolor': self._participant_to_color(p), 'facecolors': 'none', 'label': p, 's': 5}
        self.MNG_marker_size = 40

        self.fontsize = fontsize
        self._set_font_size()

        self.figsize = figsize

        self.input_data_file = 'input_data.png'
        self.VO2_predictions_file = 'VO2_predictions.png'
        self.VO2_correlation_file = 'VO2_rm_corr.png'
        self.VO2_residuals_file = 'VO2_residuals.png'
        self.VO2_bland_altman_global_file = 'VO2_bland_altman_global.png'

        self.VO2_text = "$\dot{V}O_2$"
        self.mLMin_text = "mL $\cdot$ min$^{-1}$"

        self.MNG_text = "MNG"
        self.percentage_text = "%"

    def _MNG_bland_altman_global_file(self, identifier: str) -> str:
        string = f"MNG_bland_altman_global"
        if identifier:
            string += f"_{identifier}"
        string += ".png"
        return string

    def _MNG_correlation_file(self, identifier: str) -> str:
        string = f"MNG_rm_corr"
        if identifier:
            string += f"_{identifier}"
        string += ".png"
        return string

    def _MNG_boxplot_file(self, identifier: str) -> str:
        string = f"MNG_boxplot"
        if identifier:
            string += f"_{identifier}"
        string += ".png"
        return string

    def _set_font_size(self, fontsize: int = None):
        if fontsize is None:
            fontsize = self.fontsize
        mpl.rcParams.update({'font.size': fontsize})

    def _participant_to_color(self, participant: Participant) -> tuple[float, float, float, float]:
        idx = ParticipantSelection.ALL_PARTICIPANTS.value.index(participant)
        return self.colors[idx % len(self.colors)]

    def _visit_to_marker(self, visit: Visit) -> str:
        n_visit = visit.coerce_to_normal()

        if n_visit == visit.Easy:
            return 'o'
        if n_visit == visit.Medium:
            return 'v'
        if n_visit == visit.Hard:
            return 's'

    def _add_visit_legend(self, ax, loc='best'):
        myHandle = [
            Line2D([], [],
                   marker=self._visit_to_marker(v),
                   color=self._participant_to_color('P1'),
                   linestyle='None'
                   )
            for v in VISITS_PRBS_NORMAL
        ]
        ax.legend(handles=myHandle, labels=VISITS_PRBS_NORMAL, title='Visit', loc=loc)

    def _save_plot(self, path: Path, filename: str):
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(path / filename)
        plt.close()

    def _correlation_text(self, r, p, subscript="_{rm}"):
        return f"$r{subscript}$ = {r:.3f}, $p{subscript}$ {'< ' if p < 0.001 else '= '}{0.001 if p < 0.001 else p:.3f}"

    def _draw_vlines(self, ax, sequence_start: int, sequence_length=SEQUENCE_TIME, is_prbs: bool = True):
        ax.axvline(sequence_start, color='r', linestyle='--')

        if is_prbs:
            ax.axvline(sequence_start + sequence_length, color='r', linestyle='--')
            ax.axvline(sequence_start + sequence_length * 2, color='r', linestyle='--')

    def plot_processed_data(
            self,
            visit_data: ProcessedVisitData,
            common_folder_path: Path,
            receptive_field: int
    ):
        df = visit_data.data

        fit_cols = [col for col in df.columns if File.Fit in col]
        total_cols = len(df.columns) - len(fit_cols) + 1

        rows = math.ceil(math.sqrt(total_cols))
        cols = math.ceil(total_cols / rows)
        fig, axs = plt.subplots(rows, cols, figsize=self.figsize)
        axs = axs.flat

        for i, col in enumerate(df.columns):
            if col in fit_cols:
                continue

            ax = axs[i]
            ax.plot(df[col])
            ax.set_title(col)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            self._draw_vlines(ax, receptive_field, visit_data.visit.is_prbs())

        if len(fit_cols) > 0:
            fit_data = {col: df[col].mean() for col in fit_cols}
            ax = axs[-1]
            ax.bar(fit_data.keys(), fit_data.values())
            ax.set_title('Body Comp Data')
            ax.set_xlabel('Type')
            ax.set_ylabel('Value')

        fig.suptitle(f'Input data for {visit_data.participant} - {visit_data.visit}', fontsize=self.fontsize)
        fig.tight_layout(pad=1)

        self._save_plot(
            common_folder_path / visit_data.participant / visit_data.visit,
            self.input_data_file
        )

    def plot_prediction_results(self, visit_data: ResultsVisitData, path: Path):
        fig = plt.figure(figsize=self.figsize)
        plt.plot(visit_data.data['VO2'], label='Measured')
        plt.plot(visit_data.data['predictions'], label='Predicted')
        plt.xlabel('Time (s)')
        plt.ylabel(f'{self.VO2_text} ({self.mLMin_text})')
        plt.title(f'{self.VO2_text} prediction for {visit_data.participant} - {visit_data.visit}')
        plt.legend()
        fig.tight_layout(pad=1)

        self._save_plot(
            path,
            self.VO2_predictions_file
        )

    def plot_VO2_correlation(
            self,
            VO2_df: pd.DataFrame,
            rm_corr_result: RMCorrResult,
            pearson_result: PearsonResult,
            save_path: Path,
            legend=False,
    ):
        self._set_font_size()
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)

        for participant in Utils.order_participants(VO2_df['Participant'].unique()):
            participant_df = VO2_df[VO2_df['Participant'] == participant]
            axs.scatter(participant_df['Measured'], participant_df['Predicted'], **self.scatter_kwargs(participant))

        axs.set_xlabel(f'Measured {self.VO2_text} ({self.mLMin_text})')
        axs.set_ylabel(f'Predicted {self.VO2_text} ({self.mLMin_text})')
        axs.set_xlim(left=0)
        axs.set_ylim(bottom=0)
        axs.axline((0, 0), slope=1, color='black', linestyle='--')

        if legend:
            lgnd = axs.legend()

        self._set_font_size(self.fontsize + 2)
        r_text = self._correlation_text(rm_corr_result.r, rm_corr_result.p)
        r_text += f"\n{self._correlation_text(pearson_result.r, pearson_result.p, subscript='_{p}')}"

        axs.text(.3, .99, r_text, ha='center', va='top', transform=axs.transAxes)

        fig.tight_layout()

        self._save_plot(save_path, self.VO2_correlation_file)

    def plot_VO2_residuals(
            self,
            VO2_df: pd.DataFrame,
            save_path: Path,
            plot_WR=True,
    ):
        self._set_font_size(12)

        VO2_df = VO2_df.copy()
        VO2_df['Visit'] = VO2_df['Visit'].apply(lambda x: x.coerce_to_normal())
        VO2_df['Residual'] = VO2_df['Predicted'] - VO2_df['Measured']

        available_visits = Utils.order_visits(list(VO2_df['Visit'].dropna().unique()))
        if len(available_visits) == 0:
            return

        # ------------------------------------------------------------
        # Prepare visit data first, including Ramp truncation
        # ------------------------------------------------------------
        visit_dfs = {}

        for visit in available_visits:
            visit_df = VO2_df[VO2_df['Visit'] == visit].copy()
            if len(visit_df) == 0:
                continue

            # For the Ramp visit, only plot data until the end of the shortest Ramp test.
            # This ensures that the Ramp group mean +/- SD is based on all participants throughout.
            if visit.is_ramp():
                tmp = visit_df.copy()
                tmp['_time_index'] = tmp.index

                shortest_ramp_end = tmp.groupby('Participant')['_time_index'].max().min()
                visit_df = visit_df[visit_df.index <= shortest_ramp_end]

            visit_dfs[visit] = visit_df

        if len(visit_dfs) == 0:
            return

        # ------------------------------------------------------------
        # <<< NEW: Compute per-participant VO2max (peak VO2) and the
        #          time at which it occurs, for both measured and predicted.
        #          This drives the errorbar on the ramp panel.
        # ------------------------------------------------------------
        ramp_visit = next((v for v in available_visits if v.is_ramp()), None)
        ramp_vo2max_data = None

        if ramp_visit is not None:
            ramp_full = VO2_df[VO2_df['Visit'] == ramp_visit].copy()
            # Use the FULL (untruncated) ramp data so we capture the actual peak per participant.
            ramp_full['_time_index'] = ramp_full.index

            per_participant = []
            for participant, p_df in ramp_full.groupby('Participant'):
                # Peak measured VO2 and its time (seconds -> minutes)
                idx_meas = p_df['Measured'].idxmax()
                peak_meas = p_df.loc[idx_meas, 'Measured']
                time_meas = p_df.loc[idx_meas, '_time_index'] / 60.0

                # Peak predicted VO2 and its time
                idx_pred = p_df['Predicted'].idxmax()
                peak_pred = p_df.loc[idx_pred, 'Predicted']
                time_pred = p_df.loc[idx_pred, '_time_index'] / 60.0

                per_participant.append({
                    'Participant': participant,
                    'peak_meas': peak_meas,
                    'time_meas': time_meas,
                    'peak_pred': peak_pred,
                    'time_pred': time_pred,
                })

            if per_participant:
                pp_df = pd.DataFrame(per_participant)
                ramp_vo2max_data = {
                    'mean_meas': pp_df['peak_meas'].mean(),
                    'sd_meas': pp_df['peak_meas'].std(),
                    'mean_time_meas': pp_df['time_meas'].mean(),
                    'sd_time_meas': pp_df['time_meas'].std(),
                    'mean_pred': pp_df['peak_pred'].mean(),
                    'sd_pred': pp_df['peak_pred'].std(),
                    'mean_time_pred': pp_df['time_pred'].mean(),
                    'sd_time_pred': pp_df['time_pred'].std(),
                }

        # ------------------------------------------------------------
        # Calculate shared y-axis limits across all visits
        # This avoids each subplot having a different hidden y-scale.
        # ------------------------------------------------------------
        top_y_min_values = []
        top_y_max_values = []
        res_y_min_values = []
        res_y_max_values = []

        for visit_df in visit_dfs.values():
            measured = visit_df.groupby(visit_df.index)['Measured'].apply(list)
            predicted = visit_df.groupby(visit_df.index)['Predicted'].apply(list)
            residuals = visit_df.groupby(visit_df.index)['Residual'].apply(list)

            mean_measured = measured.apply(np.mean)
            mean_predicted = predicted.apply(np.mean)
            mean_residuals = residuals.apply(np.mean)

            sd_measured = measured.apply(np.std)
            sd_predicted = predicted.apply(np.std)
            sd_residuals = residuals.apply(np.std)

            top_y_min_values.append((mean_measured - sd_measured).min())
            top_y_min_values.append((mean_predicted - sd_predicted).min())
            top_y_max_values.append((mean_measured + sd_measured).max())
            top_y_max_values.append((mean_predicted + sd_predicted).max())

            res_y_min_values.append((mean_residuals - sd_residuals).min())
            res_y_max_values.append((mean_residuals + sd_residuals).max())

        global_top_y_min = min(top_y_min_values)
        global_top_y_max = max(top_y_max_values)

        global_res_y_min = min(res_y_min_values)
        global_res_y_max = max(res_y_max_values)

        top_pad = 50
        res_pad = max(50, 0.1 * max(abs(global_res_y_min), abs(global_res_y_max)))

        # ------------------------------------------------------------
        # Create figure
        # ------------------------------------------------------------
        fig, axs = plt.subplots(
            2,
            len(available_visits),
            figsize=(4.5 * len(available_visits), 7),
            squeeze=False,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # ------------------------------------------------------------
        # Plot each visit
        # ------------------------------------------------------------
        for i, visit in enumerate(available_visits):
            if visit not in visit_dfs:
                continue

            visit_df = visit_dfs[visit]

            measured = visit_df.groupby(visit_df.index)['Measured'].apply(list)
            predicted = visit_df.groupby(visit_df.index)['Predicted'].apply(list)
            residuals = visit_df.groupby(visit_df.index)['Residual'].apply(list)

            mean_measured = measured.apply(np.mean)
            mean_predicted = predicted.apply(np.mean)
            mean_residuals = residuals.apply(np.mean)

            sd_measured = measured.apply(np.std)
            sd_predicted = predicted.apply(np.std)
            sd_residuals = residuals.apply(np.std)

            x_minutes = mean_measured.index.to_numpy(dtype=float) / 60.0

            ax_top = axs[0, i]
            ax_bot = axs[1, i]

            ax_top.set_title(f"{visit}")
            ax_bot.set_xlabel("Time (min)")

            ax_top.spines[["right", "top"]].set_visible(False)
            ax_bot.spines[["right", "top"]].set_visible(False)

            ax_top.plot(x_minutes, mean_measured, color="black", label="Measured")
            ax_top.plot(x_minutes, mean_predicted, color="red", label="Predicted")

            ax_top.fill_between(
                x_minutes,
                mean_measured - sd_measured,
                mean_measured + sd_measured,
                color="black",
                alpha=0.2,
            )
            ax_top.fill_between(
                x_minutes,
                mean_predicted - sd_predicted,
                mean_predicted + sd_predicted,
                color="red",
                alpha=0.2,
            )

            ax_bot.plot(x_minutes, mean_residuals, color="black", label="Residuals")
            ax_bot.fill_between(
                x_minutes,
                mean_residuals - sd_residuals,
                mean_residuals + sd_residuals,
                color="black",
                alpha=0.2,
            )
            ax_bot.axhline(0, color="black", linestyle="--")

            # Shared y-axes across all visits
            ax_top.set_ylim(bottom=750, top=4500)
            ax_top.set_yticks(np.arange(1000, 4501, 500))
            ax_bot.set_ylim(global_res_y_min - res_pad, global_res_y_max + res_pad)

            ax_top.set_xlim(left=x_minutes.min(), right=x_minutes.max())
            ax_bot.set_xlim(left=x_minutes.min(), right=x_minutes.max())

            ax_top.set_xticklabels([])

            if i == 0:
                ax_top.set_ylabel(f"{self.VO2_text} ({self.mLMin_text})")
                ax_bot.set_ylabel(f"Residual ({self.mLMin_text})")
            else:
                ax_top.set_yticklabels([])
                ax_bot.set_yticklabels([])

            # --------------------------------------------------------
            # <<< NEW: VO2max errorbar on the ramp panel
            # Placed just to the right of the ramp traces so it does not
            # overlap, matching the style shown in the reference figure.
            # --------------------------------------------------------
            if visit.is_ramp() and ramp_vo2max_data is not None:
                d = ramp_vo2max_data
            
                # Position the VO2max error bars beyond the end of the time-series lines.
                x_range = x_minutes.max() - x_minutes.min()
                x_offset = x_minutes.max() + 0.12 * x_range
                x_gap = 0.04 * x_range
            
                # Extend both axes equally.
                # The plotted lines still stop at x_minutes.max().
                ramp_xmax = 11.0
                ramp_xticks = np.arange(0, 11, 2)
                
                ax_top.set_xlim(left=0, right=ramp_xmax)
                ax_bot.set_xlim(left=0, right=ramp_xmax)
                
                ax_top.set_xticks(ramp_xticks)
                ax_bot.set_xticks(ramp_xticks)
                # Measured VO2max
                ax_top.errorbar(
                    x_offset,
                    d["mean_meas"],
                    xerr=d["sd_time_meas"],
                    yerr=d["sd_meas"],
                    fmt="o",
                    color="black",
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=5,
                    label=r"$\dot{V}O_{2max}$ measured",
                    zorder=5,
                )
            
                # Predicted VO2max
                ax_top.errorbar(
                    x_offset + x_gap,
                    d["mean_pred"],
                    xerr=d["sd_time_pred"],
                    yerr=d["sd_pred"],
                    fmt="o",
                    color="red",
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                    markersize=5,
                    label=r"$\dot{V}O_{2max}$ predicted",
                    zorder=5,
                )
        legend_handles = [ 
            Line2D( 
                [0], 
                [0], 
                color="black", 
                linewidth=2, 
                label=r"Measured $\dot{V}O_2$",
            ), 
            Line2D( 
                [0], 
                [0], 
                color="red", 
                linewidth=2, 
                label=r"Predicted $\dot{V}O_2$",
            ),
        ] 
        
        fig.legend(
            handles=legend_handles, 
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.99), 
            ncol=2, 
            frameon=False, 
            fontsize=14,
            handlelength=2.5,
            columnspacing=2.0, 
        )
        
        
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        
        
        self._save_plot(save_path, self.VO2_residuals_file)

    def plot_MNG_harmonics(
            self,
    ):
        data_manager = DataManager(
            required_files=VO2_WR.get_required_files(),
            visit_selection=VisitSelection.ONLY_EASY,
            participant_selection=ParticipantSelection.ALL_PARTICIPANTS
        )
        data_manager.process_data(VO2_WR)

        VO2_dfs = []
        harmonic_dfs = []

        length = 0
        harmonics = 0
        base_freq = 0

        for participant in data_manager.processed_data:
            p_data = data_manager.processed_data[participant]
            for visit in p_data:
                visit_data = p_data[visit]
                visit_data.data = visit_data.data[data_manager.model.receptive_field:]
                visit_data.data = Utils.average_repetitions(visit_data.data)
                visit_data.data.reset_index(inplace=True, drop=True)

                VO2 = visit_data.data['VO2']
                length = len(VO2)
                base_freq = 1 / length
                harmonics = math.floor(CUT_OFF_FREQUENCY / base_freq)
                coeffs = Analysis.get_fourier_coefficients(VO2, harmonics)
                reconstructed = Analysis.reconstruct_original_data(coeffs, length)

                VO2_dfs.append(pd.DataFrame({
                    'Participant': participant,
                    'Visit': visit,
                    'Measured': VO2,
                    'Reconstructed': reconstructed
                }))

                harmonic_dict = {
                    'Participant': participant,
                    'Visit': visit
                }
                for i in range(len(coeffs)):
                    harmonic_dict[f'A{i}'] = coeffs[i][0]
                    harmonic_dict[f'B{i}'] = coeffs[i][1]
                harmonic_dfs.append(pd.DataFrame(harmonic_dict, index=[len(harmonic_dfs)]))

        VO2_df = pd.concat(VO2_dfs)
        df = VO2_df.groupby(VO2_df.index)
        measured = df['Measured'].apply(list)
        reconstructed = df['Reconstructed'].apply(list)

        measured_mean = measured.apply(np.mean)
        measured_sd = measured.apply(np.std)
        reconstructed_mean = reconstructed.apply(np.mean)
        reconstructed_sd = reconstructed.apply(np.std)
        x_minutes = np.arange(0, len(measured_mean) / 60, 1 / 60)

        protocol = Utils.prbs()
        sequence = Utils.to_sequence(protocol, 25, 138)

        self._set_font_size(12)
        fig = plt.figure(figsize=(16, 14))

        gs1 = GridSpec(2, 1, left=0.06, right=0.57, top=0.98, bottom=0.04)
        ax1 = fig.add_subplot(gs1[0, 0])
        ax1.plot(x_minutes, measured_mean, color='black', alpha=0.8)
        ax1.fill_between(x_minutes, measured_mean - measured_sd, measured_mean + measured_sd, color='black', alpha=0.2)
        ax1.set_xlim(left=0, right=length / 60)
        ax1.set_ylabel(f'{self.VO2_text} ({self.mLMin_text})')
        ax1.set_xticklabels([])
        ax1.spines[['top']].set_visible(False)
        ax1.text(0.5, 1, f"Raw signal", transform=ax1.transAxes, ha='center', va='top')
        twin1 = ax1.twinx()
        twin1.set_ylim(bottom=-15, top=190)
        twin1.set_yticks([25, 138])
        twin1.plot(x_minutes, sequence, color='black', label='Work Rate', linestyle='--', alpha=0.6)
        twin1.set_ylabel('Work Rate (W)')
        twin1.spines[['top']].set_visible(False)

        ax2 = fig.add_subplot(gs1[1, 0])
        ax2.plot(x_minutes, reconstructed_mean, color='black')
        ax2.fill_between(x_minutes, reconstructed_mean - reconstructed_sd, reconstructed_mean + reconstructed_sd, color='black', alpha=0.2)
        ax2.set_xlim(left=0, right=length / 60)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel(f'{self.VO2_text} ({self.mLMin_text})')
        ax2.spines[['top']].set_visible(False)
        ax2.text(0.5, 1, f"Superposition of harmonics", transform=ax2.transAxes, ha='center', va='top')
        twin2 = ax2.twinx()
        twin2.set_ylim(bottom=-15, top=190)
        twin2.set_yticks([25, 138])
        twin2.plot(x_minutes, sequence, color='black', label='Work Rate', linestyle='--', alpha=0.6)
        twin2.set_ylabel('Work Rate (W)')
        twin2.spines[['top']].set_visible(False)

        harmonic_df = pd.concat(harmonic_dfs)

        A0 = harmonic_df['A0'].mean()
        A0_std = harmonic_df['A0'].std()
        t = np.linspace(0, length, length)
        omega = 2 * np.pi / length

        gs2 = GridSpec(harmonics, 1, left=0.67, right=0.98, top=0.98, bottom=0.04)
        for i in range(1, harmonics + 1):
            ax = fig.add_subplot(gs2[i - 1, 0])
            As = harmonic_df[f'A{i}']
            Bs = harmonic_df[f'B{i}']
            ax.text(0.05, 1, f'{base_freq * i:.5f} Hz', ha='left', va='top', transform=ax.transAxes)
            signal = A0 + As.mean() * np.cos(i * omega * t) + Bs.mean() * np.sin(i * omega * t)
            upper = (A0 - A0_std) + (As.mean() + As.std()) * np.cos(i * omega * t) + (Bs.mean() + Bs.std()) * np.sin(i * omega * t)
            lower = (A0 + A0_std) + (As.mean() - As.std()) * np.cos(i * omega * t) + (Bs.mean() - Bs.std()) * np.sin(i * omega * t)
            ax.plot(x_minutes, signal, color='black')
            ax.fill_between(x_minutes, upper, lower, color='black', alpha=0.2)
            ax.set_xlim(left=0, right=length / 60)
            ax.set_yticks([1200, 1700, 2200])
            ax.spines[['right', 'top']].set_visible(False)
            if i != harmonics:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (min)")

        # plt.tight_layout()
        plt.show()


    def plot_VO2_bland_altman_global(
            self,
            results: dict[Participant, BlandAltman],
            global_result: BlandAltmanResult,
            save_path: Path,
            legend=False,
    ):
        self._set_font_size()
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)

        for participant in Utils.order_participants(list(results.keys())):
            bland_altman = results[participant]
            axs.scatter(bland_altman.mean, bland_altman.diff, **self.scatter_kwargs(participant))

        axs.axhline(global_result.bias, color='black', linestyle='--')
        axs.axhline(global_result.bias + 1.96 * global_result.sd, color='gray', linestyle='--')
        axs.axhline(global_result.bias - 1.96 * global_result.sd, color='gray', linestyle='--')

        right_xlim = axs.get_xlim()[1]
        axs.text(right_xlim + 5, global_result.bias, int(global_result.bias), ha='left', va='center')
        axs.text(right_xlim + 5, global_result.bias + 1.96 * global_result.sd, int(global_result.bias + 1.96 * global_result.sd), ha='left', va='center')
        axs.text(right_xlim + 5, global_result.bias - 1.96 * global_result.sd, int(global_result.bias - 1.96 * global_result.sd), ha='left', va='center')

        unit = f"{self.VO2_text} ({self.mLMin_text})"
        axs.set_xlabel(f'Mean {unit}')
        axs.set_ylabel(f'Predicted - Measured {unit}')

        if legend:
            lgnd = axs.legend()

        fig.tight_layout()

        self._save_plot(save_path, self.VO2_bland_altman_global_file)

    def plot_MNG_bland_altman_global(
            self,
            results: dict[Participant, Tuple[BlandAltman, List[Visit]]],
            global_result: BlandAltmanResult,
            save_path: Path,
            legend=False,
            identifier=''
    ):
        self._set_font_size()
        fig, axs = plt.subplots(1, 1, figsize=self.figsize)

        for participant in Utils.order_participants(list(results.keys())):
            (bland_altman, visit) = results[participant]
            for (m, d, v) in zip(bland_altman.mean, bland_altman.diff, visit):
                axs.scatter(m, d, s=self.MNG_marker_size, marker=self._visit_to_marker(v))

        axs.axhline(global_result.bias, color='black', linestyle='--')
        axs.axhline(global_result.bias + 1.96 * global_result.sd, color='gray', linestyle='--')
        axs.axhline(global_result.bias - 1.96 * global_result.sd, color='gray', linestyle='--')

        right_xlim = axs.get_xlim()[1]
        axs.text(right_xlim + 5, global_result.bias, int(global_result.bias), ha='left', va='center')
        axs.text(right_xlim + 5, global_result.bias + 1.96 * global_result.sd,
                 int(global_result.bias + 1.96 * global_result.sd), ha='left', va='center')
        axs.text(right_xlim + 5, global_result.bias - 1.96 * global_result.sd,
                 int(global_result.bias - 1.96 * global_result.sd), ha='left', va='center')

        unit = f"{self.MNG_text} ({self.percentage_text})"
        axs.set_xlabel(f'Mean {unit}')
        axs.set_ylabel(f'Predicted - Measured {unit}')

        ax_handles, ax_labels = axs.get_legend_handles_labels()
        handles = []
        labels = []
        for (h, l) in zip(ax_handles, ax_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)
        l1 = axs.legend(handles, labels, title='Participant', loc='upper right')
        self._add_visit_legend(axs)
        if legend:
            axs.add_artist(l1)

        fig.tight_layout()

        self._save_plot(save_path, self._MNG_bland_altman_global_file(identifier))

    def plot_mng_boxplot(
            self,
            df: pd.DataFrame,
            save_path: Path,
            identifier='',
    ):
        # Create a figure and a set of subplots
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.boxplot(
            x='Visit',
            y=self.MNG_text,
            hue='Source',
            data=df,
            legend=True,
            order=VISITS_PRBS_NORMAL,
            ax=ax
        )

        # ax.legend(['Measured', 'Predicted'])
        ax.set_ylabel(f"{self.MNG_text} ({self.percentage_text})")
        fig.tight_layout()

        self._save_plot(save_path, self._MNG_boxplot_file(identifier))

        plt.show()

    def plot_mng_rm_corr(
            self,
            df: pd.DataFrame,
            rm_corr_result: RMCorrResult,
            pearson_result: PearsonResult = None,
            save_path: Path = None,
            legend=False,
            title='',
            identifier='',
            x='Measured',
            y='Predicted',
    ):

        fig, ax = plt.subplots(figsize=self.figsize)
        subject = 'Participant'
        visit = 'Visit'

        # Plot each subject's data points and regression line
        for subj in Utils.order_participants(df[subject].unique()):
            subj_data = df[df[subject] == subj]
            color = self._participant_to_color(subj)

            # For each visit, get the specific marker and plot scatter points with it
            for vis in subj_data[visit].unique():
                vis_data = subj_data[subj_data[visit] == vis]
                marker = self._visit_to_marker(vis)  # Get the marker style for the visit

                # Scatter plot for each visit within each subject using plt.scatter
                ax.scatter(
                    vis_data[x],
                    vis_data[y],
                    color=color,
                    marker=marker,
                    s=self.MNG_marker_size,  # Adjust size as needed
                )

            # Calculate individual intercept based on global slope
            subj_mean_x = subj_data[x].mean()
            subj_mean_y = subj_data[y].mean()
            subj_intercept = subj_mean_y - rm_corr_result.r * subj_mean_x

            # Constrain line to subject-specific x range
            min_x, max_x = subj_data[x].min(), subj_data[x].max()
            x_vals = np.array([min_x, max_x])
            y_vals = subj_intercept + rm_corr_result.r * x_vals

            # Plot the line with subject's color
            # ax.plot(x_vals, y_vals, color=color, linewidth=1)

        ax.axline((60, 60), slope=1, color='black', linestyle='--')
        # Labeling and customization
        unit = f"{self.MNG_text} ({self.percentage_text})"
        ax.set_xlabel(f'Measured {unit}')
        ax.set_ylabel(f'Predicted {unit}')

        r_text = self._correlation_text(rm_corr_result.r, rm_corr_result.p)
        if pearson_result is not None:
            r_text += f"\n{self._correlation_text(pearson_result.r, pearson_result.p, subscript='_{p}')}"

        ax.text(.3, .99, r_text, ha='center', va='top', transform=ax.transAxes)

        # Create legend for participants
        participants = Utils.order_participants(df[subject].unique())
        handles = [Line2D([], [], color=self._participant_to_color(p), marker='o', linestyle='None') for p in
                   participants]
        l1 = ax.legend(handles, participants, title='Participant', loc='lower right')

        # Add legend for visit markers
        self._add_visit_legend(ax)
        if legend:
            ax.add_artist(l1)

        if title:
            ax.set_title(title)

        fig.tight_layout()

        if save_path is not None:
            self._save_plot(save_path, self._MNG_correlation_file(identifier))

        plt.show()

    def plot_MNG(
            self,
            results: Dict[Participant, Dict[Visit, MNG]],
            save_path: Path,
            file_name: str,
            visit_selection: VisitSelection = VisitSelection.PRBS_EXCL_HARD
    ):
        fig, axs = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'{file_name} for different conditions')
        conditions = [attr for attr in MNG.__annotations__.keys() if attr not in ['participant', 'visit', 'cut_off']]
        for i, condition in enumerate(
                conditions
        ):
            ax = axs[i // 3, i % 3]
            ax.set_title(condition)
            ax.set_ylim(bottom=20, top=120)
            for participant in Utils.order_participants(list(results.keys())):
                mng_visits = Utils.order_visits([visit for visit in results[participant] if visit in visit_selection.value])
                mng_values = [results[participant][visit][condition] for visit in mng_visits]
                visits = [visit.coerce_to_normal() for visit in mng_visits]
                ax.plot(visits, mng_values, label=participant, color=self._participant_to_color(participant))
                ax.scatter(visits, mng_values, color=self._participant_to_color(participant))
            # ax.legend()
        fig.tight_layout(pad=1)

        self._save_plot(save_path, f"{file_name}_{visit_selection.name}.png")

        plt.show()

    def plot_MNG_windows(
            self,
            window_results: Dict[Participant, Dict[Visit, List[MNG_Window]]],
            averaged: bool,
            save_path: Path,
            file_name: str,
            visit_selection: VisitSelection = VisitSelection.PRBS_EXCL_HARD
    ):
        first_window = Utils.get_first_visit_data(window_results)
        count = len(first_window)
        rows = math.floor(np.sqrt(count))
        cols = math.ceil(count / np.sqrt(count))
        fig, axs = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle(f'MNG values for different window lengths {"averaged repetitions" if averaged else "both repetitions"}')

        lengths = [window['window_length'] for window in first_window]
        for i, length in enumerate(lengths):
            ax = axs[i // cols, i % cols]
            ax.set_title(f'Window length: {length}')
            ax.set_ylim(bottom=20, top=120)
            for participant in Utils.order_participants(list(window_results.keys())):
                mng_visits = Utils.order_visits([visit for visit in window_results[participant] if visit in visit_selection.value])
                mng_values = [window_results[participant][visit][i]['MNG']['mean'] for visit in mng_visits]
                visits = [visit.coerce_to_normal() for visit in mng_visits]
                ax.plot(visits, mng_values, label=participant, color=self._participant_to_color(participant))
                ax.scatter(visits, mng_values, color=self._participant_to_color(participant))
            # ax.legend()
        fig.tight_layout(pad=1)

        self._save_plot(save_path, f"{file_name}_{visit_selection.name}.png")

        plt.show()


    def plot_single_mng(
            self,
            results: Dict[Participant, Dict[Visit, float]],
            file_name: str,
            save_path: Path = None,
    ):

        fig = plt.figure()
        fig.suptitle(f'{file_name}')
        # plt.ylim(bottom=20, top=120)
        for participant in Utils.order_participants(list(results.keys())):
            mng_visits = Utils.order_visits([visit for visit in results[participant]])
            mng_values = [results[participant][visit] for visit in mng_visits]
            visits = [visit.coerce_to_normal() for visit in mng_visits]
            plt.plot(visits, mng_values, label=participant, color=self._participant_to_color(participant))
            plt.scatter(visits, mng_values, color=self._participant_to_color(participant))

        fig.tight_layout(pad=1)

        if save_path is not None:
            self._save_plot(save_path, f"{file_name}.png")
        else:
            plt.show()


    def plot_VO2_and_WR(self, VO2, WR, draw_vlines=True, sequence_start=SEQUENCE_START, sequence_length=SEQUENCE_TIME):
        fig = plt.figure(figsize=self.figsize)

        ax1 = fig.add_subplot(111, label='VO2')
        ax1.plot(VO2, color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{self.VO2_text} ({self.mLMin_text})')
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=0)

        ax2 = ax1.twinx()
        ax2.plot(WR, color='orange')
        ax2.set_ylabel('Work Rate (W)')
        ax2.set_ylim(bottom=0)

        if draw_vlines:
            self._draw_vlines(ax1, sequence_start=sequence_start, sequence_length=sequence_length)

        plt.show()