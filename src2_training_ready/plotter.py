# Class to plot data and results
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from src2_training_ready.analysis import Analysis
from src2_training_ready.train_testing.data_manager import DataManager
from src2_training_ready.data_types.base import Participant, Visit, ProcessedVisitData, File, ResultsVisitData, WorkRate
from src2_training_ready.data_types.results import RMCorrResult, PearsonResult, BlandAltman, BlandAltmanResult, MNG, MNG_Window 
import matplotlib as mpl
from matplotlib import pyplot as plt

from .constants import ParticipantSelection, SEQUENCEV2_TIME, PARTICIPANTS_ALL, VISITS_ALL, VISITS_PRBS_NORMAL, \
    VisitSelection, CUT_OFF_FREQUENCY, SEQUENCE_START
from .processing.processors import VO2_WR
from .utils import Utils


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

    def _draw_vlines(self, ax, sequence_start: int, sequence_length=SEQUENCEV2_TIME, is_prbs: bool = True):
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
        fig, axs = plt.subplots(2, 3, figsize=(14, 7), gridspec_kw={'height_ratios':[2,1]})
        VO2_df = VO2_df.copy()
        VO2_df['Visit'] = VO2_df['Visit'].apply(lambda x: x.coerce_to_normal())
        VO2_df['Residual'] = VO2_df['Predicted'] - VO2_df['Measured']

        for i, visit in enumerate(VISITS_PRBS_NORMAL):
            if visit not in VO2_df['Visit'].unique():
                continue
            visit_df = VO2_df[VO2_df['Visit'] == visit]
            measured = visit_df.groupby(visit_df.index)['Measured'].apply(list)
            predicted = visit_df.groupby(visit_df.index)['Predicted'].apply(list)
            residuals = visit_df.groupby(visit_df.index)['Residual'].apply(list)

            # Calculate the mean and standard deviation for each second
            mean_measured = measured.apply(np.mean)
            mean_predicted = predicted.apply(np.mean)
            mean_residuals = residuals.apply(np.mean)
            sd_measured = measured.apply(np.std)
            sd_predicted = predicted.apply(np.std)
            sd_residuals = residuals.apply(np.std)

            axs[0, i].set_title(f'{visit}')
            axs[1, i].set_xlabel('Time (min)')
            axs[0, i].spines[['right', 'top']].set_visible(False)
            axs[1, i].spines[['right', 'top']].set_visible(False)
            axs[0, i].set_ylim(bottom=700, top=3500)
            axs[1, i].set_ylim(bottom=-500, top=500)
            axs[0, i].set_xlim(left=-1, right=33)
            axs[1, i].set_xlim(left=-1, right=33)
            axs[0, i].set_xticklabels([])
            if i == 0:
                axs[0, i].set_ylabel(f'{self.VO2_text} ({self.mLMin_text})')
                axs[1, i].set_ylabel(f'Residual ({self.mLMin_text})')
            else:
                axs[0, i].set_yticklabels([])
                axs[1, i].set_yticklabels([])

            # Plot measured in black, predicted in red
            # Mean as a line, +- sd as a shaded area
            x_minutes = np.arange(0, len(mean_measured) / 60, 1 / 60)
            axs[0, i].plot(x_minutes, mean_measured, color='black', label='Measured')
            axs[0, i].plot(x_minutes, mean_predicted, color='red', label='Predicted')
            axs[0, i].fill_between(x_minutes, mean_measured - sd_measured, mean_measured + sd_measured, color='black', alpha=0.2)
            axs[0, i].fill_between(x_minutes, mean_predicted - sd_predicted, mean_predicted + sd_predicted, color='red', alpha=0.2)

            if plot_WR:
                wr_dict = {
                    WorkRate.W25: 25,
                    WorkRate.GET90: 138,
                    WorkRate.GET: 153,
                    WorkRate.D30: 194,
                }

                protocol = Utils.prbs() + Utils.prbs()
                sequence = Utils.to_sequence(protocol, wr_dict[visit.lower_wr], wr_dict[visit.upper_wr])
                twin = axs[0, i].twinx()
                twin.set_ylim(bottom=-10, top=250)
                twin.spines[['top']].set_visible(False)
                axs[0, i].spines[['right']].set_visible(True)
                twin.set_yticks(list(wr_dict.values()), list(wr_dict.keys()))
                if i != 2:
                    twin.set_yticklabels([])
                else:
                    twin.set_ylabel('Work Rate (W)')

                for work_rate in wr_dict.values():
                    twin.axhline(y=work_rate, color='black', linestyle='--', alpha=0.1)
                twin.plot(x_minutes, sequence, color='black', label='Work Rate', alpha=0.25)

            # Plot residuals
            axs[1, i].plot(x_minutes, mean_residuals, color='black', label='Residuals')
            axs[1, i].fill_between(x_minutes, mean_residuals - sd_residuals, mean_residuals + sd_residuals, color='black', alpha=0.2)
            axs[1, i].axhline(0, color='black', linestyle='--')

        fig.tight_layout()
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


    def plot_VO2_and_WR(self, VO2, WR, draw_vlines=True, sequence_start=210, sequence_length=SEQUENCEV2_TIME):
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
        ax1.set_title(r"$\dot{V}O_{2}$ and Work Rate during PRBS protocol", fontsize = 18)

        if draw_vlines:
            self._draw_vlines(ax1, sequence_start=sequence_start, sequence_length=sequence_length)

        plt.show()
    def plot_WR(self, WR, sequence_start=210, sequence_length=SEQUENCEV2_TIME, draw_vlines=True):
        fig = plt.figure(figsize=self.figsize)

        ax = fig.add_subplot(111)
        ax.plot(WR, color='orange')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Work Rate (W)')
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

        if draw_vlines:
            self._draw_vlines(ax, sequence_start=sequence_start, sequence_length=sequence_length)

        plt.show()


    def plot_WR_combined(self, WR_step, WR_complete,
                     draw_vlines=True, sequence_length=SEQUENCEV2_TIME):
    

        # Combine step + complete data end-to-end
        WR = np.concatenate([WR_step, WR_complete])

        time_seconds = np.arange(len(WR))
        time_minutes = time_seconds / 60.0

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # --- Background shading ---
        end_step_time = len(WR_step) / 60.0
        # Step part in very light gray
        ax.axvspan(0, end_step_time, color='#f5f5f5', alpha=1.0)
        # Warm-up in light blue
        ax.axvspan(18, 21.5, color='#deebf7', alpha=0.8)
        # Rest of PRBS in darker blue
        ax.axvspan(21.5, time_minutes.max(), color='#6baed6', alpha=0.4)

        # --- Work rate line (black) ---
        ax.plot(time_minutes, WR, color='black', linewidth=1.8)

        # Axes labels
        ax.set_xlabel('Time (min)', fontsize = 14)
        ax.set_ylabel('Work Rate (W)', fontsize = 14)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.set_title("M-H Protocol", fontsize=16)

        # Horizontal reference lines + ticks
        y_lines = [0, 25, 231]
        labels = ['0 W', '25 W', 'Δ40%']
        for y_val in y_lines:
            ax.axhline(y=y_val, color='black', linestyle=':', linewidth=1)
        ax.set_yticks(y_lines)
        ax.set_yticklabels(labels, color='black')

        # Make tick labels appear just left of the y-axis
        ax.tick_params(axis='y', which='both', direction='out', pad=5)
        ax.yaxis.set_label_coords(-0.15, 0.5)
        fig.subplots_adjust(left=0.25)

        # Draw vertical lines from your helper
        if draw_vlines:
            self._draw_vlines(ax,
                            sequence_start=210 / 60.0,
                            sequence_length=sequence_length / 60.0)

        # Vertical red dotted line at 37 min
        ax.axvline(x=37, color='red', linestyle=':', linewidth=2)

        plt.show()

    def plot_WR_PRBS_only(self, WR_complete, draw_vlines=True, sequence_length=SEQUENCEV2_TIME, get80_w=141):
    
    
        # Add 2 minutes (120 seconds) of rest at 0W at the beginning
        rest_duration = 120  # 2 minutes in seconds
        rest_period = np.zeros(rest_duration)
        
        # Replace the first 3.5 minutes (210 seconds) of WR_complete with 80% GET
        # Assuming WR_complete already has warmup + PRBS
        warmup_duration = int(3.5 * 60)  # 210 seconds
        WR_modified = WR_complete.copy()
        WR_modified[:warmup_duration] = get80_w  # Replace warmup with 80% GET
        
        # Combine rest + modified PRBS protocol
        WR_with_rest = np.concatenate([rest_period, WR_modified])
        
        time_seconds = np.arange(len(WR_with_rest))
        time_minutes = time_seconds / 60.0
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        
        # --- Time markers (in minutes) ---
        rest_end = 2.0           # 2 min rest
        warmup_end = 5.5         # 2 min rest + 3.5 min warmup = 5.5 min
        first_prbs_end = 21.0    # 5.5 + 15.5 = 21 min
        second_prbs_end = 36.5   # 21 + 15.5 = 36.5 min
        
        # --- Background shading ---
        # Rest period (light gray)
        ax.axvspan(0, rest_end, color='#f5f5f5', alpha=1.0)
        
        # Warm-up period (light blue)
        ax.axvspan(rest_end, warmup_end, color='#deebf7', alpha=0.8)
        
        # PRBS sequences (darker blue)
        ax.axvspan(warmup_end, second_prbs_end, color='#6baed6', alpha=0.4)
        
        # --- Work rate line (black) ---
        ax.plot(time_minutes, WR_with_rest, color='black', linewidth=1.8)
        
        # --- Axes labels ---
        ax.set_xlabel('Time (min)', fontsize=26)
        ax.set_ylabel('Work Rate (W)', fontsize=26)
        ax.set_ylim(bottom=-10, top=250)  # Start slightly below 0 to show baseline clearly
        ax.set_xlim(left=0, right=time_minutes.max())
        ax.set_title("PRBS Protocol", fontsize=32)
        
        # --- Horizontal reference lines + ticks ---
        y_lines = [0, 25, get80_w, 231]  # Changed from [0, 25, 231]
        labels = ['0 W', '25 W', '80% GET', 'Δ40%']  # Changed from ['0 W', '25 W', 'Δ40%']
        for y_val in y_lines:
            ax.axhline(y=y_val, color='black', linestyle=':', linewidth=1)
        
        ax.set_yticks(y_lines)
        ax.set_yticklabels(labels, color='black', fontsize = 18)
        
        # Make tick labels appear just left of the y-axis
        ax.tick_params(axis='y', which='both', direction='out', pad=5)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        fig.subplots_adjust(left=0.18)
        
        # --- Vertical lines marking sections ---
        if draw_vlines:
            # First red dotted line at 5.5 minutes (end of warmup / start of first PRBS)
            
            # Second red dotted line at 21 minutes (midpoint between two PRBS sequences)
            ax.axvline(x=first_prbs_end, color='red', linestyle=':', linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    

    def plot_WR_step_protocol_only(
        self,
        get_w=176,          # GET in Watts
        get80_w=None,       # 80% GET (computed if None)
        delta30_w=220,      # Δ25% in Watts
        delta60_w=264,      # Δ50% in Watts
        get60_w=None,       # 60% GET (computed if None)
        low_w=25
    ):
        """
        Step protocol:
        - 2:00 rest @ 0 W
        - 3:30 warm-up @ 80% GET
        - 6:00 @ 80% GET
        - 6:00 @ Δ25%
        - 6:00 @ 80% GET
        - 3:00 @ Δ50%
        - 5:00 @ 25 W
        - 5:00 @ 60% GET
        Total: 2 + 3.5 + 31 = 36.5 min
        """
        if get80_w is None:
            get80_w = 0.8 * get_w
        if get60_w is None:
            get60_w = 0.6 * get_w

        # ---- durations in seconds ----
        rest_len = 120
        warmup_len = int(3.5 * 60)  # 210 s
        
        seg1_len = 6 * 60   # 80% GET
        seg2_len = 6 * 60   # Δ25%
        seg3_len = 6 * 60   # 80% GET
        seg4_len = 3 * 60   # Δ50%
        seg5_len = 5 * 60   # 25 W
        seg6_len = 5 * 60   # 60% GET

        main = np.concatenate([
            np.full(seg1_len, get80_w),
            np.full(seg2_len, delta30_w),
            np.full(seg3_len, get80_w),
            np.full(seg4_len, delta60_w),
            np.full(seg5_len, low_w),
            np.full(seg6_len, get60_w),
        ])

        WR = np.concatenate([
            np.zeros(rest_len),
            np.full(warmup_len, get80_w),
            main
        ])

        # ---- time ----
        time_seconds = np.arange(len(WR))
        time_minutes = time_seconds / 60.0

        # ---- markers (min) ----
        rest_end = 2.0
        warmup_end = 5.5
        protocol_end = 36.5

        # ---- plot ----
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # Background shading
        ax.axvspan(0, rest_end, color="#f5f5f5", alpha=1.0)
        ax.axvspan(rest_end, warmup_end, color="#deebf7", alpha=0.8)
        ax.axvspan(warmup_end, protocol_end, color="#6baed6", alpha=0.4)

        # Work rate line
        ax.plot(time_minutes, WR, color="black", linewidth=1.8)

        # Labels/limits/title
        ax.set_xlabel("Time (min)", fontsize=26)
        ax.set_ylabel("Work Rate (W)", fontsize=26)
        ax.set_xlim(0, time_minutes.max())
        y_top = max(low_w, get60_w, get80_w, delta30_w, delta60_w) + 20
        ax.set_ylim(bottom=-10, top=y_top)
        ax.set_title("Step Protocol", fontsize=32)

        # Reference lines + ticks
        y_lines = [0, low_w, get60_w, get80_w, delta30_w, delta60_w]
        labels = ["0 W", "25 W", "60% GET", "80% GET", "Δ30%", "Δ60%"]
        for y_val in y_lines:
            ax.axhline(y=y_val, color="black", linestyle=":", linewidth=1)

        ax.set_yticks(y_lines)
        ax.set_yticklabels(labels, color="black", fontsize=18)

        ax.tick_params(axis="y", which="both", direction="out", pad=5)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        fig.subplots_adjust(left=0.18)

        plt.tight_layout()
        plt.show()


    def plot_WR_interval_protocol_only(
        self,
        get80_w=141,        # 80% GET in Watts
        delta70_w=282,      # Δ60% in Watts
        low_w=25
    ):
        """
        Interval protocol:
        - 2:00 rest @ 0 W
        - 3:30 warm-up @ 80% GET
        - 10x (2:00 @ 25 W / 1:00 @ Δ60%)
        - 1:00 @ 25 W
        Total: 2 + 3.5 + 31 = 36.5 min
        """
        # ---- durations in seconds ----
        rest_len = 120
        warmup_len = int(3.5 * 60)  # 210 s

        # 10 intervals of (2 min low + 1 min high) = 30 min
        # Plus 1 min low at the end = 31 min total
        intervals = []
        for _ in range(10):
            intervals.append(np.full(2 * 60, low_w))      # 2 min at 25 W
            intervals.append(np.full(1 * 60, delta70_w))  # 1 min at Δ60%
        intervals.append(np.full(1 * 60, low_w))          # final 1 min at 25 W

        main = np.concatenate(intervals)

        WR = np.concatenate([
            np.zeros(rest_len),
            np.full(warmup_len, get80_w),
            main
        ])

        # ---- time ----
        time_seconds = np.arange(len(WR))
        time_minutes = time_seconds / 60.0

        # ---- markers (min) ----
        rest_end = 2.0
        warmup_end = 5.5
        protocol_end = 36.5

        # ---- plot ----
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # Background shading
        ax.axvspan(0, rest_end, color="#f5f5f5", alpha=1.0)
        ax.axvspan(rest_end, warmup_end, color="#deebf7", alpha=0.8)
        ax.axvspan(warmup_end, protocol_end, color="#6baed6", alpha=0.4)

        # Work rate line
        ax.plot(time_minutes, WR, color="black", linewidth=1.8)

        # Labels/limits/title
        ax.set_xlabel("Time (min)", fontsize=26)
        ax.set_ylabel("Work Rate (W)", fontsize=26)
        ax.set_xlim(0, time_minutes.max())
        y_top = delta70_w + 20
        ax.set_ylim(bottom=-10, top=y_top)
        ax.set_title("Interval Protocol", fontsize=32)

        # Reference lines + ticks
        y_lines = [0, low_w, get80_w, delta70_w]
        labels = ["0 W", "25 W", "80% GET", "Δ70%"]
        for y_val in y_lines:
            ax.axhline(y=y_val, color="black", linestyle=":", linewidth=1)

        ax.set_yticks(y_lines)
        ax.set_yticklabels(labels, color="black", fontsize=18)

        ax.tick_params(axis="y", which="both", direction="out", pad=5)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        fig.subplots_adjust(left=0.18)

        plt.tight_layout()
        plt.show()











    


   








    def plot_VO2_and_HHb_PRBS(self, VO2_prbs, HHb_prbs, WR_prbs):
        """
        Simple plot showing VO2, HHb, and Work Rate during PRBS sequence only (no warmup)
        
        Parameters:
        - VO2_prbs: VO2 data during PRBS sequence
        - HHb_prbs: HHb data during PRBS sequence
        - WR_prbs: Work Rate data during PRBS sequence
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot VO2 on left y-axis
        ax1.plot(VO2_prbs, color='blue', linewidth=2, label='VO2')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{self.VO2_text} ({self.mLMin_text})', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(bottom=0)
        
        # Plot HHb on right y-axis
        ax2 = ax1.twinx()
        ax2.plot(HHb_prbs, color='red', linewidth=2, label='HHb')
        ax2.set_ylabel('HHb (μM·cm)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Plot Work Rate on third y-axis
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(WR_prbs, color='orange', linewidth=2, label='Work Rate', alpha=0.8)
        ax3.set_ylabel('Work Rate (W)', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        ax3.set_ylim(bottom=0)
        
        # Add vertical line at the middle to separate first and second PRBS
        middle_point = len(VO2_prbs) // 2
        ax1.axvline(x=middle_point, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Mid-point')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        plt.title('VO2, HHb, and Work Rate during PRBS sequence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()