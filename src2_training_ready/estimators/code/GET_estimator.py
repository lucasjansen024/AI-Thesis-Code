#!/usr/bin/env python
from pathlib import Path
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt

from src2_training_ready.data_types.base import ExercisePhase
from src2_training_ready.loaders.VO2_loader import VO2Loader, VO2Keys
from src2_training_ready.utils import Utils


file = askopenfilename()
file_path = Path(file)
file_name = file_path.stem
folder = file_path.parent

loader = VO2Loader(file_path)
VO2Max = Utils.calculate_VO2Max(loader.df)
df_EX = loader.df[loader.slices[ExercisePhase.EXERCISE][0]]

VO2_EX = df_EX[VO2Keys.VO2]
VCO2_EX = df_EX[VO2Keys.VCO2]
VE_EX = df_EX[VO2Keys.VE]
VEVO2_EX = df_EX[VO2Keys.VEVO2]
VEVCO2_EX = df_EX[VO2Keys.VEVCO2]
PetO2_EX = df_EX[VO2Keys.PetO2]
PetCO2_EX = df_EX[VO2Keys.PetCO2]

def create_plots():

    title = f'GET Estimation Plots {loader.details.first_name}'
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.tight_layout(pad=4)
    fig.suptitle(f'{title}, VO2MAX: {VO2Max:.1f}, Max Power: {loader.df[VO2Keys.WR].max()}W')

    sc001 = axs[0, 0].scatter(VO2_EX.index.values, VO2_EX)
    sc002 = axs[0, 0].scatter(VCO2_EX.index.values, VCO2_EX)
    axs[0, 0].set_title('VO2 and VCO2 vs time')
    axs[0, 0].set_ylabel('VO2 and VCO2')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].legend(['VO2', 'VCO2'])

    sc10 = axs[1, 0].scatter(VO2_EX, VCO2_EX)
    axs[1, 0].set_title('VO2 vs VCO2')
    axs[1, 0].set_ylabel('VCO2')
    axs[1, 0].set_xlabel('VO2')

    sc01 = axs[0, 1].scatter(VO2_EX, VEVO2_EX)
    axs[0, 1].set_title('VO2 vs VE/VO2')
    axs[0, 1].set_ylabel('VE/VO2')
    axs[0, 1].set_xlabel('VO2')

    sc11 = axs[1, 1].scatter(VO2_EX, VEVCO2_EX)
    axs[1, 1].set_title('VO2 vs VE/VCO2')
    axs[1, 1].set_ylabel('VE/VCO2')
    axs[1, 1].set_xlabel('VO2')

    sc02 = axs[0, 2].scatter(VO2_EX, PetO2_EX)
    axs[0, 2].set_title('VO2 vs PetO2')
    axs[0, 2].set_ylabel('PetO2')
    axs[0, 2].set_xlabel('VO2')

    sc12 = axs[1, 2].scatter(VO2_EX, PetCO2_EX)
    axs[1, 2].set_title('VO2 vs PetCO2')
    axs[1, 2].set_ylabel('PetCO2')
    axs[1, 2].set_xlabel('VO2')

    sc03 = axs[0, 3].scatter(VO2_EX, VE_EX)
    axs[0, 3].set_title('VE vs VO2')
    axs[0, 3].set_ylabel('VE')
    axs[0, 3].set_xlabel('VO2')

    sc13 = axs[1, 3].scatter(VCO2_EX, VE_EX)
    axs[1, 3].set_title('VE vs VCO2')
    axs[1, 3].set_ylabel('VE')
    axs[1, 3].set_xlabel('VCO2')

    scs = [sc001, sc002, sc10, sc01, sc11, sc02, sc12, sc03, sc13]

    annots = {}
    for ax in axs.flatten():
        annot = ax.annotate(
            text="",
            xy=(0,0),
            xytext=(-50, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(True)
        annots[ax] = annot

    def update_annot(annot, sc, ind):
        x, y = sc.get_offsets()[ind["ind"][0]]
        annot.xy = (x, y)
        text = f'x: {x:.0f}, y: {y:.0f}'
        annot.set_text(text)

    def onhover(event):
        for ax in axs.flatten():
            if event.inaxes == ax:
                annot = annots[ax]
                for sc in scs:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(annot, sc, ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', onhover)

    def onclick(event):
        if event.button == 1 and event.inaxes:
            event.inaxes.scatter(event.xdata, event.ydata, c='red')
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.savefig(f'{folder}/{title}.png')


def run():
    create_plots()
    plt.show()
