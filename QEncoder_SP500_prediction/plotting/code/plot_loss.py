from matplotlib import pyplot as plt
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as mp
import numpy as np

params = {
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.texsystem": "xelatex",
    "pgf.preamble": r"\usepackage{fontspec,physics}",
}

mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)
ax.xaxis.grid(linestyle=":", color="grey", linewidth=0.5)


def sliding_window(data):
    all_points = []
    buffer = [data[0], data[0], data[0], data[0], data[0]]
    for i in range(len(data)):
        buffer.append(data[i])
        buffer = buffer[1:]
        all_points.append(np.mean(buffer))
    return all_points

circE=np.load('plotting/code/experiment_losses_imdb_labelled_machine_aware_BCE_2_15_2_5_selective_1_3.npy')
circB=np.load('plotting/code/experiment_losses_imdb_labelled_machine_aware_BCE_2_15_2_5_selective_1_4.npy')

dataE=sliding_window(circE)
dataB=sliding_window(circB)
plt.ylim((.6,.8))
plt.plot(dataB)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Smoothed Loss Curve Circuit B')
plt.savefig("plotting/plots/CircBLoss.pdf", pad_inches=0.01, bbox_inches="tight")