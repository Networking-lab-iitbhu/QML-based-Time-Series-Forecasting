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
barWidth = 0.15
mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [50.7, 52.2, 47.4, 80.3, 81.3]
plt.bar(
    [i for i in range(5)],
    data,
    label="IBM Cairo",
    color="grey",
    width=barWidth,
    edgecolor="black",
)

data = [52.2, 57.5, 52.8, 85.4, 100]
plt.bar(
    [i + barWidth for i in range(5)],
    data,
    label="IBM Hanoi",
    color="blue",
    width=barWidth,
    edgecolor="black",
)

data = [56.2, 58.1, 47.5, 80, 100]
plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="IBM Kolkata",
    color="orange",
    width=barWidth,
    edgecolor="black",
)

plt.ylabel("Accuracy (\%)")
plt.xlabel("Dataset")
plt.ylim([0, 100])
ax.set_xticks([r + 1 * barWidth for r in range(5)])
mp.setp(ax.get_xticklabels(), fontsize=9.5)
ax.set_xticklabels(
    [
        "IMDB",
        "Amazon",
        "Yelp",
        "Lambeq RP",
        "Lambeq MC",
    ]
)
plt.legend(
    ncol=4,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.01, 1.065, 1.0, 0.102),
    borderaxespad=0.2,
    fontsize=10,
    handletextpad=0.5,
)
mp.setp(ax.get_yticklabels(), fontsize=14)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plots/sim_noise.pdf", pad_inches=0.01, bbox_inches="tight")
