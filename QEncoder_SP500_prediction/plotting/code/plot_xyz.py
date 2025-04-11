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
ax.xaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [50, 51, 49, 61, 50]
plt.bar(range(5), data, label="XYZ", color="green", width=barWidth)


data = [63, 62.5, 60, 81, 100]
plt.bar(
    [i + barWidth for i in range(5)], data, label="XZ", color="blue", width=barWidth
)

data = [79.2, 87.2, 92.2, 98.6, 100]
plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="SymphoniQ (Single Member)",
    color="black",
    width=barWidth,
)

plt.ylabel("Accuracy (\%)")
plt.xlabel("Dataset")
plt.ylim([0, 100])
plt.xticks(
    [r + barWidth for r in range(5)],
    [
        "IMDB",
        "Amazon",
        "Yelp",
        "Lambeq RP",
        "Lambeq MC",
    ],
)
plt.legend(
    ncol=3,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.01, 1.075, 1.0, 0.102),
    borderaxespad=0.2,
    fontsize=10,
    handletextpad=0.5,
)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
plt.savefig(f"plots/xyz")
