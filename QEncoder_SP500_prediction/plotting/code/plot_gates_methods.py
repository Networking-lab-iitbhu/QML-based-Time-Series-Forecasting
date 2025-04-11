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
cmap = mp.get_cmap("tab10")


data = [211, 208.4, 245.2]
yerr = [88.14, 91.4, 124.3]
bars = plt.bar(
    range(3),
    data,
    label="Low Depth No Padding Machine Aware",
    color="orange",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center',  color='white', rotation=90)

data = [220.3, 212.2, 241.9]
yerr = [97.8,89.4, 124.6]
bars = plt.bar(
    [i + barWidth for i in range(3)],
    data,
    label="Low Depth No Padding",
    color="purple",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
data = [422.7, 411.5, 464.4]
yerr = [181.5, 178.6,237.7 ]
bars = plt.bar(
    [i + 2*barWidth for i in range(3)],
    data,
    label="No Padding",
    color="cyan",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
data = [742, 742, 742]
bars = plt.bar(
    [i + 3 * barWidth for i in range(3)],
    data,
    label="Low Depth",
    color="magenta",
    width=barWidth,
    edgecolor="black",
)

for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
data = [1038, 1038, 1038]
bars = plt.bar(
    [i + 4 * barWidth for i in range(3)],
    data,
    label="Machine Aware",
    color="blue",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
data = [1062, 1062, 1062]
bars = plt.bar(
    [i + 5 * barWidth for i in range(3)],
    data,
    label="Unchanged",
    color="black",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2-50, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
plt.ylabel("Gate Count")
plt.xlabel("Dataset")
plt.ylim([0, 1100])
ax.set_xticks([r + 3 * barWidth for r in range(3)])
mp.setp(ax.get_xticklabels(), fontsize=9.5)
ax.set_xticklabels(
    [
        "Yelp",
        "Amazon",
        "IMDB"
    ]
)
plt.legend(
    ncol=3,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.01, 1.19, 1.0, 0.102),
    borderaxespad=0.2,
    fontsize=9,
    handletextpad=0.5,
)

mp.setp(ax.get_yticklabels(), fontsize=14)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/gates_methods.png", pad_inches=0.01, bbox_inches="tight")
