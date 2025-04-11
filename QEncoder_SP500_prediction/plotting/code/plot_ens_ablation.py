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
barWidth = 0.18
mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [66,67,72,84,100]
bars = plt.bar(
    [i for i in range(5)],
    data,
    label="No Ensemble",
    color="#BEDDE6",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='black', rotation=90, weight='extra bold', size=12)

data = [75,75,73,74,100]
bars = plt.bar(
    [i + barWidth for i in range(5)],
    data,
    label="Majority Rules",
    color="#5EA689",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

data = [81,89,81,72,100]
bars = plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="Stacking Naïve",
    color="#4B3E41",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

data = [77, 93, 88, 72, 100]
bars = plt.bar(
    [i + 3 * barWidth for i in range(5)],
    data,
    label="Homogeneous",
    color="#3f4059",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)


data = [81,92,89,73,100]
bars = plt.bar(
    [i + 4 * barWidth for i in range(5)],
    data,
    label="LexiQL",
    color="black",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

plt.ylabel("Accuracy (\%)", size=14)
plt.ylim([0, 100])
ax.set_xticks([r + 2 * barWidth for r in range(5)])
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
handles, labels = plt.gca().get_legend_handles_labels()
new_order = [0,3,1,4,2]  # Indices of the desired order
ordered_handles = [handles[idx] for idx in new_order]
ordered_labels = [labels[idx] for idx in new_order]
plt.legend(
    ordered_handles,
    ordered_labels,
    ncol=3,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.004, 1.41, 1.0, 0.102),
    borderaxespad=.1,
    fontsize=12,
    handletextpad=0.5,
    columnspacing=0.8
)
mp.setp(ax.get_yticklabels(), fontsize=12)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/ensemble_ablation.pdf", pad_inches=0.01, bbox_inches="tight")
