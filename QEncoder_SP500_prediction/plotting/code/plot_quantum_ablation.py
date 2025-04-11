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

barWidth = 0.28
mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2.4))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [41,55,44,61,87]
bars = plt.bar(
    [i for i in range(5)],
    data,
    label="W/out Incremenetal Data Injection w/out Encoder",
    color="#553A41",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

data = [64, 68,66,55,97]
bars = plt.bar(
    [i + barWidth for i in range(5)],
    data,
    label="Incremenetal Data Injection w/out Encoder",
    color="#A3E7FC",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='black', rotation=90, weight='extra bold', size=12)
data = [66,67,72,84,100]
bars = plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="Incremenetal Data Injection with Encoder",
    color="#946a6b",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
plt.ylabel("Accuracy (\%)", fontsize=14)
plt.ylim([0, 100])
ax.set_xticks([r + 1 * barWidth for r in range(5)])
mp.setp(ax.get_xticklabels(), fontsize=12)
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
    ncol=1,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.001, 1.83, 1.0, 0.102),
    borderaxespad=0.01,
    fontsize=12,
    handletextpad=0.5,
)
mp.setp(ax.get_yticklabels(), fontsize=12)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/quantum_ablation.pdf", pad_inches=0.01, bbox_inches="tight")
