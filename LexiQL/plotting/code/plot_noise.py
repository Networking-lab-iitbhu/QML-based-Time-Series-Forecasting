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
barWidth = 0.21
mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)
cmap = mp.get_cmap("tab10")


data = [73,70,69,68,100]
bars = plt.bar(
    range(5),
    data,
    label="IBM Kolkata",
    color="#8f3290",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')

data = [60,66,68,77,100]
bars = plt.bar(
    [i + barWidth for i in range(5)],
    data,
    label="IBM Cairo",
    color="#f6c0c1",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='black', rotation=90, weight='extra bold')
data = [75,75,72,72,100]
bars = plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="IBM Hanoi",
    color="#946a6b",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='black', rotation=90, weight='extra bold')

data = [81,92,89,73,100]
bars = plt.bar(
    [i + 3 * barWidth for i in range(5)],
    data,
    label="Ideal",
    color="black",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold')
plt.ylabel("Accuracy (\%)", size=14)
#plt.xlabel("Dataset", size=14)
plt.ylim([0, 100])
ax.set_xticks([r + 1.5 * barWidth for r in range(5)])
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
handles, labels = plt.gca().get_legend_handles_labels()
new_order = [0,2,1,3]  # Indices of the desired order
ordered_handles = [handles[idx] for idx in new_order]
ordered_labels = [labels[idx] for idx in new_order]
plt.legend(
    ncol=4,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.018, 1.18, 1.0, 0.102),
    borderaxespad=0.5,
    fontsize=12,
    handletextpad=0.3,
    columnspacing=1
)
mp.setp(ax.get_yticklabels(), fontsize=14)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/noise.pdf", pad_inches=0.01, bbox_inches="tight")
