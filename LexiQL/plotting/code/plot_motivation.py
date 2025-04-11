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
start_offset = 0.37
mpl.rcParams.update(params)
fig = mp.figure(figsize=(2, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.25)


data = [41]
bars = plt.bar(
    [i + start_offset for i in range(1)], data, label="IMDB", color="#2F0601", width=barWidth, edgecolor="black"
)

for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
data = [87]
bars = plt.bar(
    [i + barWidth + start_offset for i in range(1)],
    data,
    label="Lambeq MC",
    color="#26C485",
    width=barWidth,
    edgecolor="black",
)

for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
plt.ylabel("Accuracy (\%)", size=14)
#plt.xlabel("Dataset", size=14)
plt.ylim([0, 100])
plt.xlim([0,1])
ax.set_xticks([r + .5*barWidth for r in range(1)])
mp.setp(ax.get_xticklabels(), fontsize=12)
ax.set_xticklabels(
    [
        "AmpMean"
    ]
)
plt.legend(
    ncol=1,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.02, 1.3, 1.0, 0.102),
    borderaxespad=0.1,
    fontsize=12,
    handletextpad=0.4,
)
mp.setp(ax.get_yticklabels(), fontsize=14)
ax.set_xticks([start_offset + 0.5 * barWidth])
plt.setp(ax.get_xticklabels(), fontsize=12)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/motivation.pdf", pad_inches=0.01, bbox_inches="tight")
