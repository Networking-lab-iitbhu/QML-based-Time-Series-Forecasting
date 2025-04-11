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
fig = mp.figure(figsize=(6.0, 2))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [48,59,39,61,53]
bars = plt.bar(
    range(5), data, label="AmpMean [2]", color="#2F0601", width=barWidth, edgecolor="black"
)

for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
data = [60,63,63,61,100]
bars = plt.bar(
    [i + barWidth for i in range(5)],
    data,
    label="DisCoCat [25]",
    color="#26C485",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
data = [75,75,72,72,100]
bars = plt.bar(
    [i + 2 * barWidth for i in range(5)],
    data,
    label="LexiQL",
    color="black",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, data):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
plt.ylabel("Accuracy (\%)\n(IBM Hanoi)", fontsize=14)
#plt.xlabel("Dataset", size=14)
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
    ncol=4,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.008, 1.17, 1.0, 0.102),
    borderaxespad=0.2,
    fontsize=12,
    handletextpad=0.5,
)
mp.setp(ax.get_yticklabels(), fontsize=14)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/flagship_noise.pdf", pad_inches=0.01, bbox_inches="tight")
