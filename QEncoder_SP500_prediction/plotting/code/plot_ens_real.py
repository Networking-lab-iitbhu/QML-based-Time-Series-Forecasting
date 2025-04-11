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
barWidth = 0.5
mpl.rcParams.update(params)
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)


data = [
    34.5,
    48.7,
    48.7,
    66.45,
    66.77,
    62.1,
    73.87,
    74.8,
    70.96,
    68.39,
    70.6,
    73.9,
    82.2,
    78.8,
    82.2,
]
plt.bar(
    [i for i in range(15)],
    data,
    label="SymphoniQ - Kolkata",
    color="black",
    width=barWidth,
    edgecolor="black",
)


# plt.ylabel('Accuracy (\%)')
# plt.xlabel('Ensemble Members')
# plt.ylim([0, 100])
# plt.xticks([r for r in range(15)],
#         range(1,16))
# plt.legend(ncol=1, edgecolor='black', loc='upper right', bbox_to_anchor=(0.01, 1.075, 1., .102),
# borderaxespad=0.2, fontsize=10, handletextpad=0.5)
# mp.tight_layout()
# #plt.savefig(f'plots/{dataset}_full_accuracy_200')
# #plt.show()
# plt.savefig(f'plots/ensemble_real')


plt.ylabel("Accuracy (\%)")
plt.xlabel("Dataset")
plt.ylim([0, 100])
ax.set_xticks([r + 0 * barWidth for r in range(15)])
mp.setp(ax.get_xticklabels(), fontsize=9.5)
ax.set_xticklabels(range(1, 16))
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
mp.savefig("plots/ensemble_real.pdf", pad_inches=0.01, bbox_inches="tight")
