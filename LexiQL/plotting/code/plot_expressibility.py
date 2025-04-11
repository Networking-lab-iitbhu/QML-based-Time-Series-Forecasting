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
fig = mp.figure(figsize=(3.0, 2.1))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="lightgrey", linewidth=0.25)
#ax.xaxis.grid(linestyle=":", color="grey", linewidth=0.5)

#A,B,C,D
exp = [0.28728904310687065, 0.16420411619365774, 0.1175242312496448, 0.023887881252827453]
#exp = [0.014177840509161516,0.019071740490037913,0.1175242312496448,0.006585154994574126]
gate_counts = [11, 15,19, 16]
#gate_counts = [33, 30,19, 48]
plt.ylim([0, .3])

plt.scatter(gate_counts, exp,color='red')
plt.ylabel('Expressibility', fontsize=13)
plt.xlabel('Gate Count', fontsize=14)
mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)
mp.xticks([10, 15, 20])
mp.tight_layout()

# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig("plotting/plots/expressibility.pdf", pad_inches=0.01, bbox_inches="tight")
