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
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)



jump = 25
plt.ylabel("CDF")
plt.xlabel("Gate Count")

data = []
with open('yelp_2_diff_p_gates.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        data.append(number)
data=data[:300]

data.sort()
print(np.mean(data))
print(np.std(data))
print('----------')
bins = np.arange(int(min(data)), max(data)+jump, jump)
dat2 = np.histogram(data, bins)
dat2 = dat2[0]
dat2 = np.cumsum(dat2)
dat2 = dat2/dat2[-1]
ax.plot(bins[:-1], dat2, linewidth=1.5, color='maroon', label=r'\textsc{Yelp}')


data = []
with open('amazon_2_diff_p_gates.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        data.append(number)
data=data[:300]

data.sort()

print(np.mean(data))
print(np.std(data))
print('----------')
bins = np.arange(int(min(data)), max(data)+jump, jump)
dat2 = np.histogram(data, bins)
dat2 = dat2[0]
dat2 = np.cumsum(dat2)
dat2 = dat2/dat2[-1]
ax.plot(bins[:-1], dat2, linewidth=1.5, color='blue', label=r'\textsc{Amazon}')

data = []
with open('imdb_labelled_2_diff_p_gates.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        data.append(number)
data=data[:300]
data.sort()
print(np.mean(data))
print(np.std(data))
print('----------')

bins = np.arange(int(min(data)), max(data)+jump, jump)
dat2 = np.histogram(data, bins)
dat2 = dat2[0]
dat2 = np.cumsum(dat2)
dat2 = dat2/dat2[-1]
ax.plot(bins[:-1], dat2, linewidth=1.5, color='gold', label=r'\textsc{IMDB}')
















ax.set_ylim([0,1])
ax.set_yticks(np.linspace(0, 1, 5))
#ax.set_yticklabels(['0', '25', '50', '75', '100'])

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
mp.savefig("plotting/plots/cdf.png", pad_inches=0.01, bbox_inches="tight")
