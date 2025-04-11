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
fig = mp.figure(figsize=(6.0, 2.5))
fig.subplots_adjust(left=0.12, top=0.7, right=0.999, bottom=0.248)
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(linestyle=":", color="grey", linewidth=0.5)
data = {
'Depth1_Circuit5': [
    [81,87,	73,	97,	79],
    [69,70,64,95,78],
    [56,59,57,99,80],
    [66,68,66,93,82]
],
'Depth1_Circuit2': [
    [83,88,77,100,77],
    [65,66,67,100,79],
    [67,63,58,100,70],
    [66,54,58,100,74]
],
    'Depth1_Circuit2_and_5': [
    [80,87,78,100,81],
    [70,67,65,100,78],
    [74,59,50,100,77],
    [71,76,61,100,76]
],

    'Depth2_Circuit5': [
    [77,90,79,100,74],
    [53,69,64,100,71],
    [53,54,60,93,66],
    [63,55,65,100,69]
],
'Depth2_Circuit2': [
    [71,71,76,100,71],
    [68,64,60,99,68],
    [66,62,57,100,65],
    [72,68,63,100,65]
],
    'Depth2_Circuit2_and_5': [
    [74,91,80,100,68],
    [71,66,66,100,75],
    [63,57,49,100,69],
    [71,63,61,100,68]
],

    'Depth3_Circuit5': [
    [88,93,77,100,72],
    [53,52,51,100,73],
    [50,51,51,90,67],
    [49,55,48,100,65]
],
'Depth3_Circuit2': [
    [79,87,73,100,72],
    [67,58,57,98,57],
    [55,51,53,100,72],
    [67,67,65,100,61]
],
    'Depth3_Circuit2_and_5': [
    [82,91,74,100,70],
    [70,54,56,100,68],
    [71,58,47,100,55],
    [67,58,60,100,70]
],

    'Depth3_Circuit2_and_Depth1_Circuit5': [
    [81,90,68,100,81],
    [72,65,62,100,79],
    [71,60,50,100,78],
    [75,69,68,100,77]
],
    'Depth1_C': [
    [83,80,80,100,70],
],
    'Depth1_F': [
    [82,92,78,100,76],
],
    'Depth1_G': [
    [82,90,74,100,74],
],
    'Depth2_C': [
    [88,92,83,100,75],
],
    'Depth2_F': [
    [83,91,85,100,77],
],
    'Depth2_G': [
    [83,96,81,100,74],
],
    'Depth3_C': [
    [83,90,82,100,76],
],
    'Depth3_F': [
    [87,93,89,100,75],
],
    'Depth3_G': [
    [88,89,86,100,73],
],
}
order = [2,1,0,4,3]
idealE = [data['Depth1_Circuit5'][0][i] for i in order]
idealB = [data['Depth1_Circuit2'][0][i] for i in order]
idealC = [data['Depth1_C'][0][i] for i in order]
idealF = [data['Depth1_F'][0][i] for i in order]
idealG = [data['Depth1_G'][0][i] for i in order]

idealE = [data['Depth2_Circuit5'][0][i] for i in order]
idealB = [data['Depth2_Circuit2'][0][i] for i in order]
idealC = [data['Depth2_C'][0][i] for i in order]
idealF = [data['Depth2_F'][0][i] for i in order]
idealG = [data['Depth2_G'][0][i] for i in order]

idealE = [data['Depth3_Circuit5'][0][i] for i in order]
idealB = [data['Depth3_Circuit2'][0][i] for i in order]
idealC = [data['Depth3_C'][0][i] for i in order]
idealF = [data['Depth3_F'][0][i] for i in order]
idealG = [data['Depth3_G'][0][i] for i in order]
bars = plt.bar(
    range(5), idealB, label="B", color="#2F0601", width=barWidth, edgecolor="black"
)

for bar, yval in zip(bars, idealB):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

bars = plt.bar(
    [i + barWidth for i in range(5)],
    idealC,
    label="C",
    color="#C0C1F6",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, idealC):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='black', rotation=90, weight='extra bold', size=12)
bars = plt.bar(
    [i + 2 * barWidth for i in range(5)],
    idealE,
    label="E",
    color="#6A6B94",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, idealE):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)

bars = plt.bar(
    [i + 3 * barWidth for i in range(5)],
    idealF,
    label="F",
    color="#32908F",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, idealF):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)



bars = plt.bar(
    [i + 4 * barWidth for i in range(5)],
    idealG,
    label="G",
    color="#49493F",
    width=barWidth,
    edgecolor="black",
)
for bar, yval in zip(bars, idealG):
    plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)


plt.ylabel("Accuracy (\%)", size=14)
plt.xlabel("Dataset", size=14)
plt.ylim([0, 100])
ax.set_xticks([r + 2 * barWidth for r in range(5)])
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
    ncol=5,
    edgecolor="black",
    loc="upper right",
    bbox_to_anchor=(0.01, 1.1, 1.0, 0.102),
    borderaxespad=0.2,
    fontsize=12,
    handletextpad=0.5,
)
mp.setp(ax.get_yticklabels(), fontsize=14)
mp.tight_layout()
# plt.savefig(f'plots/{dataset}_full_accuracy_200')
# plt.show()
mp.savefig(f"plotting/plots/anz_d3.pdf", pad_inches=0.01, bbox_inches="tight")
mp.clf()
