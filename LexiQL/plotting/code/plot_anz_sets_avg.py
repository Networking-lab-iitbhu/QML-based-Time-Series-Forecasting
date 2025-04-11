from matplotlib import pyplot as plt
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as mp
import numpy as np





def plot(ds, env):
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
    data = {
    'D1C5': [
        [81,87,	73,	97,	79],
        [69,70,64,95,78],
        [56,59,57,99,80],
        [66,68,66,93,82]
    ],
    'D1C2': [
        [83,88,77,100,77],
        [65,66,67,100,79],
        [67,63,58,100,70],
        [66,54,58,100,74]
    ],
        'D1C2_5': [
        [80,87,78,100,81],
        [70,67,65,100,78],
        [74,59,50,100,77],
        [71,76,61,100,76]
    ],

        'D2C5': [
        [77,90,79,100,74],
        [53,69,64,100,71],
        [53,54,60,93,66],
        [63,55,65,100,69]
    ],
    'D2C2': [
        [71,71,76,100,71],
        [68,64,60,99,68],
        [66,62,57,100,65],
        [72,68,63,100,65]
    ],
        'D2C2_5': [
        [74,91,80,100,68],
        [71,66,66,100,75],
        [63,57,49,100,69],
        [71,63,61,100,68]
    ],

        'D3C5': [
        [88,93,77,100,72],
        [53,52,51,100,73],
        [50,51,51,90,67],
        [49,55,48,100,65]
    ],
    'D3C2': [
        [79,87,73,100,72],
        [67,58,57,98,57],
        [55,51,53,100,72],
        [67,67,65,100,61]
    ],
        'D3C25': [
        [82,91,74,100,70],
        [70,54,56,100,68],
        [71,58,47,100,55],
        [67,58,60,100,70]
    ],

        'Mixed': [
        [81,90,68,100,81],
        [72,65,62,100,79],
        [71,60,50,100,78],
        [75,69,68,100,77]
    ],
    
}
    order = [2,1,0,4,3]
    yelp = []
    for key in data.keys():
        if ds == 'yelp':
            for i in data[key]:
                yelp.append(i[0])
        elif ds == 'amazon':
            for i in data[key]:
                yelp.append(i[1])
        elif ds == 'imdb':
            for i in data[key]:
                yelp.append(i[2])
        elif ds == 'lambeq1':
            for i in data[key]:
                yelp.append(i[3])
        elif ds == 'lambeq2':
            for i in data[key]:
                yelp.append(i[4])

    ideal = [yelp[i] for i in range(len(yelp)) if i%4==0]
    kolkata = [yelp[i] for i in range(len(yelp)) if i%4==1]
    cairo = [yelp[i] for i in range(len(yelp)) if i%4==2]
    hanoi = [yelp[i] for i in range(len(yelp)) if i%4==3]
    if env == 'ideal':
        env_data = ideal
    if env == 'kolkata':
        env_data = kolkata
    if env == 'cairo':
        env_data = cairo
    if env == 'hanoi':
        env_data = hanoi
    bars = plt.bar(
        range(10), env_data, label=f"{ds}, {env}", color="#2F0601", width=barWidth, edgecolor="black"
    )

    for bar, yval in zip(bars, env_data):
        plt.text(bar.get_x() + bar.get_width() / 2, 3, str(round(yval)), ha='center', va='bottom', color='white', rotation=90, weight='extra bold', size=12)
    

    plt.ylabel("Accuracy (\%)", size=14)
    plt.xlabel("Dataset, Env", size=14)
    plt.ylim([0, 100])
    ax.set_xticks([r for r in range(10)])
    mp.setp(ax.get_xticklabels(), fontsize=9)
    ax.set_xticklabels(
        list(data.keys())    )
    plt.legend(
        ncol=4,
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
    mp.savefig(f"plotting/plots/anz_{ds}_{env}.pdf", pad_inches=0.01, bbox_inches="tight")
    mp.clf()
for i in ['yelp', 'imdb', 'amazon', 'lambeq1', 'lambeq2']:
    for j in ['hanoi', 'kolkata', 'ideal', 'cairo']:
        plot(i, j)
