import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from survey_info import CHOICES, CDS, CD_QS


def plot_heatmap(data, x, y, x_is_cd=True):
    if x_is_cd:
        x_ticks, y_ticks = CHOICES['CD'], CHOICES[y]
    else:
        x_ticks, y_ticks = CHOICES[x], CHOICES[y]

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_ticks)):
        for j in range(len(x_ticks)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("{} vs {}".format(x, y))
    fig.tight_layout()


def get_heatmap_data(df, x, y):
    data = []
    if x not in CHOICES:
        choices = ['High', 'Moderate', 'Low']
    else:
        choices = CHOICES[x]
    for c in choices:
        selected = df[df[x] == c]
        counts = selected[y].value_counts()
        counts = counts.reindex(CHOICES[y], fill_value=0)
        data.append(list(counts.values))
    return np.array(data).transpose()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join('data/processed', 'response.csv'))

    x, y = 'Q10.14', 'Q10.20'
    data = get_heatmap_data(df, x, y)
    plot_heatmap(np.array(data), x, y, False)

    for sc in ['icu', 'frauth']:
        for cd in CDS[:8]:
            if cd is None:
                continue
            index = CDS.index(cd)
            x = CD_QS[sc][index]
            y = 'Q10.20'
            data = get_heatmap_data(df[df['scenario'] == sc], x, y)
            plot_heatmap(data, cd, y, True)
            out_dir = os.path.join('outputs/10312021/heatmaps', sc)
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(
                os.path.join(out_dir, '{:s}_vs_{:s}'.format(cd, y)+'.pdf'),
                format='pdf')
            plt.show()
