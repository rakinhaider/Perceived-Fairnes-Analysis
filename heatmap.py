import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import survey_info
from survey_info import CHOICES, CDS, SF_MAP
from utils import get_parser, aggregate_response, merge_cd_columns
from publication_plots_util import set_rcparams, set_size


def plot_heatmap(data, x1, x2, y, fig, ax,
                 x_is_cd=True, show_ylabels=True,
                 choice=None, annot=False):
    x_ticks = [-2, -1, 0, 1, 2]
    y_ticks = CHOICES[y][::-1]
    title = '(IndFPImpact - IndFNImpact) vs (Overall preference between model X and Y)'
    y_ticks = [yt.replace('${e://Field/pref_model}', 'model X/Y')
               for yt in y_ticks]
    data = data[::-1]
    im = ax.imshow(data)

    # fontsize = matplotlib.rcParams['font.size']
    fontsize = 'x-small'
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_ticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_ticks)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    if show_ylabels:
        y_ticks = [2, 1, 0, -1, -2]
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
        plt.setp(ax.get_yticklabels(), fontsize='xx-small')
    else:
        ax.set_yticklabels([])

    # Loop over data dimensions and create text annotations.
    if annot:
        print(data)
        for i in range(len(y_ticks)):
            for j in range(len(x_ticks)):
                text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                               ha="center", va="center", color="w",
                               fontsize=fontsize)

    xlabel = f'{SF_MAP.get(x1, x1)} - {SF_MAP.get(x2, x2)}'
    ax.set_xlabel(xlabel, fontsize='x-small')
    # ylabel = f'Preference between\\\\{choice}'
    # ylabel = '\\begin{center}' + ylabel + '\\end{center}'
    ylabels = f'XY Preferences'
    ax.set_ylabel(ylabels, fontsize='x-small')
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1/25)
    pad = axes_size.Fraction(0.25, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fontsize)


def get_heatmap_data(df, x1, x2, y):
    data = []
    choices = CHOICES['CD']
    df = df.copy(deep=True)
    if x1[0] == 'B':
        df = df.replace({
            'IFPI': dict(zip(choices[::-1], range(len(choices)))),
            'SFPI': dict(zip(choices[::-1], range(len(choices))))
        })
        df = df.replace({
            'IFNI': dict(zip(choices[::-1], range(len(choices)))),
            'SFNI': dict(zip(choices[::-1], range(len(choices))))
        })
        df['BFPI'] = ((df['IFPI'] + df['SFPI']) / 2).round()
        df['BFNI'] = ((df['IFNI'] + df['SFNI']) / 2).round()
    else:
        df = df.replace({
            x1: dict(zip(choices[::-1], range(len(choices))))
        })
        df = df.replace({
            x2: dict(zip(choices[::-1], range(len(choices))))
        })
    df['differences'] = df[x1] - df[x2]
    # print(x1, x2)
    # print(df[x1] - df[x2])
    # print('##########################################################################')
    # print(df['differences'].value_counts())
    for diff, count in df['differences'].value_counts().sort_index().items():
        selected = df[df['differences'] == diff]
        # print(diff, selected[y].value_counts().reindex(CHOICES[y], fill_value=0))
        percentages = selected[y].value_counts() / len(selected)
        percentages = percentages.reindex(CHOICES[y], fill_value=0)
        print(diff, percentages)
        # print(percentages)
        data.append(list(percentages.values))
    return np.array(data).transpose()


if __name__ == "__main__":

    args = get_parser().parse_args()
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]
    response = aggregate_response(args.resp_dirs, fnames)
    response = merge_cd_columns(response)
    print(args.criteria)
    print(args.fnames)
    if args.criteria == ['None']:
        args.criteria = None

    set_rcparams(fontsize=9)

    if args.criteria:
        grouped = response.groupby(by=args.criteria)
    else:
        grouped = [('all', response)]
    choices = {'Q10.20': 'Model X vs Model Y', 'Q201': 'Model X or Y vs Model Z'}
    x1s = ['IFPI', 'IFPI', 'SFPI', 'SFPI', 'BFPI', 'BFPI']
    x2s = ['IFNI', 'IFNI', 'SFNI', 'SFNI', 'BFNI', 'BFNI']
    ys = ['Q10.20', 'Q201', 'Q10.20', 'Q201', 'Q10.20', 'Q201']
    for i, (x1, x2, y) in enumerate(zip(x1s, x2s, ys)):
        for tup, grp in grouped:
            fig, ax = plt.gcf(), plt.gca()
            print(fig.get_size_inches())
            fig.set_size_inches(set_size(width=200, fraction=0.9,
                                         aspect_ratio=.7))
            fig.tight_layout()
            heatmap_data = get_heatmap_data(grp, x1, x2, y)
            plot_heatmap(heatmap_data, x1, x2, y, fig, ax,
                         show_ylabels=True, choice=choices[y], annot=True)
            out_dir = os.path.join('outputs', "_".join(args.resp_dirs),
                                   'heatmaps', 'diff', tup)
            fig.tight_layout()
            print(fig.get_size_inches())
            os.makedirs(out_dir, exist_ok=True)
            print(out_dir)
            plt.savefig(os.path.join(
                 out_dir, 'diff({}, {})_vs_{}'.format(SF_MAP.get(x1, x1), SF_MAP.get(x2, x2), y) + '.pdf'),
                 format='pdf')

            plt.close()
