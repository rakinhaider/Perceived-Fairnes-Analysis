import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from survey_info import CHOICES, CDS
from utils import get_parser, aggregate_response, merge_cd_columns
from publication_plots_util import set_rcparams, set_size


def plot_heatmap(data, x, y, fig, ax,
                 x_is_cd=True, show_ylabels=True,
                 title=None, annot=False):
    if x_is_cd:
        x_ticks, y_ticks = CHOICES['CD'], CHOICES[y]
    else:
        x_ticks, y_ticks = CHOICES[x], CHOICES[y]

    y_ticks = [yt.replace('${e://Field/pref_model}', 'model X/Y')
               for yt in y_ticks]
    im = ax.imshow(data)

    # fontsize = matplotlib.rcParams['font.size']
    fontsize = 'small'
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_ticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_ticks)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=fontsize)

    if show_ylabels:
        ax.set_yticks(np.arange(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor",
                 fontsize=fontsize)
    else:
        ax.set_yticklabels([])

    # Loop over data dimensions and create text annotations.
    if annot:
        for i in range(len(y_ticks)):
            for j in range(len(x_ticks)):
                text = ax.text(j, i, '{:.2f}'.format(data[i, j]),
                               ha="center", va="center", color="w",
                               fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=matplotlib.rcParams['font.size'])
    else:
        ax.set_title("{} / {}".format(x, y),
                     fontsize=matplotlib.rcParams['font.size'])

    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1/25)
    pad = axes_size.Fraction(0.5, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fontsize)


def get_heatmap_data(df, x, y):
    data = []
    if x not in CHOICES:
        choices = ['High', 'Moderate', 'Low']
    else:
        choices = CHOICES[x]
    for c in choices:
        selected = df[df[x] == c]
        percentages = selected[y].value_counts() / len(selected)
        # percentages = selected[y].value_counts()
        percentages = percentages.reindex(CHOICES[y], fill_value=0)
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
    # exit()

    set_rcparams(fontsize=9)
    if False:
        x, y = 'Q10.14', 'Q10.20'
        data = get_heatmap_data(response, x, y)
        plot_heatmap(np.array(data), x, y, plt.gcf(), plt.gca(), x_is_cd=False)
        plt.show()

        x, y = 'Q10.16', 'Q10.20'
        data = get_heatmap_data(response, x, y)
        plot_heatmap(np.array(data), x, y, plt.gcf(), plt.gca(), x_is_cd=False)
        plt.show()

        x, y = 'Q10.18', 'Q10.20'
        data = get_heatmap_data(response, x, y)
        plot_heatmap(np.array(data), x, y, plt.gcf(), plt.gca(), x_is_cd=False)
        plt.show()

    if args.criteria:
        grouped = response.groupby(by=args.criteria)
    else:
        grouped = [('all', response)]
    titles = ['{}/Model X vs Model Y'] * 2 + ['{}/Model X or Y vs Model Z'] * 2
    xs = ['IFPI', 'SFPI', 'IFNI', 'SFNI']
    ys = ['Q10.20', 'Q10.20', 'Q201', 'Q201']
    for i, (x, y) in enumerate(zip(xs, ys)):
        for tup, grp in grouped:
            fig, ax = plt.gcf(), plt.gca()
            fig.set_size_inches(set_size(width=200, fraction=0.9,
                                         aspect_ratio=.7))
            heatmap_data = get_heatmap_data(grp, x, y)
            plot_heatmap(heatmap_data, x, y, fig, ax,
                         show_ylabels=True,
                         title=titles[i].format(x), annot=True)
            out_dir = os.path.join('outputs', "_".join(args.resp_dirs),
                                   'heatmaps', tup)
            os.makedirs(out_dir, exist_ok=True)
            print(out_dir)
            plt.savefig(os.path.join(
                out_dir, '{}_vs_choice'.format(x) + '.pdf'),
                format='pdf', bbox_inches='tight')

            plt.close()
            # plt.show()
            # exit()

