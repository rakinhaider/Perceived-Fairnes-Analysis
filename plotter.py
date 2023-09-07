import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from survey_info import *
from publication_plots_util import set_rcparams, set_size

results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def plot_survey(results, category_names, stats=None):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))
    # TODO: Improve the bad hack below. Reversing just based on biased labels.
    if "Very unbiased" in category_names:
        category_colors = category_colors[::-1][:]

    fig, ax = plt.subplots(figsize=set_size(width=241,
                                            fraction=.95,
                                            aspect_ratio='golden'),
                           tight_layout=True)
    print(fig.get_size_inches())
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):

        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(range(len(labels)), widths,
                        left=starts, height=0.5,
                        label=colname, color=color, tick_label=labels)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    legend = ax.legend(ncol=3, bbox_to_anchor=(0.5, 1.25),
                       loc='upper center', fontsize='xx-small')

    ax.tick_params(axis='both',
                   # labelsize=matplotlib.rcParams.get('font.size')
                   labelsize='xx-small'
                   )

    if stats is not None:
        twin = ax.twinx()
        twin.tick_params(left=False, right=True, labelright=True)
        twin.tick_params(axis='both',
                         labelsize=matplotlib.rcParams.get('font.size'))
        twin.set_ylim(ax.get_ylim())
        twin.set_yticks(ax.get_yticks())
        twin.set_yticklabels(
            ["{:.3f}".format(v) for v in stats['pvalue'].values],
            fontsize='xx-small')
    fig.tight_layout()
    print(fig.get_size_inches())
    return fig, ax


def plot_distribution(df, criteria, by='Q10.20',
                      stats=None, category_names=None):
    set_rcparams()

    df = df.copy()

    if category_names is None:
        category_names = CHOICES[by]

    grouped = df.groupby(criteria)
    percentages = pd.DataFrame()
    d = {}
    for tup, grp in grouped:
        if "DATA EXPIRED" in tup:
            continue
        if not isinstance(tup, tuple):
            tup = tuple([tup])
        counts = grp[by].value_counts() / len(grp)
        percentages[tup] = counts.reindex(category_names, fill_value=0)
        tuple_to_str = ''
        for i, t in enumerate(tup):
            if t in SCENARIO_NAME_MAP:
                tuple_to_str += SCENARIO_NAME_MAP[t]
            else:
                tuple_to_str += str(STUDY_MAP.get(t, t))
            if i != len(tup) - 1: tuple_to_str += r'\_'
        d[tuple_to_str] = percentages[tup].values.tolist()

    if stats is not None:
        stats = stats.reindex(percentages.columns)

    print(pd.DataFrame(d).transpose())
    plot_survey(d, category_names, stats)
    # ymin, ymax = plt.gca().get_ylim()
    # plt.vlines(0.5, ymin=ymin, ymax=ymax, colors='black', linestyles='dashed',
    #    linewidth=0.5)
    plt.tight_layout()


def plot_model_property(response, prop, criteria, stats=None):
    set_rcparams(fontsize=10)
    data = []
    cols = ['scenario', 'PGM', 'group', 'Ethnicity']
    qids = FAIR_QIDS if prop == 'fair' else BIAS_QIDS
    options = FAIR_OPTIONS if prop == 'fair' else BIAS_OPTIONS
    for index, row in response.iterrows():
        for qid, td in zip(qids, TRADE_OFFS):
            row_data = [row[qid], td] + [row[c] for c in cols]
            data.append(row_data)
    df = pd.DataFrame(data, columns=[prop, 'tradeoff'] + cols)
    df = df.replace(
        {'Neither biased nor unbiased.': 'Neither biased nor unbiased'})
    criteria.insert(1, 'tradeoff')
    plot_distribution(df, criteria, by=prop,
                      category_names=options, stats=stats)
    plt.show(block=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', nargs="*", default=['10312021'])
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    parser.add_argument('--what', default='test')
    args = parser.parse_args()
    response = pd.read_csv(os.path.join('data', 'processed', 'response.csv'))
    trade_off_qs = args.qid
    criteria = args.criteria

    plot_distribution(response, criteria, by=trade_off_qs)
    # plot_survey(response, criteria)
    # plot_model_property(response, criteria)
    plt.show(block=False)

    out_dir = 'outputs'
    out_dir = os.path.join(out_dir, '_'.join(args.data_dirs))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fname = '_'.join([args.what, trade_off_qs] + criteria) + '.pdf'
    plt.savefig(os.path.join(out_dir, fname), format='pdf')