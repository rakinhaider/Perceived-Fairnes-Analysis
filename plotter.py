import matplotlib.pyplot as plt
from summarizer import load_config
import pandas as pd
import numpy as np
import os
from constants import *
from survey_info import *

results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def plot_survey(results, category_names):
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

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


def plot_distribution(df, criteria):
    df = df.copy()
    trade_off_qs = 'Q10.20'
    xz_trade_off_qs = 'Q201'
    category_names = CHOICES[trade_off_qs]
    grouped = df.groupby(criteria)
    percentages = pd.DataFrame()
    d = {}
    for tup, grp in grouped:
        counts = grp[trade_off_qs].value_counts() / len(grp)
        percentages[tup] = counts.reindex(category_names, fill_value=0)
        tups = ''
        for t in tup:
            tups += str(STUDY_MAP.get(t, t)) + '_'
        d[tups] = percentages[tup].values.tolist()
    plot_survey(d, category_names)
    plt.tight_layout()


if __name__ == "__main__":
    response = pd.read_csv(os.path.join('data', 'processed', 'response.csv'))
    criteria = ['scenario', STUDY_ID, 'x_first']
    plot_distribution(response, criteria)
    # plot_survey(response, criteria)