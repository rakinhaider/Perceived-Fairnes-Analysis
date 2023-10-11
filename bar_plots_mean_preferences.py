import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_parser, map_items_to_value
from survey_response_aggregator import aggregate_response
from publication_plots_util import set_rcparams, set_size
from survey_info import SCENARIO_NAME_MAP
from hypothesis_testing import SCENARIOS, merge_cd_columns

VALUE_SHORT_FORM = {'Disadvantaged': 'Disadv.',
                    'Advantaged': 'Adv.',
                    'Caucasian': 'Cauc.',
                    'Non-Caucasian': 'Non-Cauc.'}
CRITERIA_TO_TEXT = {'group': 'Disadv. Group',
                    "Ethnicity": 'Ethnicity',
                    'PGM': 'self-identified privilege'}


if __name__ == "__main__":
    set_rcparams(fontsize=10)
    parser = get_parser()
    parser.add_argument('--x-or-y', '-xy', default=None, type=str)
    args = parser.parse_args()

    criteria = args.criteria
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    response = aggregate_response(args.resp_dirs, fnames)
    response = map_items_to_value(response)

    qid = args.qid
    xz_or_yz = args.x_or_y.upper() if args.x_or_y is not None else None
    grouping_criteria = args.criteria
    if qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']

    fig, axes = plt.subplots(2, 2,
                    figsize=set_size(430 * 0.75, 0.95,
                                     aspect_ratio=0.75),
                    tight_layout=True)

    for i, j, grouping_criteria in [(0, 0, None), (0, 1, 'group'),
                                    (1, 0, 'Ethnicity'), (1, 1, 'PGM')]:

        ax = axes[i][j]
        colors = ['green', 'blue', 'red']
        x_pos_start = 1
        for scenario, c in zip(SCENARIOS, colors):
            scneario_grp = response[response['scenario'] == scenario].copy(deep=True)
            if grouping_criteria is not None:
                grouped = scneario_grp.groupby(by=grouping_criteria)
                bar_width = 0.9
                pos_increment = 1
            else:
                grouped = [('', scneario_grp)]
                bar_width = 0.25
                pos_increment = 0.35
            columns = []
            x_tick_labels = []
            for val, grp in grouped:
                columns.append(grp[qid])
                x_tick_labels.append(val)
            means = [c.mean() for c in columns]
            errors = [c.std() for c in columns]
            errors = [c.std() / np.sqrt(len(c)) for c in columns]

            x_poss = [x_pos_start, x_pos_start + 4][:len(columns)]
            bars = ax.bar(x_poss, means, yerr=errors, width=bar_width,
                          capsize=5, label=SCENARIO_NAME_MAP[scenario])
            for bar in bars:
                bar.set_color(c)
            x_pos_start += pos_increment

        ax.tick_params(axis='x', length=0)
        ax.set_xticks([2, 6] if grouping_criteria is not None
                      else [1 + pos_increment])
        ax.set_xticklabels(x_tick_labels, fontsize='small')
        ax.set_yticks(range(-2, 3))
        ax.set_yticklabels(range(-2, 3), fontsize='small')
        ax.set_ylim(-2, 2.2)
        if grouping_criteria:
            x_label = f'Partitioned by {CRITERIA_TO_TEXT[grouping_criteria]}'
        else:  x_label = 'Overall'
        ax.set_xlabel(x_label, fontsize='small')
        ax.legend(ncols=3, loc='upper center', fontsize='4.5')

    if qid == 'Q10.20':
        fname = 'xy_preference'
    else:
        fname = f'{args.x_or_y.lower()}z_preference'

    dir = os.path.join('outputs/10312021_11092021/barplots/mean_preferences')
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, fname + '.pdf'), format='pdf')
