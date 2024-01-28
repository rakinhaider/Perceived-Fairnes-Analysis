import os
import numpy as np
import matplotlib.pyplot as plt

from constants import CRITERIA_TO_TEXT
from utils import get_parser, map_items_to_value
from utils import aggregate_response
from plotters.publication_plots_util import set_rcparams, set_size
from survey_info import SCENARIO_NAME_MAP, SCENARIOS

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


    for i, grouping_criteria in [(0, None), (1, 'group'),
                                 (2, 'Ethnicity'), (3, 'PGM')]:
        fig, ax = plt.subplots(
            figsize=set_size(107, 0.95, aspect_ratio=1))
        # ax = axes[i]
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
                x_tick_labels.append(val.capitalize())
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
        ax.set_xticklabels(x_tick_labels, fontsize='x-small')
        ax.set_yticks(range(-2, 3))
        ax.set_yticklabels(range(-2, 3), fontsize='small')
        ax.set_ylim(-2, 2.2)
        ax.legend(fontsize=5, loc='best')

        if qid == 'Q10.20':
            fname = 'xy_preference'
        else:
            fname = f'{args.x_or_y.lower()}z_preference'

        dir = os.path.join('outputs/10312021_11092021/barplots/mean_preferences')
        os.makedirs(dir, exist_ok=True)
        fname += f'_{grouping_criteria}'
        print(os.path.join(dir, fname + '.pdf'))
        plt.tight_layout()
        plt.savefig(os.path.join(dir, fname + '.pdf'), format='pdf')
