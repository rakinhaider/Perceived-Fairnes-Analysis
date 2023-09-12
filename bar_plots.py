import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_parser
from survey_response_aggregator import aggregate_response
from publication_plots_util import set_rcparams, set_size
from survey_info import SCENARIO_NAME_MAP, CHOICES

VALUE_SHORT_FORM = {'Disadvantaged': 'Disadv.',
                    'Advantaged': 'Adv.',
                    'Caucasian': 'Cauc.',
                    'Non-Caucasian': 'Non-Cauc.'}


def get_preferred_model(row):
    if row['Q201'] in CHOICES['Q201'][-2:]:
        return 'z'
    elif row['Q201'] in CHOICES['Q201'][:2]:
        if row['Q10.20'] in CHOICES['Q10.20'][-2:]:
            return 'y'
        elif row['Q10.20'] in CHOICES['Q10.20'][:2]:
            return 'x'
        else:
            return 'xy'
    else:
        if row['Q10.20'] in CHOICES['Q10.20'][-2:]:
            return 'y'
        elif row['Q10.20'] in CHOICES['Q10.20'][:2]:
            return 'x'
        else:
            return 'xyz'


if __name__ == "__main__":
    plt.style.use('default')

    set_rcparams(fontsize=10)
    parser = get_parser()
    args = parser.parse_args()

    criteria = args.criteria
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    response = aggregate_response(args.resp_dirs, fnames)

    fig, axes = plt.subplots(len(args.criteria), 1,
                    figsize=set_size(241 * 0.9, 0.95,
                                     aspect_ratio=0.85),
                    tight_layout=True)
    if len(criteria) == 1:
        axes = [axes]
        do_filter = False
        vals = [1]
    else:
        do_filter = True
        col = criteria[1]
        vals = response[col].value_counts().index
        if col == 'Ethnicity':
            vals = vals[::-1]
    for i, v in enumerate(vals):
        ax = axes[i]
        if do_filter:
            grouped = response[response[col] == v].groupby(by=criteria[0])
        else:
            grouped = response.groupby(by=criteria[0])

        count_by_config = {}
        configs = []
        for tup, grp in grouped:
            pref = grp.apply(get_preferred_model, axis=1)
            counts = pref.value_counts()
            total = sum(counts)
            print(tup, counts, total)
            efor = counts.get('z', 0) + counts.get('xyz', 0) / 3
            eo = counts.get('y', 0) + counts.get('xyz', 0) / 3 + counts.get('xy', 0) / 2
            efdr = counts.get('x', 0) + counts.get('xyz', 0) / 3 + counts.get('xy', 0) / 2
            print(efdr/total, eo/total, efor/total, efdr + eo + efor, total)
            count_by_config[tup] = (efdr, eo, efor)
            configs.append(tup)

        configs = ['icu', 'frauth', 'rent']
        print(configs)
        x = np.array([0, 5, 10])
        y = [count_by_config[k][0] / sum(count_by_config[k])
             for k in configs]
        p = ax.bar(x-1, y, label=r'$\mathds{P}(EFDiscRate \mid c)$', color=['tab:green'])
        print(x, y)
        y = [count_by_config[k][2] / sum(count_by_config[k])
             for k in configs]
        p = ax.bar(x + 1, y, label=r'$\mathds{P}(EFOmitRate \mid c)$', color=['tab:orange'])
        y = [count_by_config[k][1] / sum(count_by_config[k])
             for k in configs]
        p = ax.bar(x, y, label=r'$\mathds{P}(EOutcome \mid c)$', color=['tab:red'])


        ax.tick_params(bottom=False)
        ax.set_ylabel(r'$\mathds{P}(f \mid c)$', fontsize='small')
        if not do_filter:
            ax.set_xticks(x,
                          [f'{SCENARIO_NAME_MAP[k]}' for k in configs])
        if do_filter:
            short_v = VALUE_SHORT_FORM.get(v, v)
            tick_labels = [f'({SCENARIO_NAME_MAP[k]}, {short_v})'
                           for k in configs]
            ax.set_xticks(x, tick_labels, fontsize='xx-small')
        ax.set_yticks(np.arange(0, 1.1, 0.25), np.arange(0, 1.1, 0.25))

    if len(criteria) == 1:
        plt.legend(fontsize='x-small')
    else:
        left = 1
        above = 1.7
        if len(criteria) == 1:
            above -= 0.2
        legends = axes[0].legend(ncol=2, bbox_to_anchor=(left, above),
                        fontsize='x-small', fancybox=True, shadow=False)
        for t in legends.get_texts():
            t.set_ha('center')

    axes[-1].set_xlabel(r'Configuration $c$', fontsize='small')
    fig.tight_layout()
    # plt.show()
    dir_path = os.path.join('outputs', '10312021_11092021', 'barplots')
    plt.savefig(os.path.join(dir_path, '_'.join(criteria) + '.pdf'), format='pdf')