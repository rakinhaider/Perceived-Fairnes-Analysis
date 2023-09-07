import os
import matplotlib.pyplot as plt
import numpy as np
from utils import get_parser
from survey_response_aggregator import aggregate_response
from publication_plots_util import set_rcparams, set_size
from survey_info import SCENARIO_NAME_MAP

VALUE_SHORT_FORM = {'Disadvantaged': 'Disadv.',
                    'Advantaged': 'Adv.',
                    'Caucasian': 'Cauc.',
                    'Non-Caucasian': 'Non-Cauc.'}

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
                    figsize=set_size(241, 0.95,
                                     aspect_ratio=0.75),
                    tight_layout=True)
    if len(criteria) == 1:
        axes = [axes]
        do_filter = False
        vals = [1]
    else:
        do_filter = True
        col = criteria[1]
        vals = response[col].value_counts().index
    for i, v in enumerate(vals):
        ax = axes[i]
        if do_filter:
            grouped = response[response[col] == v].groupby(by=criteria[0])
        else:
            grouped = response.groupby(by=criteria[0])

        count_by_config = {}

        for tup, grp in grouped:
            print(tup, len(grp))
            counts = grp['Q10.20'].value_counts()
            efdr = counts.get('Definitely model X', 0) + counts.get(
                'Probably model X', 0)
            eo = counts.get('Definitely model Y', 0) + counts.get(
                'Probably model Y', 0)
            neutral = counts.get('Neither model X nor model Y', 0)
            count_by_config[tup] = (efdr, eo, neutral)
            print(efdr, eo, neutral, efdr + eo + neutral)

        x = np.array([0, 3, 6])
        bottom = np.empty(len(x)*2)
        y = [count_by_config[k][0] / sum(count_by_config[k])
             for k in count_by_config]
        bottom[0::2] = y
        p = ax.bar(x-0.5, y, label=r'$\mathds{P}(EFDR \mid c)$', color=['tab:green'])
        print(x, y)
        y = [count_by_config[k][1] / sum(count_by_config[k])
             for k in count_by_config]
        bottom[1::2] = y
        p = ax.bar(x + 0.5, y, label=r'$\mathds{P}(EO \mid c)$', color=['tab:red'])

        temp = np.empty(2*len(x))
        temp[0::2] = x - 0.5
        temp[1::2] = x + 0.5
        x = temp

        y = [count_by_config[k][2]/(2*sum(count_by_config[k])) for k in count_by_config]
        print(y)
        y = np.array([y, y]).transpose().flatten()
        print(x, y)
        p = ax.bar(x, y, bottom=bottom, color=['tab:green', 'tab:red']*3)
        ax.tick_params(bottom=False)
        ax.set_ylabel(r'$\mathds{P}(f \mid c)$', fontsize='small')
        if not do_filter:
            ax.set_xticks([0, 3, 6],
                          [f'{SCENARIO_NAME_MAP[k]}' for k in count_by_config])
        if do_filter:
            short_v = VALUE_SHORT_FORM.get(v, v)
            tick_labels = [f'({SCENARIO_NAME_MAP[k]}, {short_v})'
                           for k in count_by_config]
            ax.set_xticks([0, 3, 6], tick_labels, fontsize='xx-small')
        ax.set_yticks(np.arange(0, 1.1, 0.25), np.arange(0, 1.1, 0.25))

    above = 1.4
    if len(criteria) == 1:
        above -= 0.2
    axes[0].legend(ncol=2, bbox_to_anchor=(0.9, above),
                   fontsize='x-small', fancybox=True, shadow=False)
    axes[-1].set_xlabel(r'Configuration $c$')
    # plt.show()
    fig.tight_layout()
    dir_path = os.path.join('outputs', '10312021_11092021', 'Q10.20', 'barplots')
    plt.savefig(os.path.join(dir_path, '_'.join(criteria) + '.pdf'), format='pdf')