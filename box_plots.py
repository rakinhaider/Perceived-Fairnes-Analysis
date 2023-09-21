import os.path

import numpy as np
import matplotlib.pyplot as plt
from utils import get_parser
from survey_response_aggregator import aggregate_response
from publication_plots_util import set_rcparams, set_size
from survey_info import (
    SCENARIOS, get_scenario_qids, CHOICES, SCENARIO_NAME_MAP, SF_MAP
)

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
    qid = args.qid
    assert 'FNI' in qid or 'FPI' in qid

    response = aggregate_response(args.resp_dirs, fnames)

    fig, ax = plt.subplots(figsize=set_size(129, 0.95,
                                            aspect_ratio=1),
                           tight_layout=True)
    choices = CHOICES['CD']
    data = []
    for sc in SCENARIOS:
        scenario_grp = response[response['scenario'] == sc]
        scenario_qids = get_scenario_qids(sc, qid)
        scenario_vals = np.zeros(len(scenario_grp))
        for scenario_qid in scenario_qids:
            scenario_grp = scenario_grp.replace({
                scenario_qid: dict(zip(choices[::-1], range(len(choices))))
            })
            scenario_vals += scenario_grp[scenario_qid]
        # print(sc, scenario_vals.values)

        scenario_vals /= len(scenario_qids)
        data.append(scenario_vals)
    boxplot = ax.boxplot(data, showfliers=False, labels=SCENARIOS,
                         patch_artist=True, positions=[0, 1, 2], autorange=True)
    for bp, color in zip(boxplot['boxes'], ['lightgreen', 'lightblue', 'pink']):
        bp.set_facecolor(color)
    for ln in boxplot['medians']:
        ln.set(linewidth=2, color='red')
    ax.set_xticklabels([SCENARIO_NAME_MAP[sc] for sc in SCENARIOS],
                       fontsize='xx-small')
    if qid[0] == "B":
        type_str = ''
    elif qid[0] == 'S':
        type_str = ', Soc.'
    else:
        type_str = ', Indiv.'
    ax.set_ylabel(f'{SF_MAP[qid]} Ratings',
                  fontsize='x-small')
    ax.set_xlabel("Contexts", fontsize='x-small')
    # plt.show()
    plt.tight_layout()
    plt.gcf().get_size_inches()
    path = os.path.join('outputs/10312021_11092021/boxplots', f'{qid}.pdf')
    plt.savefig(path, format='pdf')