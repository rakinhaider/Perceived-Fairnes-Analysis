import os.path

import numpy as np
import matplotlib.pyplot as plt
from utils import get_parser
from survey_response_aggregator import aggregate_response
from publication_plots_util import set_rcparams, set_size
from utils import merge_cd_columns, map_items_to_value, combine_risk_perceptions
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
    response = merge_cd_columns(response)
    response = map_items_to_value(response)
    response = combine_risk_perceptions(response)

    fig, ax = plt.subplots(figsize=set_size(129, 0.95,
                                            aspect_ratio=1),
                           tight_layout=True)
    means = []
    stds = []
    for sc in SCENARIOS:
        scenario_grp = response[response['scenario'] == sc]
        means.append(scenario_grp[qid].mean())
        stds.append(scenario_grp[qid].std() / np.sqrt(len(scenario_grp)))
    bar_width = 0.75
    bars = ax.bar(range(len(means)), means, yerr=stds, width=bar_width,
                  capsize=5)
    for bar, c in zip(bars, ['green', 'blue', 'red']):
        bar.set_color(c)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([SCENARIO_NAME_MAP[sc] for sc in SCENARIOS],
                       fontsize='xx-small')
    ax.set_ylim(0, 2.1)

    if qid[0] == "B":
        type_str = ''
    elif qid[0] == 'S':
        type_str = ', Soc.'
    else:
        type_str = ', Indiv.'
    ax.set_ylabel(f'{SF_MAP[qid]} Ratings',
                  fontsize='small')
    ax.set_xlabel("Contexts", fontsize='small')
    plt.tight_layout()
    plt.gcf().get_size_inches()
    path = os.path.join('outputs/10312021_11092021/barplots/risk_perceptions',
                        f'{qid}.pdf')
    plt.savefig(path, format='pdf')