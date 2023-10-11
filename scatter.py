import os.path
import matplotlib.pyplot as plt
import numpy as np
from publication_plots_util import set_rcparams, set_size
from utils import (
    get_parser, aggregate_response, map_items_to_value, merge_cd_columns,
    combine_risk_perceptions
)
from survey_info import SF_MAP


if __name__ == "__main__":
    set_rcparams(fontsize=10)
    parser = get_parser()
    parser.add_argument('--x-or-y', '-xy', default=None, type=str)
    parser.add_argument('--diff-type', default='I', type=str,
                        choices=['s', 'S', 'i', 'S', 'b', 'B'])
    parser.add_argument('--type', default='overlap', type=str)
    args = parser.parse_args()

    criteria = args.criteria
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]
    diff_type = args.diff_type.upper()

    response = aggregate_response(args.resp_dirs, fnames)
    response = merge_cd_columns(response)
    response = map_items_to_value(response)
    response = combine_risk_perceptions(response)

    qid = args.qid
    xz_or_yz = args.x_or_y
    grouping_criteria = args.criteria
    if qid == 'Q10.20':
        ylabel = 'XY Preferences'
    elif qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
        ylabel = f'{xz_or_yz}Z Preferences'
    else:
        ylabel = f'X/Y_vs_Z Preferences'


    fig, axes = plt.subplots(
        figsize=set_size(190, 0.95, aspect_ratio=0.75),
        tight_layout=True)

    for sc in [None, 'icu', 'frauth', 'rent']:
        if sc is not None:
            scenario_grp = response[response['scenario'] == sc].copy(deep='True')
        else:
            scenario_grp = response.copy(deep=True)
        x1, x2 = f'{diff_type}FPI', f'{diff_type}FNI'
        scenario_grp['differences'] = scenario_grp[x1] - scenario_grp[x2]

        counts = (scenario_grp[['differences', qid]].
                  groupby(['differences', qid]).size())

        def get_color(row):
            color_map = {'icu': 'red', 'frauth': 'blue', 'rent': 'green'}
            return color_map[row]
        c = scenario_grp['scenario'].apply(get_color)

        def get_radius(row):
            return counts.get((row['differences'], row[qid]), default=0)
        radius = scenario_grp[['differences', qid]].apply(get_radius, axis=1)
        x = scenario_grp['differences']
        y = scenario_grp[qid]
        if args.type == 'noisy':
            x += np.random.rand(len(scenario_grp)) * 0.25
            y += np.random.rand(len(scenario_grp)) * 0.25
        if args.type == 'overlap':
            plt.scatter(x, y)
        elif args.type == 'size':
            plt.scatter(x, y, s=radius*10)
        elif args.type == 'noisy':
            plt.scatter(x, y, c=c, s=5)
        plt.xlabel(f'{SF_MAP[x1]} - {SF_MAP[x2]}')
        plt.ylabel(ylabel)
        out_dir = os.path.join('outputs', '_'.join(args.resp_dirs), 'scatter',
                               args.type)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{sc}_{x1}_{x2}.pdf'),
                    format='pdf')
        plt.clf()
