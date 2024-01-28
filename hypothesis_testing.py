import os
import numpy as np
import pandas as pd
import scipy.stats
from survey_info import *
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from collections import defaultdict
from scipy.stats import ttest_rel, pearsonr, mannwhitneyu, wilcoxon
from statsmodels.stats.proportion import proportions_ztest
from utils import (
    get_parser, aggregate_response, merge_cd_columns,
    map_items_to_value, combine_risk_perceptions, get_preferred_model)
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from plotters.plotter import set_rcparams, set_size


SCENARIOS = ['icu', 'frauth', 'rent']


def check_normal(response, qid, dirs, xz_or_yz):
    set_rcparams(fontsize=10)
    for sc in ['icu', 'frauth', 'rent']:
        fig, ax = plt.subplots(
            figsize=set_size(430*0.32, 0.95, 0.8)
        )
        if qid == 'Q201' and xz_or_yz is not None:
            response = response[response['pref_model'] == f'model {xz_or_yz}']
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        counts = scenario_grp[qid].value_counts()
        if 'Q' in 'qid':
            counts = counts.reindex([-2, -1, 0, 1, 2], fill_value=0)
        else:
            counts = counts.sort_index()
        heights = counts.values
        values = counts.index
        plt.xticks(counts.index, counts.index)
        plt.bar(x=values, height=heights, color='green')
        plt.xlabel('Likert Item Scores', fontsize='x-small')
        plt.ylabel('Counts', fontsize='x-small')
        dir_name = os.path.join('outputs', '_'.join(dirs), 'barplots', 'normal')
        os.makedirs(dir_name, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'{dir_name}/{sc}_{qid}.pdf', format='pdf')
        plt.cla()
        x = scenario_grp[qid]
        loc = x.mean()
        scale = x.std()
        test_stats = scipy.stats.kstest(x, 'norm', args=(loc, scale))
        print('\t&\t'.join([qid, sc,
                            f'{test_stats.statistic:.4f}',
                            f'{test_stats.pvalue:.4f}']))

def print_outputs(outputs):
    print('\t'.join([str(o) for o in outputs]))


def run_rq1_anova(response, qid):
    columns = []
    for sc in ['icu', 'frauth', 'rent']:
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        columns.append(scenario_grp[qid])

    test_stats = f_oneway(*columns)
    output = [str(qid)]
    output.extend([f'{c.mean():.3f}' for c in columns])
    output.append(f'{test_stats.statistic:.3f}')
    output.append(f'{test_stats.pvalue:.3f}')
    print_outputs(output)


def run_rq1_tukey(response, qid):
    columns = []
    for sc in ['icu', 'frauth', 'rent']:
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        columns.append(scenario_grp[qid])

    test_stats = tukey_hsd(*columns)
    for sc1, sc2 in [(0, 1), (0, 2), (1, 2)]:
        output = [SCENARIO_NAME_MAP[SCENARIOS[sc1]],
                  SCENARIO_NAME_MAP[SCENARIOS[sc2]],
                  f'{columns[sc1].mean():.3f}', f'{columns[sc2].mean():.3f}',
                  f'{test_stats.statistic[sc1][sc2]:.3f}',
                  f'{test_stats.pvalue[sc1][sc2]:.3f}']
        print_outputs(output)


def run_rq1_paired_t(response, qid1, qid2, sc, alternative='two-sided'):
    scenario_grp = response[response['scenario'] == sc].copy(deep=True)
    vals = [scenario_grp[qid1].values, scenario_grp[qid2].values]

    test_stats = ttest_rel(vals[0], vals[1], alternative=alternative)
    output = [SCENARIO_NAME_MAP[sc], f'({qid1}, {qid2})', alternative]
    output.extend([f'{v.mean():.3f}' for v in vals])
    output.append(f'{test_stats.statistic:.3f}')
    output.append(f'{test_stats.pvalue:.3f}')
    print_outputs(output)


def run_rq1_mannwhitneyu(response, qid1, qid2, sc, alternative='two-sided'):
    scenario_grp = response[response['scenario'] == sc].copy(deep=True)
    vals = [scenario_grp[qid1].values, scenario_grp[qid2].values]

    test_stats = mannwhitneyu(vals[0], vals[1], alternative=alternative)
    output = [SCENARIO_NAME_MAP[sc], f'({qid1}, {qid2})', alternative]
    output.extend([f'{v.mean():.3f}' for v in vals])
    output.append(f'{test_stats.statistic:.3f}')
    output.append(f'{test_stats.pvalue:.3f}')
    print_outputs(output)


def run_rq1_wilcoxon(response, qid1, qid2, sc, alternative='two-sided'):
    scenario_grp = response[response['scenario'] == sc].copy(deep=True)
    vals = [scenario_grp[qid1].values, scenario_grp[qid2].values]

    test_stats = wilcoxon(vals[0], vals[1], alternative=alternative)
    output = [SCENARIO_NAME_MAP[sc], f'({qid1}, {qid2})', alternative]
    print([np.sort(v) for v in vals])
    output.extend([f'{np.median(v):.3f}' for v in vals])
    output.append(f'{test_stats.statistic:.3f}')
    output.append(f'{test_stats.pvalue:.3f}')
    print_outputs(output)


def run_rq2_anova(response, qid, grouping_criteria, xz_or_yz):
    if qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    for val, grp in grouped:
        columns = []
        for scenario in ['icu', 'frauth', 'rent']:
            scneario_grp = grp[grp['scenario'] == scenario].copy(deep=True)
            columns.append(scneario_grp[qid])

        val = (val,)
        condition = {c: val[i] for i, c in enumerate(grouping_criteria)}
        test_stats = f_oneway(*columns)
        output = [' '.join([f'{key}={condition[key]}' for key in condition])]
        output.extend([f'{c.mean():.3f}' for c in columns])
        output.append(f'{test_stats.statistic:.3f}')
        output.append(f'{test_stats.pvalue:.3f}')
        print_outputs(output)


def run_rq2_tukey(response, qid, grouping_criteria, xz_or_yz):
    if qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    for val, grp in grouped:
        columns = []
        for scenario in SCENARIOS:
            scneario_grp = grp[grp['scenario'] == scenario].copy(deep=True)
            columns.append(scneario_grp[qid])

        test_stats = tukey_hsd(*columns)

        for sc1, sc2 in [(0, 1), (0, 2), (1, 2)]:
            if test_stats.pvalue[sc1][sc2] <= 0.05:
                if grouping_criteria != []:
                    output = [f'{grouping_criteria[0]} = {val}']
                else:
                    output = ['']
                output += [SCENARIO_NAME_MAP[SCENARIOS[sc1]],
                            SCENARIO_NAME_MAP[SCENARIOS[sc2]],
                            f'{test_stats.statistic[sc1][sc2]:.3f}',
                            f'{test_stats.pvalue[sc1][sc2]:.3f}']
                print_outputs(output)


def get_preference_counts(response):
    df = pd.DataFrame()
    for sc in SCENARIOS:
        scenario_grp = response[response['scenario'] == sc]
        pref = scenario_grp.apply(get_preferred_model, axis=1)
        counts = pref.value_counts().astype(float)
        for index in counts.index:
            if len(index) > 1:
                for ch in index:
                    counts[ch] += counts[index] / len(index)
                counts.drop(labels=index, inplace=True)
        counts = counts.reindex(['x', 'y', 'z'])
        df[sc] = counts
    return df


def get_preference_probabilities(response):
    counts = get_preference_counts(response)
    counts = counts.div(counts.sum(axis=0))
    return counts


def run_rq2_prop(response, qid, grouping_criteria):
    r = robjects.r
    r('source')('rscripts/proportion_test.R')
    proptest = robjects.globalenv['proptest']
    if 'scenario' in grouping_criteria:
        grouping_criteria.remove('scenario')
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    for val, grp in grouped:
        print(f'######################### {val} #############################')
        df = get_preference_counts(grp)
        print(df)
        for model in ['x', 'y', 'z']:
            numpy2ri.activate()
            counts = df.loc[model]
            nobs = df.sum(axis=0)
            prop = proptest(counts.values, nobs.values)
            prop = list(prop)
            chi_square = prop[0][0]
            pvalue = prop[2][0]
            print(rf'chi-squared {chi_square:.3f}, p-value {pvalue:.3f}')
            numpy2ri.deactivate()


def run_rq3_pearson(response, sc, x1, x2, y, xz_or_yz=None, no_diff=False):
    if sc == 'all':
        response = response.copy(deep=True)
    else:
        response = response[response['scenario'] == sc].copy(deep=True)

    if y == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if no_diff:
        x_col = x1
    else:
        response['differences'] = response[x1] - response[x2]
        x_col = 'differences'

    test_results = pearsonr(response[x_col].values, response[y].values)
    output = [sc, x1, x2, y, xz_or_yz,
              f'{test_results.statistic:.3f}', f'{test_results.pvalue:.3f}']
    print_outputs(output)


def run_freq_difference_test(f1, f2):
    f1 = defaultdict(int, f1)
    f2 = defaultdict(int, f2)
    vocab = set(list(f1.keys()) + list(f2.keys()))
    print(len(vocab))
    f1_total = sum(f1.values())
    f2_total = sum(f2.values())
    print(f1_total, f2_total)
    scores = []
    for w in vocab:
        success = [f1[w], f2[w]]
        size = [f1_total, f2_total]
        zstat, p_value = proportions_ztest(count=success, nobs=size,
                                           alternative='larger')
        scores.append([w, zstat, p_value, f1[w], f2[w],
                       f1[w] / f1_total, f2[w] / f2_total])

    scores.sort(key=lambda x: (x[2], x[0], x[1]))
    return pd.DataFrame(scores, columns=['word', 'zstat', 'pvalue', 'f1_count',
                                         'f2_count', 'f1_prop', 'f2_prop'])


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--research-question', '-rq', type=int)
    parser.add_argument('--x-or-y', '-xy', default=None, type=str)
    parser.add_argument('--no-diff', default=False, action='store_true')
    args = parser.parse_args()

    qid = args.qid
    fnames = [f + "_approved.csv" for f in args.fnames]
    grouping_criteria = args.criteria
    if "scenario" in grouping_criteria:
        grouping_criteria.remove('scenario')

    xz_or_yz = None if args.x_or_y is None else args.x_or_y.upper()

    response = aggregate_response(args.resp_dirs, fnames)
    response = merge_cd_columns(response)
    response = map_items_to_value(response)
    response = combine_risk_perceptions(response)

    if args.what == 'normal-check':
        check_normal(response, qid, args.resp_dirs, xz_or_yz)
    elif args.research_question == 1 and args.what == 'anova':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            run_rq1_anova(response, qid)

    elif args.research_question == 1 and args.what == 'tukey':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            run_rq1_tukey(response, qid)

    elif args.research_question == 1 and args.what == 'pairedt':
        scenarios = ['icu'] * 2 + ['frauth'] * 2 + ['rent'] * 2
        alternatives = ['less', 'less',
                        'greater', 'greater',
                        'greater', 'less']
        qids = [('IFPI', 'IFNI'), ('SFPI', 'SFNI')] * 3
        for sc, alternative, (qid1, qid2) in zip(scenarios, alternatives, qids):
            run_rq1_paired_t(response, qid1, qid2, sc, alternative)

    elif args.research_question == 1 and args.what == 'mwu':
        scenarios = ['icu'] * 2 + ['frauth'] * 2 + ['rent'] * 2
        alternatives = ['less', 'less',
                        'greater', 'greater',
                        'greater', 'less']
        qids = [('IFPI', 'IFNI'), ('SFPI', 'SFNI')] * 3
        for sc, alternative, (qid1, qid2) in zip(scenarios, alternatives, qids):
            run_rq1_mannwhitneyu(response, qid1, qid2, sc, alternative)

    elif args.research_question == 1 and args.what == 'wsr':
        scenarios = ['icu'] * 2 + ['frauth'] * 2 + ['rent'] * 2
        alternatives = ['less', 'less',
                        'greater', 'greater',
                        'greater', 'less']
        qids = [('IFPI', 'IFNI'), ('SFPI', 'SFNI')] * 3
        for sc, alternative, (qid1, qid2) in zip(scenarios, alternatives, qids):
            run_rq1_wilcoxon(response, qid1, qid2, sc, alternative)

    elif args.research_question == 2 and args.what == 'anova':
        run_rq2_anova(response, qid, grouping_criteria, xz_or_yz)

    elif args.research_question == 2 and args.what == 'tukey':
        run_rq2_tukey(response, qid, grouping_criteria, xz_or_yz)
    elif args.research_question == 2 and args.what == 'prop':
        run_rq2_prop(response, qid, grouping_criteria)

    elif args.research_question == 3 and args.what == 'pearson':
        no_diff = args.no_diff
        if no_diff:
            x1s = ['BFPI'] * 3 + ['BFNI'] * 3
            ys = ['Q10.20', 'Q201', 'Q201', 'Q10.20', 'Q201', 'Q201']
            x2s = [None] * 6
            xz_or_yzs = [None, 'X', 'Y'] * 2
        else:
            x1s = ['IFPI', 'IFPI', 'IFPI', 'SFPI', 'SFPI', 'SFPI',
                   'BFPI', 'BFPI', 'BFPI']
            x2s = ['IFNI', 'IFNI', 'IFNI', 'SFNI', 'SFNI', 'SFNI',
                   'BFNI', 'BFNI', 'BFNI']
            ys = ['Q10.20', 'Q201', 'Q201', 'Q10.20', 'Q201', 'Q201',
                  'Q10.20', 'Q201', 'Q201']
            xz_or_yzs = [None, 'X', 'Y', None, 'X', 'Y',
                         None, 'X', 'Y']
        # for sc in SCENARIOS + ['all']:
        for sc in ['all']:
            if not grouping_criteria:
                grouped = [('-', response)]
            else:
                grouped = response.groupby(by=grouping_criteria)
            for tup, grp in grouped:
                for x1, x2, y, xz_or_yz in zip(x1s, x2s, ys, xz_or_yzs):
                    print(tup, end='\t')
                    run_rq3_pearson(grp, sc, x1, x2, y,
                                    xz_or_yz, no_diff)
                print()