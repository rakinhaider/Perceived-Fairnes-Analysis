import numpy as np
import pandas as pd
from survey_info import *
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from collections import defaultdict
from bar_plots_pref_prob import get_preferred_model
from scipy.stats import wilcoxon, chisquare, kendalltau, ttest_rel, pearsonr
from statsmodels.stats.proportion import proportions_ztest
from utils import get_parser, aggregate_response, merge_cd_columns, map_items_to_value, combine_risk_perceptions
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri


SCENARIOS = ['icu', 'frauth', 'rent']


def run_rq1_anova(response, qid, grouping_criteria, xz_or_yz):
    # response[grouping_criteria + [qid, 'PROLIFIC_PID']]
    #   .to_csv('temp_responses.tsv', sep='\t')
    if 'scenario' in grouping_criteria:
        grouping_criteria.remove('scenario')
    if qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    choices = CHOICES[qid]
    for val, grp in grouped:
        columns = []
        for scenario in ['icu', 'frauth', 'rent']:
            scneario_grp = grp[grp['scenario'] == scenario].copy(deep=True)
            scneario_grp = scneario_grp.replace({
                qid: dict(zip(choices, [-2, -1, 0, 1, 2]))
            })
            columns.append(scneario_grp[qid])

        val = (val,)
        condition = {c: val[i] for i, c in enumerate(grouping_criteria)}
        test_stats = f_oneway(*columns)
        output = [' '.join([f'{key}={condition[key]}' for key in condition])]
        output.extend([f'{c.mean():.3f}' for c in columns])
        output.append(f'{test_stats.statistic:.3f}')
        output.append(f'{test_stats.pvalue:.3f}')
        print("\t&\t".join([str(o) for o in output]) + '\\\\')


def run_rq1_tukey(response, qid, grouping_criteria, xz_or_yz):
    if 'scenario' in grouping_criteria:
        grouping_criteria.remove('scenario')
    if qid == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    choices = CHOICES[qid]
    for val, grp in grouped:
        columns = []
        for scenario in SCENARIOS:
            scneario_grp = grp[grp['scenario'] == scenario].copy(deep=True)
            scneario_grp = scneario_grp.replace({
                qid: dict(zip(choices, [-2, -1, 0, 1, 2]))
            })
            columns.append(scneario_grp[qid])

        val = (val,)
        print({c: val[i] for i, c in enumerate(grouping_criteria)})
        test_stats = tukey_hsd(*columns)

        for sc1, sc2 in [(0, 1), (0, 2), (1, 2)]:
            print('\t&\t'.join([SCENARIO_NAME_MAP[SCENARIOS[sc1]],
                                SCENARIO_NAME_MAP[SCENARIOS[sc2]],
                                f'{test_stats.statistic[sc1][sc2]:.3f}',
                                f'{test_stats.pvalue[sc1][sc2]:.3f}'])
                  + '\\\\')


def run_rq2_anova(response, qid, grouping_criteria):
    assert qid in CDS or qid.startswith('B')
    choices = CHOICES['CD']
    columns = []
    for sc in ['icu', 'frauth', 'rent']:
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        scenario_qids = get_scenario_qids(sc, qid)
        scenario_vals = np.zeros(len(scenario_grp))
        for scenario_qid in scenario_qids:
            scenario_grp = scenario_grp.replace({
                scenario_qid: dict(zip(choices[::-1], range(len(choices))))
            })
            scenario_vals += scenario_grp[scenario_qid]
        scenario_vals /= len(scenario_qids)
        columns.append(scenario_vals)

    test_stats = f_oneway(*columns)
    output = [str(qid)]
    output.extend([f'{c.mean():.4f}' for c in columns])
    output.append(f'{test_stats.statistic:.3f}')
    output.append(f'{test_stats.pvalue:.3f}')
    print("\t&\t".join([str(o) for o in output]) + '\\\\')


def run_rq2_paired_t(response, qid1, qid2, alternative='two-sided'):
    assert qid1 in CDS or qid1.startswith('B')
    assert qid2 in CDS or qid2.startswith('B')
    choices = CHOICES['CD']
    for sc in ['icu', 'frauth', 'rent']:
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        scenario_qids1 = get_scenario_qids(sc, qid1)
        scenario_qids2 = get_scenario_qids(sc, qid2)
        vals = [np.zeros(len(scenario_grp)), np.zeros(len(scenario_grp))]
        for i, scenario_qids in enumerate([scenario_qids1, scenario_qids2]):
            for qid in scenario_qids:
                assert all(scenario_grp[qid].notna())
                scenario_grp = scenario_grp.replace({
                    qid: dict(zip(choices[::-1], range(len(choices))))
                })
                vals[i] += scenario_grp[qid].values
            vals[i] /= len(scenario_qids)

        test_stats = ttest_rel(vals[0], vals[1], alternative=alternative)
        output = [SCENARIO_NAME_MAP[sc], f'({qid1}, {qid2})', alternative]
        output.extend([f'{v.mean():.3f}' for v in vals])
        output.append(f'{test_stats.statistic:.3f}')
        output.append(f'{test_stats.pvalue:.3f}')
        print("\t".join([str(o) for o in output]) + '\\\\')


def run_rq2_tukey(response, qid, grouping_criteria):
    assert qid in CDS or qid.startswith('B')
    choices = CHOICES['CD']
    columns = []
    for sc in ['icu', 'frauth', 'rent']:
        scenario_grp = response[response['scenario'] == sc].copy(deep=True)
        scenario_qids = get_scenario_qids(sc, qid)
        scenario_vals = np.zeros(len(scenario_grp))
        for scenario_qid in scenario_qids:
            scenario_grp = scenario_grp.replace({
                scenario_qid: dict(zip(choices[::-1], range(len(choices))))
            })
            scenario_vals += scenario_grp[scenario_qid]
        scenario_vals /= len(scenario_qids)
        columns.append(scenario_vals)

    test_stats = tukey_hsd(*columns)
    for sc1, sc2 in [(0, 1), (0, 2), (1, 2)]:
        print('\t&\t'.join(['', SCENARIO_NAME_MAP[SCENARIOS[sc1]],
                            SCENARIO_NAME_MAP[SCENARIOS[sc2]],
                            f'{columns[sc1].mean():.3f}',
                            f'{columns[sc2].mean():.3f}',
                            f'{test_stats.statistic[sc1][sc2]:.3f}',
                            f'{test_stats.pvalue[sc1][sc2]:.3f}'])
              + '\\\\')


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


def run_rq1_prop(response, qid, grouping_criteria):
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
    response = response.copy(deep=True)
    if y == 'Q201' and xz_or_yz is not None:
        response = response[response['pref_model'] == f'model {xz_or_yz}']
    if no_diff:
        x_col = x1
    else:
        response['differences'] = response[x1] - response[x2]
        x_col = 'differences'

    test_results = pearsonr(response[x_col].values, response[y].values)
    print('\t&\t'.join([
        str(s) for s in [sc, x1, x2, y, xz_or_yz,
                         f'{test_results.statistic:.3f}',
                         f'{test_results.pvalue:.3f}']
    ]))
    """

    else:
        for tup, grp in grouped:
            test_results = pearsonr(grp[x_col].values, grp[y].values)
            print(f'{tup}', end='\t&\t')
            print('\t&\t'.join([
                str(s) for s in [sc, x1, x2, y, xz_or_yz,
                                 f'{test_results.statistic:.3f}',
                                 f'{test_results.pvalue:.3f}']
            ]))
        """


def run_wilcoxon(response, by, grp_criteria):
    grouped = response.groupby(grp_criteria)
    d = {}
    for tup, grp in grouped:
        if not isinstance(tup, tuple):
            tup = (tup,)
        if np.any(grp[by] != 0):
            statistics = wilcoxon(grp[by], zero_method='pratt')
            d[tup] = statistics.pvalue
        else:
            d[tup] = None
    return pd.Series(d)


def run_choice_test(response, qid, grouping_criteria):
    choices = CHOICES[qid]
    response = response.replace({
        qid: dict(zip(choices, [2, 1, 0, -1, -2]))
    })
    return run_wilcoxon(response[[qid] + grouping_criteria],
                        qid, grouping_criteria)


def run_model_property_test(response, prop, grouping_criteria):
    qids = FAIR_QIDS if prop == 'fair' else BIAS_QIDS
    d = {}
    for qid, td in zip(qids, TRADE_OFFS):
        choices = CHOICES[qid]
        response = response.replace({
            qid: dict(zip(choices, [2, 1, 0, -1, -2]))
        })
        res = run_wilcoxon(response[[qid] + grouping_criteria],
                           qid, grouping_criteria)
        for key in res.to_dict():
            if not isinstance(key, tuple):
                tup = (key, td)
            else:
                tup = key[:1] + tuple([td]) + key[1:]
            d[tup] = res[key]
    return pd.Series(d)


def run_cd_test(response, grp_criteria):
    grouped = response.groupby(grp_criteria)
    for tup, grp in grouped:
        print('##################{:s}###################'.format('_'.join(tup)))
        scenario = grp['scenario'].value_counts().index[0]
        cd_ids = CD_QS[scenario][:8]
        cd_ids = [cd_id for i, cd_id in enumerate(cd_ids) if i % 2 == 0]
        cd_ids += CD_QS[scenario][8:]
        cd_names = [i for i in CDS if i != None]
        # print(cd_ids)
        cd_stats = []
        for cd_id, cd_name in zip(cd_ids, cd_names):
            categories = len(grp[cd_id].value_counts())
            observed = grp[cd_id].value_counts()
            expected = [len(grp) / categories for i in range(categories)]
            observed.name = cd_name
            # print(observed)
            chi = chisquare(observed, expected)
            cd_stats.append((observed.idxmax(),
                             'Sig' if chi.pvalue < 0.05 else 'Not',
                             chi.pvalue))

        print(*[v for v, _, _ in cd_stats], sep='\t')
        print(*[v for _, v, _ in cd_stats], sep='\t')
        print(*['{:.3f}'.format(float(v)) for _, _, v in cd_stats], sep='\t')


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


def run_kendall_test(response, x, y):
    x_choices = CHOICES['CD'] if x not in CHOICES else CHOICES[x]
    y_choices = CHOICES['CD'] if y not in CHOICES else CHOICES[y]

    print(dict(zip(x_choices, range(len(x_choices)))))
    response = response.replace({
        x: dict(zip(x_choices, range(len(x_choices))))
    })
    response = response.replace({
        y: dict(zip(y_choices[::-1], range(len(y_choices))))
    })
    stats = kendalltau(response[x], response[y], variant='c')
    return stats


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--research-question', '-rq', type=int)
    parser.add_argument('--x-or-y', '-xy', default=None, type=str)
    parser.add_argument('--x-conditioned', '-xc', default=False,
                        action='store_true')
    parser.add_argument('--no-diff', default=False, action='store_true')
    args = parser.parse_args()

    qid = args.qid
    grouping_criteria = args.criteria
    fnames = [f + "_approved.csv" for f in args.fnames]
    response = aggregate_response(args.resp_dirs, fnames)
    if args.what == 'choice':
        print(run_choice_test(response, qid, grouping_criteria))
    elif args.what == 'cd':
        run_cd_test(response, grouping_criteria)
    elif args.what == 'model_fair':
        print(run_model_property_test(response, 'fair', grouping_criteria))
    elif args.what == 'model_bias':
        print(run_model_property_test(response, 'bias', grouping_criteria))
    elif args.what == 'kendall':
        response = merge_cd_columns(response)
        print(response)
        for qid, sf in zip(['Q10.20', 'Q10.20', 'Q201', 'Q201'],
                           ['IFPI', 'SFPI', 'IFNI', 'SFNI']):
            print(qid, sf, run_kendall_test(response, qid, sf))

    elif args.research_question == 1 and args.what == 'anova':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            run_rq2_anova(response, qid, grouping_criteria)
    elif args.research_question == 1 and args.what == 'tukey':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            print('\\midrule')
            print('\\multirow{{3}}{{*}}{{{:s}}}'.format(qid), end='')
            run_rq2_tukey(response, qid, grouping_criteria)
    elif args.research_question == 1 and args.what == 'pairedt':
        for qid1, qid2 in [('IFPI', 'IFNI'), ('SFPI', 'SFNI'), ('BFPI', 'BFNI')]:
            for alternative in ['two-sided', 'less', 'greater']:
                run_rq2_paired_t(response, qid1, qid2, alternative)

    elif args.research_question == 2 and args.what == 'anova':
        xz_or_yz = None if args.x_or_y is None else args.x_or_y.upper()
        run_rq1_anova(response, qid, grouping_criteria, xz_or_yz)
    elif args.research_question == 2 and args.what == 'tukey':
        run_rq1_tukey(response, qid, grouping_criteria, args.x_or_y.upper())
    elif args.research_question == 2 and args.what == 'prop':
        run_rq1_prop(response, qid, grouping_criteria)
    elif args.research_question == 3 and args.what == 'pearson':
        no_diff = args.no_diff
        response = merge_cd_columns(response)
        response = map_items_to_value(response)
        response = combine_risk_perceptions(response)
        choices = {'Q10.20': 'Model X vs Model Y', 'Q201': 'Model X or Y vs Model Z'}
        if no_diff:
            x1s = ['BFPI'] * 3 + ['BFNI'] * 3
            ys = ['Q10.20', 'Q201', 'Q201',
                  'Q10.20', 'Q201', 'Q201']
            x2s = [None] * 6
            if args.x_conditioned:
                xz_or_yzs = [None, 'X', 'Y'] * 2
            else: xz_or_yzs = [None] * 6
        else:
            x1s = ['IFPI', 'IFPI', 'IFPI',
                   'SFPI', 'SFPI', 'SFPI',
                   'BFPI', 'BFPI', 'BFPI']
            x2s = ['IFNI', 'IFNI', 'IFNI',
                   'SFNI', 'SFNI', 'SFNI',
                   'BFNI', 'BFNI', 'BFNI']
            ys = ['Q10.20', 'Q201', 'Q201',
                  'Q10.20', 'Q201', 'Q201',
                  'Q10.20', 'Q201', 'Q201']
            if args.x_conditioned:
                xz_or_yzs = [None, 'X', 'Y',
                            None, 'X', 'Y',
                            None, 'X', 'Y']
            else:
                xz_or_yzs = [None] * 9
        grouping_criteria = args.criteria
        for sc in SCENARIOS + ['all']:
            if sc == 'all':
                scenario_grp = response.copy(deep=True)
            else:
                scenario_grp = response[response['scenario'] == sc].copy(deep=True)

            if 'scenario' in grouping_criteria:
                grouping_criteria.remove('scenario')
            if grouping_criteria  == []:
                grouped = [('-', response)]
            else:
                grouped = scenario_grp.groupby(by=grouping_criteria)
            for tup, grp in grouped:
                for x1, x2, y, xz_or_yz in zip(x1s, x2s, ys, xz_or_yzs):
                    print(tup, end='\t&\t')
                    run_rq3_pearson(grp, sc, x1, x2, y,
                                    xz_or_yz, no_diff)
                print()