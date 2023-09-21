import numpy as np
import pandas as pd
from survey_info import *
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
from collections import defaultdict
from bar_plots import get_preferred_model
from scipy.stats import wilcoxon, chisquare, kendalltau
from statsmodels.stats.proportion import proportions_ztest
from utils import get_parser, aggregate_response, merge_cd_columns


SCENARIOS = ['icu', 'frauth', 'rent']


def run_rq1_anova(response, qid, grouping_criteria):
    # response[grouping_criteria + [qid, 'PROLIFIC_PID']].to_csv('temp_responses.tsv', sep='\t')
    if 'scenario' in grouping_criteria:
        grouping_criteria.remove('scenario')
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
            # print(scneario_grp[qid])
            columns.append(scneario_grp[qid])

        val = (val,)
        condition = {c: val[i] for i, c in enumerate(grouping_criteria)}
        test_stats = f_oneway(*columns)
        output = [' '.join([f'{key}={condition[key]}' for key in condition])]
        output.append(f'{test_stats.statistic:.3f}')
        output.append(f'{test_stats.pvalue:.3f}')
        print("\t&\t".join([str(o) for o in output]) + '\\\\')


def run_rq1_tukey(response, qid, grouping_criteria):
    if 'scenario' in grouping_criteria:
        grouping_criteria.remove('scenario')
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
            # print(scneario_grp[qid])
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
        # scenario_qid = CD_QS[sc][CDS.index(qid)]
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
    # print(test_stats.__dict__)
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
    df = get_preference_counts(response)
    print(df)
    for model in ['x', 'y', 'z']:
        counts = df.loc[model]
        nobs = df.sum(axis=0)


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
    # print(stats)
    return stats


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--research-question', '-rq', type=int)
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
        run_rq1_anova(response, qid, grouping_criteria)
    elif args.research_question == 1 and args.what == 'tukey':
        run_rq1_tukey(response, qid, grouping_criteria)
    elif args.research_question == 1 and args.what == 'prop':
        run_rq1_prop(response, qid, grouping_criteria)
    elif args.research_question == 2 and args.what == 'anova':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            run_rq2_anova(response, qid, grouping_criteria)
    elif args.research_question == 2 and args.what == 'tukey':
        for qid in ['IFPI', 'SFPI', 'IFNI', 'SFNI', 'BFPI', 'BFNI']:
            print('\\midrule')
            print('\\multirow{{3}}{{*}}{{{:s}}}'.format(qid), end='')
            run_rq2_tukey(response, qid, grouping_criteria)
