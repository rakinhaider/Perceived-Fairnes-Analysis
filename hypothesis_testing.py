import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon, chisquare, kendalltau
import pandas as pd
from utils import get_parser, aggregate_response, merge_cd_columns
from survey_info import *
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest

def run_wilcoxon(response, by, grp_criteria):
    grouped = response.groupby(grp_criteria)
    d = {}
    for tup, grp in grouped:
        if not isinstance(tup, tuple):
            tup = (tup, )
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
            expected = [len(grp)/categories for i in range(categories)]
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
        # print(w, success, size)
        zstat, p_value = proportions_ztest(count=success, nobs=size,
                                           alternative='larger')
        scores.append([w, zstat, p_value, f1[w], f2[w]])

    scores.sort(key=lambda x: (x[2], x[0], x[1]))
    return pd.DataFrame(scores, columns=['word', 'zstat', 'pvalue',
                                         'f1_count', 'f2_count'])


def run_kendall_test(response, x, y):
    x_choices = CHOICES['CD'] if x not in CHOICES else CHOICES[x]
    y_choices = CHOICES['CD'] if y not in CHOICES else CHOICES[y]

    response = response.replace({
        x: dict(zip(x_choices, range(len(x_choices))))
    })
    response = response.replace({
        y: dict(zip(y_choices[::-1], range(len(y_choices))))
    })
    stats = kendalltau(response[x], response[y], variant='c')
    print(stats)
    return stats


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    xvsy_qid = args.qid
    grouping_criteria = args.criteria
    fnames = [f + "_approved.csv" for f in args.fnames]
    response = aggregate_response(args.resp_dirs, fnames)

    if args.what == 'choice':
        print(run_choice_test(response, xvsy_qid, grouping_criteria))
    elif args.what == 'cd':
        run_cd_test(response, grouping_criteria)
    elif args.what == 'model_fair':
        print(run_model_property_test(response, 'fair', grouping_criteria))
    elif args.what == 'model_bias':
        print(run_model_property_test(response, 'bias', grouping_criteria))
    elif args.what == 'kendall':
        response = merge_cd_columns(response)
        print(run_kendall_test(response, 'Q10.20', 'IFPI'))
        print(run_kendall_test(response, 'Q10.20', 'SFPI'))
        print(run_kendall_test(response, 'Q201', 'IFNI'))
        print(run_kendall_test(response, 'Q201', 'SFNI'))
