import numpy as np
from scipy.stats import wilcoxon, chisquare
import pandas as pd
import os
from survey_info import *
import argparse


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    parser.add_argument('--what', default='choice',
                        choices=['choice', 'cd', 'model_fair', 'model_bias'])
    args = parser.parse_args()

    response = pd.read_csv(os.path.join('data', 'processed', 'response.csv'),
                           index_col=0)

    xvsy_qid = args.qid
    grouping_criteria = args.criteria

    if args.what == 'choice':
        print(run_choice_test(response, xvsy_qid, grouping_criteria))
    elif args.what == 'cd':
        run_cd_test(response, grouping_criteria)
    elif args.what == 'model_fair':
        print(run_model_property_test(response, 'fair', grouping_criteria))
    elif args.what == 'model_bias':
        print(run_model_property_test(response, 'bias', grouping_criteria))
