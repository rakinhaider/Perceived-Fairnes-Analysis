import numpy as np
from scipy.stats import wilcoxon, chisquare
import pandas as pd
import os
from survey_info import *
import argparse


def test_choices(response, qid, grp_criteria):
    choices = CHOICES[qid]
    response = response.replace({
        qid: dict(zip(choices, [2, 1, 0, -1, -2]))
    })

    grouped = response.groupby(grp_criteria)
    d = {}
    for tup, grp in grouped:
        # print(tup, 'Response Count: {:d}'.format(len(grp)))
        if np.any(grp[qid] != 0):
            # print(grp[qid].value_counts().sort_index(ascending=False))
            statistics = wilcoxon(grp[qid], zero_method='pratt')
            # print(statistics)
            d[tup] = statistics.pvalue
        else:
            # print('All zeros. Wilcoxon is not supported.')
            d[tup] = None
    return pd.Series(d)


def test_cd_choices(response, grp_criteria):
    grouped = response.groupby(grp_criteria)
    for tup, grp in grouped:
        print('##################{:s}###################'.format(tup))
        scenario = grp['scenario'].value_counts().index[0]
        cd_ids = CD_QS[scenario][:8]
        cd_ids = [cd_id for i, cd_id in enumerate(cd_ids) if i % 2 == 0]
        cd_ids += CD_QS[scenario][8:]
        print(cd_ids)
        for cd_id in cd_ids:
            categories = len(grp[cd_id].value_counts())
            observed = grp[cd_id].value_counts()
            expected = [len(grp)/categories for i in range(categories)]
            print(observed, expected)
            print(chisquare(observed, expected))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    parser.add_argument('--what', default='choice', choices=['choice', 'cd'])
    args = parser.parse_args()

    response = pd.read_csv(os.path.join('data', 'processed', 'response.csv'),
                           index_col=0)

    xvsy_qid = args.qid
    grouping_criteria = args.criteria

    if args.what == 'choice':
        test_choices(response, xvsy_qid, grouping_criteria)
    else:
        test_cd_choices(response, grouping_criteria)