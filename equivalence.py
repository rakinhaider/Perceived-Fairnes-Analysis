from statsmodels.stats.weightstats import ttost_ind
import pandas as pd
import numpy as np
import os
from survey_info import *
from utils import correct_errors

if __name__ == "__main__":
    data_dir = 'data/processed/'

    b1, b2 = '10082021', '10312021'
    df1 = pd.read_csv(os.path.join(data_dir, b1, 'APPROVED',
                                   b1 + '_approved.csv'), index_col=0)
    df2 = pd.read_csv(os.path.join(data_dir, b2, 'APPROVED',
                                   b2 + '_approved.csv'), index_col=0)

    qid = 'Q10.20'
    choices = CHOICES[qid]
    df1 = correct_errors(df1)
    df2 = correct_errors(df2)

    df1 = df1.replace({
       qid: dict(zip(choices, [2, 1, 0, -1, -2]))
    })
    df2 = df2.replace({
        qid: dict(zip(choices, [2, 1, 0, -1, -2]))
    })

    for sc in ['frauth', 'icu', 'rent']:
        print('##########################{}#################'.format(sc))
        s1 = df1[df1['scenario'] == sc]
        s2 = df2[df2['scenario'] == sc]
        print(s1[qid].value_counts())
        print(s1[qid].astype(int).mean())
        print(s1[qid].astype(int).std())
        print(s2[qid].value_counts())
        print(s2[qid].astype(int).mean())
        print(s2[qid].astype(int).std())

        print(ttost_ind(s1[qid].values, s2[qid].values,
                        low=-0.25, upp=0.25))

