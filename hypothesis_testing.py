from scipy.stats import wilcoxon
import pandas as pd
import os
from survey_info import *

def test_choices(response, qid, grp_criteria):
    choices = CHOICES[qid]
    response = response.replace({
        qid: dict(zip(choices, [2, 1, 0, -1, -2]))
    })
    response = response.replace({
        'Ethnicity': ETHNICITY_MAP
    })

    grouped = response.groupby(grp_criteria)
    for tup, grp in grouped:
        print(tup, 'Response Count: {:d}'.format(len(grp)))
        print(grp[qid].value_counts().sort_index(ascending=False))
        print(wilcoxon(grp[qid], zero_method='pratt'))


if __name__ == "__main__":

    response = pd.read_csv(os.path.join('data', 'processed', 'response.csv'),
                           index_col=0)

    xvsy_qid = 'Q10.14'
    grouping_criteria = ['scenario', 'Ethnicity']

    test_choices(response, xvsy_qid, grouping_criteria)

    for scenario in ['frauth', 'icu', 'rent']:
