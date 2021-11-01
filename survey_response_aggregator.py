import pandas as pd
import os
import numpy as np
from constants import *
from survey_info import *

def expired_data_handler(data):
    data = data.copy(deep=True)
    for index, row in data.iterrows():
        if row['Ethnicity'] not in ETHNICITY_MAP:
            # TODO: Can also update some other fields.
            #  Not necessary right now.
            data.loc[index, 'Ethnicity'] = STUDY_MAP[row[STUDY_ID]]

    return data


def aggregate_response(data_dirs, fnames):
    dfs = []
    for i, data_dir in enumerate(data_dirs):
        fname = os.path.join('data', 'processed', data_dir,
                             'approved', fnames[i])
        df = pd.read_csv(fname, index_col=0)
        dfs.append(df)

    data = pd.concat(dfs, axis=0)
    data.reset_index(drop=True, inplace=True)
    return data


def prepare_causal_data(df):
    """
        - Convert CD responses to CD columns
        - Convert Ethnicity to PGM columns
        - Create Context columns
        - Convert XY choice to Fairness Metric Columns

    :param df:
    :return:
        causal_df: Dataset df converted into causal_df
    """
    df = df.replace({'Q10.20': dict(zip(CHOICES['Q10.20'], [1, 1, 0, -1, -1]))})
    variables = [cd for cd in CDS if cd is not None] + \
                ['EFPR', 'EO', 'frauth', 'icu', 'rent']
    print(variables)
    rows = []
    print(df[df['scenario'] == 'frauth']['Q10.20'].value_counts())
    for index, row in df.iterrows():
        scenario = row['scenario']
        q_ids = CD_QS[scenario]
        values = []
        for i, cd in enumerate(CDS):
            if cd is not None:
                values.append(row[q_ids[i]])

        if row['Q10.20'] == 1:
            values.extend([1, 0])
        elif row['Q10.20'] == -1:
            values.extend([0, 1])
        else:
            """
            randint = np.random.randint(0, 10)
            if randint % 2 == 1:
                values.extend([1, 0])
            else:
                values.extend([0, 1])
            """
            values.extend([2, 2])

        values.extend([0, 0, 0])
        causal_row = pd.Series(data=values, index=variables,
                               name=row[PROLIFIC_PID])
        causal_row[row['scenario']] = 1
        rows.append(causal_row)

    causal_df = pd.DataFrame(rows, columns=variables)
    causal_df = causal_df.replace({'PGM': {'Advantaged': 1, 'Disadvantaged': 0}})
    causal_df = causal_df.replace({'DT': {'Yes': 'High', 'No': 'Low'}})
    causal_df = causal_df.replace({'Yes': 1, 'No': 0})
    causal_df = causal_df.replace({'Moderate': 'Med'})
    print(causal_df['DT'])

    return causal_df


if __name__ == "__main__":
    data_dirs = ['10082021']
    fnames = ['10082021']
    fnames = [f + '_approved.csv' for f in fnames]

    df = aggregate_response(data_dirs, fnames)
    df = expired_data_handler(df)
    # df.to_csv(os.path.join('data', 'processed', 'response.csv'))

    causal_df = prepare_causal_data(df)
    causal_df.to_csv(os.path.join('data', 'processed', 'causal_df.csv'))