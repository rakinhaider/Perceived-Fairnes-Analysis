import argparse
import os

import pandas as pd

import constants
from constants import STUDY_ID
from survey_info import ETHNICITY_MAP, CD_QS, COMMON_QS, MODELZ_QS, CDS, \
    STUDY_MAP


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--xz-qid', default='Q201')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    parser.add_argument('--resp-dirs', nargs="*", default=['10312021'])
    parser.add_argument('--fnames', nargs="*", default=['10312021'])
    parser.add_argument('--what', default='choice', choices=['choice', 'cd',
                                                             'model_fair',
                                                             'model_bias',
                                                             'kendall',
                                                             'anova', 'tukey'])
    return parser


def aggregate_response(data_dirs, fnames, root_dir='.', resp_status='APPROVED'):
    dfs = []
    for i, data_dir in enumerate(data_dirs):
        fname = os.path.join(root_dir, 'data', 'processed', data_dir,
                              resp_status, fnames[i])
        df = pd.read_csv(fname, index_col=0)
        df = correct_errors(df)
        df = df.replace({'Ethnicity': ETHNICITY_MAP})

        if resp_status == 'APPROVED':
            df = expired_data_handler(df)
            # print(df['Ethnicity'].value_counts())
        dfs.append(df)

    data = pd.concat(dfs, axis=0)
    data.reset_index(drop=True, inplace=True)
    return data


def merge_cd_columns(data):
    rows = []
    for index, row in data.iterrows():
        cols = ['scenario'] + CD_QS[row['scenario']] + COMMON_QS + MODELZ_QS
        row_data = [row[constants.PROLIFIC_PID]] + [row[c] for c in cols]
        rows.append(row_data)

    col_names = [constants.PROLIFIC_PID, 'scenario']
    col_names += [c if c is not None else 'Why?' for c in CDS]
    col_names += COMMON_QS + MODELZ_QS
    df = pd.DataFrame(rows, columns=col_names)
    df.set_index(constants.PROLIFIC_PID, inplace=True)
    return df


def expired_data_handler(data):
    data = data.copy(deep=True)
    for index, row in data.iterrows():
        if row['Ethnicity'] not in ETHNICITY_MAP and \
                row['Ethnicity'] not in ETHNICITY_MAP.values():
            # TODO: Can also update some other fields.
            #  Not necessary right now.
            data.loc[index, 'Ethnicity'] = STUDY_MAP[row[STUDY_ID]]

    return data


def correct_errors(df):
    df = df.replace({
        'Q10.20': {'Netiher model X not model Y':
                       'Neither model X nor model Y'},
        'Q201': {'Netiher ${e://Field/pref_model} nor model Z':
            'Neither ${e://Field/pref_model} nor model Z'}
    })
    return df