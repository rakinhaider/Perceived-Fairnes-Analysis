import pandas as pd
import os
from constants import *
from survey_info import ETHNICITY_MAP, STUDY_MAP

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
        print(df)
        dfs.append(df)

    data = pd.concat(dfs, axis=0)
    data.reset_index(drop=True, inplace=True)
    print(data)
    return data


if __name__ == "__main__":
    data_dirs = ['09202021', '10082021']
    fnames = ['Pilot21_v2.0_10082021', '10082021']
    fnames = [f + '_approved.csv' for f in fnames]

    df = aggregate_response(data_dirs, fnames)
    df = expired_data_handler(df)
    df.to_csv(os.path.join('data', 'processed', 'response.csv'))