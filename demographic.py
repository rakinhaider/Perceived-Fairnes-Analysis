import os
import pandas as pd
from utils import get_parser
from summarizer import (
    load_config, drop_skip_rows, SKIP_ROWS, get_pgm, keep_latest_from_pid,
    merge_demographics, AWAITING_REVIEW
)
from survey_response_aggregator import aggregate_response

if __name__ == "__main__":
    args = get_parser().parse_args()
    data_dir = 'data/processed/'
    out_dir = 'outputs'
    for resp_dir, fname in zip(args.resp_dirs, args.fnames):
        print(resp_dir, fname)
        data_subdir = os.path.join(data_dir, resp_dir)
        print(data_subdir)
        config = load_config(data_subdir)
        df = pd.read_csv(os.path.join(data_subdir, fname+'.csv'), skiprows=SKIP_ROWS)
        df = drop_skip_rows(df, config)
        questions = {c: df.iloc[0][c] for c in df.columns}
        df.drop(index=0, axis=0, inplace=True)
        df['PGM'] = df.apply(get_pgm, axis=1)
        keep_latest_from_pid(df)
        data_dir_dem = os.path.join(data_subdir, 'APPROVED')
        merged = merge_demographics(df, data_dir_dem)

        print(merged['status'].value_counts())


    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    df = aggregate_response(args.resp_dirs, fnames)
    print(df.columns)
    # print(df['Ethnicity'].value_counts())
    # print(df['Sex'].value_counts()/len(df))
    # print(df['Student Status'].value_counts())
    # print(df['Employment Status'].value_counts()/len(df))
    # print(df['PGM'].value_counts()/ len(df))
    # print(df['age'].value_counts())

    ages = df['age'].copy()
    def age_bucketing(val):
        # print(val)
        if val < 25:
            return 0
        elif val < 40:
            return 1
        else:
            return 2

    age_buckets = ages.apply(age_bucketing)
    # print(age_buckets.value_counts()/len(df))
    df.to_csv('temp_response.tsv', sep='\t')

    from datetime import datetime
    def time_diff(row):
        starttime = datetime.strptime(row['StartDate'], '%Y-%m-%d %H:%M:%S')
        endtime = datetime.strptime(row['EndDate'], '%Y-%m-%d %H:%M:%S')
        return (endtime - starttime)

    times = df.apply(lambda row: time_diff(row), axis=1)
    print(times.mean())