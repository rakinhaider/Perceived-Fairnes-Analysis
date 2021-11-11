import pandas as pd
import os
from summarizer import get_probabilities
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--xz-qid', default='Q201')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    args = parser.parse_args()

    data_dir = 'data/processed/'

    df = pd.read_csv(os.path.join(data_dir, 'response.csv'), index_col=0)
    criteria = args.criteria
    get_probabilities(df, criteria, xy_qs=args.qid,
                      xz_qid=args.xz_qid)