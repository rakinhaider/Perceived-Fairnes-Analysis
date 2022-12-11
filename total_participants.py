import os
import pandas as pd
from constants import AWAITING_REVIEW, APPROVED
from summarizer import load_config

if __name__ == "__main__":

    is_approved = False
    sub_dir = APPROVED

    batches = ['09202021', '10082021', '10312021', '11092021']
    data_dir = 'data/processed'
    counts = []

    for b in batches:
        batch_dir = os.path.join(data_dir, b)
        config = load_config(batch_dir)
        batch_dir = os.path.join(data_dir, b, APPROVED)
        fname_base = 'prolific_export_{}.csv'
        for study_id in [config['MAJ_STD_ID'], config['MIN_STD_ID']]:
            fname = os.path.join(batch_dir, fname_base.format(study_id))
            df = pd.read_csv(fname)
            counts.append(df['status'].value_counts())
            print(df['status'].value_counts())

    print(sum(counts))

    ### From Prolific
    accepts = [11, 10, 33, 41, 27, 35, 26, 31]
    rejects = [2, 3, 7, 3, 1, 2, 4, 3]

    print(sum(accepts))
    print(sum(rejects))
    print(sum(accepts) + sum(rejects))