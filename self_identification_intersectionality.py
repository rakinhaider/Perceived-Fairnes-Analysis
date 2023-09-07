import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from constants import AWAITING_REVIEW, APPROVED, PROLIFIC_PID
from publication_plots_util import set_rcparams, set_size
from summarizer import (
    format_responses, load_config, load_question
)

ETHNICITY_MAP = {'White/Caucasian': 'White',
                 'Black/African American': 'Black',
                 'Latino/Hispanic': 'Latino',
                 'African': 'Black'}


def get_pids_by_status(status, participant_fname_base, config):
    pids = []
    for studyid in [config['MAJ_STD_ID'], config['MIN_STD_ID']]:
        participants = pd.read_csv(participant_fname_base.format(studyid))
        temp = participants['status'].isin([status])
        pids.extend(participants[temp]['participant_id'].values)
    return pids


def load_reviewed_responses(batch_dir, participant_fname, fname_base):
    fname = fname_base + '_completed.csv'
    ac_pids = get_pids_by_status('APPROVED', participant_fname, config)
    rej_pids = get_pids_by_status('REJECTED', participant_fname, config)
    # Why read from AWAITING_REVIEW?
    # ANS: Because we need both approved and rejecteds.
    # In APPROVED folder rejecteds are missing.
    batch_dir = os.path.join(batch_dir, AWAITING_REVIEW)
    df = pd.read_csv(os.path.join(batch_dir, fname), sep=',', index_col=0)
    df = df[df[PROLIFIC_PID].isin(ac_pids + rej_pids)]
    return df


if __name__ == "__main__":

    batches = ['09202021', '10082021', '10312021', '11092021']
    fname_bases = ['Pilot21_v2.0_10082021', '10082021', '10312021', '11092021']
    data_dir = 'data/processed/'
    demographic_fname_base = 'prolific_export_{}.csv'

    count_stats = {'rej': defaultdict(int), 'acc': defaultdict(int)}
    total = 0
    df_by_eth = []
    df_by_eth_sex = []
    for b, fname_base in zip(batches, fname_bases):
        batch_dir = os.path.join(data_dir, b)
        config = load_config(batch_dir)

        # Approved or rejected ids
        participant_fname = os.path.join(batch_dir, APPROVED,
                                         demographic_fname_base)
        df = load_reviewed_responses(batch_dir, participant_fname, fname_base)
        df = df[df['Ethnicity'] != 'DATA EXPIRED']
        df.replace({'Ethnicity': ETHNICITY_MAP}, inplace=True)
        df_by_eth.append(df[['Ethnicity', 'PGM']])
        df_by_eth_sex.append(df[['Ethnicity', 'PGM', 'Sex']])

    agg = pd.concat(df_by_eth).reset_index(drop=True)
    # agg.replace('African', 'Black/African American', inplace=True)
    eth_count = agg['Ethnicity'].value_counts()
    print(eth_count)
    df_by_eth_agg = agg.groupby(agg.columns.tolist(), as_index=False).size()
    df_by_eth_agg['rate'] = df_by_eth_agg.apply(
        lambda row: row['size']/eth_count[row['Ethnicity']], axis=1)
    print(df_by_eth_agg)
    agg = pd.concat(df_by_eth_sex).reset_index(drop=True)
    eth_count_by_sex = agg[['Ethnicity', 'Sex']].value_counts()
    print(eth_count_by_sex)
    # agg.replace('African', 'Black/African American', inplace=True)
    df_by_eth_sex_agg = agg.groupby(agg.columns.tolist(), as_index=False).size()

    def get_rate(row, count):
        denom = count[(row['Ethnicity'], row['Sex'])]
        return row['size'] / denom
    df_by_eth_sex_agg['rate'] = df_by_eth_sex_agg.apply(
        get_rate, count=eth_count_by_sex, axis=1)
    print(df_by_eth_sex_agg.sort_values(by=['Ethnicity', 'Sex', 'PGM']))

    colors = plt.get_cmap('RdYlGn')([0.15, 0.85])
    df_by_eth_sex_agg.set_index(keys=['Ethnicity', 'Sex', 'PGM'],
                                drop=True, append=False, inplace=True)

    set_rcparams(fontsize=18)
    plt.figure(figsize=set_size(width=241, fraction=1, aspect_ratio=0.75))
    adv_bars = []
    y_ticklabels = []
    for s in eth_count.index:
        for g in ['Female', 'Male']:
            adv_bars.append(df_by_eth_sex_agg.loc[s, g, 'Advantaged']['rate'])
            y_ticklabels.append('{}_{}'.format(s, g))
    plt.barh(range(0, 6), [1 - r for r in adv_bars], label='Disadvantaged',
             color=colors[0])
    plt.barh(range(0, 6), adv_bars, label='Advantaged', color=colors[1],
             left=[1 - r for r in adv_bars])
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1.1), loc='center right',
               fontsize=8)
    plt.yticks(range(0, 6), y_ticklabels, fontsize='xx-small')
    plt.xticks(np.arange(0, 1.1,  .2), range(0, 110, 20), fontsize='xx-small')
    plt.xlim((0, 1))
    plt.xlabel('Portion of Participants (\%)', fontsize='xx-small')
    plt.ylabel('Demographic Sub-groups', fontsize='xx-small')
    plt.tight_layout()
    plt.savefig('outputs/intersectionality.pdf')
    plt.show()