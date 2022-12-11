import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from constants import AWAITING_REVIEW, APPROVED, PROLIFIC_PID
from survey_info import ATNT_QS, ATNT_ANS
from publication_plots_util import set_rcparams, set_size
from summarizer import (
    format_responses, load_config, load_question
)


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


def plot_graph_comprehension_rate_counts(count_stats):
    plt.figure(figsize=set_size(width=430, fraction=.95, aspect_ratio=0.75))
    set_rcparams(fontsize=18)

    colors = plt.get_cmap('RdYlGn')([0.15, 0.85])
    # count_stats not used right now.
    plt.bar(range(0, 5), [0, 2, 68, 57, 86], label='accepted',
            bottom=[7, 13, 3, 1, 1], color=colors[1])
    plt.bar(range(0, 5), [7, 13, 3, 1, 1], label='rejected', color=colors[0])
    plt.xticks(range(0, 5), np.linspace(0, 1, 5))
    plt.xlabel('Correct graph comprehension rate.')
    plt.ylabel('Number of participants.')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/graph_comprehension_stats.pdf')
    plt.show()


if __name__ == "__main__":

    batches = ['09202021', '10082021', '10312021', '11092021']
    fname_bases = ['Pilot21_v2.0_10082021', '10082021', '10312021', '11092021']
    data_dir = 'data/processed/'
    demographic_fname_base = 'prolific_export_{}.csv'

    count_stats = {'rej': defaultdict(int), 'acc': defaultdict(int)}
    total = 0

    for b, fname_base in zip(batches, fname_bases):
        batch_dir = os.path.join(data_dir, b)
        config = load_config(batch_dir)

        # Approved or rejected ids
        participant_fname = os.path.join(batch_dir, APPROVED,
                                         demographic_fname_base)
        df = load_reviewed_responses(batch_dir, participant_fname, fname_base)
        questions = load_question(os.path.join(batch_dir), fname_base + '.csv')
        responses, correct_counts = format_responses(df, questions)
        rej_pids = get_pids_by_status('REJECTED', participant_fname, config)
        for studyid in correct_counts:
            for count in correct_counts[studyid]:
                # rej_count = 0
                pid_list = correct_counts[studyid][count]
                rej_count = len(set(pid_list).intersection(rej_pids))
                count_stats['rej'][count] += rej_count
                count_stats['acc'][count] += (len(pid_list) - rej_count)
                total += len(pid_list)

    print(total)
    print(count_stats)
    # plot_graph_comprehension_rate_counts(count_stats)

    per_question_count = {q:[] for q in ATNT_QS}
    for b, fname_base in zip(batches, fname_bases):
        batch_dir = os.path.join(data_dir, b)
        config = load_config(batch_dir)

        # Approved or rejected ids
        participant_fname = os.path.join(batch_dir, APPROVED,
                                         'prolific_export_{}.csv')
        df = load_reviewed_responses(batch_dir, participant_fname, fname_base)
        for i, q in enumerate(ATNT_QS):
            per_question_count[q].append((df[q] == ATNT_ANS[i]).value_counts())

    rates = []
    for q in ATNT_QS:
        agg = sum(per_question_count[q])
        rates.append(agg[True] / sum(agg))

    plt.clf()
    colors = plt.get_cmap('RdYlGn')([0.15, 0.85])
    plt.figure(figsize=set_size(width=430, fraction=.95, aspect_ratio=0.75))
    set_rcparams(fontsize=18)
    plt.bar(range(0, 8, 2), rates, width=1.4, color=colors[1])
    plt.xticks(range(0, 8, 2), ['GC' + str(i) for i in range(1, 5)], rotation=45)
    plt.yticks(np.linspace(.2, 1, 5), range(20, 110, 20))
    plt.xlabel('Questions')
    plt.ylabel('Participants answered \ncorrectly (in \%)')
    for rate, loc in zip(rates, range(0, 8, 2)):
        plt.text(loc - 0.35, rate + 0.025, '{:.2f}'.format(rate*100), fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/per_question_stat.pdf')
    plt.show()