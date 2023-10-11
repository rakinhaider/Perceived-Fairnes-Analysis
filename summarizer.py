import numpy as np
import pandas as pd
import os
from collections import defaultdict
import codecs
from constants import *
from survey_info import *
from survey_info import CHOICES

TIMED_OUTS = 'TIMED_OUTS'


def load_demographic(std_id, data_dir):
    fname = os.path.join(data_dir, 'prolific_export_{}.csv'.format(std_id))
    demographic = pd.read_csv(fname)
    demographic.rename(columns={"participant_id": PROLIFIC_PID}, inplace=True)
    return demographic


def load_config(data_dir):
    config = pd.read_csv(os.path.join(data_dir, 'config'),
                         sep='\t', index_col=0, header=None).squeeze("columns")
    config['SKIP_ROWS'] = eval(config.at['SKIP_ROWS'])
    config[MAJ_RESP_COUNT] = int(config[MAJ_RESP_COUNT])
    config[MIN_RESP_COUNT] = int(config[MIN_RESP_COUNT])
    if config.notna()[MAJ_APPROVED]:
        config[MAJ_APPROVED] = int(config[MAJ_APPROVED])
    if config.notna()[MAJ_APPROVED]:
        config[MIN_APPROVED] = int(config[MIN_APPROVED])
    if config.get(TIMED_OUTS, None) == None:
        config[('%s' % TIMED_OUTS)] = []
    else:
        config[TIMED_OUTS] = eval(config[TIMED_OUTS])
        config[TIMED_OUTS] = [i - 3 for i in config[TIMED_OUTS]]
    return config


def keep_latest_from_pid(df):
    counts = df[PROLIFIC_PID].value_counts()
    for pid, count in counts.items():
        if count > 1:
            # print(pid, count)
            resp_by_pid = df[df[PROLIFIC_PID] == pid]
            # print(resp_by_pid)
            latest_start = resp_by_pid.max(axis=0)['StartDate']
            drop = resp_by_pid[resp_by_pid['StartDate'] != latest_start].index
            print('Multiple response from {}. Dropping {}'.
                  format(pid, list(drop)))
            df.drop(index=drop, inplace=True)


def merge_demographics(df, data_dir):
    study_ids = df[STUDY_ID].value_counts()
    demo_list = []
    for std_id, count in study_ids.items():
        demo = load_demographic(std_id, data_dir)
        demo_list.append(demo)

    demographics = pd.concat(demo_list)
    demographics.reset_index(inplace=True, drop=True)
    joined = df.join(demographics.set_index(PROLIFIC_PID), on=PROLIFIC_PID)
    return joined


def validate_completed_counts(df, data_dir):
    grouped = df.groupby(by=[STUDY_ID], axis=0)
    for std_id, group_df in grouped:
        if config[MAJ_STD_ID] == std_id:
            count = config[MAJ_RESP_COUNT]
        else:
            count = config[MIN_RESP_COUNT]
        incorrect_code = len(group_df[group_df['entered_code'] != COMPLETION_CODE])
        print('Maj' if std_id == config[MAJ_STD_ID] else 'MIN',
              std_id, len(group_df), count)
        ids = group_df[PROLIFIC_PID].sort_values()
        ids.to_csv(os.path.join(data_dir, std_id + '_ids.tsv'), sep='\t')
        assert len(group_df) == count
        assert all(group_df[PROLIFIC_PID].value_counts() == 1)


def validate_approved_counts(df, data_dir):
    grouped = df.groupby(by=[STUDY_ID], axis=0)
    for std_id, group_df in grouped:
        if config[MAJ_STD_ID] == std_id:
            count = config[MAJ_APPROVED]
        else:
            count = config[MIN_APPROVED]
        print('Maj' if std_id == config[MAJ_STD_ID] else 'MIN',
              std_id, len(group_df), count)
        ids = group_df[PROLIFIC_PID].sort_values()
        ids.to_csv(os.path.join(data_dir, std_id + '_ids.tsv'), sep='\t')
        assert len(group_df) == count
        assert all(group_df[PROLIFIC_PID].value_counts() == 1)


def format_responses(df, questions, verbose=False):
    correct_counts = {}
    responses = {}
    for index, row in df.iterrows():
        pid = row[PROLIFIC_PID]
        study_id = row[STUDY_ID]

        if study_id not in correct_counts:
            correct_counts[study_id] = defaultdict(list)
        s = '{}: {}\n'.format(PROLIFIC_PID, row[PROLIFIC_PID])
        s += '{}: {}\n'.format(STUDY_ID, row[STUDY_ID])
        s += '{}: {}\n'.format('scenario', row['scenario'])
        s += '{}: {}\n'.format('x_first', row['x_first'])
        s += '{}: {}\n'.format('group', row['group'])

        s += 'Attention Checks\n'.format('x_first', row['x_first'])
        correct = 4
        for i, q in enumerate(ATNT_QS):
            s += '{}: {}\n'.format(q, row[q])
            if row[q] != ATNT_ANS[i]:
                correct -= 1
                s += 'Incorrect Attention Check\n'
        s += 'Correct Attentions {}\n'.format(correct)
        correct_counts[study_id][correct].append(pid)
        s += '\n'
        for i, q in enumerate(COMMON_QS):
            s += '{}: {}\n'.format(q, str(questions[q]))
            s += 'Ans: {}\n'.format(str(row[q]))
        s += '\n'
        for i, q in enumerate(MODELZ_QS):
            formatted = str(questions[q]).replace('[Field-pref_model]',
                                                  row['pref_model'])
            s += '{}: {}\n'.format(q, formatted)
            formatted = str(row[q])
            formatted = formatted.replace('${e://Field/pref_model}',
                                          row['pref_model'])
            s += 'Ans: {}\n'.format(formatted)
        s += '\n'
        scenario_qs = CD_QS[row['scenario']]
        for i, q in enumerate(scenario_qs):
            s += '{}: {}\n'.format(q, str(questions[q]))
            s += 'Ans: {}\n'.format(str(row[q]))
        s += '\n'
        for i, q in enumerate(scenario_qs):
            if CDS[i] is not None:
                s += '{:10s}'.format(CDS[i])
        s += '\n'
        for i, q in enumerate(scenario_qs):
            if CDS[i] is not None:
                s += '{:10s}'.format(str(row[q]))
        s += '\n'
        responses[(pid, study_id)] = s

    if verbose:
        for std_id in correct_counts:
            print(STUDY_ID, std_id)
            for count in sorted(correct_counts[std_id].keys()):
                print('Correct Count:', count)
                print(*sorted(correct_counts[std_id][count]), sep='\n')
    return responses, correct_counts


def write_responses_to_files(responses, data_dir):
    for (pid, study_id) in responses:
        s = responses[(pid, study_id)]
        fname = os.path.join(data_dir, study_id, 'responses')
        os.makedirs(fname, exist_ok=True)
        fname = os.path.join(fname, pid + '.txt')
        with codecs.open(fname, 'w', encoding='utf-8') as f:
            f.write(s)
            f.flush()


def get_cd_aggregate(df, cds, study_id, verbose=False):
    grouped = df.groupby(by=['scenario'], axis=0)
    cd_dataframe = pd.DataFrame(index=[c for c in cds if c is not None])
    for scenario, group_df in grouped:
        if verbose:
            print('\nScenario', scenario, 'Study id', study_id)
            # print(group_df['Q10.20'].value_counts())
        scenario_qs = CD_QS[scenario]
        if study_id is not None:
            group_df = group_df[group_df[STUDY_ID] == study_id]
        if verbose:
            for i, q in enumerate(scenario_qs):
                if cds[i] is not None:
                    counts = group_df[q].value_counts()
                    counts.name = cds[i]
                    print(counts)
        cd_col = []
        for i, q in enumerate(scenario_qs):
            if cds[i] is not None:
                counts = group_df[q].value_counts()
                max_idx = counts.argmax()
                cd_col.append(counts.index[max_idx])

        if study_id is None:
            cd_dataframe[scenario] = cd_col[:-1] + ['x']
        else:
            cd_dataframe[scenario] = cd_col

    cd_dataframe = cd_dataframe.transpose()
    cd_dataframe.replace({'AB': {'Yes': 1, 'No': 0}}, inplace=True)
    cd_dataframe.replace({'DB': {'Yes': 1, 'No': 0}}, inplace=True)
    cd_dataframe.replace({'DT': {'Yes': 'High', 'No': 'Low'}}, inplace=True)
    cd_dataframe = cd_dataframe.transpose()
    if verbose:
        print(cd_dataframe)
    return cd_dataframe


def get_probabilities(df, criteria,
                      xy_qs='Q10.20',
                      xz_qid='Q201'):
    df = df.copy()
    grouped = df.groupby(criteria)
    res = pd.DataFrame(columns=['EFPR', 'EO', 'EFNR', 'n'])
    for tup, grp in grouped:
        if not isinstance(tup, tuple):
            tup = (tup,)
        prob = pd.DataFrame()
        grp = grp.replace({
            xy_qs: dict(zip(CHOICES[xy_qs], [1, 1, 0, -1, -1]))
        })
        grp = grp.replace({
            xz_qid: dict(
                zip(CHOICES[xz_qid], [1, 1, 0, -1, -1]))
        })

        counts = grp[xy_qs].value_counts().reindex(
            [1, 0, -1], fill_value=0
        )
        # print(grp[trade_off_qs].value_counts())
        n_efpr, n_none, n_eo = counts[1], counts[0], counts[-1]
        print(tup, 'Total {:d} responses'.format(len(grp)))
        print(n_efpr, n_none, n_eo)
        np = sum(counts)
        prob['EFPR'] = [(n_eo + n_none/2) / np, (n_efpr + n_none/2) / np]
        prob['EO'] = [(n_efpr + n_none/2) / np, (n_eo + n_none/2) / np]
        counts = grp[xz_qid].value_counts().reindex(
            [1, 0, -1], fill_value=0
        )
        n_efnr = counts[-1]
        n_none = counts[0]
        n_x_or_y = counts[1]
        print(n_x_or_y, n_none, n_efnr)
        prob['EFNR'] = [(n_x_or_y + n_none/2) / np, (n_efnr + n_none/2) / np]
        res = res.append(pd.Series({'EFPR': prob.loc[1]['EFPR'],
                                    'EO': prob.loc[1]['EO'],
                                    'EFNR': prob.loc[1]['EFNR'],
                                    'n': len(grp)}, name=tup))
        # print(df[[trade_off_qs, xz_trade_off_qs]].value_counts())
        nx, ny, nz = 0, 0, 0
        for i, row in grp[[xy_qs, xz_qid]].iterrows():
            if row[xz_qid] == -1:
                nz += 1
            elif row[xz_qid] == 0:
                if row[xy_qs] == 1:
                    nx += 1
                elif row[xy_qs] == -1:
                    ny += 1
                else:
                    nx += 1
                    ny += 1
                nz += 1
            else:
                if row[xy_qs] == 1:
                    nx += 1
                elif row[xy_qs] == -1:
                    ny += 1
                else:
                    nx += 1
                    ny += 1
            # print(row)
            # print(nx, ny, nz)
        # print(nx/len(grp), ny/len(grp), nz/len(grp))
    return res


def get_pgm(row):
    for col in ['Q12.1', 'Q12.2', 'Q12.3']:
        if isinstance(row[col], str):
            return row[col]


def drop_skip_rows(df, config):
    for index in config['SKIP_ROWS']:
        index = index - 3
        row = df.loc[index]
        # print(row[PROLIFIC_PID])
        if row[STUDY_ID] == config[MAJ_STD_ID]:
            config[MAJ_RESP_COUNT] -= 1
        else:
            config[MIN_RESP_COUNT] -= 1
        df.drop(index=[index], inplace=True)
    return df


def load_raw_responses(data_dir, fname, config, skip_rows=SKIP_ROWS):
    print(os.path.join(data_dir, fname))
    df = pd.read_csv(os.path.join(data_dir, fname), skiprows=SKIP_ROWS)
    df = drop_skip_rows(df, config)
    df.drop(index=0, axis=0, inplace=True)
    df['PGM'] = df.apply(get_pgm, axis=1)
    keep_latest_from_pid(df)
    return df


def load_question(data_dir, fname):
    df = pd.read_csv(os.path.join(data_dir, fname), nrows=2)
    questions = {c: df.iloc[0][c] for c in df.columns}
    return questions


if __name__ == "__main__":
    data_dir = 'data/processed/'
    out_dir = 'outputs'
    batch = '11092021'
    data_dir = os.path.join(data_dir, batch)
    out_dir = os.path.join(out_dir, batch)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    fname = '11092021.csv'
    criteria = ['scenario', 'STUDY_ID', 'x_first']

    config = load_config(data_dir)
    questions = load_question(data_dir, fname)
    df = load_raw_responses(data_dir, fname, config, questions)

    # AWAITING REVIEW
    data_dir_ar = os.path.join(data_dir, AWAITING_REVIEW)
    merged = merge_demographics(df, data_dir_ar)
    completed = merged[(merged['entered_code'] == COMPLETION_CODE) |
                       merged.index.isin(config[TIMED_OUTS])]
    completed.to_csv(os.path.join(data_dir, AWAITING_REVIEW,
                                  fname.replace('.csv', '_completed.csv')))
    validate_completed_counts(completed, data_dir_ar)
    responses, graph_comprehension_stat = format_responses(
        completed, questions, verbose=True)
    write_responses_to_files(responses, data_dir_ar)


    if False:
        print(completed)
        criteria = ['scenario']
        print('Criteria:', criteria)
        get_probabilities(completed, criteria)
        criteria = ['scenario', 'group', STUDY_ID]
        print('Criteria:', criteria)
        get_probabilities(completed, criteria)
        criteria = ['scenario', 'PGM']
        print('Criteria:', criteria)
        get_probabilities(completed, criteria)
        criteria = ['scenario', STUDY_ID]
        print('Criteria:', criteria)
        get_probabilities(completed, criteria)

    # APPROVED
    data_dir_ap = os.path.join(data_dir, APPROVED)
    if os.path.exists(data_dir_ap):
        merged = merge_demographics(df, data_dir_ap)
        approved = merged[merged['status'] == APPROVED]
        approved.to_csv(os.path.join(data_dir, APPROVED,
                                     fname.replace('.csv', '_approved.csv')))

        validate_approved_counts(approved, data_dir_ap)
        responses, _ = format_responses(approved, questions)
        write_responses_to_files(responses, data_dir_ap)


        if not AWAITING_REVIEW in approved['status'].values:
            cd_agg = get_cd_aggregate(approved, CDS, study_id=None)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg.tsv'), sep='\t')

            std_ids = approved[STUDY_ID].value_counts().index
            for s in std_ids:
                cd_agg = get_cd_aggregate(approved, CDS, study_id=s, verbose=False)
                cd_agg.to_csv(os.path.join(
                    out_dir, 'cd_agg_{:s}.tsv'.format(s)), sep='\t')

            print(approved)
            criteria = ['scenario']
            print('Criteria:', criteria)
            get_probabilities(approved, criteria)
            criteria = ['scenario', 'group', STUDY_ID]
            print('Criteria:', criteria)
            get_probabilities(approved, criteria)
            criteria = ['scenario', 'PGM']
            print('Criteria:', criteria)
            get_probabilities(approved, criteria)
            criteria = ['scenario', STUDY_ID]
            print('Criteria:', criteria)
            get_probabilities(approved, criteria)

            scenarios = approved['scenario'].value_counts().index
            for sc in scenarios:
                sc_df = approved[approved['scenario'] == sc]
                # print(sc_df[[PROLIFIC_PID, 'scenario']])