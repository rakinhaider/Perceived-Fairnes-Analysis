import pandas as pd
import os
from collections import defaultdict

from constants import *
from survey_info import *
from survey_info import CHOICES


def load_demographic(std_id, data_dir):
    fname = os.path.join(data_dir, 'prolific_export_{}.csv'.format(std_id))
    demographic = pd.read_csv(fname)
    demographic.rename(columns={"participant_id": PROLIFIC_PID}, inplace=True)
    return demographic


def load_config(data_dir):
    config = pd.read_csv(os.path.join(data_dir, 'config'),
                         sep='\t', index_col=0, squeeze=True, header=None)
    # print(config_09202021)
    if not pd.isna(config['SKIP_ROWS']):
        global SKIP_ROWS
        SKIP_ROWS = SKIP_ROWS + config['SKIP_ROWS'].split(',')
        SKIP_ROWS = [int(s) for s in SKIP_ROWS]

    return config


def keep_latest_from_pid(df):
    counts = df[PROLIFIC_PID].value_counts()
    for pid, count in counts.iteritems():
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
    for std_id, count in study_ids.iteritems():
        # print(std_id)
        demo = load_demographic(std_id, data_dir)
        # print(demo)
        demo_list.append(demo)

    demographics = pd.concat(demo_list)
    demographics.reset_index(inplace=True, drop=True)
    # print(demographics)
    joined = df.join(demographics.set_index(PROLIFIC_PID), on=PROLIFIC_PID)
    print(joined[PROLIFIC_PID])
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
        assert len(group_df) == int(count)
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
        assert len(group_df) == int(count)
        assert all(group_df[PROLIFIC_PID].value_counts() == 1)


def get_scenario_qs(scenario):
    if scenario == 'icu':
        scenario_qs = ICU_QS
    elif scenario == 'frauth':
        scenario_qs = FRAUTH_QS
    else:
        scenario_qs = RENT_QS
    return scenario_qs


def format_responses(df, questions, data_dir):
    correct_count = {}
    for index, row in df.iterrows():
        pid = row[PROLIFIC_PID]
        study_id = row[STUDY_ID]
        fname = os.path.join(data_dir, study_id)
        if not os.path.exists(fname):
            os.mkdir(fname)
        fname = os.path.join(fname, 'responses')
        if not os.path.exists(fname):
            os.mkdir(fname)

        if study_id not in correct_count:
            correct_count[study_id] = defaultdict(list)

        fname = os.path.join(fname, pid + '.txt')
        with open(fname, 'w') as f:
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
            correct_count[study_id][correct].append(pid)
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
            scenario_qs = get_scenario_qs(row['scenario'])
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
            f.write(s)
            f.flush()

    """
    for std_id in correct_count:
        print(std_id)
        std_dict = correct_count[std_id]
        print([len(std_dict[i]) for i in range(5)])
        for i in range(5):
            print(i)
            print(*std_dict[i], sep='\n')
            print()
    """


def get_cd_aggregate(df, cds, privileged, verbose=False):
    grouped = df.groupby(by=['scenario'], axis=0)
    cd_dataframe = pd.DataFrame(index=[c for c in cds if c is not None])
    for scenario, group_df in grouped:
        if verbose:
            print('\nScenario', scenario)
            print(group_df['Q10.20'].value_counts())
        scenario_qs = get_scenario_qs(scenario)
        # TODO: Correct the following computation.
        #  The selection should be done on ethnicity, not on self-declaration.
        # group_df = group_df[group_df[STUDY_ID] == study_id]
        pgm_col = scenario_qs[-1]
        if privileged:
            group_df = group_df[group_df[pgm_col] == 'Advantaged']
        elif privileged == False:
            group_df = group_df[group_df[pgm_col] == 'Disadvantaged']
        """
        for i, q in enumerate(scenario_qs):
            if cds[i] is not None:
                counts = group_df[q].value_counts()
                counts.name = cds[i]
                print(counts)
        """
        cd_col = []
        for i, q in enumerate(scenario_qs):
            if cds[i] is not None:
                counts = group_df[q].value_counts()
                max_idx = counts.argmax()
                cd_col.append(counts.index[max_idx])

        if privileged is None:
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


def get_probabilities(df, criteria):
    df = df.copy()
    trade_off_qs = 'Q10.20'
    xz_trade_off_qs = 'Q201'
    df = df.replace({
        trade_off_qs: dict(zip(CHOICES[trade_off_qs], [1, 1, 0, -1, -1]))
    })
    df = df.replace({
        xz_trade_off_qs: dict(zip(CHOICES[xz_trade_off_qs], [1, 1, 0, -1, -1]))
    })

    grouped = df.groupby(criteria)
    for tup, grp in grouped:
        prob = pd.DataFrame()
        counts = grp[trade_off_qs].value_counts().reindex(
            [1, 0, -1], fill_value=0
        )
        n_efpr, n_none, n_eo = counts[1], counts[0], counts[-1]
        print(tup, 'Total {:d} responses'.format(len(grp)))
        print(n_efpr, n_none, n_eo)
        np = sum(counts)
        prob['EFPR'] = [(n_eo + n_none/2) / np, (n_efpr + n_none/2) / np]
        prob['EO'] = [(n_efpr + n_none/2) / np, (n_eo + n_none/2) / np]
        counts = grp[xz_trade_off_qs].value_counts().reindex(
            [1, 0, -1], fill_value=0
        )
        n_efnr = counts[-1]
        n_none = counts[0]
        n_x_or_y = counts[1]
        prob['EFNR'] = [(n_x_or_y + n_none/2) / np, (n_efnr + n_none/2) / np]
        print(prob)
        # print(df[[trade_off_qs, xz_trade_off_qs]].value_counts())
        nx, ny, nz = 0, 0, 0
        for i, row in grp[[trade_off_qs, xz_trade_off_qs]].iterrows():
            if row[xz_trade_off_qs] == -1:
                nz += 1
            elif row[xz_trade_off_qs] == 0:
                if row[trade_off_qs] == 1:
                    nx += 1
                elif row[trade_off_qs] == -1:
                    ny += 1
                else:
                    nx += 1
                    ny += 1
                nz += 1
            else:
                if row[trade_off_qs] == 1:
                    nx += 1
                elif row[trade_off_qs] == -1:
                    ny += 1
                else:
                    nx += 1
                    ny += 1
            # print(row)
            # print(nx, ny, nz)
        # print(nx/len(grp), ny/len(grp), nz/len(grp))


if __name__ == "__main__":

    data_dir = 'data/processed/'
    out_dir = 'outputs'
    batch = '10082021'
    data_dir = os.path.join(data_dir, batch)
    out_dir = os.path.join(out_dir, batch)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    fname = '10082021.csv'
    criteria = ['scenario', 'STUDY_ID', 'x_first']

    config = load_config(data_dir)
    df = pd.read_csv(os.path.join(data_dir, fname), skiprows=SKIP_ROWS)
    questions = {c: df.iloc[0][c] for c in df.columns}
    df.drop(index=0, axis=0, inplace=True)
    keep_latest_from_pid(df)

    # AWAITING REVIEW
    data_dir_ar = os.path.join(data_dir, AWAITING_REVIEW)
    merged = merge_demographics(df, data_dir_ar)
    completed = merged[merged['entered_code'] == COMPLETION_CODE]
    completed.to_csv(os.path.join(data_dir, AWAITING_REVIEW,
                                  fname.replace('.csv', '_completed.csv')))
    validate_completed_counts(completed, data_dir_ar)
    format_responses(completed, questions, data_dir_ar)

    get_probabilities(completed, criteria)

    # APPROVED
    data_dir_ap = os.path.join(data_dir, APPROVED)
    if os.path.exists(data_dir_ap):
        merged = merge_demographics(df, data_dir_ap)
        approved = merged[merged['status'] == APPROVED]
        print(approved[PROLIFIC_PID])
        approved.to_csv(os.path.join(data_dir, APPROVED,
                                     fname.replace('.csv', '_approved.csv')))

        validate_approved_counts(approved, data_dir_ap)
        format_responses(approved, questions, data_dir_ap)

        if not AWAITING_REVIEW in approved['status'].values:
            cd_agg = get_cd_aggregate(approved, CDS, privileged=None,
                                      verbose=False)
            # print(cd_agg)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg.tsv'), sep='\t')

            cd_agg = get_cd_aggregate(approved, CDS, privileged=True,
                                      verbose=False)
            # print(cd_agg)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg_priv.tsv'), sep='\t')

            cd_agg = get_cd_aggregate(approved, CDS, privileged=False,
                                      verbose=False)
            # print(cd_agg)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg_unpriv.tsv'), sep='\t')

            get_probabilities(approved, criteria)

            scenarios = approved['scenario'].value_counts().index
            for sc in scenarios:
                sc_df = approved[approved['scenario'] == sc]
                # print(sc_df[[PROLIFIC_PID, 'scenario']])