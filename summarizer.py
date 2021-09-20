import pandas as pd
import os

APPROVED = 'APPROVED'

AWAITING_REVIEW = 'AWAITING REVIEW'

STUDY_ID = 'STUDY_ID'
PROLIFIC_PID = 'PROLIFIC_PID'
MAJ_RESP_COUNT = 'MAJ_RESP_COUNT'
MIN_RESP_COUNT = 'MIN_RESP_COUNT'
MAJ_STD_ID = 'MAJ_STD_ID'
MIN_STD_ID = 'MIN_STD_ID'
MAJ_APPROVED = 'MAJ_APPROVED'
MIN_APPROVED = 'MIN_APPROVED'
SKIP_ROWS = [2]


def load_demographic(std_id, data_dir):
    fname = os.path.join(data_dir, 'prolific_export_{}.csv'.format(std_id))
    demographic = pd.read_csv(fname)
    demographic.rename(columns={"participant_id": PROLIFIC_PID}, inplace=True)
    return demographic


def load_config(data_dir):
    config = pd.read_csv(os.path.join(data_dir, 'config'),
                         sep='\t', index_col=0, squeeze=True, header=None)
    # print(config)
    global SKIP_ROWS
    if not pd.isna(config['SKIP_ROWS']):
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
    return joined


def validate_completed_counts(df, data_dir):
    grouped = df.groupby(by=[STUDY_ID], axis=0)
    for std_id, group_df in grouped:
        if config[MAJ_STD_ID] == std_id:
            count = config[MAJ_RESP_COUNT]
        else:
            count = config[MIN_RESP_COUNT]
        print('Maj' if std_id == config[MAJ_STD_ID] else 'MIN',
              std_id, len(group_df), count)
        ids = group_df[PROLIFIC_PID].sort_values()
        ids.to_csv(os.path.join(data_dir, std_id + '_ids.tsv'), sep='\t')
        assert len(group_df) == int(count)
        assert all(group_df[PROLIFIC_PID].value_counts() == 1)


def validate_approved_counts(df, data_dir):
    print(APPROVED)
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

ATNT_QS = ['Q240', 'Q241', 'Q242', 'Q243']

ATNT_ANS = ['Model X is more likely to predict correctly for female applicants than male applicants.',
            'Figure 1',
            'Female applicants will be more likely to mistakenly be rejected than male applicants.',
            'Figure 2']

COMMON_QS = [
    'Q5.7', 'Q5.8', 'Q5.9', 'Q5.10', 'Q5.11',
    'Q5.12', 'Q6.7', 'Q6.8', 'Q6.9', 'Q6.10', 'Q6.11', 'Q6.12', 'Q10.14',
    'Q10.15', 'Q10.16', 'Q10.17', 'Q10.18', 'Q10.19', 'Q10.20', 'Q10.21'
]

ICU_QS = ['Q11.6', 'Q11.9', 'Q11.10', 'Q11.13', 'Q11.14', 'Q11.17',
          'Q11.18', 'Q11.21', 'Q11.22', 'Q11.25', 'Q11.28', 'Q12.1'
]

FRAUTH_QS = ['Q11.7', 'Q11.9', 'Q11.11', 'Q11.13', 'Q11.15', 'Q11.17',
             'Q11.19', 'Q11.21', 'Q11.23', 'Q11.26', 'Q11.29', 'Q12.2'
]

RENT_QS = ['Q11.8', 'Q11.9', 'Q11.12', 'Q11.13', 'Q11.16', 'Q11.17',
           'Q11.20', 'Q11.21', 'Q11.24', 'Q11.27', 'Q11.30', 'Q12.3'
]

CDS = ['IFPI', None, 'IFNI', None, 'SFPI', None, 'SFNI', None,
       'AB', 'DB', 'DT', 'PGM']


def get_scenario_qs(scenario):
    if scenario == 'icu':
        scenario_qs = ICU_QS
    elif scenario == 'frauth':
        scenario_qs = FRAUTH_QS
    else:
        scenario_qs = RENT_QS
    return scenario_qs


def format_responses(df, questions, data_dir):
    for index, row in df.iterrows():
        pid = row[PROLIFIC_PID]
        study_id = row[STUDY_ID]
        fname = os.path.join(data_dir, study_id)
        if not os.path.exists(fname):
            os.mkdir(fname)
        fname = os.path.join(fname, 'responses')
        if not os.path.exists(fname):
            os.mkdir(fname)

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
            s += '\n'
            for i, q in enumerate(COMMON_QS):
                s += '{}: {}\n'.format(q, str(questions[q]))
                s += 'Ans: {}\n'.format(str(row[q]))
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


def get_aggregate(df, cds, privileged):
    grouped = df.groupby(by=['scenario'], axis=0)
    cd_dataframe = pd.DataFrame(index=[c for c in cds if c is not None])
    for scenario, group_df in grouped:
        print('\nScenario', scenario)
        print(group_df['Q10.20'].value_counts())
        scenario_qs = get_scenario_qs(scenario)
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
    print(cd_dataframe)
    return cd_dataframe


def get_probabilities(df, scenario):
    df = df.copy()
    df = df.replace({'Q10.20': {
        'Definitely model X': 1,
        'Probably model X': 1,
        'Netiher model X not model Y': 0,
        'Probably model Y': -1,
        'Definitely model Y': -1
    }})
    df = df[df['scenario'] == scenario]

    prob = pd.DataFrame()
    counts = df['Q10.20'].value_counts().reindex(
        [1, 0, -1], fill_value=0
    )
    print(counts)
    na, nn, nd = counts[1], counts[0], counts[-1]
    np = sum(counts)
    prob['EFPR'] = [(nd + nn/2) / np, (na + nn/2) / np]
    prob['EO'] = [(na + nn/2) / np, (nd + nn/2) / np]
    print(prob)


if __name__ == "__main__":

    data_dir = 'data/processed/'
    out_dir = 'outputs'
    batch = '09202021'
    data_dir = os.path.join(data_dir, batch)
    out_dir = os.path.join(out_dir, batch)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    fname = 'Pilot21_v2.0.csv'

    config = load_config(data_dir)
    print(config)
    df = pd.read_csv(os.path.join(data_dir, fname), skiprows=SKIP_ROWS)
    questions = {c: df.iloc[0][c] for c in df.columns}
    df.drop(index=0, axis=0, inplace=True)
    # print(*[c for c in df.columns], sep='\n')
    keep_latest_from_pid(df)

    # AWAITING REVIEW
    data_dir_ar = os.path.join(data_dir, AWAITING_REVIEW)
    merged = merge_demographics(df, data_dir_ar)
    completed = merged[merged['entered_code'] == '19AE28A9']
    validate_completed_counts(completed, data_dir_ar)
    format_responses(completed, questions, data_dir_ar)

    # APPROVED
    data_dir_ap = os.path.join(data_dir, APPROVED)
    if os.path.exists(data_dir_ap):
        merged = merge_demographics(df, data_dir_ap)
        approved = merged[merged['status'] == APPROVED]
        validate_approved_counts(approved, data_dir_ap)
        format_responses(approved, questions, data_dir_ap)

        if not AWAITING_REVIEW in approved['status'].values:
            print(False)
            cd_agg = get_aggregate(approved, CDS, privileged=None)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg.tsv'), sep='\t')

            cd_agg = get_aggregate(approved, CDS, privileged=True)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg_priv.tsv'), sep='\t')

            cd_agg = get_aggregate(approved, CDS, privileged=False)
            cd_agg.to_csv(os.path.join(out_dir, 'cd_agg_unpriv.tsv'), sep='\t')

            get_probabilities(approved, 'icu')
            get_probabilities(approved, 'frauth')
            get_probabilities(approved, 'rent')

            scenarios = approved['scenario'].value_counts().index
            for sc in scenarios:
                sc_df = approved[approved['scenario'] == sc]
                print(sc_df[[PROLIFIC_PID, 'scenario']])

            whites = approved[approved['Ethnicity'] == 'White/Caucasian']
            blacks = approved[approved['Ethnicity'] != 'White/Caucasian']
            durations = approved['time_taken'].astype(int)
            white_duration = whites['time_taken'].astype(int)
            black_duration = blacks['time_taken'].astype(int)
            print(durations.mean()/60)
            print(durations.median()/60)
            print(white_duration.mean()/60)
            print(black_duration.mean()/60)
            # print(approved[approved[PROLIFIC_PID] == '60dc0bcf44e3c6c623a104a5']
            # ['time_taken'])
            # 00:14:05