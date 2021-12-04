DY = 'Definitely model Y'
PY = 'Probably model Y'
PX = 'Probably model X'
DX = 'Definitely model X'

FAIR_OPTIONS = ['Very unfair', 'Mildly unfair', 'Neither fair nor unfair',
                'Acceptably fair', 'Very fair']
BIAS_OPTIONS = ['Very unbiased', 'Mildly unbiased',
                'Neither biased nor unbiased',
                'Acceptably biased', 'Very biased']

FAIR_QIDS = ['Q5.7', 'Q6.7']
BIAS_QIDS = ['Q5.9', 'Q6.9']
TRADE_OFFS = ['=FPR \u2260Outcome', '\u2260FPR =Outcome']

ATNT_QS = ['Q240', 'Q241', 'Q242', 'Q243']
ATNT_ANS = [
    'Model X is more likely to predict correctly for female applicants than '
    'male applicants.',
    'Figure 1',
    'Female applicants will be more likely to mistakenly be rejected than '
    'male applicants.',
    'Figure 2']
COMMON_QS = [
    'Q5.7', 'Q5.8', 'Q5.9', 'Q5.10', 'Q5.11',
    'Q5.12', 'Q6.7', 'Q6.8', 'Q6.9', 'Q6.10', 'Q6.11', 'Q6.12', 'Q10.14',
    'Q10.15', 'Q10.16', 'Q10.17', 'Q10.18', 'Q10.19', 'Q10.20', 'Q10.21',
]
MODELZ_QS = ['Q199', 'Q203', 'Q200', 'Q204', 'Q201', 'Q205']

CD_QS = {
    'icu': ['Q11.6', 'Q11.9', 'Q11.10', 'Q11.13', 'Q11.14', 'Q11.17',
            'Q11.18', 'Q11.21', 'Q11.22', 'Q11.25', 'Q11.28', 'Q12.1'],
    'frauth': ['Q11.7', 'Q11.9', 'Q11.11', 'Q11.13', 'Q11.15', 'Q11.17',
               'Q11.19', 'Q11.21', 'Q11.23', 'Q11.26', 'Q11.29', 'Q12.2'],
    'rent': ['Q11.8', 'Q11.9', 'Q11.12', 'Q11.13', 'Q11.16', 'Q11.17',
             'Q11.20', 'Q11.21', 'Q11.24', 'Q11.27', 'Q11.30', 'Q12.3']
}

CDS = ['IFPI', None, 'IFNI', None, 'SFPI', None, 'SFNI', None,
       'AB', 'DB', 'DT', 'PGM']

CHOICES = {'Q10.20': [DX, PX, 'Neither model X nor model Y', PY, DY],
           'Q10.14': [DX, PX, 'Models X and Y are equally fair', PY, DY],
           'Q10.16': [DX, PX, 'Models X and Y are equally biased', PY, DY],
           'Q10.18': [DX, PX, 'Models X and Y are equally useful', PY, DY],
           'Q199': ['Definitely ${e://Field/pref_model}',
                    'Probably ${e://Field/pref_model}',
                    'Both ${e://Field/pref_model} and Z are equally fair',
                    'Probably model Z', 'Definitely model Z'],
           'Q201': ['Definitely ${e://Field/pref_model}',
                    'Probably ${e://Field/pref_model}',
                    'Neither ${e://Field/pref_model} nor model Z',
                    'Probably model Z', 'Definitely model Z'],
           'Q5.7': FAIR_OPTIONS,
           'Q5.9': [op if i != 2 else op + '.'
                    for i, op in enumerate(BIAS_OPTIONS)
                    ],
           'Q6.7': FAIR_OPTIONS,
           'Q6.9': BIAS_OPTIONS,
           'CD': ['High', 'Moderate', 'Low']}

ETHNICITY_MAP = {'White/Caucasian': 'Maj',
                 'Black/African American': 'Min', 'Latino/Hispanic': 'Min',
                 'African': 'Min'}

STUDY_MAP = {'615f94f07f89d7a8afda6025': 'Maj',
             '614391dcdf09127f013fe60a': 'Maj',
             '615f943ec32164d0f282bd34': 'Min',
             '61439223ecf74e491e22a39c': 'Min',
             '617f102667bd25701aa8461e': 'Maj',
             '617f10dcb9390b6c13530629': 'Min',
             '618b3dcb32ad5f5cc3f00d86': 'Maj',
             '618b3e1d59be0d9e09f6f943': 'Min'
             }
