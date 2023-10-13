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

COMPLETION_CODE = '19AE28A9'

TEX_TABLE = """
\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{|{:s}|}}
        \\hline
        {:s} & EFPR & EO & EFNR & n & p-value\\\\
        \\hline
        {:s}
        \\hline
    \\end{{tabular}}
    \\caption{{Grouped by {:s}}}
    \\label{{tab:my_label}}
\\end{{table}}"""

TEX_FIGURE = """
\\begin{{figure}}[h]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{figures/{:s}/{:s}/{:s}}}
    \\caption{{Grouped by {:s}}}
    \\label{{fig:my_label}}
\\end{{figure}}
"""
VALUE_SHORT_FORM = {'Disadvantaged': 'Disadv.',
                    'Advantaged': 'Adv.',
                    'Caucasian': 'Cauc.',
                    'Non-Caucasian': 'Non-Cauc.'}
CRITERIA_TO_TEXT = {'group': 'Disadv. Group',
                    "Ethnicity": 'Ethnicity',
                    'PGM': 'self-identified privilege'}
