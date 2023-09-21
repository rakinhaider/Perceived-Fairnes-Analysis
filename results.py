from survey_info import SCENARIO_NAME_MAP
from hypothesis_testing import get_preference_probabilities
from utils import get_parser, aggregate_response

MODEL_TO_METRIC = {'x': 'efdr', 'y': 'eo', 'z': 'efor'}


def run_rq1_preference_probability(response, qid, grouping_criteria):
    if len(grouping_criteria) == 0:
        grouped = [('all', response)]
    else:
        grouped = response.groupby(grouping_criteria)
    for tup, grp in grouped:
        counts = get_preference_probabilities(grp)
        counts = counts.transpose()
        condition = list(tup) if len(grouping_criteria) > 0 else []
        print('\\midrule')
        for index, row in counts.iterrows():
            print('\t&\t'.join([SCENARIO_NAME_MAP[index]] + condition
                               + [f'{val:.3f}' for val in row]) + '\\\\')


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--research-question', '-rq', type=int)
    parser.add_argument('--result-type', '-rt', default='mean')
    args = parser.parse_args()

    qid = args.qid
    grouping_criteria = args.criteria
    fnames = [f + "_approved.csv" for f in args.fnames]
    response = aggregate_response(args.resp_dirs, fnames)
    grouping_criteria.remove('scenario')
    if args.research_question == 1 and args.result_type == 'mean':
        run_rq1_preference_probability(response, qid, grouping_criteria)
