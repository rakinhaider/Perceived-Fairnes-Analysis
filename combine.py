import argparse
import os
import matplotlib.pyplot as plt
from summarizer import get_probabilities
from survey_response_aggregator import aggregate_response
from hypothesis_testing import test_choices, test_cd_choices
from plotter import plot_distribution
from constants import TEX_TABLE, TEX_FIGURE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid', default='Q10.20')
    parser.add_argument('--xz-qid', default='Q201')
    parser.add_argument('--criteria', nargs="*", default=['scenario'])
    parser.add_argument('--data-dirs', nargs="*", default=['10312021'])
    parser.add_argument('--fnames', nargs="*", default=['10312021'])
    parser.add_argument('--what', default='choice', choices=['choice', 'cd'])

    args = parser.parse_args()

    data_dir = 'data/processed/'
    out_dir = 'outputs'
    out_dir = os.path.join(out_dir, '_'.join(args.data_dirs), args.qid)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    criteria = args.criteria

    data_dirs = args.data_dirs
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    response = aggregate_response(data_dirs, fnames)

    # Should return something.
    res = get_probabilities(response, criteria, xy_qs=args.qid,
                            xz_qid=args.xz_qid)
    # Test Hypothesis
    if args.what == 'choice':
        stats = test_choices(response, args.qid, args.criteria)
        res['pvalue'] = stats
    else:
        test_cd_choices(response, args.criteria)

    print(res)
    # Plot
    plot_distribution(response, criteria, xy_qs=args.qid, stats=res)
    fname = '_'.join([args.qid] + criteria) + '.pdf'
    plt.savefig(os.path.join(out_dir, fname), format='pdf')

    # Generate LaTeX file
    fig_name = fname
    fname = '_'.join([args.qid] + criteria) + '.tex'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    with open(os.path.join(out_dir, fname), 'w') as f:
        s = "\\subsection{{{:s} data {:s} grouped by {:s}}}\n\n".format(
            args.qid, ' '.join(data_dirs), ' \& '.join(criteria)
        )
        s += '\\begin{{comment}}\n{:s}\n\\end{{comment}}\n'.format(str(res))
        table_rows = ''
        for index, row in res.iterrows():
            index = index if isinstance(index, tuple) else (index, )
            data = list(index)
            for i, v in enumerate(list(row.values)):
                if i <= 2:
                    if v > 0.5:
                        data.append('\\textbf{{{:.3f}}}'.format(v))
                    else:
                        data.append('{:.3f}'.format(v))
                elif i == 4:
                    if v < 0.05:
                        data.append('\\textbf{{{:.3f}}}'.format(v))
                    else:
                        data.append('{:.3f}'.format(v))
                else:
                    data.append(str(v))
            table_rows += ' & '.join(data)
            table_rows += '\\\\\n\t\t'
        s += TEX_TABLE.format('|'.join(['c']*(len(criteria) + 5)),
            ' & '.join(criteria), table_rows, ' '.join(criteria))
        s += TEX_FIGURE.format(args.qid, '_'.join(data_dirs), fig_name,
                               ' \& '.join(criteria))
        f.write(s)
        f.flush()
        f.close()