from summarizer import get_probabilities
from hypothesis_testing import *
from plotter import *
from constants import TEX_TABLE, TEX_FIGURE
from utils import get_parser, aggregate_response

if __name__ == "__main__":

    args = get_parser().parse_args()

    data_dir = 'data/processed/'
    out_dir = 'outputs'
    out_dir = os.path.join(out_dir, '_'.join(args.resp_dirs))

    criteria = args.criteria
    fnames = args.fnames
    fnames = [f + '_approved.csv' for f in fnames]

    response = aggregate_response(args.resp_dirs, fnames)

    if args.what == 'choice':
        out_dir = os.path.join(out_dir, args.qid)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        print(criteria)
        res = get_probabilities(response, criteria, xy_qs=args.qid,
                                xz_qid=args.xz_qid)
        # Hypothesis Tests
        stats = run_choice_test(response, args.qid, args.criteria)
        res['pvalue'] = stats
        print(res)
        # Plot
        plot_distribution(response, criteria, by=args.qid, stats=res)
        fname = '_'.join([args.qid] + criteria) + '.pdf'
        plt.savefig(os.path.join(out_dir, fname), format='pdf')

        # Generate LaTeX file
        fig_name = fname
        fname = '_'.join([args.qid] + criteria) + '.tex'
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        with open(os.path.join(out_dir, fname), 'w') as f:
            s = "\\subsubsection{{{:s} data {:s} grouped by {:s}}}\n\n".format(
                args.qid, ' '.join(args.resp_dirs), ' \& '.join(criteria)
            )
            s += '\\begin{{comment}}\n{:s}\n\\end{{comment}}\n'.format(str(res))
            table_rows = ''
            for index, row in res.iterrows():
                index = index if isinstance(index, tuple) else (index, )
                data = [str(i) for i in index]
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
            s += TEX_FIGURE.format(args.qid, '_'.join(args.resp_dirs), fig_name,
                                   ' \& '.join(criteria))
            f.write(s)
            f.flush()
            f.close()
    elif args.what == 'cd':
        run_cd_test(response, args.criteria)
    elif args.what == 'model_fair':
        res = run_model_property_test(response, 'fair', args.criteria)
        res = pd.DataFrame(res, columns=['pvalue'])
        plot_model_property(response, 'fair', args.criteria, res)
        fname = '_'.join(['model_fair'] + criteria) + '.pdf'
        plt.savefig(os.path.join(out_dir, fname), format='pdf')
    elif args.what == 'model_bias':
        res = run_model_property_test(response, 'bias', args.criteria)
        res = pd.DataFrame(res, columns=['pvalue'])
        plot_model_property(response, 'bias', args.criteria, res)
        fname = '_'.join(['model_bias'] + criteria) + '.pdf'
        plt.savefig(os.path.join(out_dir, fname), format='pdf')
