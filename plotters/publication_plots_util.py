import matplotlib


def set_rcparams(**kwargs):
    fontsize = kwargs.get('fontsize', 10)
    params = {
        'pdf.fonttype': 42,
        "font.family": "serif",
        "font.serif": "Linux Libertine",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": 'medium',
        "xtick.labelsize": 'x-small',
        "ytick.labelsize": 'x-small',
        "mathtext.fontset": 'cm',
        "mathtext.default": 'bf',
        # "figure.figsize": set_size(width, fraction)
        "text.usetex": True,
        'text.latex.preamble': r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
            \usepackage{amsmath}
            \usepackage{dsfont}
        """
    }
    if kwargs.get('titlepad') is not None:
        params["axes.titlepad"] = kwargs.get('titlepad')
    if kwargs.get('labelpad') is not None:
        params["axes.labelpad"] = kwargs.get('labelpad')
    if kwargs.get('markersize') is not None:
        params["lines.markersize"] = kwargs.get('markersize')
    matplotlib.rcParams.update(params)
    # print(matplotlib.rcParams.get('font.size'))


# Source: https://tobiasraabe.github.io/blog/matplotlib-for-publications.html
def set_size(width, fraction=1, aspect_ratio='golden'):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if aspect_ratio == 'golden':
        aspect_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect_ratio

    return fig_width_in, fig_height_in
