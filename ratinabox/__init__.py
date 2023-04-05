verbose = False  # verbosity of ratinabox, recommend False unless debugging

# SOME PLOT FORMATTING STUFF
# These two variables are bools but default to None so they trigger a warning recommending them to be set
autosave_plots = "undefined"  # whether to save figures automatically
figure_directory = "undefined"  # where to save figures, must be a director i.e. end in "/", for example "../figures/"

_save_plot_warnings_on = True  # whether to warn that autosave is turned off
_stylize_plot_warnings_on = (
    True  # whether to warn that rcParams haven't been set to make plots look good
)
_stylized_plots = False  # whether rcParams have been set to make plots look good
from .Environment import *
from .Agent import *
from .Neurons import *

from . import contribs


def stylize_plots():
    """Stylises plots to look like they do on the repo/paper by setting a bunch of matplotlib rcParams.
    This will effect other plots you may make (we think they look good).
    """
    from matplotlib import rcParams, rc
    from cycler import cycler

    darkgrey = [0.3, 0.3, 0.3, 1]
    # FONT
    # for now we leave the font formatting out since it throws warnings on some systems and doesn't really add much. TODO: fix this
    # matplotlib.rcParams['pdf.fonttype'] = 42 #this is super weird, see http://phyletica.org/matplotlib-fonts/
    # matplotlib.rcParams['pdf.fonttype'] = 3
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = 'Helvetica'
    rcParams["text.color"] = darkgrey
    rcParams["axes.labelcolor"] = darkgrey
    rcParams["xtick.color"] = darkgrey
    rcParams["ytick.color"] = darkgrey
    # FIGURE
    rcParams["figure.dpi"] = 150
    rcParams["figure.figsize"] = [1, 1]  # 2 x 2 inches
    rcParams["figure.titlesize"] = "medium"
    # AXES
    rcParams["axes.labelsize"] = 8
    rcParams["axes.labelpad"] = 3
    rcParams["axes.titlepad"] = 3
    rcParams["axes.titlesize"] = 8
    rcParams["axes.xmargin"] = 0
    rcParams["axes.ymargin"] = 0
    rcParams["axes.facecolor"] = [1, 1, 1, 0]
    rcParams["axes.edgecolor"] = darkgrey
    rcParams["axes.linewidth"] = 1
    # TICKS
    rcParams["xtick.major.width"] = 1
    rcParams["xtick.color"] = darkgrey
    rcParams["ytick.major.width"] = 1
    rcParams["ytick.color"] = darkgrey
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["xtick.major.pad"] = 2
    rcParams["xtick.minor.pad"] = 2
    rcParams["ytick.major.pad"] = 2
    rcParams["ytick.minor.pad"] = 2
    # GRIDS
    rcParams["grid.linewidth"] = 0.1
    # LEGEND
    rcParams["legend.fontsize"] = 6
    rcParams["legend.facecolor"] = [1, 1, 1, 0.3]
    rcParams["legend.edgecolor"] = darkgrey
    # LINES
    rcParams["lines.linewidth"] = 1
    rcParams["lines.markersize"] = 1
    rcParams["lines.markeredgewidth"] = 0.0
    # IMSHOWS
    rcParams["image.cmap"] = "inferno"
    # BOXPLOTS
    rcParams["boxplot.flierprops.linewidth"] = 1
    rcParams["boxplot.meanprops.linewidth"] = 1
    rcParams["boxplot.medianprops.linewidth"] = 1
    rcParams["boxplot.boxprops.linewidth"] = 1
    rcParams["boxplot.whiskerprops.linewidth"] = 1
    rcParams["boxplot.capprops.linewidth"] = 1
    # COLORSCHEME
    rcParams["axes.prop_cycle"] = cycler(
        "color",
        [
            "#7b699a",
            "#37738f",
            "#2eb37f",
            "#bed539",
            "#523577",
            "#e97670",
            "#f6d444",
            "#9a539b",
        ],
    )

    ratinabox._stylized_plots = True


# print(
#     "RatInABox successfully imported.\n    (optional) To automatically format and save RatInABox figures \n           • set  `ratinabox.autosave_plots = True`\n           • set  `ratinabox.figure_directory = 'path/to/figures/'`\n           • call `ratinabox.stylize_plots()`",end="\n\n"
# )
