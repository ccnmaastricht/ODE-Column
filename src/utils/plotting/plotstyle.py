import matplotlib as mpl

mpl.rcParams.update({
    # Font
    "font.family": "serif",
    "font.size": 8,

    # Axes
    "axes.labelsize": 6,
    "axes.titlesize": 6,

    # Tick labels
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,

    # Legend
    "legend.fontsize": 5,

    # Line widths
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.4,

    # Figure dpi
    "figure.dpi": 300,
    "savefig.dpi": 300,

    # PDF output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Tick lengths
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.size": 1.0,
    "ytick.minor.size": 1.0,

    # Tick widths
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.2,
    "ytick.minor.width": 0.2,

    # Distance between ticks and labels
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
})

panel_label = {
    "fontsize": mpl.rcParams["font.size"],
    "ha": "left",
    "va": "bottom",
}
