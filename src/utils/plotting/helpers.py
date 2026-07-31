from src.utils.plotting.plotstyle import panel_label


def finish_plot(fig, filename):
    """
    Save figure artifact to file with tight bounding box cropping and transparent
    background.

    Args:
        fig (matplotlib.figure.Figure): Target Matplotlib figure instance.
        filename (str | Path): Output file path for saving figure.
    """
    fig.savefig(
        filename,
        bbox_inches="tight",
        transparent=True)

def add_sub_fig_labels(mapping):
    """
    Annotate subplot axes with panel labels (e.g. 'A', 'B') using predefined plot style
    properties.

    Args:
        mapping (dict[str, matplotlib.axes.Axes]): Dictionary mapping label strings to
            target Matplotlib subplot Axes.
    """
    for label, sub_fig in mapping.items():
        sub_fig.text(
            -0.15,
            1.05,
            label,
            transform=sub_fig.transAxes,
            **panel_label)
