from src.utils.plotting.plotstyle import panel_label



def finish_plot(fig, filename):
    fig.savefig(
        filename,
        bbox_inches="tight",
        transparent=True)

def add_sub_fig_labels(mapping):
    for label, sub_fig in mapping.items():
        sub_fig.text(
            -0.15,
            1.05,
            label,
            transform=sub_fig.transAxes,
            **panel_label)
