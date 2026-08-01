"""
Matplotlib plotting utilities with config-driven styling.
"""

import matplotlib.pyplot as plt
from functools import partial


def apply_plot_style(ax, config):
    """Apply matplotlib styling from config to an axes object."""
    style = config["plotting"]["style"]

    [ax.spines[name].set_visible(visible) for name, visible in style["spines"].items()]

    ax.grid(style["grid"], alpha=0.3 * style["grid"])

    return ax


def _normalize_axes(axes, nrows, ncols):
    """Normalize axes to list format."""
    axes_map = {
        (True, True): lambda ax: [ax],
        (True, False): lambda ax: ax.flatten(),
        (False, True): lambda ax: ax.flatten(),
        (False, False): lambda ax: ax.flatten(),
    }

    key = (nrows == 1, ncols == 1)
    return axes_map[key](axes)


def setup_figure(config, nrows=1, ncols=1):
    """Create a figure with styling from config."""
    style = config["plotting"]["style"]
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(style["figure"]["figsize"]))

    axes_list = _normalize_axes(axes, nrows, ncols)
    [apply_plot_style(ax, config) for ax in axes_list]

    return (fig, axes_list[0]) if len(axes_list) == 1 else (fig, axes_list)


def apply_legend(ax, config, **kwargs):
    """Apply legend styling from config."""
    legend_style = config["plotting"]["style"]["legend"]
    ax.legend(frameon=legend_style["frameon"], loc=legend_style["loc"], **kwargs)


def save_plot(fig, output_path, config):
    """Save plot with settings from config."""
    style = config["plotting"]["style"]
    fig.savefig(
        output_path,
        dpi=style["figure"]["dpi"],
        bbox_inches="tight",
        format=config["output"]["plot_format"],
    )
