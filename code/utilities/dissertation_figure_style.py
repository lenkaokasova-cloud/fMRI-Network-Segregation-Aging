from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FIG_DPI = 300

TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 10

SUMMARY_AXIS_LABEL_SIZE = 12.5
SUMMARY_TICK_LABEL_SIZE = 11.5
SUMMARY_TITLE_SIZE = 15
SUMMARY_ANNOTATION_SIZE = 10.8

FONT_FAMILY = "serif"
FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "STIX Two Text",
    "Liberation Serif",
    "DejaVu Serif",
]

TEXT_COLOR = "#253547"
AXIS_TEXT_COLOR = "#18222D"
SUBTLE_TEXT_COLOR = "#5A6573"
GRID_COLOR = "#D7DCE4"
REFERENCE_LINE_COLOR = "#253547"
MUTED_REFERENCE_LINE_COLOR = "#6A7D92"
REFERENCE_RANGE_COLOR = "#6B8EB5"

YOUNG_COLOR = "#5E7FA6"
OLDER_COLOR = "#C07A5B"
POSITIVE_COLOR = YOUNG_COLOR
NEGATIVE_COLOR = OLDER_COLOR

OVERALL_COLOR = "#355D4C"
NETWORK_COLOR = "#7A9C8B"
HIGHLIGHT_COLOR = "#5E7B5F"
NEUTRAL_COLOR = "#C5CBD4"
BURDEN_COLOR = "#B55223"

HEATMAP_CMAP = "coolwarm"


def apply_dissertation_rcparams(
    *,
    title_size: float = TITLE_SIZE,
    label_size: float = LABEL_SIZE,
    tick_size: float = TICK_SIZE,
    legend_size: float = LEGEND_SIZE,
) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "font.size": tick_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "figure.titlesize": title_size,
            "axes.titleweight": "semibold",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.linewidth": 0.9,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "mathtext.fontset": "stix",
        }
    )


def save_figure(fig: plt.Figure, outpath: Path, *, dpi: int = FIG_DPI) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", length=4.2, width=0.8, color=TEXT_COLOR)


def darken_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.tick_params(axis="both", labelcolor=AXIS_TEXT_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(AXIS_TEXT_COLOR)


def emphasize_axis(
    ax: plt.Axes,
    *,
    axis_label_size: float = SUMMARY_AXIS_LABEL_SIZE,
    tick_label_size: float = SUMMARY_TICK_LABEL_SIZE,
    title_size: float = SUMMARY_TITLE_SIZE,
    bold_title: bool = True,
) -> None:
    darken_axis_text(ax)
    ax.xaxis.label.set_fontsize(axis_label_size)
    ax.yaxis.label.set_fontsize(axis_label_size)
    ax.title.set_fontsize(title_size)
    if bold_title:
        ax.title.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_label_size)


def rotate_xticklabels(ax: plt.Axes, xrotation: float = 0.0) -> None:
    ax.tick_params(axis="x", labelrotation=xrotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right" if xrotation else "center")


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xrotation: float = 0.0,
    grid_axis: str | None = None,
    grid_alpha: float = 0.55,
    grid_linewidth: float = 0.8,
    zero_line: str | None = None,
    zero_line_color: str = MUTED_REFERENCE_LINE_COLOR,
    zero_line_width: float = 1.2,
    zero_line_style: str = "-",
    zero_line_alpha: float = 0.92,
) -> None:
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    style_spines(ax)
    darken_axis_text(ax)
    rotate_xticklabels(ax, xrotation=xrotation)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, alpha=grid_alpha, linewidth=grid_linewidth)
    if zero_line == "x":
        ax.axvline(
            0.0,
            color=zero_line_color,
            linewidth=zero_line_width,
            linestyle=zero_line_style,
            alpha=zero_line_alpha,
        )
    elif zero_line == "y":
        ax.axhline(
            0.0,
            color=zero_line_color,
            linewidth=zero_line_width,
            linestyle=zero_line_style,
            alpha=zero_line_alpha,
        )


def style_colorbar(
    colorbar,
    *,
    label_size: float = SUMMARY_AXIS_LABEL_SIZE,
    tick_size: float = SUMMARY_TICK_LABEL_SIZE,
    label_color: str = AXIS_TEXT_COLOR,
) -> None:
    colorbar.ax.tick_params(labelsize=tick_size, colors=label_color)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    colorbar.outline.set_linewidth(0.9)
    if colorbar.ax.yaxis.label:
        colorbar.ax.yaxis.label.set_color(label_color)
        colorbar.ax.yaxis.label.set_fontsize(label_size)
