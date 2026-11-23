from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# All sizes are in inches or points as they appear in the paper, times SCALE. Each figure
# is included at 1/SCALE of its natural width, so panels and type match across figures.
SCALE = 2.0
AX_W = 2.33
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.87, 0.24, 0.60, 0.53
GAP_W, GAP_H = 0.63, 0.55
LEGEND_PAD = 0.16
SUPTITLE_TOP = 0.05
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 7.5, 7.0, 6.0, 5.8
LW, MS, MS_STAR, MEW, ELW = (SCALE * x for x in (1.5, 5.0, 9.0, 0.6, 0.7))

METHODS = {
    "at": ("#F6AE2D", "o", "additive thresholding (AT)"),
    "at_calibrated": ("#F26419", "s", "AT + confidence calibration"),
    "oe": ("#33658A", "D", "optimally efficient"),
    "oe_uncapped": ("#7EB6D9", "^", "optimally efficient (no cap)"),
}
MAIN = ["at", "at_calibrated", "oe"]
UNCAPPED = MAIN + ["oe_uncapped"]

ORACLE = dict(color="#808080", alpha=0.8, ls=(0, (4, 3)), lw=1.4, marker="*",
              ms=0.85 * MS_STAR, mec="white", mew=MEW)
RANDOM = dict(color="#9A9FA6", ls=(0, (4, 3)), lw=1.4)
VALID_C = "#8A9A5B"

SAFETY = [("sorrybench", "SORRY-Bench"), ("strongreject", "StrongREJECT")]
LABELING = [("bias", "Bias"), ("stance", "Stance"), ("imagenet", "ImageNet"),
            ("imagenetv2", "ImageNet-V2")]
ANNOTATION_TITLE = "Qualitative Interview Annotation"


def annotation(cutoff):
    return [(f"ucsb_{a}_cutoff{cutoff}", f"Annotator: {name}")
            for a, name in (("qwen14b", "Qwen2.5-14B"), ("gpt4omini", "GPT-4o-mini"))]


def _extent(draw, fig_w=40.0):
    fig = plt.figure(figsize=(fig_w, 4))
    bb = draw(fig).get_window_extent(fig.canvas.get_renderer())
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    plt.close(fig)
    return bb.width, bb.height


def axes_height():
    _, h = _extent(lambda f: f.text(0, 0, "Audit effort saved (%)", rotation=90,
                                    fontsize=FS_LABEL * SCALE), 4.0)
    return max(0.95 * AX_W, h + 0.08)


def handles(methods):
    h = [Line2D([0], [0], color=c, marker=m, lw=LW, ms=MS, mec="white", mew=1.5 * MEW, label=lab)
         for c, m, lab in (METHODS[k] for k in methods)]
    h.append(Line2D([0], [0], label="oracle threshold", **ORACLE))
    h.append(Line2D([0], [0], label="random audit", **RANDOM))
    return h


def legend_height(h, ncol, fontsize):
    return _extent(lambda f: f.legend(handles=h, ncol=ncol, frameon=False, fontsize=fontsize,
                                      loc="center"))[1]


def draw_panels(ax_sav, ax_err, df, methods, title, ylabels):
    alphas = np.sort(df.alpha.unique())
    xlo, xhi = alphas[0] - 0.02, alphas[-1] + 0.02
    xs = np.linspace(xlo, xhi, 200)
    ax_sav.plot(alphas, 100 * alphas, zorder=1, **RANDOM)
    ax_err.fill_between(xs, 0, 100 * xs, color=VALID_C, alpha=0.2, lw=0, zorder=0)

    for name in methods:
        color, marker, _ = METHODS[name]
        sub = df[df.algorithm == name].sort_values("alpha")
        for ax, col in ((ax_sav, "saved"), (ax_err, "error")):
            _, _, bars = ax.errorbar(sub.alpha, sub[col], yerr=sub[f"{col}_sd"], fmt="-" + marker,
                                     color=color, lw=LW, ms=MS, mec="white", mew=1.5 * MEW,
                                     elinewidth=ELW, zorder=3)
            for b in bars:
                b.set_alpha(0.38)
    o = df[df.algorithm == "oracle"].sort_values("alpha")
    ax_sav.plot(o.alpha, o.saved, zorder=2, **ORACLE)

    ax_sav.set_title(title, pad=8.0)
    if ylabels:
        ax_sav.set_ylabel("Audit effort saved (%)")
        ax_err.set_ylabel("Error rate (%)")
    for ax in (ax_sav, ax_err):
        ax.set_xlabel("α", labelpad=2.0)
        ax.set_xlim(xlo, xhi)
        ax.grid(True, alpha=0.22, lw=0.7 * SCALE)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5 * SCALE, width=1.0 * SCALE, color="#7a7a7a", pad=2.0)
    shown = df[df.algorithm.isin(methods + ["oracle"])].saved
    pad = max(0.12 * (shown.max() - shown.min()), 1.0)
    ax_sav.set_ylim(max(0.0, shown.min() - pad), min(100.0, shown.max() + pad))
    ax_err.set_ylim(bottom=0)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def figure(name, datasets, methods, legend_ncol=None, suptitle=None):
    n = len(datasets)
    ax_h = axes_height()
    h = handles(methods)
    band = legend_height(h, legend_ncol, FS_LEGEND * SCALE) + LEGEND_PAD if legend_ncol else 0.0
    fig_w = MARGIN_L + n * AX_W + (n - 1) * GAP_W + MARGIN_R
    fig_h = MARGIN_T + 2 * ax_h + GAP_H + MARGIN_B + band
    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)
    for j, (stream, title) in enumerate(datasets):
        df = pd.read_csv(RESULTS / f"{stream}.csv")
        draw_panels(axes[0][j], axes[1][j], df, methods, title, ylabels=(j == 0))
    fig.subplots_adjust(left=MARGIN_L / fig_w, right=1 - MARGIN_R / fig_w,
                        top=1 - (band + MARGIN_T) / fig_h, bottom=MARGIN_B / fig_h,
                        wspace=GAP_W / AX_W, hspace=GAP_H / ax_h)
    if legend_ncol:
        fig.legend(handles=h, loc="center", ncol=legend_ncol, frameon=False,
                   fontsize=FS_LEGEND * SCALE, bbox_to_anchor=(0.5, 1 - 0.5 * band / fig_h))
    if suptitle:
        fig.text(0.5, 1 - (band + SUPTITLE_TOP) / fig_h, suptitle, ha="center", va="top",
                 fontsize=FS_TITLE * SCALE)
    save(fig, name)


def legend_strip(name, methods, fontsize):
    """Legend shared by two side-by-side two-column figures."""
    h = handles(methods)
    width = 2 * (MARGIN_L + 2 * AX_W + GAP_W + MARGIN_R)
    fig = plt.figure(figsize=(width, legend_height(h, len(h), fontsize) + LEGEND_PAD))
    fig.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=fontsize)
    save(fig, name)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#3d3d3d",
        "axes.linewidth": 1.0 * SCALE,
        "axes.titlesize": FS_TITLE * SCALE,
        "axes.labelsize": FS_LABEL * SCALE,
        "xtick.labelsize": FS_TICK * SCALE,
        "ytick.labelsize": FS_TICK * SCALE,
        "figure.dpi": 150,
    })

    legend_strip("legend", MAIN, FS_LEGEND * SCALE)
    figure("figure1_safety", SAFETY, MAIN)
    figure("figure2_annotation", annotation(0.8), MAIN, suptitle=ANNOTATION_TITLE)
    figure("figure3_labeling", LABELING, MAIN, legend_ncol=5)

    legend_strip("legend_uncapped", UNCAPPED, 4.9 * SCALE)
    figure("figure4_safety_uncapped", SAFETY, UNCAPPED)
    figure("figure5_annotation_uncapped", annotation(0.8), UNCAPPED, suptitle=ANNOTATION_TITLE)
    figure("figure6_labeling_uncapped", LABELING, UNCAPPED, legend_ncol=3)
    for cutoff in (0.7, 0.6, 0.5):
        figure(f"figure7_annotation_cutoff{cutoff}", annotation(cutoff), UNCAPPED,
               suptitle=f"{ANNOTATION_TITLE}, cutoff {cutoff}")


if __name__ == "__main__":
    main()
