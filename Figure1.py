"""Reproduce a figure from the generalized ER experiment."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
Z = 1.959963984540054
UNIFORM_NS = (2048, 4096, 8192, 16384)

def set_style():
    use_tex = bool(shutil.which("latex") and shutil.which("dvipng"))
    plt.rcParams.update({"text.usetex": use_tex, "font.family": "serif",
                         "mathtext.fontset": "stix",
                         "axes.labelsize": 17, "axes.titlesize": 18,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 13.5})
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage[mathscr]{eucal}"

def homogeneous(uniform, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.45))
    delta = uniform["uniform_delta"].mean(axis=1)
    sd = uniform["uniform_delta"].std(axis=1, ddof=1)
    ax = axes[0]
    ax.errorbar(UNIFORM_NS, delta, yerr=sd, fmt="o-", color="#344e61",
                lw=1.7, ms=5.5, capsize=3.5, elinewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(UNIFORM_NS, [rf"$2^{{{int(np.log2(n))}}}$" for n in UNIFORM_NS])
    ax.set(xlabel=r"$n$", ylabel=r"$\Delta_n$")
    ax.set_ylim(min(delta-sd)*.92, max(delta+sd)*1.08)
    q_factor = float(uniform["qs"][0] * np.sqrt(UNIFORM_NS[0]))
    yticks = ([2.5, 3, 4, 4.5] if q_factor < .9
              else [1.5, 2, 2.5, 3, 3.5, 4])
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in yticks]))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax = axes[1]
    colors = plt.get_cmap("viridis")(np.linspace(.12, .88, len(UNIFORM_NS)))
    x_all, y_all = [], []
    for tier, n in enumerate(UNIFORM_NS):
        x = np.sqrt(uniform["oracle_r"][tier])
        y = np.abs(uniform["errors"][tier]).mean(axis=0)
        ax.scatter(x, y, color=colors[tier], s=12, alpha=.68,
                   edgecolors="none", rasterized=True,
                   label=rf"$2^{{{int(np.log2(n))}}}$")
        x_all.append(x)
        y_all.append(y)
    x, y = np.concatenate(x_all), np.concatenate(y_all)
    xpad=.06*np.ptp(x)
    ypad=.06*np.ptp(y)
    ax.set(xlim=(x.min()-xpad,x.max()+xpad),
           ylim=(y.min()-ypad,y.max()+ypad),
           xlabel=r"$\sqrt{\mathscr{R}_{\mathcal{G}_n^*}(i\leftrightarrow j)}$",
           ylabel=r"$|(\widehat u_i-\widehat u_j)-(u_i^*-u_j^*)|$")
    ax.legend(title=r"$n$", loc="upper left", frameon=False,
              fontsize=13, title_fontsize=13, handletextpad=.35,
              labelspacing=.25, markerscale=1.2)

    ax = axes[2]
    errors = uniform["errors"]
    plugin_r = uniform["plugin_r"]
    pair_coverage = (np.abs(errors) <= Z*np.sqrt(plugin_r)).mean(axis=1)
    mean_coverage = pair_coverage.mean(axis=1)
    pair_sd = pair_coverage.std(axis=1, ddof=1)
    ax.errorbar(UNIFORM_NS, mean_coverage, yerr=pair_sd, fmt="o-",
                color="#345a78", lw=1.7, ms=5.5,
                capsize=3.5, elinewidth=1.2)
    ax.axhline(.95, color=".50", ls=(0, (5, 3)), lw=1.2, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(UNIFORM_NS, [rf"$2^{{{int(np.log2(n))}}}$" for n in UNIFORM_NS])
    ax.set(xlabel=r"$n$", ylabel=r"95\% interval coverage",
           ylim=(.925, .975))
    ax.set_yticks([.93, .94, .95, .96, .97])

    for ax in axes:
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.96,
                        wspace=.43)
    dest = out_dir / "er_homogeneous_three_panel"
    fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(dest.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    set_style()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    from package.er_homogeneous import run
    data = output / "er_homogeneous.npz"
    run(data, args.reps, args.workers, 50, 20261026)
    with np.load(data) as values:
        homogeneous(values, output)
    print(output / "er_homogeneous_three_panel.png")
