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
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle

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

class SegmentedLegendKey:
    def __init__(self, colors):
        self.colors = colors


class SegmentedLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        colors = orig_handle.colors
        cell_width = width / len(colors)
        artists = [Rectangle((xdescent + k*cell_width, ydescent),
                             cell_width, height, facecolor=color,
                             edgecolor="none", transform=trans)
                   for k, color in enumerate(colors)]
        artists.extend(Line2D([xdescent + k*cell_width]*2,
                              [ydescent, ydescent + height],
                              color=".32", linewidth=.35, transform=trans)
                       for k in range(1, len(colors)))
        artists.append(Rectangle((xdescent, ydescent), width, height,
                                 facecolor="none", edgecolor=".32",
                                 linewidth=.8, transform=trans))
        return artists


def heterogeneous(sweep, out_dir):
    ps = sweep["ps"]
    q_factor = float(sweep["q"] * np.sqrt(sweep["u"].size))
    base_expr = (r"n^{-1/2}" if np.isclose(q_factor, 1)
                 else rf"{q_factor:g}n^{{-1/2}}")
    hh = sweep["pair_group"] == "high-high"
    ll = sweep["pair_group"] == "low-low"
    colors = plt.get_cmap("viridis")(np.linspace(.12,.88,len(ps)))
    fig, axes = plt.subplots(1,2,figsize=(13.2,4.7),
                            gridspec_kw={"width_ratios":[1.1,1]})
    ax = axes[0]
    for t, p in enumerate(ps):
        x = np.sqrt(sweep["oracle_r"][t])
        y = np.abs(sweep["errors"][t]).mean(axis=0)
        ax.scatter(x[ll],y[ll],color=".62",s=11,alpha=.13,
                   edgecolors="none",rasterized=True,zorder=1)
        ax.scatter(x[hh],y[hh],color=colors[t],s=16,alpha=.55,
                   edgecolors="none",rasterized=True,zorder=2)
    ax.set(xscale="log",yscale="log",xlim=(.034,.52),ylim=(.025,.49),
           xlabel=r"$\sqrt{\mathscr{R}_{\mathcal{G}_n^*}(i\leftrightarrow j)}$",
           ylabel=r"$|(\widehat u_i-\widehat u_j)-(u_i^*-u_j^*)|$")
    ax.set_xticks([.05, .1, .2, .4],
                  [r"$5\times10^{-2}$", r"$10^{-1}$",
                   r"$2\times10^{-1}$", r"$4\times10^{-1}$"])
    ax.set_yticks([.03, .1, .3],
                  [r"$3\times10^{-2}$", r"$10^{-1}$", r"$3\times10^{-1}$"])
    handles = [Line2D([],[],marker="o",ls="none",color=colors[t],
                      label=rf"$p_{{\mathcal{{V}}_0}}={p:g}$" if t
                      else rf"$p_{{\mathcal{{V}}_0}}={base_expr}$")
               for t,p in enumerate(ps)]
    handles.append(Line2D([],[],marker="o",ls="none",color=".62",
                          label=r"pairs outside $\mathcal{V}_0$"))
    ax.legend(handles=handles,frameon=False,ncol=2,loc="upper left",
              fontsize=12,handletextpad=.2,columnspacing=.6)

    ax = axes[1]
    covered = (np.abs(sweep["errors"])
               <=Z*np.sqrt(sweep["plugin_r"])).mean(axis=1)
    tier_positions = np.arange(1, len(ps)+1, dtype=float)
    tier_positions[1:] += .65
    for t,p in enumerate(ps):
        for offset,mask,color in ((-.17,hh,colors[t]),(.17,ll,".76")):
            ax.boxplot(covered[t,mask],positions=[tier_positions[t]+offset],widths=.28,
                       patch_artist=True,showfliers=False,whis=(5,95),
                       showmeans=True,
                       boxprops={"facecolor":color,"edgecolor":".32",
                                 "linewidth":.8},
                       whiskerprops={"color":".32","linewidth":.9},
                       capprops={"color":".32","linewidth":.9},
                       medianprops={"color":"white","linewidth":1.15},
                       meanprops={"marker":"o","markerfacecolor":".13",
                                  "markeredgecolor":".13","markersize":4})
    ax.axhline(.95,color=".5",ls=(0,(5,3)),lw=1.2,zorder=0)
    ax.set(xlim=(.45,tier_positions[-1]+.55),ylim=(.79,.985),
           xlabel=r"$p_{\mathcal{V}_0}$",
           ylabel=r"95\% interval coverage")
    ax.set_xticks(tier_positions,
                  [rf"${base_expr}$"]+[rf"${p:g}$" for p in ps[1:]])
    ax.legend(handles=[SegmentedLegendKey(colors),
                       Patch(facecolor=".76",edgecolor=".32",
                             label=r"pairs outside $\mathcal{V}_0$")],
              labels=[r"pairs within $\mathcal{V}_0$",
                      r"pairs outside $\mathcal{V}_0$"],
              handler_map={SegmentedLegendKey: SegmentedLegendHandler()},
              frameon=False,loc="lower left",fontsize=12)
    for ax in axes:
        ax.grid(alpha=.13)
        ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(left=.08,right=.985,bottom=.21,top=.96,wspace=.38)
    dest = out_dir / "er_heterogeneous_allpairs"
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
    from package.er_heterogeneous import run
    data = output / "er_heterogeneous.npz"
    run(data, args.reps, args.workers, 50, 20261007)
    with np.load(data) as values:
        heterogeneous(values, output)
    print(output / "er_heterogeneous_allpairs.png")
