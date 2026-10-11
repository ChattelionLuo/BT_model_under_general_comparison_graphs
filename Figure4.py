"""Reproduce a bipartite manuscript figure."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from package.bipartite import case
PAIR_ERROR = r"$\Delta_{12}$"
BIAS_LABEL = r"$c_2/\lambda_2(\boldsymbol{L}_{\mathcal{G}_n})$"
NOISE_LABEL = (r"$c_1" r"\sqrt{\mathscr{R}_{\mathcal{G}_n}(1\leftrightarrow2)}$")

def set_style():
    use_tex=bool(shutil.which("latex") and shutil.which("dvipng"))
    plt.rcParams.update({"text.usetex":use_tex,"font.family":"serif",
                         "mathtext.fontset":"stix",
                         "axes.labelsize":13,"xtick.labelsize":11,
                         "ytick.labelsize":11,"legend.fontsize":10.5,
                         "axes.titlesize":13})
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage[mathscr]{eucal}"

def plot_sequences(frame,name):
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.4))
    specs=[(.5,r"$L_n=\lceil n^{1/2}\rceil$"),
           (1.,r"$L_n=\lceil0.1n\rceil$"),
           (1.5,r"$L_n=\lceil n^{3/2}\rceil$")]
    for ax,(beta,label) in zip(axes,specs):
        data=frame[(frame.design=="sequence")&(frame.beta==beta)].sort_values("n")
        n=data.n.to_numpy()
        ax.plot(n,data.mean_abs,"o-",color="#a22635",lw=1.8,label="MLE")
        ax.plot(n,data.pred_bias,"--",color="#d18639",lw=1.6,label=BIAS_LABEL)
        ax.plot(n,data.pred_noise_mae,"--",color="#253f66",lw=1.6,
                label=NOISE_LABEL)
        ax.set_xscale("log",base=2); ax.set_yscale("log")
        ax.set_xticks(n,labels=[rf"$2^{{{int(np.log2(v))}}}$" for v in n])
        ax.set_xlabel("$n$"); ax.grid(alpha=.17)
        ax.spines[["top","right"]].set_visible(False)
        ax.text(.59,.91,label,transform=ax.transAxes,
                fontsize=11.5,color=".16",ha="left",va="center")
    axes[0].set_ylabel(PAIR_ERROR)
    axes[0].set_yticks([.003, .01, .03],
                       [r"$3\times10^{-3}$", r"$10^{-2}$", r"$3\times10^{-2}$"])
    handles,labels=axes[0].get_legend_handles_labels()
    order=(0,2,1)
    fig.legend([handles[i] for i in order],[labels[i] for i in order],
               loc="lower center",ncol=3,frameon=False,
               bbox_to_anchor=(.5,.055),fontsize=10.5)
    fig.tight_layout(rect=[0,.15,1,1],w_pad=1.5)
    fig.savefig(name.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(name.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    set_style()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    rows = []
    for beta, mult in [(.5, 1.), (1., .10), (1.5, 1.)]:
        for n in [256, 512, 1024, 2048, 4096, 8192]:
            L = int(np.ceil(mult * n**beta))
            case(rows, "sequence", n, L, args.reps,
                 300007 + int(beta * 100) * 1000003 + n * 31, beta)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "bipartite_sequences.csv", index=False)
    plot_sequences(frame, output / "two_item_abs_sequences_full")
    print(output / "two_item_abs_sequences_full.png")
