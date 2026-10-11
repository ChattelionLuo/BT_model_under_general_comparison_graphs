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
from scipy.special import ndtr
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from package.bipartite import case, coefficients, folded_mean
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

def plot_crossover(frame, name, save_pdf=True):
    fixed = frame[frame.design=="fixed_n"].sort_values("L")
    collapse = frame[frame.design=="collapse"]
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.6))
    ax=axes[0]
    ax.plot(fixed.L,fixed.mean_abs,"o-",color="#a22635",lw=1.8,
            label="MLE")
    ax.plot(fixed.L,fixed.pred_mean_abs,":",color="#222222",lw=1.8,
            label=r"$m_{n,L}$")
    ax.plot(fixed.L,fixed.pred_bias,"--",color="#d18639",lw=1.6,
            label=BIAS_LABEL)
    ax.plot(fixed.L,fixed.pred_noise_mae,"--",color="#253f66",lw=1.6,
            label=NOISE_LABEL)
    Lc=fixed.Lc_abs.iloc[0]
    ax.axvline(Lc,color="#777777",ls="-.",lw=1)
    ax.text(Lc*1.08,.035,rf"$L_c\approx{Lc:.0f}$",
            transform=ax.get_xaxis_transform(),fontsize=11.5,
            color=".28",va="bottom")
    ax.set_xlabel("Comparisons per edge $L$")
    ax.set_ylabel(PAIR_ERROR)
    handles,labels=ax.get_legend_handles_labels()
    order=(0,1,3,2)
    ax.legend([handles[i] for i in order],[labels[i] for i in order],
              frameon=False,loc="upper right")
    ax=axes[1]
    grid=np.geomspace(.08,12,200)
    z=np.sqrt(2/np.pi)/np.sqrt(grid)
    theory=(np.sqrt(2/np.pi)*np.exp(-z*z/2)+z*(2*ndtr(z)-1))/np.sqrt(2/np.pi)
    ax.plot(grid,theory,"-",color="#222222",lw=1.8,label="Predicted crossover")
    for n,color in [(512,"#a22635"),(2048,"#5b568b"),(8192,"#253f66")]:
        data=collapse[collapse.n==n].sort_values("L")
        ax.plot(data.L/data.Lc_abs,data.mean_abs/data.pred_noise_mae,
                "o",color=color,ms=4,label=rf"$n={n}$")
    ax.axvline(1,color="#777777",ls="-.",lw=1)
    ax.axhline(1,color="#777777",ls=":",lw=.8)
    ax.text(.07,.10,
            r"$\frac{1}{\lambda_2(\boldsymbol{L}_{\mathcal{G}_n})}$ dominates",
            transform=ax.transAxes,color="black",fontsize=10.5)
    ax.text(.56,.27,
            r"$\sqrt{\mathscr{R}_{\mathcal{G}_n}(1\leftrightarrow2)}$ dominates",
            transform=ax.transAxes,color="black",fontsize=10.5)
    ax.set_xlabel(r"$L/L_c$")
    ax.set_ylabel(r"$\Delta_{12}/"
                  r"[c_1\sqrt{\mathscr{R}_{\mathcal{G}_n}(1\leftrightarrow2)}]$")
    ax.legend(frameon=False,loc="upper right")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.16)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    fig.savefig(name.with_suffix(".png"),dpi=300,bbox_inches="tight")
    if save_pdf:
        fig.savefig(name.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    set_style()
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    rows = []
    # Vary L at n=2048, then compare the normalized crossover at other n.
    b, s = coefficients(2048)
    lc = (b / (s * np.sqrt(2 / np.pi)))**2 * 2046
    for ratio in [.09, .13, .20, .30, .44, .66, 1., 1.5, 2.25, 3.4, 5., 7.5]:
        L = int(np.ceil(ratio * lc))
        case(rows, "fixed_n", 2048, L, args.reps, 100003 + L * 23)
    for n in [512, 2048, 8192]:
        b, s = coefficients(n)
        lc = (b / (s * np.sqrt(2 / np.pi)))**2 * (n - 2)
        for index, ratio in enumerate([.125, .25, .5, 1., 2., 4., 8.]):
            L = int(np.ceil(ratio * lc))
            case(rows, "collapse", n, L, args.reps,
                 200003 + n * 101 + index * 10007)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "bipartite_crossover.csv", index=False)
    plot_crossover(frame, output / "two_item_abs_crossover_full")
    print(output / "two_item_abs_crossover_full.png")
