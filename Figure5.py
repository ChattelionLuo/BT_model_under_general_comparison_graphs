"""Reproduce a manuscript figure for the original ATP graph."""
from __future__ import annotations
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATA=ROOT/"data/tennis"
RESULTS=ROOT/"results"
POSITIONS=DATA/"positions.npz"

def set_style():
    use_tex=bool(shutil.which("latex") and shutil.which("dvipng"))
    plt.rcParams.update({'text.usetex':use_tex,'font.family':'serif',
                         'mathtext.fontset':'stix',
                         'axes.labelsize':19,'xtick.labelsize':15,
                         'ytick.labelsize':16,'legend.fontsize':14})
    if use_tex:
        plt.rcParams['text.latex.preamble']=r'\usepackage[mathscr]{eucal}'

def style(ax):
    ax.grid(color='#BFC7CC',alpha=.22,zorder=0)
    ax.spines[['top','right']].set_visible(False)

def save(fig,stem):
    if not plt.rcParams['text.usetex']:
        for label in fig.findobj(match=matplotlib.text.Text):
            label.set_text(label.get_text().replace(r'\mathscr R',r'\mathscr{R}')
                           .replace(r'\mathcal G',r'\mathcal{G}'))
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS/f'{stem}.png',dpi=300,bbox_inches='tight')
    fig.savefig(RESULTS/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)

def graph_geometry():
    from package.bt import fisher_laplacian, pseudoinverse
    with np.load(DATA/'graph_fit.npz') as graph:
        a, b, matches, u = (graph[key] for key in ('a', 'b', 'matches', 'u_star'))
    fisher = pseudoinverse(fisher_laplacian(len(u), a, b, matches, u))
    i,j=np.triu_indices(len(fisher),1)
    diagonal=np.diag(fisher)
    sqrt_r=np.sqrt(np.maximum(0,diagonal[i]+diagonal[j]-2*fisher[i,j]))
    return i,j,sqrt_r

def structure_figure(players, edges, sqrt_r):
    layout = np.load(POSITIONS)
    lookup = dict(zip(layout['players'].tolist(), layout['coordinates']))
    pos = np.stack([lookup[name] for name in players.name])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.75),
                            gridspec_kw={'width_ratios': [1, 1.11], 'wspace': .24})
    ax = axes[0]
    draw = np.random.default_rng(20261004).choice(len(edges), size=15000, replace=False)
    shown = edges.iloc[draw]
    segments = np.stack([pos[shown.a.to_numpy()], pos[shown.b.to_numpy()]], axis=1)
    ax.add_collection(LineCollection(segments, colors='#70849A', lw=.22,
                                     alpha=.055, rasterized=True, zorder=1))
    size = 3 + 1.5*np.log1p(players.matches.to_numpy())
    utility = players.u_star.to_numpy()
    order = np.argsort(utility)
    dots = ax.scatter(pos[order, 0], pos[order, 1],
                      c=utility[order], cmap='viridis',
                      vmin=-2.1, vmax=3.0, s=size[order], alpha=.88,
                      edgecolors='none', rasterized=True, zorder=2)
    labels = [
        ('Bjorn Borg', 'Borg', (-.76, .53), 'right'),
        ('Pete Sampras', 'Sampras', (-.63, -.54), 'right'),
        ('Roger Federer', 'Federer', (.46, .57), 'left'),
        ('Novak Djokovic', 'Djokovic', (.47, -.27), 'left'),
        ('Carlos Alcaraz', 'Alcaraz', (.33, -.61), 'left'),
    ]
    lookup_index = dict(zip(players.name, players['index'].astype(int)))
    for name, label, text_pos, align in labels:
        idx = lookup_index[name]
        ax.scatter(*pos[idx], s=62, facecolor='white', edgecolor='#242424',
                   linewidth=1, zorder=4)
        ax.scatter(*pos[idx], s=27,
                   color=dots.cmap(dots.norm(utility[idx])), zorder=5)
        ax.annotate(label, pos[idx], xytext=text_pos, textcoords='data',
                    ha=align, fontsize=12.0, color='#202020', zorder=6,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': .8},
                    arrowprops={'arrowstyle': '-', 'color': '.27', 'lw': .7})
    colorbar = fig.colorbar(dots, ax=ax, fraction=.043, pad=.035)
    colorbar.set_label(r'Utility $u_i^*$', fontsize=13, labelpad=8)
    colorbar.ax.yaxis.set_label_position('left')
    colorbar.ax.yaxis.set_ticks_position('right')
    colorbar.ax.tick_params(labelsize=11)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()

    ax = axes[1]
    bins = np.linspace(.09, .65, 110)
    mid = (bins[:-1] + bins[1:])/2
    density, _ = np.histogram(sqrt_r, bins=bins, density=True)
    density = gaussian_filter1d(density, sigma=1.5)
    ax.fill_between(mid, density, color='#3D6285', alpha=.15)
    ax.plot(mid, density, color='#3D6285', lw=2.2)
    ax.set_xlim(.09, .65)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\sqrt{\mathscr R_{\mathcal G_n^*}(i\leftrightarrow j)}$')
    ax.set_ylabel('Density')
    style(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'tennis_organic_structure')



if __name__=='__main__':
    set_style()
    players=pd.read_csv(DATA/'players.csv')
    edges=pd.read_csv(DATA/'edges.csv')
    _,_,sqrt_r=graph_geometry()
    structure_figure(players,edges,sqrt_r)
    print(RESULTS/'tennis_organic_structure.png')
