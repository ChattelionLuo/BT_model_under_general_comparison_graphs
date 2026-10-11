"""Reproduce a manuscript figure for the original ATP graph."""
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
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
DATA=ROOT/"data/tennis"
RESULTS=ROOT/"results"
HIGHLIGHTS=(
    ('low','Federer--Nadal','Roger Federer--Rafael Nadal'),
    ('low','Sampras--Agassi','Pete Sampras--Andre Agassi'),
    ('low','Connors--Lendl','Jimmy Connors--Ivan Lendl'),
    ('middle','Borg--Rune','Bjorn Borg--Holger Rune'),
    ('middle','Federer--Fils','Roger Federer--Arthur Fils'),
    ('middle','Sinner--Shelton','Jannik Sinner--Ben Shelton'),
    ('high','Federer--Shang','Roger Federer--Juncheng Shang'),
    ('high','Alcaraz--Shang','Carlos Alcaraz--Juncheng Shang'),
    ('high','Sampras--Medjedovic','Pete Sampras--Hamad Medjedovic'),
)
COLORS={'low':'#345C80','middle':'#4F8E98','high':'#C5844D'}
LABELS={'low':r'Lowest 20\%','middle':r'Middle 40--60\%',
        'high':r'Highest 20\%'}
Z=1.959963984540054

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

def selected_pairs(i_all, j_all, sqrt_r, errors, all_pair_coverage, refresh=False):
    candidate = pd.read_csv(DATA / 'named_pairs.csv').set_index('pair')
    ordered_r = np.sort(sqrt_r)
    rows = []
    for regime, label, full_name in HIGHLIGHTS:
        entry = candidate.loc[full_name].copy()
        if refresh:
            pair_i,pair_j=int(entry.i),int(entry.j)
            entry['mean_abs_error']=float(np.abs(errors[:,pair_i]-errors[:,pair_j]).mean())
            k=pair_i*(2*errors.shape[1]-pair_i-1)//2+pair_j-pair_i-1
            entry['coverage']=float(all_pair_coverage[k])
        percentile = float(np.searchsorted(ordered_r, entry.sqrt_fisher_r,
                                           side='right') / len(ordered_r))
        assert ((regime == 'low' and percentile <= .2) or
                (regime == 'middle' and .4 <= percentile <= .6) or
                (regime == 'high' and percentile >= .8))
        rows.append({**entry.to_dict(), 'regime': regime, 'pair': label,
                     'fisher_r_percentile': percentile})
    highlight = pd.DataFrame(rows).sort_values('sqrt_fisher_r').reset_index(drop=True)
    highlight.to_csv(RESULTS / 'highlighted_pairs.csv', index=False)

    ranks = np.searchsorted(np.sort(sqrt_r), sqrt_r, side='right')/len(sqrt_r)
    rng = np.random.default_rng(20261004)
    choices = []
    regimes = []
    for regime, lo, hi in [('low', 0, .2), ('middle', .4, .6), ('high', .8, 1)]:
        eligible = np.flatnonzero((ranks > lo) & (ranks <= hi))
        draw = rng.choice(eligible, size=30, replace=False)
        choices.extend(draw.tolist())
        regimes.extend([regime]*len(draw))
    pi = i_all[choices]
    pj = j_all[choices]
    cloud = pd.DataFrame({
        'i': pi, 'j': pj, 'regime': regimes,
        'sqrt_fisher_r': sqrt_r[choices],
        'mean_abs_error': np.abs(errors[:, pi] - errors[:, pj]).mean(axis=0),
    })
    cloud.to_csv(RESULTS / 'resistance_stratified_pairs.csv', index=False)
    return highlight, cloud


def inference_figure(cloud, highlight, all_pair_coverage):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5),
                            gridspec_kw={'width_ratios': [1.08, 1], 'wspace': .65})
    ax, forest = axes
    for regime in ('low', 'middle', 'high'):
        part = cloud[cloud.regime == regime]
        ax.scatter(part.sqrt_fisher_r, part.mean_abs_error, s=40,
                   color=COLORS[regime], alpha=.8, edgecolor='white',
                   linewidth=.4, label=LABELS[regime], zorder=2)
    ax.scatter(highlight.sqrt_fisher_r, highlight.mean_abs_error,
               s=40, c=[COLORS[regime] for regime in highlight.regime],
               alpha=.8, edgecolors='black', linewidth=1.2, zorder=4)
    ax.scatter([], [], s=40, facecolors='none', edgecolors='black',
               linewidth=1.2, label='Highlighted pairs')
    annotations = [
        ('Connors--Lendl', (.205, .120), 'left'),
        ('Federer--Nadal', (.205, .080), 'left'),
        ('Sampras--Agassi', (.205, .040), 'left'),
        ('Borg--Rune', (.245, .350), 'right'),
        ('Sinner--Shelton', (.245, .310), 'right'),
        ('Federer--Fils', (.245, .270), 'right'),
        ('Alcaraz--Shang', (.475, .465), 'left'),
        ('Sampras--Medjedovic', (.475, .420), 'left'),
        ('Federer--Shang', (.475, .375), 'left'),
    ]
    for label, text_pos, alignment in annotations:
        row = highlight[highlight.pair == label].iloc[0]
        ax.plot([row.sqrt_fisher_r, text_pos[0]],
                [row.mean_abs_error, text_pos[1]],
                color='#343B42', lw=.65, zorder=4)
        ax.text(*text_pos, label, ha=alignment, va='center',
                fontsize=10.5, color='#202D38',
                bbox={'facecolor': 'white', 'edgecolor': 'none',
                      'alpha': .86, 'pad': .4}, zorder=5)
    ax.set_xlim(.080, .665)
    ax.set_ylim(0, .69)
    ax.set_xlabel(r'$\sqrt{\mathscr R_{\mathcal G_n^*}(i\leftrightarrow j)}$')
    ax.set_ylabel(r'$|(\widehat u_i-\widehat u_j)-(u_i^*-u_j^*)|$')
    ax.legend(loc='upper left', frameon=False, handletextpad=.6,
              labelspacing=.35)

    ordered = highlight.sort_values('sqrt_fisher_r', ascending=False).reset_index(drop=True)
    coverage_lo, coverage_hi = np.quantile(all_pair_coverage, [.025, .975])
    forest.axvspan(coverage_lo, coverage_hi, color='#E4E7E8', zorder=0)
    forest.axvline(.95, color='#666666', lw=1.2, ls=(0, (4, 3)), zorder=1)
    for y, row in enumerate(ordered.itertuples()):
        forest.scatter(row.coverage, y, color=COLORS[row.regime],
                       s=66, edgecolors='none', zorder=3)
    forest.set_yticks(np.arange(len(ordered)))
    forest.set_yticklabels([f'{row.pair} ({row.direct_matches})'
                            for row in ordered.itertuples()])
    forest.set_xlim(.918, .98)
    forest.set_ylim(-.65, len(ordered)-.35)
    forest.set_xlabel(r'$95\%$ interval coverage')
    for target in axes:
        style(target)
    forest.grid(axis='y', color='#BFC7CC', alpha=.28)
    save(fig, 'tennis_resistance_regimes')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    set_style()
    from package.tennis_recompute import run
    errors, all_pair_coverage = run(DATA, RESULTS, args.reps, args.workers)
    i, j, sqrt_r = graph_geometry()
    highlight, cloud = selected_pairs(i, j, sqrt_r, errors,
                                      all_pair_coverage, refresh=True)
    inference_figure(cloud, highlight, all_pair_coverage)
    print(RESULTS / 'tennis_resistance_regimes.png')
