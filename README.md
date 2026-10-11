# Reproducibility file for "Uniform consistency of maximum likelihood estimation in the Bradley–Terry model under general graphs"

This folder contains the code for the six figures in the paper's numerical section. Each script generates its own simulation outcomes and writes a 300 dpi PNG and a PDF to `results/`. The processed ATP comparison graph and the tennis-network display coordinates are included. No simulated outcomes or finished figures are included.

## 1. Folder structure

```text
BT_graph_experiments/
├── Figure1.py              Homogeneous Erdős–Rényi experiment
├── Figure2.py              Locally heterogeneous Erdős–Rényi experiment
├── Figure3.py              Bipartite crossover at fixed n
├── Figure4.py              Bipartite rate sequences
├── Figure5.py              ATP network and resistance distribution
├── Figure6.py              ATP pairwise errors and interval coverage
├── run_all.py              Runs Figures 1–6 in order
├── package/                BT estimation, graph calculations, and generators
├── data/tennis/            Processed comparisons, fitted utilities, and layout
├── requirements.txt
└── README.md
```

The `results/` directory is created automatically. It holds the regenerated figures and numerical arrays or tables.

## 2. Installation

Python 3.11 or newer is required. From this directory, create an environment and install the listed dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Use the virtual environment's Python to run the commands below (`.venv/bin/python` on Linux/macOS or `.venv\Scripts\python.exe` on Windows). A LaTeX installation gives the figures the same typography as the manuscript; Matplotlib can also draw them without LaTeX.

## 3. Reproducing the figures

Run all figures with `python run_all.py`, or run `python Figure1.py` through `python Figure6.py` individually. The graph, utilities, comparison counts, and random seeds are fixed as specified in each experiment.

`Figure1.py`, `Figure2.py`, and `Figure6.py` support parallel workers, for example `python Figure1.py --workers 32` or `python run_all.py --workers 32`. On Linux the ER scripts use processes; on Windows they use threads. The remaining scripts run directly. As a timing reference, on server8 (256 logical AMD EPYC cores), the amortized wall time per replication across all design points was approximately 0.34 s for Figure 1, 0.11 s for Figure 2, 0.03 s for Figure 3, 0.02 s for Figure 4, and 0.21 s for Figure 6, using 96, 64, and 48 workers for Figures 1, 2, and 6, respectively. Figure 5 is deterministic and took about 10 s.

### Erdős–Rényi graphs

- **Figure 1:** Generates a graph with $p_{ij}^{(n)}=0.75n^{-1/2}$ for $n=2^{11},\ldots,2^{14}$ and centered utilities drawn from `Unif(-1.5, 1.5)`. It reports the uniform error, the mean pairwise error against $\sqrt{\mathscr R_{\mathcal G_n^*}(i\leftrightarrow j)}$ for 500 fixed pairs at each $n$, and the coverage of MLE plug-in 95% pairwise intervals.
- **Figure 2:** Fixes $n=2^{13}$ and raises the edge probability incident to a set of 20 vertices through six levels. It compares all 190 within-set pairs with 190 pairs outside the set, showing how pairwise error and plug-in interval coverage change with local graph density.

### Imbalanced complete bipartite graphs

- **Figure 3:** Uses $K_{2,n-2}$ with hub utilities $\pm\log 2$ and zero leaf utilities. At fixed $n=2048$, it varies the comparisons per edge $L$ to show the crossover between the resistance and spectral-gap contributions. It also compares the normalized crossover at three values of $n$.
- **Figure 4:** Uses the same design with $L_n$ chosen along three graph sequences. It checks the distinct error rates as $n$ grows.

### ATP tennis graph

- **Figure 5:** Draws the observed comparison network, colors vertices by their fitted utilities, and shows the distribution of square-root pairwise Fisher effective resistance. The saved `positions.npz` fixes the network layout.
- **Figure 6:** Keeps the observed match counts and uses the original-data BT MLE as the generating utility. It resamples outcomes, refits the joint MLE, and constructs 95% pairwise intervals using the MLE plug-in Fisher resistance. The left panel samples pairs from three resistance ranges and marks nine named pairs. The right panel shows their coverage against the coverage range of all 578,350 pairs.

The processed ATP data contain 1,076 players, 63,314 compared pairs, and 124,005 matches. See [`data/tennis/SOURCE.md`](data/tennis/SOURCE.md) for preprocessing, provenance, and attribution.
