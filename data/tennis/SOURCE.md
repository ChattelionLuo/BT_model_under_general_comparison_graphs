# ATP data provenance

The processed comparison graph in this directory was derived from ATP match records compiled by Jeff Sackmann / Tennis Abstract (historical repository: `JeffSackmann/tennis_atp`; a [public archive](https://github.com/Aneeshers/tennis-sackmann-archive) preserves the attribution and source). It covers 1979–2024. Players were identified by player ID, then iteratively restricted to the largest strongly connected component while removing players with fewer than 10 wins or 10 losses among retained matches. The resulting graph contains 1,076 players, 63,314 compared pairs, and 124,005 matches.

`edges.csv` records the observed wins and losses on each pair. `players.csv` records the centered BT MLE fitted to those outcomes. `graph_fit.npz` is the same graph in array form. `positions.npz` fixes the display coordinates. `named_pairs.csv` records the nine named pairs used in the figure. The Fisher Laplacian pseudoinverse and all Monte Carlo outcomes are recomputed by the scripts.

The underlying ATP data are attributed to Jeff Sackmann and distributed for noncommercial research under [Creative Commons Attribution–NonCommercial–ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please retain that attribution and license when redistributing the derived tennis data.
