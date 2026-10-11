from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.special import expit

from .bt import fit_mle, strongly_connected_outcomes

Z = 1.959963984540054


def _coverage_one(error, a, b, matches, truth, pair_i, pair_j):
    n = len(truth)
    fitted = truth + error
    p = expit(fitted[a] - fitted[b])
    weight = matches * p * (1 - p)
    row, col = np.r_[a, b, a, b], np.r_[a, b, b, a]
    lap = coo_matrix((np.r_[weight, weight, -weight, -weight], (row, col)),
                     shape=(n, n)).tocsc()
    inverse = splu(lap[:-1, :-1]).solve(np.eye(n - 1))
    diagonal = np.r_[np.diag(inverse), 0.0]
    resistance = diagonal[pair_i] + diagonal[pair_j]
    both = pair_j < n - 1
    resistance[both] -= 2 * inverse[pair_i[both], pair_j[both]]
    if np.any(resistance <= 0):
        raise RuntimeError("Nonpositive plug-in resistance")
    contrast_error = error[pair_i] - error[pair_j]
    return np.abs(contrast_error) <= Z * np.sqrt(resistance)


def run(data_dir: Path, out_dir: Path, reps: int, workers: int):
    with np.load(data_dir / "graph_fit.npz") as graph:
        a, b, matches, truth = (graph[key] for key in
                                ("a", "b", "matches", "u_star"))
    n = len(truth)
    probability = expit(truth[a] - truth[b])
    rng = np.random.default_rng(20260929)
    errors = np.empty((reps, n))
    accepted = attempted = 0
    while accepted < reps:
        attempted += 1
        if attempted > 20 * reps:
            raise RuntimeError("Too few finite MLEs")
        wins_a = rng.binomial(matches.astype(np.int64), probability).astype(float)
        wins_b = matches - wins_a
        if not strongly_connected_outcomes(n, a, b, wins_a, wins_b):  # finite MLE
            continue
        errors[accepted] = fit_mle(a, b, wins_a, wins_b, n, start=truth) - truth
        accepted += 1
        if accepted % 50 == 0 or accepted == reps:
            print(f"Tennis MLEs: {accepted}/{reps} ({attempted} draws)", flush=True)

    pair_i, pair_j = np.triu_indices(n, 1)
    counts = np.zeros(len(pair_i), dtype=np.uint16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, covered in enumerate(pool.map(
                lambda error: _coverage_one(error, a, b, matches, truth,
                                            pair_i, pair_j), errors), 1):
            counts += covered
            if index % 50 == 0 or index == reps:
                print(f"Tennis plug-in intervals: {index}/{reps}", flush=True)
    coverage = counts / reps
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / "tennis_errors.npz", errors=errors,
                        attempts=attempted)
    np.savez_compressed(out_dir / "tennis_coverage.npz", coverage=coverage,
                        counts=counts, reps=reps)
    return errors, coverage
