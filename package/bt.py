from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.special import expit


@njit(nogil=True)
def _newman_sweep(u, indptr, neighbors, wins, losses):
    largest = 0.0
    for i in range(len(u)):
        numerator = 0.0
        denominator = 0.0
        for pos in range(indptr[i], indptr[i + 1]):
            j = neighbors[pos]
            p = 1.0 / (1.0 + np.exp(u[j] - u[i]))
            numerator += wins[pos] * (1.0 - p)
            denominator += losses[pos] * p
        if numerator <= 0.0 or denominator <= 0.0:
            return np.inf
        step = np.log(numerator) - np.log(denominator)
        u[i] += step
        largest = max(largest, abs(step))
    u -= np.mean(u)
    return largest


def fit_mle(a, b, wins_a, wins_b, n, start=None, tolerance=1e-8):
    source = np.r_[a, b]
    target = np.r_[b, a]
    wins = np.r_[wins_a, wins_b].astype(float)
    losses = np.r_[wins_b, wins_a].astype(float)
    order = np.argsort(source, kind="stable")
    indptr = np.r_[0, np.cumsum(np.bincount(source, minlength=n))].astype(np.int64)
    target = target[order].astype(np.int64)
    wins, losses = wins[order], losses[order]
    u = np.zeros(n) if start is None else np.asarray(start, dtype=float).copy()
    u -= u.mean()
    matches = wins_a + wins_b
    for sweep in range(3000):
        update = _newman_sweep(u, indptr, target, wins, losses)
        if not np.isfinite(update):
            raise RuntimeError("The MLE is not finite")
        if update < tolerance:
            p = expit(u[a] - u[b])
            edge_score = wins_a - matches * p
            score = (np.bincount(a, weights=edge_score, minlength=n)
                     - np.bincount(b, weights=edge_score, minlength=n))
            if np.max(np.abs(score)) < 1e-6:
                return u
    raise RuntimeError("MLE iteration did not converge")


def strongly_connected_outcomes(n, a, b, wins_a, wins_b):
    source = np.r_[a[wins_a > 0], b[wins_b > 0]]
    target = np.r_[b[wins_a > 0], a[wins_b > 0]]
    directed = csr_matrix((np.ones(len(source), dtype=np.int8),
                           (source, target)), shape=(n, n))
    return connected_components(directed, directed=True, connection="strong")[0] == 1


def laplacian(n, a, b, weights):
    adjacency = coo_matrix((np.r_[weights, weights],
                            (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency


def fisher_laplacian(n, a, b, matches, u):
    p = expit(u[a] - u[b])
    return laplacian(n, a, b, matches * p * (1 - p))


def pseudoinverse(lap):
    n = lap.shape[0]
    factor = cho_factor(lap.toarray() + np.full((n, n), 1 / n),
                       lower=True, check_finite=False)
    inverse = cho_solve(factor, np.eye(n), check_finite=False) - 1 / n  # center
    return (inverse + inverse.T) / 2
