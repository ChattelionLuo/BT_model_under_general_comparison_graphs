"""All within-group contrasts on an incident-community polynomial ER graph."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
import json
import os
import sys
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import expit

from .bt import (_newman_sweep, fisher_laplacian, fit_mle,
                 strongly_connected_outcomes)

N = 8192
H = 20
A = 1.5
Q_FACTOR = .75
BASE_DEGREE_POWER = .5
Z = 1.959963984540054
GRAPH_SEED = 20261007 + N + int(100*A) + 125

GRAPHS = []
U = None
GROUPS = []
PAIR_I = None
PAIR_J = None
ORACLE_STARTS = []


def fisher_lap(a, b, u):
    return fisher_laplacian(N, a, b, np.ones(len(a)), u)


def group_resistances(lap, groups, initial=None):
    """Get every within-group resistance using H-1 Laplacian solves per group."""
    diagonal = lap.diagonal()
    pre = LinearOperator((N,N), matvec=lambda x:x/diagonal)
    all_r, all_solutions = [], []
    for gidx, vertices in enumerate(groups):
        anchor = vertices[0]
        solutions = np.empty((len(vertices)-1,N))
        for k, target in enumerate(vertices[1:]):
            rhs = np.zeros(N)
            rhs[target], rhs[anchor] = 1, -1
            x0 = None if initial is None else initial[gidx][k]
            solution, info = cg(lap, rhs, M=pre, x0=x0,
                                rtol=2e-7, atol=0, maxiter=5*N)
            if info:
                raise RuntimeError(f"CG failed for group {gidx}, vertex {k}: {info}")
            solutions[k] = solution
        gram = solutions[:,vertices[1:]]-solutions[:,[anchor]]
        gram = (gram+gram.T)/2
        resistance = np.zeros((len(vertices),len(vertices)))
        resistance[0,1:] = resistance[1:,0] = np.diag(gram)
        resistance[1:,1:] = (np.diag(gram)[:,None]
                              +np.diag(gram)[None,:]-2*gram)
        tri = np.triu_indices(len(vertices),1)
        all_r.append(resistance[tri])
        all_solutions.append(solutions)
    return np.concatenate(all_r), all_solutions


def prepare():
    global U, PAIR_I, PAIR_J
    GRAPHS.clear()
    GROUPS.clear()
    ORACLE_STARTS.clear()
    q = Q_FACTOR*N**(BASE_DEGREE_POWER-1)
    ps = np.asarray([q,.04,.08,.16,.32,.8])
    rng = np.random.default_rng(GRAPH_SEED)
    high = np.sort(rng.choice(N,size=H,replace=False))
    indicator = np.zeros(N,dtype=bool)
    indicator[high] = True
    rows,cols,touch_rows,draw_rows=[],[],[],[]
    for i in range(N-1):
        js=np.arange(i+1,N)
        touches=indicator[i] | indicator[js]
        uniforms=rng.random(len(js))
        keep=np.where(touches,uniforms<ps[-1],uniforms<q)
        rows.append(np.full(np.count_nonzero(keep),i,dtype=np.int32))
        cols.append(js[keep].astype(np.int32))
        touch_rows.append(touches[keep])
        draw_rows.append(uniforms[keep])
    max_a,max_b=np.concatenate(rows),np.concatenate(cols)
    max_touch,max_uniform=np.concatenate(touch_rows),np.concatenate(draw_rows)
    U = rng.uniform(-A,A,N)
    U -= U.mean()
    pair_rng = np.random.default_rng(GRAPH_SEED+1)
    low = np.sort(pair_rng.choice(np.setdiff1d(np.arange(N),high),
                                  size=H,replace=False))
    GROUPS.extend((high,low))
    tri = np.triu_indices(H,1)
    PAIR_I = np.r_[high[tri[0]],low[tri[0]]]
    PAIR_J = np.r_[high[tri[1]],low[tri[1]]]
    pair_group = np.array(["high-high"]*len(tri[0])
                          +["low-low"]*len(tri[0]))
    oracle_r = []
    for p in ps:
        keep = np.where(max_touch,max_uniform<p,max_uniform<q)
        a,b = max_a[keep].astype(np.int32),max_b[keep].astype(np.int32)
        GRAPHS.append((a,b,expit(U[a]-U[b])))
        r,starts = group_resistances(fisher_lap(a,b,U),GROUPS)
        oracle_r.append(r)
        ORACLE_STARTS.append(starts)
        print(f"prepared p={p:.6g}, edges={len(a)}",flush=True)
    return q,ps,pair_group,np.asarray(oracle_r)


def shard(task):
    tier,start,count,seed = task
    a,b,win_p = GRAPHS[tier]
    rng = np.random.default_rng(seed)
    truth = U[PAIR_I]-U[PAIR_J]
    errors = np.empty((count,len(PAIR_I)))
    plugin_r = np.empty_like(errors)
    attempts = accepted = 0
    while accepted<count:
        attempts += 1
        if attempts>30*count:
            raise RuntimeError(f"Finite MLE too rare in tier {tier}")
        wins = rng.binomial(1,win_p).astype(float)
        # Ensure the win-directed graph is strongly connected.
        if not strongly_connected_outcomes(N,a,b,wins,1-wins):
            continue
        fit = fit_mle(a,b,wins,1-wins,N,start=U,tolerance=1e-8)
        r,_ = group_resistances(fisher_lap(a,b,fit),GROUPS,
                                ORACLE_STARTS[tier])
        errors[accepted] = fit[PAIR_I]-fit[PAIR_J]-truth
        plugin_r[accepted] = r
        accepted += 1
    return tier,start,errors,plugin_r,attempts


def run(out_path,reps,workers,shards_per_tier,seed):
    started = time.perf_counter()
    q,ps,pair_group,oracle_r = prepare()
    _newman_sweep(np.zeros(2),np.array([0,1,2]),
                  np.array([1,0]),np.ones(2),np.ones(2))
    counts = [reps//shards_per_tier+(j<reps%shards_per_tier)
              for j in range(shards_per_tier)]
    tasks = []
    for tier in range(len(ps)):
        start = 0
        for part,count in enumerate(counts):
            if count:
                tasks.append((tier,start,count,seed+10000*tier+part))
            start += count
    errors = np.empty((len(ps),reps,len(PAIR_I)))
    plugin_r = np.empty_like(errors)
    attempts = np.zeros(len(ps),dtype=int)
    workers = min(workers,os.cpu_count() or workers,len(tasks))
    print(f"running {len(tasks)} shards with {workers} workers",flush=True)
    # Forked workers share the prepared fixed graphs without copying them.
    # Windows retains threads because its spawned workers cannot inherit globals.
    if sys.platform.startswith("linux"):
        executor = ProcessPoolExecutor
        options = {"mp_context": get_context("fork")}
    else:
        executor = ThreadPoolExecutor
        options = {}
    with executor(max_workers=workers, **options) as pool:
        futures = [pool.submit(shard,task) for task in tasks]
        for done,future in enumerate(as_completed(futures),1):
            tier,start,e,r,tried = future.result()
            end = start+len(e)
            errors[tier,start:end] = e
            plugin_r[tier,start:end] = r
            attempts[tier] += tried
            if done%12==0 or done==len(tasks):
                print(f"completed {done}/{len(tasks)} shards, "
                      f"elapsed {time.perf_counter()-started:.1f}s",flush=True)
    covered = np.abs(errors)<=Z*np.sqrt(plugin_r)
    report = dict(n=N,community_size=H,amplitude=A,q=q,
                  degree_power=BASE_DEGREE_POWER,q_factor=Q_FACTOR,
                  ps=ps.tolist(),reps=reps,
                  pairs_per_group=H*(H-1)//2,
                  workers=workers,shards_per_tier=shards_per_tier,
                  attempts_by_tier=attempts.tolist(),
                  elapsed_seconds=time.perf_counter()-started,tiers=[])
    for t,p in enumerate(ps):
        row = dict(p=float(p),edges=len(GRAPHS[t][0]),
                   balance_ratio=float((N*q)**2/((N-1)*p)))
        for label,code in (("hh","high-high"),("ll","low-low")):
            mask = pair_group==code
            mae = np.abs(errors[t][:,mask]).mean(axis=0)
            coverage = covered[t][:,mask].mean(axis=0)
            row[f"{label}_mae_mean"] = float(mae.mean())
            row[f"{label}_mae_max"] = float(mae.max())
            row[f"{label}_coverage_mean"] = float(coverage.mean())
            row[f"{label}_coverage_q25"] = float(np.quantile(coverage,.25))
            row[f"{label}_coverage_q75"] = float(np.quantile(coverage,.75))
            row[f"{label}_median_sqrt_r"] = float(np.median(
                np.sqrt(plugin_r[t][:,mask])))
        report["tiers"].append(row)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,ps=ps,q=q,u=U,
                        high_vertices=GROUPS[0],low_vertices=GROUPS[1],
                        pair_i=PAIR_I,pair_j=PAIR_J,pair_group=pair_group,
                        oracle_r=oracle_r,errors=errors,plugin_r=plugin_r)
    out_path.with_suffix(".json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


