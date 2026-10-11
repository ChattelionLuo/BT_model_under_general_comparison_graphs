"""Polynomial-degree ER errors and exact MLE plug-in intervals."""
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

NS = (2048, 4096, 8192, 16384)
COEFFICIENT = .75
DEGREE_POWER = .5
AMPLITUDE = 1.5
PAIRS = 500
Z = 1.959963984540054
GRAPHS = []
ORACLE_STARTS = []


def fisher_lap(n, a, b, u):
    return fisher_laplacian(n, a, b, np.ones(len(a)), u)


def fisher_resistances(n, lap, pair_i, pair_j, initial=None):
    diagonal = lap.diagonal()
    pre = LinearOperator((n,n), matvec=lambda x:x/diagonal)
    values = np.empty(len(pair_i))
    solutions = np.empty((len(pair_i),n))
    for k,(i,j) in enumerate(zip(pair_i,pair_j)):
        rhs = np.zeros(n)
        rhs[i],rhs[j] = 1,-1
        voltage,info = cg(lap,rhs,M=pre,
                          x0=None if initial is None else initial[k],
                          rtol=2e-7,atol=0,maxiter=5*n)
        if info:
            raise RuntimeError(f"CG failed on n={n}, pair={k}")
        values[k] = voltage[i]-voltage[j]
        solutions[k] = voltage
    return values,solutions


def prepare():
    GRAPHS.clear()
    ORACLE_STARTS.clear()
    qs=[]; edges=[]; resistance=[]; pairs_i=[]; pairs_j=[]
    for n in NS:
        q=COEFFICIENT*n**(DEGREE_POWER-1)
        if q>=1:
            raise ValueError(f"invalid p at n={n}")
        rng=np.random.default_rng(20261026+n)
        u=rng.uniform(-AMPLITUDE,AMPLITUDE,n);u-=u.mean()
        # Row-wise draws preserve the original RNG order without a dense n-by-n array.
        rows,cols=[],[]
        for i in range(n-1):
            chosen=np.flatnonzero(rng.random(n-i-1)<q)
            rows.append(np.full(len(chosen),i,dtype=np.int32))
            cols.append((i+1+chosen).astype(np.int32))
        a,b=np.concatenate(rows),np.concatenate(cols)
        selected=set()
        while len(selected)<PAIRS:
            i,j=rng.integers(n,size=2)
            if i!=j:
                selected.add((min(i,j),max(i,j)))
        pi,pj=map(np.asarray,zip(*sorted(selected)))
        r,starts=fisher_resistances(n,fisher_lap(n,a,b,u),pi,pj)
        ORACLE_STARTS.append(starts)
        win_p=expit(u[a]-u[b])
        GRAPHS.append((n,a,b,u,win_p,pi,pj))
        qs.append(q);edges.append(len(a));resistance.append(r)
        pairs_i.append(pi);pairs_j.append(pj)
        print(f"n={n}, p={q:.6f}, mean degree={2*len(a)/n:.1f}",flush=True)
    return np.asarray(qs),np.asarray(edges),np.asarray(resistance),np.asarray(pairs_i),np.asarray(pairs_j)


def shard(task):
    tier,start,count,seed=task
    n,a,b,u,win_p,pi,pj=GRAPHS[tier]
    rng=np.random.default_rng(seed)
    true=u[pi]-u[pj]
    delta=np.empty(count)
    errors=np.empty((count,len(pi)))
    plugin_r=np.empty_like(errors)
    attempts=0;accepted=0
    while accepted<count:
        attempts+=1
        if attempts>20*count:
            raise RuntimeError(f"finite MLE too rare for n={n}")
        wins=rng.binomial(1,win_p).astype(float)
        # Ensure the win-directed graph is strongly connected.
        if not strongly_connected_outcomes(n,a,b,wins,1-wins):
            continue
        fit=fit_mle(a,b,wins,1-wins,n,start=u,tolerance=1e-8)
        node_error=fit-u
        delta[accepted]=np.ptp(node_error)
        errors[accepted]=fit[pi]-fit[pj]-true
        plugin_r[accepted],_=fisher_resistances(
            n,fisher_lap(n,a,b,fit),pi,pj,ORACLE_STARTS[tier])
        accepted+=1
    return tier,start,delta,errors,plugin_r,attempts


def run(out_path,reps,workers,shards_per_n,seed):
    started=time.perf_counter()
    qs,edges,resistance,pair_i,pair_j=prepare()
    _newman_sweep(np.zeros(2),np.array([0,1,2]),
                  np.array([1,0]),np.ones(2),np.ones(2))
    counts=[reps//shards_per_n+(j<reps%shards_per_n)
            for j in range(shards_per_n)]
    tasks=[]
    for tier in range(len(NS)):
        start=0
        for part,count in enumerate(counts):
            if count:
                tasks.append((tier,start,count,seed+10000*tier+part))
            start+=count
    delta=np.empty((len(NS),reps))
    errors=np.empty((len(NS),reps,PAIRS))
    plugin_r=np.empty_like(errors)
    attempts=np.zeros(len(NS),dtype=int)
    workers=min(workers,os.cpu_count() or workers,len(tasks))
    # Forked workers share the prepared fixed graphs without copying them.
    # Windows retains threads because its spawned workers cannot inherit globals.
    if sys.platform.startswith("linux"):
        executor = ProcessPoolExecutor
        options = {"mp_context": get_context("fork")}
    else:
        executor = ThreadPoolExecutor
        options = {}
    with executor(max_workers=workers, **options) as pool:
        futures=[pool.submit(shard,task) for task in tasks]
        for done,future in enumerate(as_completed(futures),1):
            tier,start,d,e,r,tried=future.result()
            delta[tier,start:start+len(d)]=d
            errors[tier,start:start+len(d)]=e
            plugin_r[tier,start:start+len(d)]=r
            attempts[tier]+=tried
            if done%20==0 or done==len(tasks):
                print(f"completed {done}/{len(tasks)}; {time.perf_counter()-started:.1f}s",flush=True)
    report=dict(ns=list(NS),coefficient=COEFFICIENT,
                degree_power=DEGREE_POWER,
                amplitude=AMPLITUDE,reps=reps,pair_count=PAIRS,workers=workers,
                elapsed_seconds=time.perf_counter()-started,rows=[])
    for tier,n in enumerate(NS):
        report["rows"].append(dict(n=n,p=float(qs[tier]),
            mean_degree=float(2*edges[tier]/n),edges=int(edges[tier]),
            attempts=int(attempts[tier]),
            mean_delta=float(delta[tier].mean()),
            sd_delta=float(delta[tier].std(ddof=1)),
            mean_pair_mae=float(np.abs(errors[tier]).mean()),
            coverage=float((np.abs(errors[tier])
                            <=Z*np.sqrt(plugin_r[tier])).mean())))
    out_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_path,ns=np.asarray(NS),qs=qs,
        edges=edges,oracle_r=resistance,pair_i=pair_i,pair_j=pair_j,
        uniform_delta=delta,errors=errors,plugin_r=plugin_r)
    out_path.with_suffix(".json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


