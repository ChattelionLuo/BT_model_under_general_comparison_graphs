"""Joint MLE and analytic approximation for the bipartite design."""
from __future__ import annotations
import numpy as np
from scipy.special import expit, ndtr
DELTA, AMP = 2*np.log(2), 0.0

def truth(n):
    m = n-2
    if m % 2: raise ValueError("n must be even")
    ability = np.tile([-AMP, AMP], m//2)
    return ability, expit(DELTA/2-ability), expit(-DELTA/2-ability)

def coefficients(n):
    _, p1, p2 = truth(n)
    w1, w2 = p1*(1-p1), p2*(1-p2)
    total = w1+w2
    info = w1*w2/total
    b = np.sum(info*(p1-p2)/total)/np.sum(info)
    s = np.sqrt((n-2)/np.sum(info))
    return float(b), float(s)

def profiled_p1(q, delta):
    r = np.exp(delta[:, None])
    B = 1-q+r*(q+1)
    D = np.maximum(B*B+4*(1-r)*r*q, 0)
    return np.clip(2*r*q/(B+np.sqrt(D)), 0, 1)

def fit_full_mle(y1, y2, L):
    if np.any((y1+y2==0)|(y1+y2==2*L)):
        raise ValueError("Infinite student nuisance MLE")
    q = (y1+y2)/L
    a, b = y1.mean(axis=1)/L, y2.mean(axis=1)/L
    delta = np.clip(np.log(a/(1-a))-np.log(b/(1-b)), 0, 8)
    for _ in range(12):
        p1 = profiled_p1(q, delta)
        p2 = q-p1
        w1, w2 = p1*(1-p1), p2*(1-p2)
        info = np.sum(w1*w2/np.maximum(w1+w2, 1e-15), axis=1)
        score = np.sum(y1/L-p1, axis=1)
        step = score/info
        delta += step
        if np.max(np.abs(step))<1e-11: break
    residual = np.abs(np.sum(y1/L-profiled_p1(q, delta), axis=1))
    if np.max(residual)>1e-7: raise RuntimeError("Joint MLE score did not converge")
    return delta

def simulate(n, L, reps, seed):
    _, p1, p2 = truth(n)
    rng = np.random.default_rng(seed)
    errors = []
    while len(errors) < reps:
        count = min(100, reps-len(errors))
        y1 = rng.binomial(L, p1, size=(count,n-2))
        y2 = rng.binomial(L, p2, size=(count,n-2))
        finite = np.all((y1+y2 > 0) & (y1+y2 < 2*L), axis=1)
        if np.any(finite):
            errors.extend(fit_full_mle(y1[finite], y2[finite], L)-DELTA)
    return np.asarray(errors)

def folded_mean(mu, sd):
    z = mu/sd
    return sd*np.sqrt(2/np.pi)*np.exp(-z*z/2)+mu*(2*ndtr(z)-1)

def case(rows, design, n, L, reps, seed, beta=np.nan):
    e = simulate(n,L,reps,seed)
    b,s = coefficients(n)
    bias, sd = b/L, s/np.sqrt((n-2)*L)
    noise_mae = sd*np.sqrt(2/np.pi)
    rows.append(dict(
        design=design,n=n,L=L,beta=beta,reps=reps,
        mean_abs=np.mean(np.abs(e)),median_abs=np.median(np.abs(e)),
        q25_abs=np.quantile(np.abs(e),.25),q75_abs=np.quantile(np.abs(e),.75),
        signed_bias=np.mean(e),sd_signed=np.std(e,ddof=1),
        pred_bias=bias,pred_sd=sd,pred_noise_mae=noise_mae,
        pred_mean_abs=folded_mean(bias,sd),
        Lc_abs=(b/(s*np.sqrt(2/np.pi)))**2*(n-2),
        lambda2=2*L,resistance_items=2/((n-2)*L)))

