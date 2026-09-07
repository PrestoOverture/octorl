"""Candidate replacement rules for the A-side fork bar (final review 2026-09-02). Explicit seeds; null rows are synthetic controls."""
import numpy as np
from math import lgamma
G=8; _gl=np.vectorize(lgamma)
def f(p): p=np.asarray(p,float); return 1-p**G-(1-p)**G
def Ef(a,b):
    t1=_gl(a+G)+_gl(a+b)-_gl(a)-_gl(a+b+G); t2=_gl(b+G)+_gl(a+b)-_gl(b)-_gl(a+b+G); return 1-np.exp(t1)-np.exp(t2)
def bp(mu,sd): nu=mu*(1-mu)/sd**2-1; return mu*nu,(1-mu)*nu
def sd_for_gap(mu,gap):
    lo,hi=1e-3,np.sqrt(mu*(1-mu))*0.95
    for _ in range(60):
        m=(lo+hi)/2; g=float(f(mu))-float(Ef(*bp(mu,m)))
        if g<gap: lo=m
        else: hi=m
    return (lo+hi)/2
rng=np.random.default_rng(77)
def gap_hat(K): return float(f(K.mean()/G))-float(np.mean((K>0)&(K<G)))
def boot_lb(K,B=300,q=0.05):
    n=len(K); idx=rng.integers(0,n,(B,n)); vals=np.array([gap_hat(K[i]) for i in idx]); return np.quantile(vals,q)
def run(mus,gaps,T,thr,lb_q,trials=1500):
    passes=0
    for _ in range(trials):
        hit=False
        for mu,gap in zip(mus,gaps):
            p=np.full(T,mu) if gap==0 else rng.beta(*bp(mu,sd_for_gap(mu,gap)),T)
            K=rng.binomial(G,p)
            gh=gap_hat(K)
            if gh>=thr and (lb_q is None or boot_lb(K,q=lb_q)>0): hit=True; break
        passes+=hit
    return passes/trials
hard=(0.15,0.08)
print("Rule = pass iff gap_hat >= thr AND bootstrap lower bound(q) > 0 on >=1 of the 2 HARD buckets (mu=0.15, 0.08)")
print(f"{'T':>4} {'thr':>5} {'LBq':>5} | {'null':>6} {'gap.03':>7} {'gap.05':>7} {'gap.08':>7} {'gap.12':>7}")
for T,thr,q in ((100,0.02,None),(200,0.02,None),(200,0.05,0.05),(300,0.05,0.05),(300,0.05,0.025),(400,0.05,0.025)):
    row=[run(hard,(0,0),T,thr,q)]+[run(hard,(g,g),T,thr,q) for g in (0.03,0.05,0.08,0.12)]
    print(f"{T:4d} {thr:5.2f} {str(q):>5} | "+" ".join(f"{r:7.2f}" for r in row))
print("Cost of 2 hard buckets at T: 2*T*8 trajectories -> T=300: 4,800; T=400: 6,400 (vs 3,200 for 4x100 today)")
