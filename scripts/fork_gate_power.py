"""Power audit of the pre-registered Fork Gate bars, the D4 staleness SE claim, and the rollout-budget arithmetic (final review 2026-09-02).
Explicit seeds. Synthetic controls (null inputs) included for every statistic."""
import numpy as np
from math import lgamma, comb
G = 8
_gammaln = np.vectorize(lgamma)
def f(p): p=np.asarray(p,float); return 1-p**G-(1-p)**G
def Ef_beta(a,b):
    t1=_gammaln(a+G)+_gammaln(a+b)-_gammaln(a)-_gammaln(a+b+G)
    t2=_gammaln(b+G)+_gammaln(a+b)-_gammaln(b)-_gammaln(a+b+G)
    return 1-np.exp(t1)-np.exp(t2)
def beta_params(mu,sd): nu=mu*(1-mu)/sd**2-1; return mu*nu,(1-mu)*nu
def est_b2(K):
    ph=K/G; mu=ph.mean()
    if mu<=0 or mu>=1: return float(f(mu))
    v=(ph.var(ddof=1)-mu*(1-mu)/G)/(1-1/G)
    if v<=0 or v>=mu*(1-mu): return float(f(mu))
    a,b=beta_params(mu,np.sqrt(v)); return float(Ef_beta(a,b))

# ---------- 1. A-side viability bar: "gap >= 0.02 on >=1 of 4 buckets", T=100 ----------
print("="*78); print("1. A viability bar (gap >= 0.02 on >=1 of 4 buckets), T tasks x G=8")
print("   gap estimate = f(mu_hat) - q_hat_emp  (plug-in minus observed mixed fraction)")
rng=np.random.default_rng(11)
def gap_est(K): return float(f(K.mean()/G)) - float(np.mean((K>0)&(K<G)))
def gap_est_b2(K): return float(f(K.mean()/G)) - est_b2(K)
for T in (100,200,400):
    for label,mus,sd in (("NULL: 4 homogeneous buckets",(0.5,0.3,0.15,0.08),0.0),
                         ("ALT: true gap ~0.02-0.03 (Beta sd small)",(0.5,0.3,0.15,0.08),None)):
        trials=4000; pass_emp=0; pass_b2=0; per_bucket=np.zeros(4)
        for t in range(trials):
            hit_e=False; hit_b=False
            for j,mu in enumerate(mus):
                if sd==0.0: p=np.full(T,mu)
                else:
                    # choose sd giving oracle gap ~0.02-0.03 at this mu (from the PRD grid)
                    s={0.5:0.125,0.3:0.085,0.15:0.05,0.08:0.035}[mu]
                    a,b=beta_params(mu,s); p=rng.beta(a,b,T)
                K=rng.binomial(G,p)
                ge=gap_est(K); gb=gap_est_b2(K)
                if ge>=0.02: hit_e=True; per_bucket[j]+=1
                if gb>=0.02: hit_b=True
            pass_emp+=hit_e; pass_b2+=hit_b
        print(f"  T={T:3d} {label:42s} P(pass via B_emp gap)={pass_emp/trials:.2f}  P(pass via B2 gap)={pass_b2/trials:.2f}  per-bucket hit rate={np.round(per_bucket/trials,2)}")
print("   (NULL rows are the false-pass rate; ALT rows are power at the smallest gap the bar is meant to catch.)")
# true oracle gaps of the ALT populations for reference
print("   ALT true oracle gaps:", [round(float(f(mu))-float(Ef_beta(*beta_params(mu,s))),4) for mu,s in ((0.5,0.125),(0.3,0.085),(0.15,0.05),(0.08,0.035))])

# ---------- 2. C-G1: paired A0 vs A1, 2 cells x 40 instances, McNemar exact ----------
print("="*78); print("2. C-G1 bar: pooled lift >= +15pp AND exact McNemar p<0.05, n=80 paired")
from statistics import NormalDist
from math import comb
def binom_two_sided(k,n):
    # exact two-sided binomial test p-value at p=0.5 (McNemar exact)
    pk=lambda i: comb(n,i)*0.5**n
    p0=pk(k)
    return min(1.0,sum(pk(i) for i in range(n+1) if pk(i)<=p0+1e-12))
def mcnemar_p(b,c):
    n=b+c
    return 1.0 if n==0 else binom_two_sided(min(b,c),n)
rng=np.random.default_rng(22)
def sim_g1(p0,p1,n=80,trials=6000,rho=0.3):
    # paired binary outcomes with correlation via shared latent
    passes=0; lifts=[]
    for _ in range(trials):
        z=rng.normal(size=n)
        u0=rng.normal(size=n)*np.sqrt(1-rho)+z*np.sqrt(rho)
        u1=rng.normal(size=n)*np.sqrt(1-rho)+z*np.sqrt(rho)
        y0=u0<NormalDist().inv_cdf(p0); y1=u1<NormalDist().inv_cdf(p1)
        b=int(np.sum(y1&~y0)); c=int(np.sum(~y1&y0))
        lift=(y1.mean()-y0.mean())
        lifts.append(lift)
        if lift>=0.15 and mcnemar_p(b,c)<0.05: passes+=1
    lifts=np.array(lifts)
    return passes/trials, np.percentile(lifts,[2.5,97.5])
for p0,p1 in ((0.15,0.15),(0.15,0.25),(0.15,0.30),(0.15,0.35),(0.15,0.45)):
    pw,ci=sim_g1(p0,p1)
    print(f"  A0={p0:.2f} A1={p1:.2f} (true lift {p1-p0:+.2f}): P(pass)={pw:.2f}   95% range of observed lift=[{ci[0]:+.2f},{ci[1]:+.2f}]")

# ---------- 3. C-F: rho_c = (A3-A0)/(A1-A0) >= 0.5 point rule, n=80 same-repo-new ----------
print("="*78); print("3. C-F bar: rho_c=(A3-A0)/(A1-A0) >= 0.5 with McNemar(A3>A0) p<0.05, n=80 (E1, 2 cells x 40)")
rng=np.random.default_rng(33)
def sim_f(p0,p1,p3,n=80,trials=6000,rho=0.3):
    res={"pass":0,"kill":0,"grey":0}; rc=[]
    for _ in range(trials):
        z=rng.normal(size=n)
        def draw(p): return (rng.normal(size=n)*np.sqrt(1-rho)+z*np.sqrt(rho))<NormalDist().inv_cdf(p)
        y0,y1,y3=draw(p0),draw(p1),draw(p3)
        l1=y1.mean()-y0.mean(); l3=y3.mean()-y0.mean()
        r=l3/l1 if l1>0 else np.nan
        rc.append(r)
        b=int(np.sum(y3&~y0)); c=int(np.sum(~y3&y0))
        if not np.isnan(r) and r>=0.5 and mcnemar_p(b,c)<0.05: res["pass"]+=1
        elif np.isnan(r) or r<=0.25: res["kill"]+=1
        else: res["grey"]+=1
    rc=np.array(rc); rc=rc[~np.isnan(rc)]
    return {k:v/trials for k,v in res.items()}, np.percentile(rc,[2.5,50,97.5])
for p0,p1,p3,lab in ((0.15,0.35,0.15,"NULL consolidation (rho_c=0)"),
                     (0.15,0.35,0.20,"rho_c=0.25 (kill boundary)"),
                     (0.15,0.35,0.25,"rho_c=0.50 (pass boundary)"),
                     (0.15,0.35,0.30,"rho_c=0.75"),
                     (0.15,0.55,0.35,"rho_c=0.50 with a 40pp G1 lift")):
    r,q=sim_f(p0,p1,p3)
    print(f"  {lab:34s} P(pass)={r['pass']:.2f} P(kill)={r['kill']:.2f} P(grey)={r['grey']:.2f}   rho_c 2.5/50/97.5% = {np.round(q,2)}")

# ---------- 4. D4 staleness-cost SE vs affordable batch counts ----------
print("="*78); print("4. D4 staleness cost SE = SE[Brier(L)-Brier(0)] from the run's own groups")
print("   PRD claim: SE ~1.5e-3 at 400 batches x 64 groups = 25,600 groups = 204,800 trajectories.")
for groups in (25600, 5440, 1216, 640):
    traj=groups*G
    print(f"   groups={groups:6d} trajectories={traj:7d}  SE ~ {1.5e-3*np.sqrt(25600/groups):.4f}")
print("   Signal the PRD quotes for 'large' staleness (0.02 drift/update, L=16): +0.0031.")
print("="*78); print("5. Rollout budget arithmetic (assumptions explicit)")
rate=2.18  # CNY/h measured on AutoDL 2026-09-01
for name,cny in (("P2 (gates+D1/D2)",100),("Fork window",190),("P3 first RL run",250),("P4 matrix (9 runs)",500)):
    hours=cny/rate
    for thr_name,thr in (("serial 40/h (90s gate, no concurrency)",40),("8x concurrent ~320/h",320)):
        traj=hours*thr
        print(f"   {name:22s} {cny:4d} CNY = {hours:5.0f} h @ {thr_name:38s} -> {traj:8.0f} trajectories = {traj/G:6.0f} groups = {traj/512:5.1f} updates of 64 groups")
