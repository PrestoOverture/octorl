#!/usr/bin/env python3
"""
T2 (compounding audit) — reseeded-incumbent null gate: power / false-commit simulation.

Pre-registration support for v2.7 (2026-09-02). Every number in the v2.7 delta that
concerns Gate G2 / the T2 main design is reproducible from this file with the seeds below.

Model
-----
A floor-regime cell has per-instance success probabilities p_i ~ Beta(a, b) with mean mu
and noise-corrected sd (the same (mu, sd) grid as prd.md §3.5 / estimator_study.py).
A "reseeded-incumbent null" round evaluates the *same* harness under a fresh sampling seed:
candidate outcomes are Bernoulli(p_i) independent of the incumbent's — the true delta is 0
by construction, so every commit is a false commit.

Acceptors simulated
-------------------
greedy_stored : classic DGM/SICA/AHE-style "commit if dev score > incumbent's recorded
                dev score" on a reused dev set of n_dev instances (winner's-curse variant).
greedy_fresh  : incumbent re-evaluated each round on the same dev set; commit if strictly
                higher (best-case greedy; removes the stale-score effect, keeps the noise).
eprocess      : PACE-style paired betting e-process on discordant pairs drawn from FRESH
                generated instances (OctoRL can generate them); commit when wealth >= 1/alpha,
                give up after n_max paired instances.

Outputs
-------
1. Null: commits per run (30 rounds) for each acceptor, across the (mu, sd) grid.
2. Discordance rate on the null (the observable that makes the phenomenon exist at all).
3. Alternative: P(commit) for a true edit of size delta, per acceptor (power).
4. Audit labelling error at n_audit fresh paired instances.
5. Budget arithmetic for the fork window and the P3–P4 T2 design.
6. Synthetic checks: e-process false-commit <= alpha on the null; greedy_fresh commit
   probability on a homogeneous cell equals the closed-form P(X > Y) for iid binomials.

Run: python3 scripts/null_gate_power.py
"""
from __future__ import annotations

import math
import numpy as np

SEED = 20260902
N_SIMS = 4000
N_DEV = 40
ROUNDS = 30
ALPHA = 0.05
LAMBDA = 0.4          # fixed betting fraction; registered value
N_MAX_E = 200         # e-process paired-instance budget per candidate (registered; sweep below covers 120)
FUTILITY = 0.25       # abandon a candidate when wealth <= FUTILITY (never raises false-commit prob)
N_AUDIT = (100, 200, 400)
GRID_MU = (0.08, 0.15, 0.25)
GRID_SD = (0.0, 0.07, 0.12)
DELTAS = (0.05, 0.10, 0.20)


def beta_params(mu: float, sd: float) -> tuple[float, float] | None:
    if sd == 0.0:
        return None
    v = sd * sd
    k = mu * (1 - mu) / v - 1
    if k <= 0:
        raise ValueError(f"sd={sd} infeasible at mu={mu}")
    return mu * k, (1 - mu) * k


def draw_p(rng, mu, sd, n):
    bp = beta_params(mu, sd)
    if bp is None:
        return np.full(n, mu)
    return rng.beta(bp[0], bp[1], size=n)


def eprocess_commit(rng, p_inc, p_cand, n_max, lam, alpha, futility=None):
    """Sequential paired betting on fresh instances. Returns (committed, n_used)."""
    if futility is None:
        futility = FUTILITY
    wealth = 1.0
    thresh = 1.0 / alpha
    for t in range(n_max):
        a = rng.random() < p_inc[t]
        b = rng.random() < p_cand[t]
        if a != b:
            w = 1.0 if (b and not a) else 0.0
            wealth *= 1.0 + lam * (2 * w - 1)
            if wealth >= thresh:
                return True, t + 1
            if futility > 0 and wealth <= futility:
                return False, t + 1
    return False, n_max


def simulate_null(rng, mu, sd):
    """One null run of ROUNDS rounds. Returns dict of commits per acceptor + discordance."""
    p = draw_p(rng, mu, sd, N_DEV)
    inc_stored = int((rng.random(N_DEV) < p).sum())
    commits = {"greedy_stored": 0, "greedy_fresh": 0, "eprocess": 0}
    disc = 0
    e_used = 0
    for _ in range(ROUNDS):
        cand = rng.random(N_DEV) < p
        inc_fresh = rng.random(N_DEV) < p
        disc += int((cand != inc_fresh).sum())
        cs = int(cand.sum())
        if cs > inc_stored:
            commits["greedy_stored"] += 1
            inc_stored = cs
        if cs > int(inc_fresh.sum()):
            commits["greedy_fresh"] += 1
        # e-process on fresh instances: new p_i for fresh instances from the same cell
        p_fresh = draw_p(rng, mu, sd, N_MAX_E)
        ok, used = eprocess_commit(rng, p_fresh, p_fresh, N_MAX_E, LAMBDA, ALPHA)
        commits["eprocess"] += int(ok)
        e_used += used
    return commits, disc / (ROUNDS * N_DEV), e_used / ROUNDS


def simulate_alt(rng, mu, sd, delta):
    """One candidate with true per-instance lift delta (clipped). Returns commit flags."""
    p = draw_p(rng, mu, sd, N_DEV)
    pc = np.clip(p + delta, 0, 1)
    inc_stored = int((rng.random(N_DEV) < p).sum())
    cand = rng.random(N_DEV) < pc
    inc_fresh = rng.random(N_DEV) < p
    cs = int(cand.sum())
    g_stored = cs > inc_stored
    g_fresh = cs > int(inc_fresh.sum())
    p_fresh = draw_p(rng, mu, sd, N_MAX_E)
    ok, used = eprocess_commit(rng, p_fresh, np.clip(p_fresh + delta, 0, 1), N_MAX_E, LAMBDA, ALPHA)
    return g_stored, g_fresh, ok, used


def audit_label(rng, mu, sd, delta, n_audit):
    """Fresh paired audit; label 'true' if one-sided exact binomial on discordant pairs p<0.05."""
    p = draw_p(rng, mu, sd, n_audit)
    pc = np.clip(p + delta, 0, 1)
    a = rng.random(n_audit) < p
    b = rng.random(n_audit) < pc
    n01 = int((b & ~a).sum())  # candidate wins
    n10 = int((a & ~b).sum())
    m = n01 + n10
    if m == 0:
        return False
    # one-sided exact binomial P(X >= n01 | m, 0.5)
    tail = sum(math.comb(m, k) for k in range(n01, m + 1)) / 2 ** m
    return tail < 0.05


def pct(x, q):
    return float(np.percentile(x, q))


def main():
    rng = np.random.default_rng(SEED)

    # ---------- synthetic checks ----------
    print("== Synthetic checks ==")
    # (a) greedy_fresh on homogeneous cell vs closed form P(X>Y), X,Y ~ Bin(n, mu)
    mu = 0.15
    from math import comb
    pm = [comb(N_DEV, k) * mu**k * (1 - mu) ** (N_DEV - k) for k in range(N_DEV + 1)]
    closed = sum(pm[i] * pm[j] for i in range(N_DEV + 1) for j in range(N_DEV + 1) if i > j)
    sim = np.mean([
        int((rng.random(N_DEV) < mu).sum()) > int((rng.random(N_DEV) < mu).sum())
        for _ in range(20000)
    ])
    print(f"greedy_fresh P(commit) homogeneous mu=0.15: closed-form {closed:.3f}, sim {sim:.3f}")
    # (b) e-process on null must commit <= alpha
    e_null = np.mean([eprocess_commit(rng, np.full(N_MAX_E, mu), np.full(N_MAX_E, mu),
                                      N_MAX_E, LAMBDA, ALPHA)[0] for _ in range(20000)])
    print(f"eprocess P(commit | null, n_max={N_MAX_E}): {e_null:.4f}  (must be <= {ALPHA})")

    # ---------- null runs across the grid ----------
    print(f"\n== Reseeded-incumbent NULL: commits per run ({ROUNDS} rounds, n_dev={N_DEV}, {N_SIMS} sims) ==")
    print("mu    sd    | greedy_stored mean [q05,q95] | greedy_fresh mean [q05,q95] | eprocess mean [q95] | discord | e-inst/round")
    null_table = {}
    for mu in GRID_MU:
        for sd in GRID_SD:
            gs, gf, ep, dc, eu = [], [], [], [], []
            for _ in range(N_SIMS):
                c, d, u = simulate_null(rng, mu, sd)
                gs.append(c["greedy_stored"]); gf.append(c["greedy_fresh"]); ep.append(c["eprocess"])
                dc.append(d); eu.append(u)
            gs, gf, ep = np.array(gs), np.array(gf), np.array(ep)
            null_table[(mu, sd)] = (gs, gf, ep, np.mean(dc), np.mean(eu))
            print(f"{mu:.2f}  {sd:.2f}  | {gs.mean():5.2f} [{pct(gs,5):.0f},{pct(gs,95):.0f}]"
                  f"              | {gf.mean():5.2f} [{pct(gf,5):.0f},{pct(gf,95):.0f}]"
                  f"             | {ep.mean():5.2f} [{pct(ep,95):.0f}]"
                  f"          | {np.mean(dc):.3f}   | {np.mean(eu):.0f}")

    # pooled over the registered design: 2 cells x 2 seeds = 4 null runs
    print("\n== Pooled null over 4 runs (2 cells x 2 seeds): total commits, q05 / q50 / q95 ==")
    for mu in GRID_MU:
        for sd in GRID_SD:
            gs, gf, ep, _, _ = null_table[(mu, sd)]
            pool = lambda x: x.reshape(-1, 4)[: N_SIMS // 4].sum(axis=1)
            ps, pf, pe = pool(gs), pool(gf), pool(ep)
            print(f"mu={mu:.2f} sd={sd:.2f} | greedy_stored {pct(ps,5):.0f}/{pct(ps,50):.0f}/{pct(ps,95):.0f}"
                  f" | greedy_fresh {pct(pf,5):.0f}/{pct(pf,50):.0f}/{pct(pf,95):.0f}"
                  f" | eprocess {pct(pe,5):.0f}/{pct(pe,50):.0f}/{pct(pe,95):.0f}")

    # ---------- power: a real edit of size delta ----------
    print(f"\n== Alternative: P(commit) for a true edit of +delta (single candidate) ==")
    print("mu    sd    delta | greedy_stored | greedy_fresh | eprocess(n_max=N_MAX_E) | e-inst used")
    for mu in (0.08, 0.15):
        for sd in (0.0, 0.12):
            for delta in DELTAS:
                r = np.array([simulate_alt(rng, mu, sd, delta) for _ in range(N_SIMS)], dtype=float)
                print(f"{mu:.2f}  {sd:.2f}  {delta:.2f}  |    {r[:,0].mean():.2f}       |    {r[:,1].mean():.2f}"
                      f"      |        {r[:,2].mean():.2f}         | {r[:,3].mean():.0f}")

    # ---------- audit labelling ----------
    print(f"\n== Audit label 'true improvement' rate at n_audit fresh paired instances (one-sided exact, 0.05) ==")
    print("mu    sd    delta | " + " | ".join(f"n={n}" for n in N_AUDIT))
    for mu in (0.08, 0.15):
        for sd in (0.0, 0.12):
            for delta in (0.0,) + DELTAS:
                row = []
                for n in N_AUDIT:
                    row.append(np.mean([audit_label(rng, mu, sd, delta, n) for _ in range(2000)]))
                print(f"{mu:.2f}  {sd:.2f}  {delta:.2f}  | " + " | ".join(f"{v:.3f}" for v in row))

    # ---------- e-process design sweep: futility x n_max ----------
    print("\n== E-process design sweep (mu=0.15, sd=0.07): null commit prob / mean instances; power at delta ==")
    print("futility n_max | P(commit|null) inst(null) | pow d=.05 inst | pow d=.10 inst | pow d=.20 inst")
    for fut in (0.0, 0.25, 0.5):
        for nmax in (120, 200):
            null_c, null_u = [], []
            for _ in range(3000):
                pf = draw_p(rng, 0.15, 0.07, nmax)
                ok, u = eprocess_commit(rng, pf, pf, nmax, LAMBDA, ALPHA, fut)
                null_c.append(ok); null_u.append(u)
            row = f"{fut:.2f}     {nmax:4d}  |    {np.mean(null_c):.3f}        {np.mean(null_u):5.0f}    "
            for d in DELTAS:
                c, u = [], []
                for _ in range(3000):
                    pf = draw_p(rng, 0.15, 0.07, nmax)
                    ok, uu = eprocess_commit(rng, pf, np.clip(pf + d, 0, 1), nmax, LAMBDA, ALPHA, fut)
                    c.append(ok); u.append(uu)
                row += f"|  {np.mean(c):.2f}  {np.mean(u):4.0f}  "
            print(row)

    # ---------- budget arithmetic ----------
    print("\n== Budget arithmetic (trajectories) ==")
    cells, seeds = 2, 2
    # Fork window G2 null: greedy_stored needs only candidate evals; greedy_fresh adds incumbent evals;
    # e-process null capped at 15 rounds x mean instances used (from sim at mu=0.15, sd=0.07, with futility).
    e_inst = null_table[(0.15, 0.07)][4]
    g2 = cells * seeds * (ROUNDS * N_DEV + ROUNDS * N_DEV) + cells * seeds * 15 * e_inst
    # G1 characterization: 3 arms x 8 candidates x 40 dev + held-out 60 x 3 arms, per cell
    g1 = cells * (3 * 8 * N_DEV + 3 * 60)
    print(f"Fork window: G1 characterization {g1:.0f} + G2 null {g2:.0f} = {g1+g2:.0f}")
    for rate in (100, 200, 320):
        h = (g1 + g2) / rate
        print(f"  at {rate} traj/h: {h:.0f} h = {h*2.18:.0f} CNY")
    # P3-P4 main design
    G, K = 6, 8
    greedy_arm = G * K * N_DEV + 5 * 200          # ~5 commits/run audited at n=200
    e_arm = G * K * e_inst + 2 * 200               # ~2 commits/run audited
    bon = greedy_arm                               # matched-compute best-of-N baseline
    heldout = 3 * 200
    per_run = greedy_arm + e_arm + bon + heldout
    total = per_run * cells * seeds
    print(f"P3-P4 T2 main: per cell-seed {per_run:.0f}; x{cells} cells x{seeds} seeds = {total:.0f}")
    for rate in (100, 200, 320):
        h = total / rate
        print(f"  at {rate} traj/h: {h:.0f} h = {h*2.18:.0f} CNY")


if __name__ == "__main__":
    main()
