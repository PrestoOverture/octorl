"""Descriptive G0 decision sensitivity; never changes a registered gate.

Run from the repository root: .venv/bin/python artifacts/research/g0_decision_20260912/analyze.py
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEED = 20260912
N_SIM = 20000
N_DEV = 40
ROUNDS = 30


def rng_for(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{label}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def exact_greedy(mu: float) -> float:
    pmf = binom.pmf(np.arange(N_DEV + 1), N_DEV, mu)
    return float((1 - np.sum(pmf**2)) / 2)


def describe(rows: list[dict]) -> dict:
    k = np.array([row["K"] for row in rows])
    mu = float(k.mean() / 8)
    d = float(np.mean(2 * k * (8 - k) / (8 * 7)))
    var_raw = float((np.var(k / 8, ddof=1) - mu * (1 - mu) / 8) / (1 - 1 / 8)) if len(k) > 1 else None
    return {"groups": len(rows), "rollouts": 8 * len(rows), "successes": int(k.sum()),
            "mu": mu, "discordance_all_pairs": d,
            "discordance_independent_homogeneous_upper_at_mu": 2 * mu * (1 - mu),
            "discordance_any_coupling_upper_at_mu": 2 * mu,
            "variance_moment_raw": var_raw,
            "sd_moment_clamped": math.sqrt(max(0, var_raw)) if var_raw is not None else None,
            "K_histogram": dict(sorted(Counter(map(int, k)).items())),
            "unique_task_ids": len({json.dumps(row['task'], sort_keys=True) for row in rows}),
            "summed_group_wall_hours": sum(row["wall_time_s"] for row in rows) / 3600,
            "observed_rollouts_per_group_wall_hour": 8 * len(rows) * 3600 / sum(row["wall_time_s"] for row in rows)}


def simulate(label: str, cells: list[tuple[float, float]], seeds_per_cell: int = 2) -> dict:
    """Registered greedy mechanics, fixed 40-item dev set per cell/seed.

    Distinct simulation replicates draw fresh dev sets. The two seeds within
    each replicate share the cell's dev set, as specified by the manifest design.
    sd=0 is a homogeneous Bernoulli scenario, otherwise a Beta population.
    """
    rng = rng_for(label)
    fresh = np.zeros(N_SIM, dtype=int)
    stored = np.zeros(N_SIM, dtype=int)
    disc = np.zeros(N_SIM, dtype=int)
    for mu, sd in cells:
        if sd == 0:
            p = np.full((N_SIM, N_DEV), mu)
        else:
            concentration = mu * (1 - mu) / sd**2 - 1
            assert concentration > 0
            p = rng.beta(mu * concentration, (1 - mu) * concentration, size=(N_SIM, N_DEV))
        for _ in range(seeds_per_cell):
            incumbent = (rng.random(p.shape) < p).sum(axis=1)
            for _ in range(ROUNDS):
                candidate = rng.random(p.shape) < p
                reevaluated = rng.random(p.shape) < p
                cs = candidate.sum(axis=1)
                fresh += cs > reevaluated.sum(axis=1)
                stored += cs > incumbent
                incumbent = np.maximum(incumbent, cs)
                disc += (candidate != reevaluated).sum(axis=1)
    denom = len(cells) * seeds_per_cell * ROUNDS * N_DEV
    a = disc / denom >= .10
    b = (fresh >= 30) & (stored >= 3)
    def rate(flags):
        count = int(flags.sum())
        p = count / N_SIM
        return {"count": count, "probability": p, "mc_se": math.sqrt(p * (1-p) / N_SIM),
                "zero_event_95_upper": 1 - .05**(1 / N_SIM) if count == 0 else None}
    return {"cells_mu_sd": cells, "seeds_per_cell": seeds_per_cell,
            "dev_shared_across_seeds": True,
            "mean_discordance": float(np.mean(disc / denom)),
            "discordance_q05_q50_q95": np.quantile(disc / denom, [.05,.5,.95]).tolist(),
            "fresh_mean": float(fresh.mean()), "fresh_q05_q50_q95": np.quantile(fresh,[.05,.5,.95]).tolist(),
            "stored_mean": float(stored.mean()), "stored_q05_q50_q95": np.quantile(stored,[.05,.5,.95]).tolist(),
            "G2a": rate(a), "G2b": rate(b), "G2a_and_b": rate(a & b)}


def synthetic_checks() -> dict:
    # Known-answer iid null: all 28 pairings estimate 2*p*(1-p).
    rng = rng_for("synthetic")
    k = rng.binomial(8, .025, size=200000)
    d = 2 * k * (8-k) / 56
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    expected = 2 * .025 * .975
    assert abs(d.mean() - expected) <= se, (d.mean(), expected, se)
    # Exact enumeration agrees with the symmetry identity used for greedy.
    p = .125
    pmf = binom.pmf(np.arange(41), 40, p)
    enumerated = sum(pmf[i] * pmf[j] for i in range(41) for j in range(i))
    assert abs(enumerated - exact_greedy(p)) < 1e-12
    # Completely deterministic instances have zero within-instance discordance.
    assert np.all(2 * np.array([0,8]) * (8-np.array([0,8])) / 56 == 0)
    return {"seed": SEED, "null_mu": .025, "null_samples": len(k),
            "expected_discordance": expected, "measured_discordance": float(d.mean()),
            "SE": se, "within_one_SE": True, "greedy_exact_check": True,
            "deterministic_check": True}


def main():
    checks = synthetic_checks()
    buckets = {}
    for path in sorted((ROOT / "artifacts/p2/d1").glob("prequential_groups_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        key = rows[0]["bucket_id"]
        buckets[key] = describe(rows)
        buckets[key]["by_source"] = {src: describe([r for r in rows if r['task']['source']==src])
                                     for src in sorted({r['task']['source'] for r in rows})}
        buckets[key]["by_repo"] = {repo: describe([r for r in rows if r['task']['repo']==repo])
                                   for repo in sorted({r['task']['repo'] for r in rows})}
    c1 = buckets['c1-L0-single-function']
    c3 = buckets['c3-L0-single-function']
    scenarios = {
        'registered_low_mu_reference': ([(.08,0),(.08,0)],2),
        'A_old_point_estimates_homogeneous': ([(.125,0),(.025,0)],2),
        'A_old_point_estimates_sd_07': ([(.125,.07),(.025,.07)],2),
        'A_D1_point_estimates_homogeneous': ([(c1['mu'],0),(c3['mu'],0)],2),
        'A_D1_point_estimates_moment_sd': ([(c1['mu'],c1['sd_moment_clamped']),(c3['mu'],c3['sd_moment_clamped'])],2),
        'A_D1_mutation_only_point_estimates': ([(c1['by_source']['mutation']['mu'],c1['by_source']['mutation']['sd_moment_clamped']),(c3['mu'],c3['sd_moment_clamped'])],2),
        'B_old_c1_homogeneous': ([(.125,0)],2),
        'B_D1_c1_homogeneous': ([(c1['mu'],0)],2),
        'B_D1_c1_moment_sd': ([(c1['mu'],c1['sd_moment_clamped'])],2),
    }
    results = {name: simulate(name,*spec) for name,spec in scenarios.items()}
    inputs = list((ROOT/'docs').glob('*.md')) + list((ROOT/'artifacts/p2/d1').glob('prequential_groups_*.jsonl'))
    inputs += [ROOT/'artifacts/p2/bucket_table.json', ROOT/'artifacts/p2/d1/run_metadata.json',
               ROOT/'scripts/null_gate_power.py', Path(__file__).resolve()]
    output = {"status": "descriptive sensitivity, not a G2 outcome or preregistered test",
              "master_seed": SEED,"N_SIM":N_SIM,"N_DEV":N_DEV,"ROUNDS":ROUNDS,
              "numpy":np.__version__,"checks":checks,"D1":buckets,"simulations":results,
              "exact_B_fresh_at_old_mu": {"per_round":exact_greedy(.125),
                                         "P_at_least_30_in_60":float(binom.sf(29,60,exact_greedy(.125)))},
              "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"checks":checks,"D1":{k:{x:v[x] for x in ['mu','discordance_all_pairs','sd_moment_clamped','observed_rollouts_per_group_wall_hour']} for k,v in buckets.items()},
                      "simulations":results,"exact_B":output['exact_B_fresh_at_old_mu']},indent=2))


if __name__ == '__main__':
    main()
