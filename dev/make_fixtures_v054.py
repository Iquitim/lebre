#!/usr/bin/env python3
"""make_fixtures_v054.py — generates tests/fixtures_v054/ from the PROMOTED research implementation of LEBRE v0.54.

Requires a checkout of the research record (https://github.com/Iquitim/lebre-research) at commit f9906c4 or later (the
prototype is frozen); set LEBRE054_PROTO to its folder experiments/LEBRE-V0.54-PROTO-01 and run, from this repository's
root:  python dev/make_fixtures_v054.py
For the ten scenarios of tests/fixtures_v053/ (inputs reused) and three new ones (seeds 5461-5463: an input-driven target
where M1 sleeps and a joint dynamics that appears later is absorbed by the core, so M1 stays asleep; a relation that
vanishes for good and is removed by a triggered test; forty inputs in correlated groups, with the report of equivalent
inputs queried in the middle of an update cycle), it runs the canonical call of the promoted prototype (m5a=m5b=m4=True, m6=False) and stores the expected forecasts, interval
bounds, the event log (core and gate audit), the analytic cost, the coverage, the final M1 weight, the dormancy state,
the number of accepted triggered removals and the reports of equivalent inputs.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["LEBRE054_PROTO"])
from lebre054.model054 import Lebre054  # noqa: E402

CANONICAL = dict(m5a=True, m5b=True, m4=True, m6=False)
FIX1, FIX2, OUT = (HERE.parent / "tests" / f for f in ("fixtures", "fixtures_v053", "fixtures_v054"))
QUERY_AT = (5003, None)          # steps after which every report of equivalent inputs is stored (None: at the end)


def ar1(rng, rho, n, d):
    e = rng.standard_normal((n, d)) * np.sqrt(1 - rho ** 2); z = np.zeros((n, d)); z[0] = rng.standard_normal(d)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + e[t]
    return z


def lag(v, k):
    return np.r_[np.zeros(k), v[:-k]]


def new_scenarios():
    S = {}
    rng = np.random.default_rng(5461); T = 16000; X = ar1(rng, 0.6, T, 3); e = 0.3 * rng.standard_normal(T)
    y = np.zeros(T); drive = 0.8 * lag(X[:, 0], 2)
    for t in range(2, T):                                       # joint dynamics only after step 9000
        y[t] = (1.2 * y[t - 1] - 0.35 * y[t - 2] if t >= 9000 else 0.0) + drive[t] + e[t]
    S["late_joint_dynamics"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None, reference=None)
    rng = np.random.default_rng(5462); T = 16000; X = ar1(rng, 0.5, T, 3)
    y = np.where(np.arange(T) < 6000, 1.0, 0.0) * lag(X[:, 1], 3) + 0.4 * lag(X[:, 2], 1) + 0.3 * rng.standard_normal(T)
    S["relation_vanishes"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None, reference=None)
    rng = np.random.default_rng(5463); T = 9000; base = ar1(rng, 0.7, T, 10)
    X = np.repeat(base, 4, axis=1) + 0.1 * rng.standard_normal((T, 40))   # ten groups of four near-copies
    y = 0.7 * lag(base[:, 0], 1) + 0.5 * lag(base[:, 3], 2) + 0.3 * rng.standard_normal(T)
    S["many_inputs_equivalents"] = dict(X=X, y=y, q=np.zeros(T, bool), season=24, season2=None, reference=None,
                                        query=True)
    return S


def reports(m, d):
    return {u: m.equivalent_inputs(u) for u in [f"x{i}" for i in range(d)] + ["own_past"]}


def main():
    OUT.mkdir(exist_ok=True)
    S = {}
    for p in sorted(FIX2.glob("*.npz")):
        meta = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
        Z = np.load(FIX1 / p.name) if meta["inputs"] == "fixtures" else np.load(p)
        S[p.stem] = dict(X=Z["X"], y=Z["y"], q=Z["q"], season=meta["season"], season2=meta["season2"],
                         reference=meta["reference"], inputs="fixtures" if meta["inputs"] == "fixtures" else "fixtures_v053")
    S.update(new_scenarios())
    for name, sc in S.items():
        X, y, q = sc["X"], sc["y"], sc["q"]
        m = Lebre054(X.shape[1], season=sc["season"], season2=sc["season2"], reference=sc["reference"], **CANONICAL)
        f, lo, hi = (np.empty(len(y)) for _ in range(3))
        eq = {}
        for t in range(len(y)):
            fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
            f[t], lo[t], hi[t] = fc.value, fc.lower, fc.upper
            if sc.get("query") and t + 1 == QUERY_AT[0]:
                eq[str(t + 1)] = reports(m, X.shape[1])
        if sc.get("query"):
            eq["end"] = reports(m, X.shape[1])
        arrays = dict(forecast=f, lower=lo, upper=hi)
        if sc.get("inputs") is None:
            arrays.update(X=X, y=y, q=q)
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        ev = [[e.t, e.outcome, e.change, None if e.log_evidence is None else float(e.log_evidence), e.samples] for e in m.events]
        st = m.m1_state
        meta = dict(inputs=sc.get("inputs", "here"), season=sc["season"], season2=sc["season2"], reference=sc["reference"],
                    events=ev, cost_per_step=m.cost_per_step, coverage=m.coverage, m1_weight=m.layer_weights["m1"],
                    dormancy={k: st[k] for k in ("dormant", "sleeps", "wakes", "trials", "dormant_steps")},
                    triggered_removals=m.core._core.n_acc_rem, equivalents=eq or None)
        (OUT / f"{name}.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
        print(name, len(y), "events", len(ev), "m1", round(meta["m1_weight"], 3), "cost", round(meta["cost_per_step"], 1),
              "dormancy", meta["dormancy"], "triggered", meta["triggered_removals"],
              "superseded", sum(e[1] == "superseded" for e in ev), flush=True)


if __name__ == "__main__":
    main()
