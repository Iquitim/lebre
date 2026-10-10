#!/usr/bin/env python3
"""make_fixtures_v053.py — generates tests/fixtures_v053/ from the PROMOTED research implementation of LEBRE v0.53.

Requires a checkout of the research record (https://github.com/Iquitim/lebre-research) at the promoted commit 4a2620e
or later (the prototype is frozen); set LEBRE053_PROTO to its folder experiments/LEBRE-V0.53-PROTO-01 and run, from this
repository's root:  python dev/make_fixtures_v053.py
For the seven scenarios of tests/fixtures/ (inputs reused) and three new ones (seeds 5311-5313: joint dynamics where M1
takes over; returns with reference zero and a shock; a random-walk level with a target gap and reference persistence),
it runs the canonical call of the promoted prototype and stores the expected forecasts, interval bounds, the event log
(core and gate audit), the analytic cost, the coverage and the final M1 weight.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["LEBRE053_PROTO"])
from lebre053.model053 import Lebre053  # noqa: E402

CANONICAL = dict(porta=True, eps_porta=0.002, recriar=True, alpha_porta=0.01, observar_quarentena_entradas=True,
                 saida="prod_q2", precisao="r5")
REF = {None: None, "zero": "zero", "persistence": "persistencia", "seasonal": "sazonal"}
FIX1, OUT = HERE.parent / "tests" / "fixtures", HERE.parent / "tests" / "fixtures_v053"


def ar1(rng, rho, n, d):
    e = rng.standard_normal((n, d)) * np.sqrt(1 - rho ** 2); z = np.zeros((n, d)); z[0] = rng.standard_normal(d)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + e[t]
    return z


def lag(v, k):
    return np.r_[np.zeros(k), v[:-k]]


def new_scenarios():
    S = {}
    rng = np.random.default_rng(5311); T = 12000; X = ar1(rng, 0.6, T, 4); y = np.zeros(T); e = 0.3 * rng.standard_normal(T)
    drive = 0.6 * np.mean([lag(X[:, 0], k) for k in range(2, 4)], 0) + 0.4 * np.mean([lag(X[:, 1], k) for k in range(4, 8)], 0)
    for t in range(2, T):
        y[t] = 1.2 * y[t - 1] - 0.35 * y[t - 2] + drive[t] + e[t]
    S["joint_dynamics"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None, reference=None)
    rng = np.random.default_rng(5312); T = 6000; X = ar1(rng, 0.1, T, 3); y = 0.01 * rng.standard_t(3, T); y[4000] = -0.25
    S["returns_zero"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None, reference="zero")
    rng = np.random.default_rng(5313); T = 8000; X = ar1(rng, 0.95, T, 2)
    y = np.cumsum(0.5 * rng.standard_normal(T)) + 0.05 * X[:, 0]; y[2500:2520] = np.nan
    S["random_walk_level"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None, reference="persistence")
    return S


def main():
    OUT.mkdir(exist_ok=True)
    S = {}
    for p in sorted(FIX1.glob("*.npz")):
        Z = np.load(p); meta = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
        S[p.stem] = dict(X=Z["X"], y=Z["y"], q=Z["q"], season=meta["season"], season2=meta["season2"], reference=None, inputs="fixtures")
    S.update(new_scenarios())
    for name, sc in S.items():
        X, y, q = sc["X"], sc["y"], sc["q"]
        m = Lebre053(X.shape[1], season=sc["season"], season2=sc["season2"], referencia=REF[sc["reference"]], **CANONICAL)
        f, lo, hi = (np.empty(len(y)) for _ in range(3))
        for t in range(len(y)):
            fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
            f[t], lo[t], hi[t] = fc.value, fc.lower, fc.upper
        arrays = dict(forecast=f, lower=lo, upper=hi)
        if sc.get("inputs") != "fixtures":
            arrays.update(X=X, y=y, q=q)
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        ev = [[e.t, e.outcome, e.change, None if e.log_evidence is None else float(e.log_evidence), e.samples] for e in m.events]
        meta = dict(inputs=sc.get("inputs", "here"), season=sc["season"], season2=sc["season2"], reference=sc["reference"],
                    events=ev, cost_per_step=m.cost_per_step, coverage=m.coverage, m1_weight=m.precisao.peso())
        (OUT / f"{name}.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
        print(name, len(y), "events", len(ev), "m1", round(meta["m1_weight"], 3), "cost", round(meta["cost_per_step"], 1))


if __name__ == "__main__":
    main()
