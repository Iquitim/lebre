#!/usr/bin/env python3
"""make_fixtures_v055.py — generates tests/fixtures_v055/ from the FROZEN research implementation of LEBRE v0.55.

Requires a checkout of the research record (https://github.com/Iquitim/lebre-research) at commit b0522a0 or later (the
prototype folder is frozen); set LEBRE055_PROTO to its folder experiments/LEBRE-V0.55-PROTO-01 and run, from this
repository's root:  python dev/make_fixtures_v055.py
For the thirteen scenarios of tests/fixtures_v054/ (inputs reused) and three new ones (seeds 5591-5593: an integrated
target whose differences are driven by an input; a random walk with three inputs that also wander and are unrelated to
it; an integrated target driven by one input, with three unrelated wandering inputs), it runs the canonical call of the
frozen prototype (Lebre055(..., variant="B10"), on top of the canonical v0.54) and stores the expected forecasts,
interval bounds, the event log (core and gate audit), the analytic cost, the coverage, the final M1 weight, the dormancy
state, the number of accepted triggered removals, the final weights wS and wI of the core and the reports of equivalent
inputs.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["LEBRE055_PROTO"])
from lebre055.model055 import Lebre055  # noqa: E402

TESTS = HERE.parent / "tests"
SRC = {k: TESTS / k for k in ("fixtures", "fixtures_v053", "fixtures_v054")}
OUT = TESTS / "fixtures_v055"
QUERY_AT = 5003                  # step after which every report of equivalent inputs is stored (also at the end)


def new_scenarios():
    S = {}
    T = 9000
    rng = np.random.default_rng(5591)                           # I(1): the differences follow an input
    x = np.zeros((T, 2)); e = rng.standard_normal((T, 2))
    for t in range(1, T):
        x[t] = 0.5 * x[t - 1] + e[t]
    dy = np.r_[0.0, 0.0, 0.5 * x[:-2, 0]] + 0.5 * rng.standard_normal(T)
    S["integrated_with_input"] = dict(X=x, y=np.cumsum(dy), q=np.zeros(T, bool), season=None, season2=None,
                                      reference=None)
    rng = np.random.default_rng(5592)                           # random walk and unrelated wandering inputs
    X = np.cumsum(rng.standard_normal((T, 3)), axis=0)
    S["random_walk_wandering_inputs"] = dict(X=X, y=np.cumsum(rng.standard_normal(T)), q=np.zeros(T, bool), season=None,
                                             season2=None, reference=None)
    rng = np.random.default_rng(5593)                           # I(1) driven by x0, plus three wandering decoys
    x0 = np.zeros(T); e = rng.standard_normal(T)
    for t in range(1, T):
        x0[t] = 0.5 * x0[t - 1] + e[t]
    X = np.column_stack([x0, np.cumsum(rng.standard_normal((T, 3)), axis=0)])
    dy = np.r_[0.0, 0.0, 0.5 * x0[:-2]] + 0.5 * rng.standard_normal(T)
    S["integrated_with_decoys"] = dict(X=X, y=np.cumsum(dy), q=np.zeros(T, bool), season=None, season2=None,
                                       reference=None)
    return S


def reports(m, d):
    return {u: m.equivalent_inputs(u) for u in [f"x{i}" for i in range(d)] + ["own_past"]}


def main():
    OUT.mkdir(exist_ok=True)
    S = {}
    for p in sorted(SRC["fixtures_v054"].glob("*.npz")):
        meta = json.load(open(p.with_suffix(".json"), encoding="utf-8"))
        where = "fixtures_v054" if meta["inputs"] == "here" else meta["inputs"]
        Z = np.load(SRC[where] / p.name)
        S[p.stem] = dict(X=Z["X"], y=Z["y"], q=Z["q"], season=meta["season"], season2=meta["season2"],
                         reference=meta["reference"], inputs=where, query=bool(meta["equivalents"]))
    S.update(new_scenarios())
    for name, sc in S.items():
        X, y, q = sc["X"], sc["y"], sc["q"]
        m = Lebre055(X.shape[1], season=sc["season"], season2=sc["season2"], reference=sc["reference"], variant="B10")
        f, lo, hi = (np.empty(len(y)) for _ in range(3))
        eq = {}
        for t in range(len(y)):
            fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
            f[t], lo[t], hi[t] = fc.value, fc.lower, fc.upper
            if sc.get("query") and t + 1 == QUERY_AT:
                eq[str(t + 1)] = reports(m, X.shape[1])
        if sc.get("query"):
            eq["end"] = reports(m, X.shape[1])
        arrays = dict(forecast=f, lower=lo, upper=hi)
        if sc.get("inputs") is None:
            arrays.update(X=X, y=y, q=q)
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        ev = [[e.t, e.outcome, e.change, None if e.log_evidence is None else float(e.log_evidence), e.samples] for e in m.events]
        st = m.m1_state; c = m.core._core
        meta = dict(inputs=sc.get("inputs", "here"), season=sc["season"], season2=sc["season2"], reference=sc["reference"],
                    events=ev, cost_per_step=m.cost_per_step, coverage=m.coverage, m1_weight=m.layer_weights["m1"],
                    dormancy={k: st[k] for k in ("dormant", "sleeps", "wakes", "trials", "dormant_steps")},
                    triggered_removals=c.n_acc_rem, wS=c.wS, wI=c.wI, structure=m.structure, equivalents=eq or None)
        (OUT / f"{name}.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
        print(name, len(y), "events", len(ev), "wS", round(c.wS, 3), "wI", round(c.wI, 3), "structure", m.structure,
              "cost", round(meta["cost_per_step"], 1), flush=True)


if __name__ == "__main__":
    main()
