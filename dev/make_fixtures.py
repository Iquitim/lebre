#!/usr/bin/env python3
"""make_fixtures.py — generates the regression fixtures of the `lebre` package from the FROZEN research implementation.

Requires a checkout of the research record (lebre-research, https://doi.org/10.5281/zenodo.23049103); run from
there as  python packages/lebre/dev/make_fixtures.py . For each synthetic scenario it runs the frozen pipeline used in
every evaluation (causal input standardiser + LebreV052H with the canonical configuration) and stores the raw inputs,
target, quarantine flags, expected forecasts, the interval half-width available at each forecast, the event log and the
analytic cost. The package tests require bit-for-bit equality with these references.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, ROOT / "experiments" / "LEBRE-V0.52-PROTO-01"):
    sys.path.insert(0, str(p))
from experiments.bench01.streams import CausalStandardScaler  # noqa: E402
from lebre_v052h import LebreV052H  # noqa: E402

CANONICAL = json.load(open(ROOT / "configs" / "lebre_v052_canonical.json", encoding="utf-8"))["config"]
OUT = HERE.parent / "tests" / "fixtures"


def ar1(rng, rho, n, d):
    e = rng.standard_normal((n, d)) * np.sqrt(1 - rho ** 2); z = np.zeros((n, d)); z[0] = rng.standard_normal(d)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + e[t]
    return z


def lag(v, k):
    return np.r_[np.zeros(k), v[:-k]]


def scenarios():
    S = {}
    rng = np.random.default_rng(5301); T = 20000; X = ar1(rng, 0.5, T, 3)
    y = 0.8 * lag(X[:, 0], 12) + 0.3 * np.mean([lag(X[:, 1], k) for k in range(4, 8)], axis=0) + 0.5 * rng.standard_normal(T)
    S["lag_and_band"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None)
    rng = np.random.default_rng(5302); T = 30000; X = ar1(rng, 0.5, T, 6)
    y = (0.8 * lag(X[:, 0], 2) + 0.7 * lag(X[:, 1], 5) + 0.6 * lag(X[:, 2], 9) + 0.5 * lag(X[:, 3], 14)
         + 0.45 * lag(X[:, 4], 20) + 0.5 * rng.standard_normal(T))
    S["many_inputs"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None)
    rng = np.random.default_rng(5303); T = 25000; X = ar1(rng, 0.3, T, 2)
    y = 0.7 * lag(X[:, 0], 9) - 0.7 * lag(X[:, 0], 13) + 0.5 * rng.standard_normal(T)
    S["opposite_lags"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None)
    rng = np.random.default_rng(5304); T = 20000; X = 5.0 + 2.0 * ar1(rng, 0.9, T, 2); t = np.arange(T)
    v = np.zeros(T); e = rng.standard_normal(T)
    for i in range(1, T):
        v[i] = 0.7 * v[i - 1] + 0.4 * e[i]
    y = 10 + 3 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 168) + 0.5 * lag(X[:, 0], 3) + v
    for a, b in ((3000, 3024), (9000, 9300), (15000, 15050)):
        y[a:b] = np.nan                                              # target gaps
    X[[5000, 12000, 12001], 1] += 200.0                              # inputs far outside the contract (|x| > 8 sigma)
    q = np.zeros(T, bool); q[7000:7010] = True                       # externally quarantined targets
    S["hourly_gaps"] = dict(X=X, y=y, q=q, season=24, season2=168)
    rng = np.random.default_rng(5306); T = 40000; X = ar1(rng, 0.5, T, 6); h = np.arange(T) >= T // 2
    y = (np.where(h, 0.0, 0.8) * lag(X[:, 0], 3) + 0.6 * lag(X[:, 1], 6) + 0.4 * lag(X[:, 2], 2) + 0.3 * lag(X[:, 3], 11)
         + np.where(h, 0.9, 0.0) * lag(X[:, 4], 4) + 0.5 * rng.standard_normal(T))
    S["regime_change"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None)
    rng = np.random.default_rng(5307); T = 30000; X = ar1(rng, 0.5, T, 3); h = np.arange(T) >= 10000
    y = np.where(h, 0.0, 0.9) * lag(X[:, 0], 4) + 0.6 * lag(X[:, 1], 7) + 0.5 * rng.standard_normal(T)
    S["effect_vanishes"] = dict(X=X, y=y, q=np.zeros(T, bool), season=None, season2=None)
    rng = np.random.default_rng(5305); T = 3000
    S["short_null"] = dict(X=rng.standard_normal((T, 1)), y=rng.standard_normal(T), q=np.zeros(T, bool), season=None, season2=None)
    return S


def run(sc):
    X, y, q = sc["X"], sc["y"], sc["q"]; T, d = X.shape
    scaler = CausalStandardScaler(d=d)
    m = LebreV052H(d=d, season=sc["season"], **dict(CANONICAL, season2=sc["season2"]))
    f = np.empty(T); qh = np.full(T, np.nan)
    for t in range(T):
        xs = scaler.transform(X[t]); scaler.update(X[t])
        qh[t] = m.qhat if m.qhat is not None else np.nan
        f[t] = m.step(xs, float(y[t]) if np.isfinite(y[t]) else None, quarantine=bool(q[t]))
    ev = [[int(e[0]), e[1], e[2], repr(e[3]), repr(e[4]), (None if e[5] is None else float(e[5])), int(e[6])] for e in m.events]
    return f, qh, ev, m.fp_total(), m.coverage(), sorted(repr(u) for u in m.active)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sc in scenarios().items():
        f, qh, ev, fp, cov, act = run(sc)
        np.savez_compressed(OUT / f"{name}.npz", X=sc["X"], y=sc["y"], q=sc["q"], forecast=f, qhat=qh)
        meta = {"season": sc["season"], "season2": sc["season2"], "events": ev, "fp_total": fp, "coverage": cov, "final_structure": act}
        json.dump(meta, open(OUT / f"{name}.json", "w", encoding="utf-8"), indent=1)
        kinds = sorted({(e[1], e[2]) for e in ev})
        print(f"{name}: T={len(f)} accepted={sum(e[1] == 'accepted' for e in ev)} kinds={kinds} final={act}")
