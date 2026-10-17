"""Regression of LEBRE v0.53 (structural core + M1 + M2), i.e. `Lebre` with the three v0.54 changes switched off,
against the promoted research implementation of v0.53: lebre 0.3.0 with dormancy, cheap_equivalents and
triggered_removal False must reproduce lebre 0.2.0.

The fixtures (tests/fixtures_v053/) were produced by dev/make_fixtures_v053.py from the promoted prototype (commit
4a2620e, canonical call). They reuse the seven scenarios of tests/fixtures/ and add three where the new layers act:
joint dynamics (M1 takes over), returns with reference zero and a shock, and a random-walk level with a target gap.

Two levels, as for the core: bit for bit in the reference environment (Windows, CPython 3.11, NumPy 2.2.5); elsewhere,
identical decisions (core and gate audit) and forecasts within 1e-9 of their scale. A third test checks that
core_only=True with the v0.54 and v0.55 core changes off reproduces lebre 0.1.0 exactly.
"""
import functools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from lebre import Lebre, LebreCore

V053 = dict(dormancy=False, cheap_equivalents=False, triggered_removal=False, anchored=False)

FIX1 = Path(__file__).parent / "fixtures"
FIX = Path(__file__).parent / "fixtures_v053"
NAMES = sorted(p.stem for p in FIX.glob("*.npz"))
REFERENCE_ENV = (platform.system() == "Windows" and sys.version_info[:2] == (3, 11) and np.__version__ == "2.2.5")


def _inputs(name, meta):
    Z = np.load(FIX1 / f"{name}.npz") if meta["inputs"] == "fixtures" else np.load(FIX / f"{name}.npz")
    return Z["X"], Z["y"], Z["q"]


@functools.lru_cache(maxsize=None)                       # each scenario runs once for both test levels
def _run(name):
    meta = json.load(open(FIX / f"{name}.json", encoding="utf-8")); E = np.load(FIX / f"{name}.npz")
    X, y, q = _inputs(name, meta)
    m = Lebre(X.shape[1], season=meta["season"], season2=meta["season2"], reference=meta["reference"], **V053)
    f, lo, hi = (np.empty(len(y)) for _ in range(3))
    for t in range(len(y)):
        fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
        f[t], lo[t], hi[t] = fc.value, fc.lower, fc.upper
    return E, meta, m, f, lo, hi


@pytest.mark.parametrize("name", NAMES)
def test_same_decisions_and_forecasts(name):
    E, meta, m, f, lo, hi = _run(name)
    assert [(e.t, e.outcome, e.change, e.samples) for e in m.events] == [(t, o, c, n) for t, o, c, le, n in meta["events"]]
    le_got = np.array([np.nan if e.log_evidence is None else e.log_evidence for e in m.events], float)
    le_exp = np.array([np.nan if e[3] is None else e[3] for e in meta["events"]], float)
    assert np.allclose(le_got, le_exp, rtol=0, atol=0.011, equal_nan=True)
    for got, key in ((f, "forecast"), (lo, "lower"), (hi, "upper")):
        exp = E[key]; ok = np.isfinite(exp)
        assert np.array_equal(np.isfinite(got), ok)
        assert np.allclose(got[ok], exp[ok], rtol=0, atol=1e-9 * np.nanmax(np.abs(exp))), f"{name}/{key}"
    assert m.cost_per_step == pytest.approx(meta["cost_per_step"], rel=1e-12)
    assert m.coverage == meta["coverage"]
    assert m.layer_weights["m1"] == pytest.approx(meta["m1_weight"], abs=1e-9)


@pytest.mark.skipif(not REFERENCE_ENV, reason="bit-for-bit equality is defined in the reference environment only")
@pytest.mark.parametrize("name", NAMES)
def test_bit_for_bit_in_reference_environment(name):
    E, meta, m, f, lo, hi = _run(name)
    assert np.array_equal(f, E["forecast"])
    assert np.array_equal(lo, E["lower"], equal_nan=True) and np.array_equal(hi, E["upper"], equal_nan=True)
    assert m.cost_per_step == meta["cost_per_step"] and m.layer_weights["m1"] == meta["m1_weight"]


@pytest.mark.parametrize("name", sorted(p.stem for p in FIX1.glob("*.npz")))
def test_core_only_is_lebre_0_1_0(name):
    Z = dict(np.load(FIX1 / f"{name}.npz")); meta = json.load(open(FIX1 / f"{name}.json", encoding="utf-8"))   # load once
    X, y, q = Z["X"], Z["y"], Z["q"]
    a = Lebre(X.shape[1], season=meta["season"], season2=meta["season2"], core_only=True, cheap_equivalents=False,
              triggered_removal=False, anchored=False)
    f = np.array([a.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), bool(q[t])).value for t in range(len(y))])
    assert isinstance(a.core, LebreCore) and a.layer_weights is None
    assert [(e.t, e.outcome, e.samples) for e in a.events] == [(t, o, n) for t, o, k, ad, r, le, n in meta["events"]]
    assert a.cost_per_step * a.n_steps == pytest.approx(meta["fp_total"], rel=0, abs=1e-6)
    if REFERENCE_ENV:
        assert np.array_equal(f, Z["forecast"])
    else:
        assert np.allclose(f, Z["forecast"], rtol=0, atol=1e-9 * np.nanmax(np.abs(Z["forecast"])))
