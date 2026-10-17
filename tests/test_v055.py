"""Regression of the complete LEBRE v0.55 (anchored structural core with the innovation structure, the soft gate and the
one-step test scale, triggered removals and the cheap report of equivalent inputs + M1 with dormancy + M2) against the
frozen research implementation.

The fixtures (tests/fixtures_v055/) were produced by dev/make_fixtures_v055.py from the frozen prototype (commit b0522a0,
variant B10). They reuse the thirteen scenarios of tests/fixtures_v054/ and add three that exercise v0.55: an integrated
target whose differences follow an input, a random walk with unrelated wandering inputs, and an integrated target driven
by one input with three wandering decoys.

Two levels, as before: bit for bit in the reference environment (Windows, CPython 3.11, NumPy 2.2.5); elsewhere,
identical decisions (core and gate audit), dormancy, triggered removals, structure and reports, and forecasts within
1e-9 of their scale.
"""
import functools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from lebre import Lebre

TESTS = Path(__file__).parent
FIX = TESTS / "fixtures_v055"
NAMES = sorted(p.stem for p in FIX.glob("*.npz"))
REFERENCE_ENV = (platform.system() == "Windows" and sys.version_info[:2] == (3, 11) and np.__version__ == "2.2.5")


def _inputs(name, meta):
    Z = np.load((FIX if meta["inputs"] == "here" else TESTS / meta["inputs"]) / f"{name}.npz")
    return Z["X"], Z["y"], Z["q"]


def _reports(m, d):
    return {u: m.equivalent_inputs(u) for u in [f"x{i}" for i in range(d)] + ["own_past"]}


@functools.lru_cache(maxsize=None)                       # each scenario runs once for both test levels
def _run(name):
    meta = json.load(open(FIX / f"{name}.json", encoding="utf-8")); E = np.load(FIX / f"{name}.npz")
    X, y, q = _inputs(name, meta)
    m = Lebre(X.shape[1], season=meta["season"], season2=meta["season2"], reference=meta["reference"])
    f, lo, hi = (np.empty(len(y)) for _ in range(3))
    eq = {}
    for t in range(len(y)):
        fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
        f[t], lo[t], hi[t] = fc.value, fc.lower, fc.upper
        if meta["equivalents"] and str(t + 1) in meta["equivalents"]:
            eq[str(t + 1)] = _reports(m, X.shape[1])
    if meta["equivalents"]:
        eq["end"] = _reports(m, X.shape[1])
    return E, meta, m, f, lo, hi, eq


@pytest.mark.parametrize("name", NAMES)
def test_same_decisions_and_forecasts(name):
    E, meta, m, f, lo, hi, eq = _run(name)
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
    assert m.dormancy_state == meta["dormancy"]
    c = m.core._core
    assert c.n_acc_rem == meta["triggered_removals"]
    assert c.wS == pytest.approx(meta["wS"], abs=1e-9) and c.wI == pytest.approx(meta["wI"], abs=1e-9)
    assert m.structure == meta["structure"]
    assert eq == (meta["equivalents"] or {})


@pytest.mark.skipif(not REFERENCE_ENV, reason="bit-for-bit equality is defined in the reference environment only")
@pytest.mark.parametrize("name", NAMES)
def test_bit_for_bit_in_reference_environment(name):
    E, meta, m, f, lo, hi, eq = _run(name)
    assert np.array_equal(f, E["forecast"])
    assert np.array_equal(lo, E["lower"], equal_nan=True) and np.array_equal(hi, E["upper"], equal_nan=True)
    assert m.cost_per_step == meta["cost_per_step"] and m.layer_weights["m1"] == meta["m1_weight"]
    assert m.core._core.wS == meta["wS"] and m.core._core.wI == meta["wI"]


def test_the_scenarios_exercise_v055():
    metas = {n: json.load(open(FIX / f"{n}.json", encoding="utf-8")) for n in NAMES}
    assert any(mt["wI"] > 0.5 for mt in metas.values())                 # the innovation structure carries weight
    assert sum(mt["triggered_removals"] for mt in metas.values()) > 0
    assert any(mt["equivalents"] for mt in metas.values())
    assert sum(mt["dormancy"]["sleeps"] for mt in metas.values()) > 0
