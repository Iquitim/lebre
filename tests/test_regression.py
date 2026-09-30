"""Regression against the frozen research implementation of LEBRE v0.52-r1.

The fixtures were produced by dev/make_fixtures.py from the frozen research code (causal input standardiser +
LebreV052H, canonical configuration). They cover: acceptance of inputs and of the residual state, hierarchical splits,
swaps with four active units, removal tests, target gaps, inputs far outside the contract, external quarantine,
daily and weekly declared cycles, and a structureless target.

Two levels:
- in the reference environment (Windows, CPython 3.11, NumPy 2.2.5 — where the fixtures were generated) the forecasts
  must be identical bit for bit;
- on any other platform or NumPy version, floating-point rounding differs in the last bits, so the structural decisions
  (every event, in order, with its evidence) must be identical and the forecasts must agree to a tight tolerance.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from lebre import Lebre

FIX = Path(__file__).parent / "fixtures"
NAMES = sorted(p.stem for p in FIX.glob("*.npz"))
KIND = {"add": "add", "rem": "remove", "swap": "swap", "split": "split"}
REFERENCE_ENV = (platform.system() == "Windows" and sys.version_info[:2] == (3, 11) and np.__version__ == "2.2.5")


def _unit(model, s):
    return None if s == "None" else model.unit_name(eval(s))   # keys are stored as repr() of tuples of ints/str


def _run(name):
    Z = np.load(FIX / f"{name}.npz"); meta = json.load(open(FIX / f"{name}.json", encoding="utf-8"))
    X, y, q = Z["X"], Z["y"], Z["q"]
    m = Lebre(X.shape[1], season=meta["season"], season2=meta["season2"])
    f = np.empty(len(y)); half = np.empty(len(y))
    for t in range(len(y)):
        fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
        f[t] = fc.value; half[t] = fc.upper - fc.value if np.isfinite(fc.upper) else np.nan
    return Z, meta, m, f, half


@pytest.mark.parametrize("name", NAMES)
def test_same_decisions_and_forecasts(name):
    Z, meta, m, f, half = _run(name)
    got = [(e.t, e.outcome, e.change, e.added, e.removed) for e in m.events]
    exp = [(t, o, KIND[k], _unit(m, a), _unit(m, r)) for t, o, k, a, r, le, n in meta["events"]]
    assert got == exp, f"{name}: structural decisions differ"
    assert [e.samples for e in m.events] == [n for *_, n in meta["events"]]
    le_got = np.array([np.nan if e.log_evidence is None else e.log_evidence for e in m.events], float)
    le_exp = np.array([np.nan if e[5] is None else e[5] for e in meta["events"]], float)
    assert np.allclose(le_got, le_exp, rtol=0, atol=0.011, equal_nan=True)          # stored rounded to 2 decimals
    scale = np.nanmax(np.abs(Z["forecast"]))
    assert np.allclose(f, Z["forecast"], rtol=0, atol=1e-9 * scale), f"{name}: max |diff| {np.nanmax(np.abs(f - Z['forecast']))}"
    ok = np.isfinite(Z["qhat"])
    assert np.array_equal(np.isfinite(half), ok)
    assert np.allclose(half[ok], Z["qhat"][ok], rtol=0, atol=1e-9 * np.nanmax(np.abs(Z["qhat"])))
    assert m.cost_per_step * m.n_steps == pytest.approx(meta["fp_total"], rel=0, abs=1e-6)
    assert m.coverage == pytest.approx(meta["coverage"], rel=0, abs=0)
    assert sorted(m.structure) == sorted(_unit(m, s) for s in meta["final_structure"])


@pytest.mark.skipif(not REFERENCE_ENV, reason="bit-for-bit equality is defined in the reference environment only")
@pytest.mark.parametrize("name", NAMES)
def test_bit_for_bit_in_reference_environment(name):
    Z, meta, m, f, half = _run(name)
    assert np.array_equal(f, Z["forecast"])
    assert [(e.t, e.outcome, e.change, e.added, e.removed, e.log_evidence, e.samples) for e in m.events] == \
           [(t, o, KIND[k], _unit(m, a), _unit(m, r), le, n) for t, o, k, a, r, le, n in meta["events"]]
