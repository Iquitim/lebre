"""Bit-for-bit regression against the frozen research implementation of LEBRE v0.52-r1.

The fixtures were produced by dev/make_fixtures.py from the frozen research code (causal input standardiser +
LebreV052H, canonical configuration). They cover: acceptance of inputs and of the residual state, hierarchical splits,
swaps with four active units, removal tests, target gaps, inputs far outside the contract, external quarantine,
daily and weekly declared cycles, and a structureless target.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from lebre import Lebre

FIX = Path(__file__).parent / "fixtures"
NAMES = sorted(p.stem for p in FIX.glob("*.npz"))
KIND = {"add": "add", "rem": "remove", "swap": "swap", "split": "split"}


def _unit(model, s):
    return None if s == "None" else model.unit_name(eval(s))   # keys are stored as repr() of tuples of ints/str


@pytest.mark.parametrize("name", NAMES)
def test_bit_for_bit(name):
    Z = np.load(FIX / f"{name}.npz"); meta = json.load(open(FIX / f"{name}.json", encoding="utf-8"))
    X, y, q = Z["X"], Z["y"], Z["q"]
    m = Lebre(X.shape[1], season=meta["season"], season2=meta["season2"])
    f = np.empty(len(y)); half = np.empty(len(y))
    for t in range(len(y)):
        fc = m.step(X[t], None if not np.isfinite(y[t]) else float(y[t]), quarantine=bool(q[t]))
        f[t] = fc.value; half[t] = fc.upper - fc.value if np.isfinite(fc.upper) else np.nan
    assert np.array_equal(f, Z["forecast"]), f"{name}: forecasts differ (max |diff| {np.nanmax(np.abs(f - Z['forecast']))})"
    ok = np.isfinite(Z["qhat"])
    assert np.array_equal(np.isfinite(half), ok)
    assert np.allclose(half[ok], Z["qhat"][ok], rtol=0, atol=1e-9 * np.nanmax(np.abs(Z["qhat"])))
    got = [(e.t, e.outcome, e.change, e.added, e.removed, e.log_evidence, e.samples) for e in m.events]
    exp = [(t, o, KIND[k], _unit(m, a), _unit(m, r), le, n) for t, o, k, a, r, le, n in meta["events"]]
    assert got == exp
    assert m.cost_per_step * m.n_steps == pytest.approx(meta["fp_total"], rel=0, abs=1e-6)
    assert m.coverage == pytest.approx(meta["coverage"], rel=0, abs=0)
    assert sorted(m.structure) == sorted(_unit(m, s) for s in meta["final_structure"])
