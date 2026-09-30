import math

import numpy as np
import pytest

from lebre import ALGORITHM_VERSION, Event, Forecast, Lebre, __version__


def _data(T=3000, seed=0):
    rng = np.random.default_rng(seed); X = rng.standard_normal((T, 2))
    y = 0.8 * np.r_[np.zeros(3), X[:-3, 0]] + 0.3 * rng.standard_normal(T)
    return X, y


def test_versions():
    assert __version__ == "0.1.0" and ALGORITHM_VERSION == "LEBRE v0.52-r1"


def test_predict_observe_equals_step():
    X, y = _data()
    a, b = Lebre(2), Lebre(2)
    for t in range(len(y)):
        fa = a.predict(X[t]); a.observe(y[t])
        fb = b.step(X[t], y[t])
        assert fa.t == fb.t and np.array_equal([fa.value, fa.lower, fa.upper], [fb.value, fb.lower, fb.upper], equal_nan=True)


def test_forecast_and_interval():
    X, y = _data()
    m = Lebre(2)
    f0 = m.predict(X[0])
    assert isinstance(f0, Forecast) and math.isnan(f0.lower) and f0.t == 0
    m.observe(y[0])
    for t in range(1, len(y)):
        f = m.step(X[t], y[t])
    assert f.lower <= f.value <= f.upper
    assert 0.8 < m.coverage < 0.98


def test_call_order_is_enforced():
    m = Lebre(1)
    with pytest.raises(RuntimeError):
        m.observe(1.0)
    m.predict([0.0])
    with pytest.raises(RuntimeError):
        m.predict([0.0])


def test_input_validation():
    m = Lebre(2)
    with pytest.raises(ValueError):
        m.predict([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Lebre(0)


def test_missing_target_and_inputs():
    X, y = _data(500)
    m = Lebre(2)
    for t in range(len(y)):
        x = X[t].copy()
        if t % 50 == 0:
            x[1] = np.nan
        f = m.step(x, None if t % 7 == 0 else y[t])
        assert math.isfinite(f.value)


def test_events_and_structure_are_reported():
    X, y = _data(8000)
    m = Lebre(2)
    for t in range(len(y)):
        m.step(X[t], y[t])
    assert all(isinstance(e, Event) for e in m.events)
    acc = [e for e in m.events if e.outcome == "accepted"]
    assert any(e.added == "x0" for e in acc)
    assert "x0" in m.structure
    resp = m.response("x0")
    assert all(isinstance(seg, tuple) for seg, _ in resp)
    assert "x0" in m.equivalent_inputs("x0")
    assert 100 < m.cost_per_step < 2000


def test_save_load_roundtrip(tmp_path):
    X, y = _data(2000)
    a = Lebre(2, season=24)
    for t in range(1000):
        a.step(X[t], y[t])
    p = tmp_path / "state.pkl"; a.save(p)
    b = Lebre.load(p)
    for t in range(1000, 2000):
        fa, fb = a.step(X[t], y[t]), b.step(X[t], y[t])
        assert np.array_equal([fa.value, fa.lower, fa.upper], [fb.value, fb.lower, fb.upper], equal_nan=True)


def test_load_rejects_foreign_pickle(tmp_path):
    import pickle
    p = tmp_path / "x.pkl"; pickle.dump({"a": 1}, open(p, "wb"))
    with pytest.raises(ValueError):
        Lebre.load(p)
