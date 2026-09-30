"""Public API of the LEBRE online forecaster (implements LEBRE v0.52-r1)."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ._core import Core

_FORMAT = "lebre-state"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Forecast:
    """One-step-ahead forecast issued before the target of that step is observed."""
    value: float
    lower: float          # lower bound of the adaptive 90% interval (NaN until the interval is initialised)
    upper: float          # upper bound of the adaptive 90% interval
    t: int                # step index (0-based)


@dataclass(frozen=True)
class Event:
    """Audit record of one structural experiment outcome."""
    t: int                # step at which the decision was taken
    outcome: str          # "accepted", "closed" (not accepted; evidence kept) or "superseded"
    change: str           # "add", "remove", "swap" or "split"
    added: Optional[str]  # unit that enters (None for a removal)
    removed: Optional[str]  # unit that leaves (None unless remove/swap)
    log_evidence: Optional[float]  # log of the e-process at the decision
    samples: int          # evidence samples behind the decision


class _CausalScaler:
    """Causal exponential standardiser used in every LEBRE evaluation (rate 1e-4, from mean 0 / variance 1).
    Missing input values (NaN) are passed on as missing and do not update the statistics."""

    def __init__(self, d, alpha=1e-4, eps=1e-6):
        self.alpha, self.eps = float(alpha), float(eps)
        self.mean = np.zeros(d, dtype=np.float64); self.var = np.ones(d, dtype=np.float64)

    def transform(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    def update(self, x):
        ok = np.isfinite(x)
        if ok.all():
            delta = x - self.mean
            self.mean += self.alpha * delta
            self.var = (1.0 - self.alpha) * self.var + self.alpha * (x - self.mean) * delta
            self.var = np.maximum(self.var, 1e-4)
        else:
            delta = np.where(ok, x - self.mean, 0.0)
            new_mean = self.mean + self.alpha * delta
            new_var = np.maximum((1.0 - self.alpha) * self.var + self.alpha * (np.where(ok, x, 0.0) - new_mean) * delta, 1e-4)
            self.mean = np.where(ok, new_mean, self.mean); self.var = np.where(ok, new_var, self.var)


class Lebre:
    """Online one-step-ahead forecaster for a target driven by exogenous inputs.

    Parameters
    ----------
    n_inputs : number of exogenous inputs (the target's own past is added internally).
    season : period of the declared cycle in steps (e.g. 24 for hourly data with a daily cycle), or None.
    season2 : optional second declared cycle (e.g. 168 for the weekly cycle of hourly building data), or None.
    standardize : standardise the inputs causally (the evaluated configuration). Set False only if the inputs are
        already standardised by the caller with past-only statistics.

    Usage: for each step, ``f = model.predict(x)`` with the current inputs, then ``model.observe(y)`` once the target
    is known (``None``/NaN for a missing target). ``step(x, y)`` does both. Every structural change is recorded in
    ``events``.
    """

    def __init__(self, n_inputs: int, season: Optional[int] = None, season2: Optional[int] = None, standardize: bool = True):
        if n_inputs < 1:
            raise ValueError("n_inputs must be >= 1")
        self.n_inputs, self.season, self.season2, self.standardize = int(n_inputs), season, season2, bool(standardize)
        self._core = Core(self.n_inputs, season=season, season2=season2)
        self._scaler = _CausalScaler(self.n_inputs) if standardize else None
        self._last: Optional[Forecast] = None

    # ------------------------------------------------------------------ stepping
    def predict(self, x: Sequence[float]) -> Forecast:
        """Forecast of the current target from the current inputs x (length n_inputs)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {x.shape[0]}")
        if self._scaler is not None:
            xs = self._scaler.transform(x); self._scaler.update(x)
        else:
            xs = x
        c = self._core
        q = c.qhat
        f = c.predict(xs)
        lo, hi = (f - q, f + q) if q is not None else (math.nan, math.nan)
        self._last = Forecast(float(f), float(lo), float(hi), c.t)
        return self._last

    def observe(self, y: Optional[float], quarantine: bool = False) -> None:
        """Reveal the target of the step just predicted. y = None or NaN marks a missing target; quarantine=True
        marks a value that must not be learned from (e.g. flagged by the data source)."""
        self._core.observe(None if y is None else float(y), quarantine=bool(quarantine))

    def step(self, x: Sequence[float], y: Optional[float], quarantine: bool = False) -> Forecast:
        """predict(x) then observe(y); returns the forecast made before y was used."""
        f = self.predict(x)
        self.observe(y, quarantine)
        return f

    # ------------------------------------------------------------------ audit and structure
    def unit_name(self, key) -> str:
        if key is None:
            return None
        if key[0] == "in":
            return "own_past" if key[1] == self.n_inputs else f"x{key[1]}"
        if key[0] == "res":
            return "residual_state"
        if key[0] == "split":
            base = "own_past" if key[1] == self.n_inputs else f"x{key[1]}"
            return f"{base}[{key[2]}-{key[3]}]"
        return str(key)

    @property
    def events(self) -> list:
        kind = {"add": "add", "rem": "remove", "swap": "swap", "split": "split"}
        return [Event(int(t), out, kind[k], self.unit_name(a), self.unit_name(r), le, int(n))
                for t, out, k, a, r, le, n in self._core.events]

    @property
    def structure(self) -> list:
        """active units: names of the inputs (x0, x1, ..., own_past) and/or residual_state"""
        return [self.unit_name(u) for u in sorted(self._core.active)]

    def response(self, unit: str) -> list:
        """recovered response of an active input: [((lag_lo, lag_hi), weight per lag), ...] in standardised units"""
        for u in self._core.active:
            if self.unit_name(u) == unit and u[0] == "in":
                return self._core.response(u)
        raise KeyError(f"{unit} is not an active input")

    def equivalent_inputs(self, unit: str) -> list:
        """inputs whose causal correlation with `unit` is >= 0.95 (the accepted input may stand for any of them)"""
        for u in self._core.units:
            if self.unit_name(u) == unit:
                return [self.unit_name(("in", j)) for j in sorted(self._core.equivalents(u))]
        raise KeyError(unit)

    @property
    def n_steps(self) -> int:
        return self._core.t

    @property
    def coverage(self) -> float:
        """empirical coverage of the 90% interval so far"""
        c = self._core
        return c.hits_ok / c.hits_obs if c.hits_obs else math.nan

    @property
    def cost_per_step(self) -> float:
        """mean analytic floating-point operations per step so far (operation-count model of the research code)"""
        return self._core.fp_total() / self._core.t if self._core.t else math.nan

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """Save the full state (pickle). Only load files you trust: unpickling can execute code."""
        with open(Path(path), "wb") as fh:
            pickle.dump({"format": _FORMAT, "version": _FORMAT_VERSION, "model": self}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Lebre":
        """Load a state saved by `save`. Only load files you trust: unpickling can execute code."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, dict) or obj.get("format") != _FORMAT:
            raise ValueError("not a LEBRE state file")
        if obj.get("version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported state format version {obj.get('version')}")
        return obj["model"]

    def __repr__(self):
        return (f"Lebre(n_inputs={self.n_inputs}, season={self.season}, season2={self.season2}, "
                f"steps={self.n_steps}, structure={self.structure})")
