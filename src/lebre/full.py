"""Public API of the complete LEBRE (implements LEBRE v0.53: structural core + M1 + M2, canonical configuration).

The structural core is `lebre.model.Lebre` (LEBRE v0.52-r1, unchanged; exported as `LebreCore`). On top of it:
- M1 combines the core forecast L with a precision expert E (recursive regression on the target's past and on lag-band
  means of the most correlated inputs) by AdaHedge on squared errors: L' = (1 - omega) L + omega E.
- M2 starts the output from a declared trivial reference R (zero, persistence or seasonal naive) and mixes R and L'
  through two paths, H (AdaHedge) and W (one-way switch R -> L'), combined by a horizon-free (A,B)-Prod with losses
  normalised by a forgetting running maximum. An e-process with its own budget audits the switches (events only).
Transcribed from the promoted research prototype (lebre-research, experiments/LEBRE-V0.53-PROTO-01, commit 4a2620e,
canonical call saida="prod_q2", precisao="r5"), keeping the order of every floating-point operation."""
from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

from ._aggregation import FP_ADAHEDGE_STEP, FP_PROD_STEP, FP_SWITCH_STEP, AdaHedge, OneWaySwitch, ProdAB
from ._core import ALPHA_COV, CLIP_K, DECIDE_EVERY, EVERY, GAMMA_Q, LAM, N_MIN, SCALE_FLOOR, T_MAX
from ._engine import ChangeEngine
from ._precision import PrecisionExpert
from .model import Event, Forecast
from .model import Lebre as LebreCore

REFERENCES = ("zero", "persistence", "seasonal")
GATE_EPS, GATE_ALPHA = 0.002, 0.01
FP_GATE_STEP = 16             # clipped losses, increment and exponential means of the gate audit
FP_NORMALISATION = 3          # running maximum with forgetting and the two divisions
_FORMAT = "lebre-state"
_FORMAT_VERSION = 2


def _fin(v):
    return v if math.isfinite(v) else 1e300                  # a non-finite loss never propagates into the weights


class Lebre:
    """Online one-step-ahead forecaster for a target driven by exogenous inputs (LEBRE v0.53).

    Parameters
    ----------
    n_inputs : number of exogenous inputs (the target's own past is added internally).
    season : period of the declared cycle in steps (e.g. 24 for hourly data with a daily cycle), or None.
    season2 : optional second declared cycle (e.g. 168 for the weekly cycle of hourly building data), or None.
    standardize : standardise the inputs causally (the evaluated configuration).
    reference : trivial reference of layer M2 — "zero" (only for targets that are changes, e.g. financial returns),
        "persistence" (last observed value) or "seasonal" (value one season ago). None: "seasonal" if a season is
        declared, otherwise "persistence".
    core_only : True switches M1 and M2 off; the forecasts are then identical, bit for bit, to `LebreCore` (lebre 0.1.0).

    Usage: for each step, ``f = model.predict(x)`` with the current inputs, then ``model.observe(y)`` once the target
    is known (``None``/NaN for a missing target). ``step(x, y)`` does both.
    """

    def __init__(self, n_inputs: int, season: Optional[int] = None, season2: Optional[int] = None,
                 standardize: bool = True, reference: Optional[str] = None, core_only: bool = False):
        self.core = LebreCore(n_inputs, season=season, season2=season2, standardize=standardize)
        self.n_inputs, self.season, self.season2 = self.core.n_inputs, season, season2
        self.core_only = bool(core_only)
        if reference is None:
            reference = "seasonal" if season else "persistence"
        if reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}")
        if reference == "seasonal" and not season:
            raise ValueError("reference 'seasonal' needs a declared season")
        self.reference = reference
        if self.core_only:
            return
        self._m1 = PrecisionExpert(self.n_inputs)
        self._L_core = None
        self._hist = deque(maxlen=season) if reference == "seasonal" else None
        self._last_y = None
        self._mode = "REF"
        self._e2 = {"REF": None, "LEBRE": None}; self._sig = {"REF": 1.0, "LEBRE": 1.0}
        self._gate_engine = ChangeEngine(GATE_ALPHA, 2, 1, N_MIN, T_MAX)
        self._hyp = None; self._hkey = None; self._inst = 0; self._ep_n = 0; self._ep_S = 0.0
        self._gate_events = []; self.switches = 0; self._fp_extra = 0.0
        self._out = None; self._R = None; self._L = None; self._R_def = False
        self._prod = ProdAB(); self._H = AdaHedge(2); self._W = OneWaySwitch()
        self._h = None; self._w = None; self._N = 0.0
        self._e2o = None; self._sigo = 1.0; self._qhat = None; self._qacc = 0.0; self._hits_obs = 0; self._hits_ok = 0
        self._new_hypothesis()

    # ------------------------------------------------------------------ gate audit
    def _new_hypothesis(self):
        self._inst += 1
        direction = "promote" if self._mode == "REF" else "demote"
        self._hkey = ("gate", direction, self._inst)
        self._hyp = self._gate_engine.hypothesis(self._hkey, GATE_EPS)
        self._ep_n = 0; self._ep_S = 0.0

    def _review(self, t):
        h = self._hyp
        le = h.log_e(); self._fp_extra += 3 * len(h.lam)
        if le >= h.log_thr:
            new = "LEBRE" if self._mode == "REF" else "REF"
            self._gate_events.append(Event(int(t), "accepted", "gate", "adaptive" if new == "LEBRE" else "reference",
                                           None, round(float(le), 2), int(h.n)))
            self._gate_engine.n_accepted += 1
            self._gate_engine.hyps.pop(self._hkey, None)
            self._mode = new; self.switches += 1
            self._new_hypothesis()
        elif (self._ep_n >= N_MIN and self._ep_S <= 0.0) or self._ep_n >= T_MAX:
            self._gate_engine.hyps.pop(self._hkey, None)
            self._new_hypothesis()

    def _reference(self):
        if self.reference == "zero":
            return 0.0
        if self.reference == "seasonal" and len(self._hist) == self.season and math.isfinite(self._hist[0]):
            return float(self._hist[0])
        return float(self._last_y) if self._last_y is not None else math.nan

    # ------------------------------------------------------------------ stepping
    def predict(self, x: Sequence[float]) -> Forecast:
        """Forecast of the current target from the current inputs x (length n_inputs)."""
        fb = self.core.predict(x)
        if self.core_only:
            return fb
        self._L_core = fb.value
        self._L = self._m1.predict(x, fb.value)
        self._R = self._reference()
        self._R_def = math.isfinite(self._R) and (self.reference != "seasonal" or (
            len(self._hist) == self.season and math.isfinite(self._hist[0])))
        if self._R_def:
            wd, wm, sp = self._H.weights(), self._W.weights(), self._prod.s()
            self._h = float(wd[0] * self._R + wd[1] * self._L)
            self._w = float(wm[0] * self._R + wm[1] * self._L)
            out = sp * self._w + (1.0 - sp) * self._h
        else:
            out = self._L
        self._out = out
        q = self._qhat
        lo, hi = (out - q, out + q) if q is not None else (math.nan, math.nan)
        return Forecast(float(out), float(lo), float(hi), fb.t)

    def observe(self, y: Optional[float], quarantine: bool = False) -> None:
        """Reveal the target of the step just predicted. y = None or NaN marks a missing target; quarantine=True
        marks a value that must not be learned from."""
        if self.core_only:
            self.core.observe(y, quarantine)
            return
        c = self.core._core
        t0, quar, ysv = c.t, c.quar_until, c.y_sv
        y_ok = y is not None and math.isfinite(y)
        learn = y_ok and not quarantine
        floor = SCALE_FLOOR * math.sqrt(max(ysv, 0.0)) if t0 >= 200 else 0.0     # past only (before seeing y)
        self.core.observe(y, quarantine)
        if y_ok:
            self._m1.observe(float(y), self._L_core, (not quarantine) and not (t0 <= quar), floor)
        R, L = self._R, self._L
        if learn and math.isfinite(R) and math.isfinite(L):
            cur, alt = (R, L) if self._mode == "REF" else (L, R)
            B = CLIP_K * max(self._sig[self._mode], floor)
            lc = min((y - cur) ** 2 / (B * B), 1.0); la = min((y - alt) ** 2 / (B * B), 1.0)
            d = lc - la - GATE_EPS
            self._hyp.observe(d); self._ep_n += 1; self._ep_S += d
            if self._R_def:
                self._H.update((_fin((y - R) ** 2), _fin((y - L) ** 2)))
                self._fp_extra += FP_ADAHEDGE_STEP
                if self._e2["REF"] is not None:                      # W and Prod: scales from the past only
                    s2 = max(min(self._e2["REF"], self._e2["LEBRE"]), floor * floor, 1e-300)
                    self._W.update((_fin((y - R) ** 2 / (2.0 * s2)), _fin((y - L) ** 2 / (2.0 * s2))))
                    lh, lw = _fin((y - self._h) ** 2), _fin((y - self._w) ** 2)
                    self._N = max(self._N * LAM, lh, lw)
                    if self._N > 0:
                        self._prod.update(lh / self._N, lw / self._N)
                    self._fp_extra += FP_NORMALISATION
                    self._fp_extra += FP_SWITCH_STEP + FP_PROD_STEP
            for k, v in (("REF", R), ("LEBRE", L)):
                e2 = (y - v) ** 2
                self._e2[k] = e2 if self._e2[k] is None else self._e2[k] + (1 - LAM) * (e2 - self._e2[k])
            if t0 % EVERY == 0:
                for k in ("REF", "LEBRE"):
                    self._sig[k] = math.sqrt(max(self._e2[k], 1e-300))
            self._fp_extra += FP_GATE_STEP
        if y_ok:                                                        # output interval
            e = y - self._out
            self._e2o = e * e if self._e2o is None else self._e2o + (1 - LAM) * (e * e - self._e2o)
            if self._qhat is None:
                self._qhat = 1.645 * math.sqrt(self._e2o)
            miss = 1.0 if abs(e) > self._qhat else 0.0
            self._hits_obs += 1; self._hits_ok += miss == 0.0; self._qacc += miss - ALPHA_COV
            if t0 % EVERY == 0:
                self._sigo = math.sqrt(max(self._e2o, 1e-300))
                self._qhat = max(0.0, self._qhat + GAMMA_Q * self._sigo * self._qacc); self._qacc = 0.0
            self._last_y = float(y)
        if self._hist is not None:
            self._hist.append(float(y) if y_ok else (self._last_y if self._last_y is not None else math.nan))
        if t0 % DECIDE_EVERY == 0 and t0 > 0:
            self._review(t0)

    def step(self, x: Sequence[float], y: Optional[float], quarantine: bool = False) -> Forecast:
        """predict(x) then observe(y); returns the forecast made before y was used."""
        f = self.predict(x)
        self.observe(None if y is None or (isinstance(y, float) and math.isnan(y)) else y, quarantine)
        return f

    # ------------------------------------------------------------------ audit and structure
    @property
    def events(self) -> list:
        """structural changes of the core and, with change="gate", the switches documented by the gate audit"""
        if self.core_only:
            return self.core.events
        return sorted(self.core.events + self._gate_events, key=lambda e: e.t)

    @property
    def structure(self) -> list:
        return self.core.structure

    def response(self, unit: str) -> list:
        return self.core.response(unit)

    def equivalent_inputs(self, unit: str) -> list:
        return self.core.equivalent_inputs(unit)

    def unit_name(self, key) -> str:
        return self.core.unit_name(key)

    @property
    def layer_weights(self) -> Optional[dict]:
        """current weights of M1 and M2 (None with core_only): m1 = weight omega of the precision expert in L';
        h = [R, L'] weights in path H; w = [R, L'] weights in path W; s = fraction of the output given to W"""
        if self.core_only:
            return None
        return {"m1": self._m1.weight(), "m1_inputs": None if self._m1.sel is None else [f"x{i}" for i in self._m1.sel],
                "h": [float(v) for v in self._H.weights()], "w": [float(v) for v in self._W.weights()],
                "s": float(self._prod.s())}

    @property
    def n_steps(self) -> int:
        return self.core.n_steps

    @property
    def coverage(self) -> float:
        """empirical coverage of the 90% interval of the output so far"""
        if self.core_only:
            return self.core.coverage
        return self._hits_ok / self._hits_obs if self._hits_obs else math.nan

    @property
    def cost_per_step(self) -> float:
        """mean analytic floating-point operations per step so far (operation-count model of the research code)"""
        if self.core_only:
            return self.core.cost_per_step
        t = self.core.n_steps
        extra = self._fp_extra + self._m1.fp
        return self.core.cost_per_step + (extra / t if t else 0.0)

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        """Save the full state (pickle). Only load files you trust: unpickling can execute code."""
        with open(Path(path), "wb") as fh:
            pickle.dump({"format": _FORMAT, "version": _FORMAT_VERSION, "model": self}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Lebre":
        """Load a state saved by `save` (version 2: Lebre; version 1: LebreCore of lebre 0.1.0). Only load files you
        trust: unpickling can execute code."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, dict) or obj.get("format") != _FORMAT:
            raise ValueError("not a LEBRE state file")
        if obj.get("version") not in (1, _FORMAT_VERSION):
            raise ValueError(f"unsupported state format version {obj.get('version')}")
        return obj["model"]

    def __repr__(self):
        return (f"Lebre(n_inputs={self.n_inputs}, season={self.season}, season2={self.season2}, "
                f"reference={self.reference!r}, core_only={self.core_only}, steps={self.n_steps}, structure={self.structure})")
