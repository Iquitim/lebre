"""Layer M1 of LEBRE v0.53: the precision expert E and its combination with the core forecast L (transcribed from the
promoted research prototype, commit 4a2620e: expert of draft 2 + AdaHedge combination of draft 5, keeping the order of
every floating-point operation and the operation count).

E = clip(w . z, L +- CLIP_K * max(sigma_L, floor)), z = [1, y(t-1), y(t-2), means of the lag bands [0], [1], [2-3],
[4-7], [8-15] of the m = min(d, 5) inputs most correlated with the target], weights by recursive least squares with
forgetting, updated every EVERY learned targets. L' = (1 - omega) L + omega E, omega from AdaHedge on raw squared
errors, with a shadow entry of N_MIN updates."""
import math
from collections import deque

import numpy as np

from ._aggregation import FP_ADAHEDGE_STEP, AdaHedge
from ._core import CLIP_K, EVERY, LAM, N_MIN, T_MAX

M_MAX = 5
BANDS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15))
LAMBDA_RLS, DELTA_RLS = 0.999, 100.0


class PrecisionExpert:
    def __init__(self, d):
        self.d = int(d)
        self.m = np.zeros(5 * self.d).reshape(5, self.d)    # exponential means of x, x^2, x*y (per input), rows 3-4 unused
        self.my = 0.0; self.my2 = 0.0; self.started = False
        self.sel = None; self.mu = None; self.sd = None; self.muy = None; self.sdy = None
        self.w = None; self.P = None
        self.last_x = np.zeros(self.d); self.has_x = np.zeros(self.d, bool)
        self.n_obs = 0; self.n_sel = 0; self.n_rls = 0
        self.e2_core = None
        self.fp = 0.0; self.z = None; self.e = None; self._x = None; self._floor = 0.0
        self.buf = deque(maxlen=16)
        self.bufz = deque(maxlen=16)                          # the same rows, standardised (chosen inputs only)
        self.yobs = deque(maxlen=2)
        self.ag = AdaHedge(2)                                 # [core, expert]

    # ----------------------------------------------------------------- inputs
    def _fill(self, x):
        x = np.asarray(x, float).copy()
        ok = np.isfinite(x)
        self.last_x[ok] = x[ok]; self.has_x |= ok
        x[~ok] = self.last_x[~ok]
        x[~self.has_x] = 0.0
        return x

    # ----------------------------------------------------------------- forecast
    def predict(self, x, L):
        """L' (the combination with the core forecast L); sets self.e (None while E does not exist)."""
        x = self._fill(x)
        self._x = x
        self.buf.append(x)
        if self.sel is None:
            self.z = None; self.e = None
            return L
        self.bufz.append((x[self.sel] - self.mu) / self.sd)   # only the new row is standardised
        H = np.asfortranarray(list(reversed(self.bufz)))     # newest first, column order (summation order of the means)
        bands = [H[a:b + 1].mean(0) if len(H) > a else np.zeros(len(self.sel)) for a, b in BANDS]
        ys = list(self.yobs)[::-1] + [self.muy] * (2 - len(self.yobs))
        z = np.concatenate([[1.0], (np.array(ys[:2]) - self.muy) / self.sdy, np.ravel(np.array(bands).T)])
        self.z = z
        k = len(z)
        self.fp += 2 * k + 18 * len(self.sel) + 4             # prediction 2k; new row 2m; band means 16m; target lags 4
        if self.n_rls < k:                                    # regression not yet determined
            self.e = None
            return L
        self.e = float(self.w @ z)
        if self.e2_core is None:                              # no scale yet: the expert does not enter
            self.e = None
            return L
        B = CLIP_K * max(math.sqrt(self.e2_core), self._floor)
        self.e = min(max(self.e, L - B), L + B)
        self.fp += 4
        if self.ag.n < N_MIN:                                 # shadow: AdaHedge learns, the output does not use it yet
            return L
        w = self.ag.weights()
        return float(w[0] * L + w[1] * self.e)

    def weight(self):
        """current weight omega of the expert in L'"""
        return float(self.ag.weights()[1])

    # ----------------------------------------------------------------- learning
    def _screen(self, x, y):
        a = 1.0 - LAM
        if not self.started:
            self.m[0] = x; self.m[1] = x * x; self.m[2] = x * y; self.my = y; self.my2 = y * y; self.started = True
        else:
            self.m[0] += a * (x - self.m[0]); self.m[1] += a * (x * x - self.m[1]); self.m[2] += a * (x * y - self.m[2])
            self.my += a * (y - self.my); self.my2 += a * (y * y - self.my2)
        self.fp += 5 * self.d + 4

    def _select(self):
        vx = np.maximum(self.m[1] - self.m[0] ** 2, 0.0)
        vy = max(self.my2 - self.my ** 2, 0.0)
        cov = self.m[2] - self.m[0] * self.my
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((vx > 0) & (vy > 0), np.abs(cov) / np.sqrt(vx * vy), 0.0)
        sel = np.sort(np.argsort(-r, kind="stable")[:min(M_MAX, self.d)])
        if self.sel is None or not np.array_equal(sel, self.sel):
            self.sel = sel
            self.mu = self.m[0][sel].copy(); self.sd = np.sqrt(np.maximum(vx[sel], 1e-24))
            self.muy = self.my; self.sdy = math.sqrt(max(vy, 1e-24))
            k = 3 + len(BANDS) * len(sel)
            self.w = np.zeros(k); self.P = np.eye(k) * DELTA_RLS; self.n_rls = 0
            self.ag = AdaHedge(2)                             # new expert: the combination restarts
            self.bufz = deque(((row[sel] - self.mu) / self.sd for row in self.buf), maxlen=16)
            self.fp += 2 * len(self.buf) * len(sel)           # standardisation of the buffer with the new set
        self.n_sel = self.n_obs
        self.fp += 6 * self.d

    def _rls(self, z, y):
        Pz = self.P @ z
        g = Pz / (LAMBDA_RLS + z @ Pz)
        self.w = self.w + g * (y - self.w @ z)
        P = (self.P - np.outer(g, Pz)) / LAMBDA_RLS
        iu = np.triu_indices(len(z), 1)                       # each pair (i, j) symmetrised once; equal bit for bit
        s = (P[iu] + P.T[iu]) / 2                             # to (P + P.T) / 2
        P[iu] = s
        P[(iu[1], iu[0])] = s
        self.P = P
        if not np.isfinite(self.P).all() or np.diag(self.P).min() <= 0:
            self.P = np.eye(len(z)) * DELTA_RLS
        self.n_rls += 1
        k = len(z)
        self.fp += 180
        self.fp += 6 * k * k + 5 * k - 180                    # Pz 2k^2 - k; gain and weights ~7k; P update 3k^2; symmetry

    def observe(self, y, L_core, learn, floor):
        """y observed (float); L_core = core forecast of this step; learn = step outside quarantine and input contract."""
        e_core = (y - L_core) ** 2
        if self.e is not None:
            le = (y - self.e) ** 2
            if math.isfinite(e_core) and math.isfinite(le):
                self.ag.update((e_core, le))
            self.fp += FP_ADAHEDGE_STEP
        self.e2_core = e_core if self.e2_core is None else self.e2_core + (1 - LAM) * (e_core - self.e2_core)
        self._floor = floor
        if learn:
            self.n_obs += 1
            if self.n_obs % EVERY == 0:
                self._screen(self._x, y)
                if self.z is not None:
                    self._rls(self.z, y)
            if self.n_obs >= N_MIN and (self.sel is None or self.n_obs - self.n_sel >= T_MAX):
                self._select()
        self.yobs.append(float(y))
