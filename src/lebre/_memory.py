"""Memory expert M of LEBRE v0.52-r1: forecast of the target from its own past, relative to the last value
(mean reversion, last difference, seasonal naive and one-cycle increment, per-phase profile of increments, optional
second declared cycle), learned by NLMS with scale-relative floors.

Extracted from the frozen research implementation (lebre_v052.Memory and lebre_v052h.MemoryW); arithmetic is unchanged.
"""
import numpy as np


class Memory:
    def __init__(self, season=None, mu=0.05, a_prof=0.1, a_mean=0.01, eps_rel=1e-3):
        self.s, self.mu, self.ap, self.am, self.eps_rel = season, mu, a_prof, a_mean, eps_rel
        self.feats = ("mean", "d1", "sn", "si", "prof") if season else ("mean", "d1")
        self.Lb = 2 * season + 2 if season else 3
        self.buf = np.zeros(self.Lb); self.n = 0
        self.mew = None; self.G = np.zeros(season) if season else None
        self.w = np.zeros(len(self.feats)); self.phi = None; self.pp_ema = None; self.y2 = None; self.fp = 0.0

    def _b(self, k):
        return self.buf[(self.n - k) % self.Lb]

    def predict(self):
        if self.n < self.Lb:
            self.phi = None
            return self._b(1) if self.n else 0.0
        last, s = self._b(1), self.s
        v = [(self.mew - last) if self.mew is not None else 0.0, last - self._b(2)]
        if s:
            v += [self._b(s) - last, self._b(s - 1) - self._b(s), self.G[self.n % s]]
        self.phi = np.array(v); self.fp += 3 * len(v) + 1
        return last + float(self.w @ self.phi)

    def update(self, y, y_hat, learn=True):
        if y is None or not np.isfinite(y):
            y, learn = y_hat, False
        if learn and self.phi is not None:
            pp = float(self.phi @ self.phi)
            self.pp_ema = pp if self.pp_ema is None else self.pp_ema + 0.01 * (pp - self.pp_ema)
            self.y2 = y * y if self.y2 is None else self.y2 + 1e-4 * (y * y - self.y2)
            den = pp + self.eps_rel * self.pp_ema + 1e-4 * self.y2
            if den > 0:
                self.w += (self.mu * (y - y_hat) / den) * self.phi; self.fp += 4 * len(self.w) + 6
        if learn and self.s and self.n >= 1:
            ph = self.n % self.s; self.G[ph] += self.ap * ((y - self._b(1)) - self.G[ph]); self.fp += 4
        if learn:
            self.mew = y if self.mew is None else self.mew + self.am * (y - self.mew); self.fp += 3
        self.buf[self.n % self.Lb] = y; self.n += 1


class MemoryW(Memory):
    """memory with a second declared cycle (e.g. the week, 168 h, on top of the day, 24 h)"""

    def __init__(self, season, season2, **kw):
        super().__init__(season, **kw)
        self.s2 = season2
        self.feats = self.feats + ("sn2", "si2")
        self.Lb = max(self.Lb, season2 + 2)
        self.buf = np.zeros(self.Lb); self.w = np.zeros(len(self.feats))

    def predict(self):
        if self.n < self.Lb:
            self.phi = None
            return self._b(1) if self.n else 0.0
        last, s = self._b(1), self.s
        v = [(self.mew - last) if self.mew is not None else 0.0, last - self._b(2)]
        if s:
            v += [self._b(s) - last, self._b(s - 1) - self._b(s), self.G[self.n % s]]
        v += [self._b(self.s2) - last, self._b(self.s2 - 1) - self._b(self.s2)]
        self.phi = np.array(v); self.fp += 3 * len(v) + 1
        return last + float(self.w @ self.phi)
