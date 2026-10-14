"""Cheap report of equivalent inputs (LEBRE v0.54, M5b): the part of screening whose cost is quadratic in d and has no
effect on forecasts or decisions.

The groups of equivalent inputs (report only, `equivalent_inputs`) are updated with one sample every EQ_EVERY steps instead
of every SCREEN_EVERY, with the weight scaled (1e-3 x EQ_EVERY) to keep the same time window. EQ_EVERY is prime, so the
samples go through every phase of cycles such as 24, 168, 7 and 12. On the sample step the mean is updated and the centred
vector is stored; the matrix is then updated with it one row at a time on the following steps (up to ceil(d/30) rows per
step; row i updates the columns j >= i and is mirrored). At the end of each cycle the matrix is the one of a single full
update with that sample, bit for bit; a query of equivalents first applies the pending rows. The total cost is
3d(d+1)/2 + 2d per cycle and the per-step peak stays low. The rest of screening is the core's, unchanged, and the method
below is the original with this single change. Forecasts and decisions are identical to those of the original core.
Transcribed from the promoted research prototype (lebre-research, experiments/LEBRE-V0.54-PROTO-01/lebre054/screening.py,
commit f9906c4, revision 2 of errata 01), keeping the order of every floating-point operation."""
import numpy as np

from ._core import BANDS, POLES, SCREEN_EVERY, SCREEN_W, Core, halves

EQ_EVERY = 31                # prime: the updates go through every phase of cycles such as 24, 168, 7 and 12


class CheapEquivalenceCore(Core):
    def __init__(self, d, season=None, season2=None):
        super().__init__(d, season=season, season2=season2)
        self._eq_xc = None; self._eq_next = 0                        # centred vector of the cycle and next row to apply

    def _eq_rows(self, n):
        """applies the cycle's sample to the next n rows (row i: columns j >= i, mirrored); at the end of the cycle the
        matrix is bit for bit that of a full update (same sample and same arithmetic per element)"""
        if self._eq_xc is None:
            return
        gw, xc = 1e-3 * EQ_EVERY, self._eq_xc
        for i in range(self._eq_next, min(self._eq_next + n, self.d)):
            row = self.g_c[i, i:] + gw * (np.outer(xc[i:i + 1], xc[i:])[0] - self.g_c[i, i:])
            self.g_c[i, i:] = row; self.g_c[i:, i] = row
            self.fp["screening"] += 3 * (self.d - i)
        self._eq_next = min(self._eq_next + n, self.d)
        if self._eq_next >= self.d:
            self._eq_xc = None

    def _eq_flush(self):
        self._eq_rows(self.d)

    def equivalents(self, unit, thr=0.95):
        self._eq_flush()                                             # the query sees the matrix with the cycle complete
        return super().equivalents(unit, thr)

    def _screen_spread(self, res):
        w, se, t = SCREEN_W, SCREEN_EVERY, self.t
        act = {u[1] for u in self.active if u[0] == "in"}
        due = [i for i in range(self.d) if (t + i) % se == 0]
        x, x1, ra = self.H[0], self.H[1], 1e-3 * se
        for i in due:
            self.a_c0[i] += ra * (x[i] * x[i] - self.a_c0[i]); self.a_c1[i] += ra * (x[i] * x1[i] - self.a_c1[i])
            self.fp["screening"] += 6
            if i in act:
                continue
            a = float(self._a()[i]); pres = res - a * self.res_prev
            for b, (lo, hi) in enumerate(BANDS):
                ub = (self.bs[b, i] - a * (self.bs[b, i] - self.H[lo, i] + self.H[hi + 1, i])) / (hi - lo + 1)
                self.sc_xy[b, i] += w * (pres * ub - self.sc_xy[b, i]); self.sc_xx[b, i] += w * (ub * ub - self.sc_xx[b, i])
            self.sc_yy[i] += w * (pres * pres - self.sc_yy[i])
            self.fp["screening"] += 11 * len(BANDS) + 5
        if t % se == self.d % se:                                   # residual latent unit
            self.sr_xy += w * (res * self.r - self.sr_xy); self.sr_xx += w * (self.r * self.r - self.sr_xx)
            self.sr_yy += w * (res * res - self.sr_yy); self.fp["screening"] += 12 * len(POLES)
        if t % EQ_EVERY == (self.d + 1) % EQ_EVERY:                 # equivalence groups (reporting only)
            self._eq_flush()                                         # finishes the previous cycle, if rows are pending
            gw = 1e-3 * EQ_EVERY
            self.g_m += gw * (x - self.g_m); self._eq_xc = x - self.g_m; self._eq_next = 0
            self.fp["screening"] += 2 * self.d
        self._eq_rows(-(-self.d // (EQ_EVERY - 1)))                  # rows per step: the cycle ends within 30 calls
        for j, key in enumerate(self.split_keys()):
            if (t + j) % se:
                continue
            i, sg = key[1], key[2:]
            l_, r_ = halves(sg); dv = self._mean(i, l_) - self._mean(i, r_)
            st = self.sp.setdefault(key, [0.0, 1e-3, 1.0])
            st[0] += w * (res * dv - st[0]); st[1] += w * (dv * dv - st[1]); st[2] += w * (res * res - st[2])
            self.fp["screening"] += 14
