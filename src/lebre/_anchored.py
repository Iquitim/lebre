"""Anchored structural core (LEBRE v0.55): each unit is tested against the forecast it can improve, an innovation
structure carries accepted units to the forecast of integrated targets, a soft gate in screening and a predictable
one-step test scale.

- External units (each input and its splits) are tested against the core forecast L, which already contains the
  memory of the target; removing one discounts its exact contribution in L. Internal units (the target's own past, the
  residual state and the splits of the target's past) are tested against the level structure s, as before. In a swap,
  the kind of the entering unit applies. The test scale comes from the error of L.
- Innovation structure: s_d predicts y - m with the same accepted units and features as s, its own weights, NLMS with
  the same step and denominator. Nested combination: L1 = m + wS (s - m), L = L1 + wI ((m + s_d) - L1), with wI by the
  same dynamic averaging as wS (lambda 0.99, eta 0.5, every 8 steps). Contribution of an external unit in L:
  (1 - wI) wS c_s + wI c_d.
- Weights from whoever learned: an accepted unit enters s with the challenger's weights; it enters s_d with the same
  weights if external and with zero if internal (a split of an internal unit gives both halves the parent's weight).
- Soft gate: internal units are screened with the residual of L, like external ones.
- Test scale floor: 0.1 sqrt(min(variance of the level, variance of the first difference)), both estimated causally,
  computed before the target is seen.

Transcribed from the frozen research prototype (lebre-research, experiments/LEBRE-V0.55-PROTO-01/lebre055/anchored.py,
commit b0522a0, variant B10: REF_SPLIT, INNOV, SCREEN_CORE_INT, INN_FROM_OWN_RESIDUAL, PRED_FLOOR and DIFF_FLOOR on;
ANCHOR, SLOW, REF_CORE and GATE_INTERNAL off), keeping the order of every floating-point operation; the branches of the
switches that are off in v0.55 were left out."""
import math

import numpy as np

from ._core import (ALPHA_COV, BANDS, CHAL_WARM, CLIP_K, CLIP_X, DECIDE_EVERY, ETA, EVERY, GAMMA_Q, LAM, MU, OUT_MARGIN,
                    OUT_RHO, PEAK_UNTIL, POLES, SCALE_FLOOR, SCREEN_EVERY, SCREEN_W, halves)
from ._removal import TriggeredRemovalCheapCore
from ._screening import EQ_EVERY


class AnchoredMixin:
    def __init__(self, d, season=None, season2=None):
        super().__init__(d, season=season, season2=season2)
        self._refs = None; self.res_prev_int = 0.0                  # test reference of each running experiment
        self.base_i = self.base.copy(); self.inn_w = {}             # innovation structure
        self.d_sm = None; self.d_sv = 0.0; self.n_d = 0              # causal variance of the first difference
        self.Di = 0.0; self.wI = 0.5; self._inn = None

    def _internal(self, ex):
        """internal unit (target's own past, residual state and splits of the target's past); in a swap, the entering one"""
        u = self.unit_of(ex.add_key) if ex.add_key else ex.rem_key
        return u[0] == "res" or (u[0] == "in" and u[1] == self.d - 1)

    def _contrib_L(self, u):
        """exact contribution of unit u to the core forecast with the innovation structure"""
        b = self.block(u)
        return (1.0 - self.wI) * self.wS * float(self.active[u]["w"] @ b) + self.wI * float(self.inn_w[u] @ b)

    def _contrib_L_seg(self, key):
        i, sg = key[1], key[2:]
        u = ("in", i); st = self.active[u]; j = st["segs"].index(sg); mu_ = self._mean(i, sg)
        return (1.0 - self.wI) * self.wS * float(st["w"][j]) * mu_ + self.wI * float(self.inn_w[u][j]) * mu_

    def _seg_contrib_ref(self, key):
        i, sg = key[1], key[2:]
        st = self.active[("in", i)]
        return float(st["w"][st["segs"].index(sg)]) * self._mean(i, sg)

    # ---------------- one step, split in two halves
    def predict(self, x):
        if self._pending is not None:
            raise RuntimeError("predict() called twice without observe()")
        v = self.ylast
        x = np.append(np.asarray(x, float), (v - self.ym) / math.sqrt(self.yv + 1e-6))
        dlt = v - self.ym; self.ym += 1e-4 * dlt
        self.yv = max((1 - 1e-4) * self.yv + 1e-4 * (v - self.ym) * dlt, 1e-4); self.fp["structure"] += 9
        x = np.nan_to_num(x)
        if np.any(np.abs(x) > CLIP_X):
            self.quar_until = self.t + self.L
        x = np.clip(x, -CLIP_X, CLIP_X)
        self._shift(x)
        m = self.M.predict()
        s, xb = self._struct_pred(x)
        f = m + self.wS * (s - m)
        si = float(self.base_i @ xb) + sum(float(self.inn_w[u] @ self.block(u)) for u in self.active)
        L1, Li = f, m + si
        f = L1 + self.wI * (Li - L1); self._inn = (L1, Li)
        self.fp["structure"] += 2 * (self.d + 1) + sum(3 * len(st["w"]) for st in self.active.values()) + 4
        f_core = f
        if self.env_n >= 50:
            w_ = OUT_MARGIN * (self.env_hi - self.env_lo)
            fc = min(max(f, self.env_lo - w_), self.env_hi + w_) if math.isfinite(f) else self.last_obs
            self.n_clipped += fc != f; f = fc; self.fp["control"] += 5
        outs, rrs = [], []
        refL = f_core
        self._refs = []
        for ex in self.engine.slots:
            internal = self._internal(ex)
            rx, kx = (s, 1.0) if internal else (refL, self.wS)
            self._refs.append(rx)
            if ex.kind == "split":
                if not internal:
                    c = self._contrib_L_seg(ex.add_key)
                else:
                    c = kx * self._seg_contrib_ref(ex.add_key)
                outs.append(rx - c + float(ex.w @ self._cf(ex))); rrs.append(c)
                continue
            a = float(ex.w @ self._cf(ex)) if ex.add_key else 0.0
            if ex.rem_key and not internal:
                rm = self._contrib_L(ex.rem_key)
            else:
                rm = kx * float(self.active[ex.rem_key]["w"] @ self.block(ex.rem_key)) if ex.rem_key else 0.0
            outs.append(rx + a - rm); rrs.append(rm)
        ref = refL
        self.fp["experiments"] += 6 * len(outs)
        self._pending = (xb, m, s, f, ref, outs, rrs)
        return f

    def observe(self, y, quarantine=False):
        if self._pending is None:
            raise RuntimeError("observe() called without a preceding predict()")
        xb, m, s, f, ref, outs, rrs = self._pending; self._pending = None
        y_ok = y is not None and np.isfinite(y)
        mem_learn = y_ok and not quarantine
        learn = mem_learn and not (self.t <= self.quar_until)
        if y_ok:
            e = y - f; er = y - ref
            self.e2r = er * er if self.e2r is None else self.e2r + (1 - LAM) * (er * er - self.e2r)
            v_ = min(self.y_sv, self.d_sv) if self.n_d > 0 else self.y_sv          # floor from the past only
            floor = SCALE_FLOOR * math.sqrt(max(v_, 0.0)) if self.t >= 200 else 0.0
            if self.last_obs is not None:                            # variance of the first difference, after the floor
                dd = y - self.last_obs
                if self.d_sm is None:
                    self.d_sm = dd; self.n_d = 0
                self.n_d += 1; rd = max(1e-4, 1.0 / self.n_d)
                e_ = dd - self.d_sm; self.d_sm += rd * e_; self.d_sv += rd * (e_ * (dd - self.d_sm) - self.d_sv)
            if self.y_sm is None:
                self.y_sm = y; self.n_y = 0
            self.n_y += 1; ry = max(1e-4, 1.0 / self.n_y)
            dy = y - self.y_sm; self.y_sm += ry * dy; self.y_sv += ry * (dy * (y - self.y_sm) - self.y_sv)
            B = CLIP_K * max(self.sigr, floor)
            self.fp["control"] += 8
            res = y - s
            res_ref = y - ref                                        # residual of the core forecast
            refs = self._refs
            if learn:
                for j, (ex, g, back) in enumerate(zip(self.engine.slots, outs, rrs)):
                    ex.k += 1
                    lfx = min((y - refs[j]) ** 2 / (B * B), 1.0)
                    if ex.k > CHAL_WARM or not ex.add_key:
                        ex.observe(lfx - min((y - g) ** 2 / (B * B), 1.0) - ex.eps)
                    if ex.add_key:                                   # RLS on the residual the change would explain
                        rr = (y - refs[j]) + back
                        if ex.add_key[0] == "in" and ex.k <= PEAK_UNTIL:
                            self._peak_stats(ex, rr)
                        fx = self._cf(ex)
                        lam_ = 1.0
                        Pf = ex.P @ fx; kg = Pf / (lam_ + fx @ Pf)
                        ex.w = ex.w + kg * (rr - ex.w @ fx); ex.P = (ex.P - np.outer(kg, Pf)) / lam_
                        self.fp["experiments"] += 4 * len(fx) ** 2 + 6 * len(fx)
                self.fp["experiments"] += 8 * len(outs)
                blocks = {u: self.block(u) for u in self.active}
                bpow = sum(float(b @ b) for b in blocks.values())
                den = float(xb @ xb) + bpow + 1e-6
                g_ = MU * res / den
                self.base += g_ * xb
                for u, b in blocks.items():
                    st = self.active[u]; st["w"] = st["w"] + g_ * b
                    c = abs(float(st["w"] @ b))
                    self.contrib_ema[u] = c if u not in self.contrib_ema else self.contrib_ema[u] + 0.01 * (c - self.contrib_ema[u])
                self.fp["structure"] += 4 * (len(xb) + sum(len(b) for b in blocks.values())) + 4
                gi = MU * (y - self._inn[1]) / den                   # NLMS of the innovation structure
                self.base_i += gi * xb
                for u, b in blocks.items():
                    self.inn_w[u] = self.inn_w[u] + gi * b
                self.fp["structure"] += 4 * (len(xb) + sum(len(b) for b in blocks.values())) + 4
                rint = res_ref                                       # soft gate: internal units screened by y - L
                self._screen_spread_split(res_ref, rint)
                self.res_prev = res_ref; self.res_prev_int = rint
                self.res2 = res * res if self.res2 is None else self.res2 + (1 - LAM) * (res * res - self.res2)
            es, em = y - s, y - m
            self.D = LAM * self.D + (es * es - em * em) / (self.sig ** 2 + 1e-12)
            self.fp["control"] += 7
            e1, ei = y - self._inn[0], y - self._inn[1]              # dynamic averaging between L1 and m + s_d
            self.Di = LAM * self.Di + (ei * ei - e1 * e1) / (self.sig ** 2 + 1e-12)
            self.fp["control"] += 7
            if self.t % EVERY == 0:
                z = ETA * self.D
                self.wS = 1.0 / (1.0 + math.exp(z)) if z < 700 else 0.0
                self.fp["control"] += 4
                zi = ETA * self.Di
                self.wI = 1.0 / (1.0 + math.exp(zi)) if zi < 700 else 0.0
                self.fp["control"] += 4
            self.e2 = e * e if self.e2 is None else self.e2 + (1 - LAM) * (e * e - self.e2)
            if self.qhat is None:
                self.qhat = 1.645 * math.sqrt(self.e2)
            miss = 1.0 if abs(e) > self.qhat else 0.0
            self.hits_obs += 1; self.hits_ok += miss == 0.0
            self.qacc += miss - ALPHA_COV
            if self.t % EVERY == 0:
                self.sig = math.sqrt(max(self.e2, 1e-300)); self.sigr = math.sqrt(max(self.e2r, 1e-300))
                self.qhat = max(0.0, self.qhat + GAMMA_Q * self.sig * self.qacc); self.qacc = 0.0
            self.fp["control"] += 10
            u_ = e / self.sig
        else:
            u_ = 0.0
        if not y_ok and self.last_obs is not None:
            self.M.update(self.last_obs, m, learn=False)             # hold: history gets the last observed value
        else:
            self.M.update(y if y_ok else None, m, learn=mem_learn)
        if y_ok:
            self.last_obs = float(y)
            if self.env_hi is None:
                self.env_hi = self.env_lo = self.env_m = float(y)
            self.env_n += 1; r_ = OUT_RHO
            self.env_m += max(r_, 1.0 / self.env_n) * (y - self.env_m)
            self.env_hi = max(float(y), self.env_hi - r_ * (self.env_hi - self.env_m))
            self.env_lo = min(float(y), self.env_lo - r_ * (self.env_lo - self.env_m))
            self.fp["control"] += 10
        if y_ok:
            self.ylast = float(y)
        self.r = np.array(POLES) * self.r + (1 - np.array(POLES)) * u_
        self.fp["structure"] += 3 * len(POLES)
        if self.t % DECIDE_EVERY == 0 and self.t > 0:
            self._decide()
        self.t += 1

    # ---------------- screening with two residuals (the cheap screening of v0.54, M5b)
    def _screen_spread_split(self, res, res_int):
        w, se, t = SCREEN_W, SCREEN_EVERY, self.t
        act = {u[1] for u in self.active if u[0] == "in"}
        due = [i for i in range(self.d) if (t + i) % se == 0]
        x, x1, ra = self.H[0], self.H[1], 1e-3 * se
        own = self.d - 1
        for i in due:
            self.a_c0[i] += ra * (x[i] * x[i] - self.a_c0[i]); self.a_c1[i] += ra * (x[i] * x1[i] - self.a_c1[i])
            self.fp["screening"] += 6
            if i in act:
                continue
            rz, rp = (res_int, self.res_prev_int) if i == own else (res, self.res_prev)
            a = float(self._a()[i]); pres = rz - a * rp
            for b, (lo, hi) in enumerate(BANDS):
                ub = (self.bs[b, i] - a * (self.bs[b, i] - self.H[lo, i] + self.H[hi + 1, i])) / (hi - lo + 1)
                self.sc_xy[b, i] += w * (pres * ub - self.sc_xy[b, i]); self.sc_xx[b, i] += w * (ub * ub - self.sc_xx[b, i])
            self.sc_yy[i] += w * (pres * pres - self.sc_yy[i])
            self.fp["screening"] += 11 * len(BANDS) + 5
        if t % se == self.d % se:                                   # residual latent unit (internal)
            self.sr_xy += w * (res_int * self.r - self.sr_xy); self.sr_xx += w * (self.r * self.r - self.sr_xx)
            self.sr_yy += w * (res_int * res_int - self.sr_yy); self.fp["screening"] += 12 * len(POLES)
        if t % EQ_EVERY == (self.d + 1) % EQ_EVERY:                 # equivalence groups (reporting only), M5b
            self._eq_flush()
            gw = 1e-3 * EQ_EVERY
            self.g_m += gw * (x - self.g_m); self._eq_xc = x - self.g_m; self._eq_next = 0
            self.fp["screening"] += 2 * self.d
        self._eq_rows(-(-self.d // (EQ_EVERY - 1)))
        for j, key in enumerate(self.split_keys()):
            if (t + j) % se:
                continue
            i, sg = key[1], key[2:]
            rz = res_int if i == own else res
            l_, r_ = halves(sg); dv = self._mean(i, l_) - self._mean(i, r_)
            st = self.sp.setdefault(key, [0.0, 1e-3, 1.0])
            st[0] += w * (rz * dv - st[0]); st[1] += w * (dv * dv - st[1]); st[2] += w * (rz * rz - st[2])
            self.fp["screening"] += 14

    # ---------------- lifecycle
    def _realign(self, store, old, ex, segs_of, zero_new=False):
        """weights of the innovation structure realigned with the active units after an acceptance: an unchanged unit
        keeps them; a split applies the core's merge; a new unit gets the challenger's weights (zero if internal)"""
        new = {}
        for u, st in self.active.items():
            if u in old and old[u][0] == segs_of(st):
                new[u] = old[u][1]
            elif ex.kind == "split" and u == self.unit_of(ex.add_key) and u in old:
                segs_o, w_o = old[u]; sg = ex.add_key[2:]; j = segs_o.index(sg); l_, r_ = halves(sg)
                segs = segs_o[:j] + [l_, r_] + segs_o[j + 1:]
                halves_w = np.full(2, w_o[j]) if zero_new else ex.w        # internal split: halves with the parent's weight
                w = np.concatenate([w_o[:j], halves_w, w_o[j + 1:]])
                merged = {}
                for sg_, w_ in zip(segs, w):
                    merged[sg_] = merged.get(sg_, 0.0) + float(w_)
                new[u] = np.array(list(merged.values()))
            else:
                new[u] = np.zeros(len(st["w"])) if zero_new else np.array(st["w"], float)
            assert len(new[u]) == len(st["w"])
        return new

    def _accept(self, ex, le):
        segs_of = lambda st: None if st["segs"] is None else list(st["segs"])  # noqa: E731
        old_i = {u: (segs_of(st), self.inn_w[u].copy()) for u, st in self.active.items()}
        super()._accept(ex, le)
        internal = bool(ex.add_key) and self._internal(ex)
        self.inn_w = self._realign(self.inn_w, old_i, ex, segs_of, zero_new=internal)
        if ex.kind != "split" and ex.add_key and ex.add_key[0] == "in" and not internal:
            self.base_i[1 + ex.add_key[1]] += float(ex.w[-1])


class AnchoredCore(AnchoredMixin, TriggeredRemovalCheapCore):
    """structural core of LEBRE v0.55: the core of v0.54 (triggered removals, cheap report of equivalent inputs) anchored"""
