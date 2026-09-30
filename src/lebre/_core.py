"""Core of LEBRE v0.52-r1 (canonical configuration only).

Extracted from the frozen research implementation `lebre_v052h.LebreV052H` with the canonical configuration
{scale_floor 0.1, peak_persist, evidence psiE1, lazy_halves, screen_skip_active, track_groups, mu 0.05, stagger,
screen_spread, peak_split, peak_until 200, chal_warm 250, gap_hold, out_contract}. Branches used only by research
ablations were removed; every arithmetic expression on the canonical path is kept in the same order, so the forecasts
are bit-for-bit identical (checked by the regression tests).

The only structural change is that one research `step(x, y)` is split into `predict(x)` (everything computed before
the target is read) and `observe(y)` (everything after); `step` = `predict` then `observe`.
"""
import math

import numpy as np

from ._engine import ChangeEngine
from ._memory import Memory, MemoryW

POLES = (0.8, 0.95)
BANDS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31))

# canonical constants (identical to the research defaults + canonical configuration)
M_MAX, M_SLOTS, MU, ALPHA = 4, 2, 0.05, 0.05
EPS = {"add": 0.002, "swap": 0.002, "rem": -0.002, "split": 0.002}
CLIP_K, DECIDE_EVERY, N_MIN, T_MAX, COOLDOWN, REM_PERIOD = 2.0, 10, 100, 5000, 2000, 2000
LAM, ETA, ALPHA_COV, GAMMA_Q, EVERY, CLIP_X = 0.99, 0.5, 0.1, 0.1, 8, 8.0
CHAL_WARM, SCREEN_EVERY, SCREEN_W, SCREEN_Q, SPLIT_Q, RLS_P0 = 250, 4, 0.02, 15.09, 6.63, 10.0
PEAK_UNTIL, SCALE_FLOOR, OUT_MARGIN, OUT_RHO = 200, 0.1, 0.5, 1e-3


def halves(seg):
    lo, hi = seg; mid = (lo + hi) // 2
    return (lo, mid), (mid + 1, hi)


class Core:
    def __init__(self, d, season=None, season2=None):
        self.ylast = 0.0; self.ym = 0.0; self.yv = 1.0
        d = d + 1                                                    # the target's own past is one more input
        self.d, self.L = d, BANDS[-1][1]
        self.M = MemoryW(season, season2) if season2 else Memory(season)
        self.quar_until = -1
        self.y_sm = None; self.y_sv = 0.0; self.n_y = 0
        self.g_m = np.zeros(d); self.g_c = np.eye(d)
        self.units = [("in", i) for i in range(d)] + [("res",)]
        nb = len(BANDS)
        self.H = np.zeros((self.L + 2, d))                           # H[k] = x_{t-k}
        self.bs = np.zeros((nb, d))                                  # running sums of the octave bands
        self.seg = {}                                                # (i, lo, hi) -> running sum of an active segment
        self.r = np.zeros(len(POLES))
        self.base = np.zeros(d + 1)
        self.active = {}                                             # unit -> {"segs": [(lo, hi)...] | None, "w": array}
        self.contrib_ema = {}
        self.a_c0 = np.ones(d); self.a_c1 = np.zeros(d); self.res_prev = 0.0
        self.sc_xy = np.zeros((nb, d)); self.sc_xx = np.ones((nb, d)) * 1e-3; self.sc_yy = np.ones(d)
        self.sr_xy = np.zeros(len(POLES)); self.sr_xx = np.ones(len(POLES)) * 1e-3; self.sr_yy = 1.0
        self.sp = {}                                                 # split key -> [xy, xx, yy] EMA
        self.engine = ChangeEngine(ALPHA, 2 * len(self.units), M_SLOTS, N_MIN, T_MAX)
        self.last_tested = {}; self.last_rem = {}
        self.events = []; self.t = 0
        self.e2 = None; self.sig = 1.0; self.e2r = None; self.sigr = 1.0; self.res2 = None
        self.D = 0.0; self.wS = 0.5
        self.qhat = None; self.qacc = 0.0; self.hits_obs = 0; self.hits_ok = 0
        self.fp = {"memory": 0.0, "structure": 0.0, "experiments": 0.0, "screening": 0.0, "control": 0.0}
        self.last_obs = None; self.env_hi = None; self.env_lo = None; self.env_m = None; self.env_n = 0; self.n_clipped = 0
        self._pending = None

    # ---------------- segment sums
    def _track(self, i, seg):
        k = (i, seg[0], seg[1])
        if k not in self.seg:
            self.seg[k] = float(self.H[seg[0]:seg[1] + 1, i].sum())
            self.fp["structure"] += seg[1] - seg[0] + 1

    def _mean(self, i, seg):
        k = (i, seg[0], seg[1])
        if k in self.seg:
            return self.seg[k] / (seg[1] - seg[0] + 1)
        self.fp["structure"] += seg[1] - seg[0] + 1
        return float(self.H[seg[0]:seg[1] + 1, i].sum()) / (seg[1] - seg[0] + 1)

    def _retrack(self):
        """keep exactly the active segments as running sums (halves are summed on demand)"""
        need = set()
        for u, st in self.active.items():
            if u[0] == "in":
                for sg in st["segs"]:
                    need.add((u[1],) + sg)
        for k in need:
            self._track(k[0], k[1:])
        for k in [k for k in self.seg if k not in need]:
            del self.seg[k]

    # ---------------- features (values available before y_t)
    def block(self, unit):
        if unit[0] == "in":
            i = unit[1]
            st = self.active.get(unit)
            if st is None:
                return np.array([self.bs[b, i] / (hi - lo + 1) for b, (lo, hi) in enumerate(BANDS)])
            return np.array([self._mean(i, sg) for sg in st["segs"]])
        return self.r.copy()

    def cfeat(self, key):
        if key[0] == "split":
            i, sg = key[1], key[2:]
            return np.array([self._mean(i, h) for h in halves(sg)])
        f = self.block(key)
        return np.append(f, self.H[0, key[1]]) if key[0] == "in" else f

    def _cf(self, ex):
        key = ex.add_key
        if key[0] == "in" and ex.kstar:
            f = self.cfeat(key)
            return np.concatenate([f[:-1], [self.H[ex.kstar, key[1]]], f[-1:]])
        return self.cfeat(key)

    def _seg_contrib(self, key):
        i, sg = key[1], key[2:]
        st = self.active[("in", i)]
        return float(st["w"][st["segs"].index(sg)]) * self._mean(i, sg)

    def response(self, unit):
        st = self.active[unit]
        return [(sg, float(w) / (sg[1] - sg[0] + 1)) for sg, w in zip(st["segs"], st["w"])]

    def _struct_pred(self, x):
        xb = np.concatenate(([1.0], x))
        s = float(self.base @ xb) + sum(float(st["w"] @ self.block(u)) for u, st in self.active.items())
        self.fp["structure"] += 2 * (self.d + 1) + sum(3 * len(st["w"]) for st in self.active.values())
        return s, xb

    def _a(self):
        return np.clip(self.a_c1 / np.maximum(self.a_c0, 1e-6), -0.999, 0.999)

    def _shift(self, x):
        self.H = np.roll(self.H, 1, axis=0); self.H[0] = x
        for b, (lo, hi) in enumerate(BANDS):
            self.bs[b] += self.H[lo] - self.H[hi + 1]
        for k in self.seg:
            i, lo, hi = k
            self.seg[k] += self.H[lo, i] - self.H[hi + 1, i]
        self.fp["structure"] += 2 * len(BANDS) * self.d + 2 * len(self.seg)

    # ---------------- one step, split in two halves
    def predict(self, x):
        """everything computed before y_t is read; returns the (bounded) forecast"""
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
        if self.env_n >= 50:
            w_ = OUT_MARGIN * (self.env_hi - self.env_lo)
            fc = min(max(f, self.env_lo - w_), self.env_hi + w_) if math.isfinite(f) else self.last_obs
            self.n_clipped += fc != f; f = fc; self.fp["control"] += 5
        ref = s
        outs, rrs = [], []
        for ex in self.engine.slots:
            if ex.kind == "split":
                c = self._seg_contrib(ex.add_key)
                outs.append(ref - c + float(ex.w @ self._cf(ex))); rrs.append(c)
                continue
            a = float(ex.w @ self._cf(ex)) if ex.add_key else 0.0
            r = float(self.active[ex.rem_key]["w"] @ self.block(ex.rem_key)) if ex.rem_key else 0.0
            outs.append(ref + a - r); rrs.append(r)
        self.fp["experiments"] += 6 * len(outs)
        self._pending = (xb, m, s, f, ref, outs, rrs)
        return f

    def observe(self, y, quarantine=False):
        """everything after y_t is read (None or NaN = missing target)"""
        if self._pending is None:
            raise RuntimeError("observe() called without a preceding predict()")
        xb, m, s, f, ref, outs, rrs = self._pending; self._pending = None
        y_ok = y is not None and np.isfinite(y)
        mem_learn = y_ok and not quarantine
        learn = mem_learn and not (self.t <= self.quar_until)
        if y_ok:
            e = y - f; er = y - ref
            self.e2r = er * er if self.e2r is None else self.e2r + (1 - LAM) * (er * er - self.e2r)
            if self.y_sm is None:
                self.y_sm = y; self.n_y = 0
            self.n_y += 1; ry = max(1e-4, 1.0 / self.n_y)
            dy = y - self.y_sm; self.y_sm += ry * dy; self.y_sv += ry * (dy * (y - self.y_sm) - self.y_sv)
            floor = SCALE_FLOOR * math.sqrt(max(self.y_sv, 0.0)) if self.t >= 200 else 0.0
            B = CLIP_K * max(self.sigr, floor)
            self.fp["control"] += 8
            lf = min(er * er / (B * B), 1.0)
            res = y - s
            if learn:
                for ex, g, back in zip(self.engine.slots, outs, rrs):
                    ex.k += 1
                    if ex.k > CHAL_WARM or not ex.add_key:
                        ex.observe(lf - min((y - g) ** 2 / (B * B), 1.0) - ex.eps)
                    if ex.add_key:                                   # RLS on the residual the change would explain
                        rr = res + back
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
                self._screen_spread(res)
                self.res_prev = res
                self.res2 = res * res if self.res2 is None else self.res2 + (1 - LAM) * (res * res - self.res2)
            es, em = y - s, y - m
            self.D = LAM * self.D + (es * es - em * em) / (self.sig ** 2 + 1e-12)
            self.fp["control"] += 7
            if self.t % EVERY == 0:
                z = ETA * self.D
                self.wS = 1.0 / (1.0 + math.exp(z)) if z < 700 else 0.0
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

    def _peak_stats(self, ex, rr):
        i = ex.add_key[1]; a = self._a()[i]
        u = self.H[1:self.L + 1, i] - a * self.H[2:self.L + 2, i]       # prewhitened lags 1..L
        pr = rr - a * getattr(ex, "rprev", 0.0); ex.rprev = rr
        if getattr(ex, "cxy", None) is None:
            prev = ex.hyp.peak_stats
            if prev is not None:
                ex.cxy, ex.cxx = prev
            else:
                ex.cxy = np.zeros(self.L); ex.cxx = np.full(self.L, 1e-9)
            ex.hyp.peak_stats = (ex.cxy, ex.cxx)                        # same arrays: updated in place
        msk = (np.arange(self.L) + ex.k) % 2 == 0
        ex.cxy[msk] += pr * u[msk]; ex.cxx[msk] += u[msk] * u[msk]
        self.fp["experiments"] += 3 * self.L + 4
        if ex.k == PEAK_UNTIL:
            k = int(np.argmax(ex.cxy ** 2 / ex.cxx)) + 1
            if k > 1:                                                   # lag 1 is already its own band
                ex.kstar = k
                n = len(ex.w)
                w = np.insert(ex.w, n - 1, 0.0)
                P = np.insert(np.insert(ex.P, n - 1, 0.0, axis=0), n - 1, 0.0, axis=1); P[n - 1, n - 1] = RLS_P0
                ex.w, ex.P = w, P

    # ---------------- screening (each part on its own phase, every SCREEN_EVERY steps)
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
        if t % se == (self.d + 1) % se:                             # equivalence groups (reporting only)
            gw = 1e-3 * se
            self.g_m += gw * (x - self.g_m); xc = x - self.g_m
            self.g_c += gw * (np.outer(xc, xc) - self.g_c)
            self.fp["screening"] += 3 * self.d * (self.d + 1) // 2 + 2 * self.d
        for j, key in enumerate(self.split_keys()):
            if (t + j) % se:
                continue
            i, sg = key[1], key[2:]
            l_, r_ = halves(sg); dv = self._mean(i, l_) - self._mean(i, r_)
            st = self.sp.setdefault(key, [0.0, 1e-3, 1.0])
            st[0] += w * (res * dv - st[0]); st[1] += w * (dv * dv - st[1]); st[2] += w * (res * res - st[2])
            self.fp["screening"] += 14

    def equivalents(self, unit, thr=0.95):
        if unit[0] != "in":
            return set()
        i = unit[1]; sd = np.sqrt(np.maximum(np.diag(self.g_c), 1e-12))
        r = np.abs(self.g_c[i]) / (sd[i] * sd)
        return {int(j) for j in np.flatnonzero(r >= thr)}

    def split_keys(self):
        out = []
        for u, st in self.active.items():
            if u[0] == "in":
                out += [("split", u[1]) + sg for sg in st["segs"] if sg[1] > sg[0]]
        return out

    def screen_stat(self, key):
        k = SCREEN_W / (2 - SCREEN_W)
        if key[0] == "split":
            st = self.sp.get(key)
            return 0.0 if st is None else st[0] ** 2 / (st[1] * st[2] + 1e-12) / k
        if key[0] == "in":
            i = key[1]
            c2 = self.sc_xy[:, i] ** 2 / (self.sc_xx[:, i] * self.sc_yy[i] + 1e-12)
        else:
            c2 = self.sr_xy ** 2 / (self.sr_xx * self.sr_yy + 1e-12)
        return float(c2.sum()) / k

    # ---------------- lifecycle
    @staticmethod
    def unit_of(key):
        return ("in", key[1]) if key[0] == "split" else key

    def _open(self, kind, add_key, rem_key):
        hkey = ("out", rem_key) if kind == "rem" else ("in", add_key)
        ex = self.engine.open(kind, add_key, rem_key, hkey, EPS[kind], self.t)
        if add_key:
            n_ = len(self.cfeat(add_key)); ex.w = np.zeros(n_); ex.P = RLS_P0 * np.eye(n_)
            if kind == "split":                                      # start from the parent's weight on both halves
                st = self.active[self.unit_of(add_key)]
                ex.w[:] = float(st["w"][st["segs"].index(add_key[2:])])
        for k in (add_key, rem_key):
            if k:
                self.last_tested[k] = self.t

    def _decide(self):
        changed = set()
        for decision, ex, le in self.engine.review():
            if decision == "accepted":
                self._accept(ex, le)
                changed |= {self.unit_of(k) for k in (ex.add_key, ex.rem_key) if k}
            else:
                self.events.append((self.t, "closed", ex.kind, ex.add_key, ex.rem_key, round(le, 2), ex.n))
        if changed:   # experiments about units that just changed no longer describe the live model
            keep = []
            for ex in self.engine.slots:
                if {self.unit_of(k) for k in (ex.add_key, ex.rem_key) if k} & changed:
                    self.events.append((self.t, "superseded", ex.kind, ex.add_key, ex.rem_key, None, ex.n))
                else:
                    keep.append(ex)
            self.engine.slots = keep
            self._retrack()
            self.sp = {k: v for k, v in self.sp.items() if k in set(self.split_keys())}
        self._fill()

    def _accept(self, ex, le):
        if ex.kind == "split":
            u = self.unit_of(ex.add_key); st = self.active[u]; sg = ex.add_key[2:]
            j = st["segs"].index(sg); l_, r_ = halves(sg)
            st["segs"] = st["segs"][:j] + [l_, r_] + st["segs"][j + 1:]
            st["w"] = np.concatenate([st["w"][:j], ex.w, st["w"][j + 1:]])
            merged = {}                                              # a half equal to the peak segment: one weight
            for sg_, w_ in zip(st["segs"], st["w"]):
                merged[sg_] = merged.get(sg_, 0.0) + float(w_)
            st["segs"], st["w"] = list(merged), np.array(list(merged.values()))
        else:
            if ex.rem_key and ex.rem_key in self.active:
                del self.active[ex.rem_key]; self.contrib_ema.pop(ex.rem_key, None)
            if ex.add_key:
                if ex.add_key[0] == "in":
                    segs = list(BANDS) + ([(ex.kstar, ex.kstar)] if ex.kstar else [])
                    self.active[ex.add_key] = {"segs": segs, "w": ex.w[:-1].copy()}
                    self.base[1 + ex.add_key[1]] += float(ex.w[-1])
                else:
                    self.active[ex.add_key] = {"segs": None, "w": ex.w.copy()}
        self.engine.consume(ex)
        self._retrack()
        self.events.append((self.t, "accepted", ex.kind, ex.add_key, ex.rem_key, round(le, 2), ex.hyp.n))

    def _fill(self):
        busy = {self.unit_of(k) for k in self.engine.busy_keys()}
        while self.engine.free_slots() > 0:
            due = [a for a in self.active if a not in busy and self.t - self.last_rem.get(a, -10 ** 9) >= REM_PERIOD
                   and self.t - self.last_tested.get(a, -10 ** 9) >= N_MIN]
            if due:
                a = due[0]; self.last_rem[a] = self.t; self._open("rem", None, a); busy.add(a); continue
            if any(ex.add_key and ex.k <= CHAL_WARM for ex in self.engine.slots):
                break                                                # one warm-up at a time
            cands = []
            pool = [u for u in self.units if u not in self.active and u not in busy
                    and self.t - self.last_tested.get(u, -10 ** 9) >= COOLDOWN]
            if pool:
                c = max(pool, key=self.screen_stat)
                if self.screen_stat(c) >= SCREEN_Q:
                    cands.append((self.screen_stat(c) / SCREEN_Q, "unit", c))
            sp = [k for k in self.split_keys() if self.unit_of(k) not in busy
                  and self.t - self.last_tested.get(k, -10 ** 9) >= COOLDOWN]
            if sp:
                c = max(sp, key=self.screen_stat)
                if self.screen_stat(c) >= SPLIT_Q:
                    cands.append((self.screen_stat(c) / SPLIT_Q, "split", c))
            if not cands:
                break
            _, typ, c = max(cands)
            if typ == "split":
                self._open("split", c, None)
            elif len(self.active) < M_MAX:
                self._open("add", c, None)
            else:
                a = min((k for k in self.active if k not in busy), key=lambda k: self.contrib_ema.get(k, float("inf")), default=None)
                if a is None:
                    break
                self._open("swap", c, a); busy.add(a)
            busy.add(self.unit_of(c))
            self.fp["control"] += (len(self.units) + len(self.sp)) / DECIDE_EVERY

    def fp_total(self):
        return sum(self.fp.values()) + self.M.fp + self.engine.fp
