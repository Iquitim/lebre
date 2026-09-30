"""Evidence engine of LEBRE v0.52-r1: persistent hypotheses with a one-sided sub-exponential mixture e-process,
e-LOND-type acceptance levels fixed at creation, and experiment slots closed by a scan rule.

Extracted from the frozen research implementation (change_engine.py, evidence="psiE1"); arithmetic is unchanged.
"""
import math

import numpy as np


def _gamma_const(n=10 ** 6):
    k = np.arange(1, n + 1, dtype=float)
    s = float(np.sum(1.0 / (k * np.log(k + 1) ** 2)))
    return 1.0 / (s + 1.0 / math.log(n + 1))          # tail bound of the integral


GAMMA_C = _gamma_const()


def gamma_k(k):
    return GAMMA_C / (k * math.log(k + 1) ** 2)


class Hypothesis:
    """Weak null: over the steps at which this change was tested, the challenger did not improve the clipped loss of the
    reference by more than eps on average. One-sided mixture e-process with c = 1 + max(eps, 0) and predictable
    centring gamma_i = min(past mean, 0) (valid because d_i - gamma_i >= -c)."""
    LAMBDAS_FRAC = 0.9 * 2.0 ** -np.arange(8)            # lambda grid as a fraction of 1/c (uniform mixture)

    def __init__(self, eps, alpha_level, idx):
        self.eps = eps
        self.c = 1.0 + max(eps, 0.0)
        self.alpha = alpha_level; self.log_thr = math.log(1.0 / alpha_level); self.idx = idx
        self.S = 0.0; self.V = 0.0; self.n = 0; self.mean = 0.0
        self.lam = self.LAMBDAS_FRAC / self.c
        self.psi = (-np.log(1 - self.c * self.lam) - self.c * self.lam) / self.c ** 2
        self.peak_stats = None

    def observe(self, d):
        gamma = max(-self.c, min(0.0, self.mean))
        self.S += d; self.V += (d - gamma) ** 2; self.n += 1
        dm = d - self.mean; self.mean += dm / self.n

    def log_e(self):
        a = self.lam * self.S - self.psi * self.V
        m = a.max()
        return float(m + math.log(np.mean(np.exp(a - m))))


class Experiment:
    """One episode on a slot; evidence goes to the persistent hypothesis. The challenger's parameters (w, P, k, kstar)
    are owned and updated by the model."""

    def __init__(self, kind, add_key, rem_key, hyp, t0):
        self.kind, self.add_key, self.rem_key, self.hyp, self.t0 = kind, add_key, rem_key, hyp, t0
        self.S = 0.0; self.n = 0; self.k = 0; self.kstar = None
        self.eps = hyp.eps

    def observe(self, d):
        self.hyp.observe(d); self.S += d; self.n += 1


class ChangeEngine:
    def __init__(self, alpha, family_size, m_slots, n_min, T_max):
        self.alpha, self.P, self.m_slots, self.n_min, self.T_max = alpha, family_size, m_slots, n_min, T_max
        self.hyps = {}; self.n_hyp = 0; self.n_accepted = 0; self.k_started = 0
        self.slots = []; self.fp = 0.0

    def hypothesis(self, hkey, eps):
        if hkey not in self.hyps:
            self.n_hyp += 1
            g = 1.0 / (2 * self.P) if self.n_hyp <= self.P else 0.5 * gamma_k(self.n_hyp - self.P)   # sum <= 1
            self.hyps[hkey] = Hypothesis(eps, min(self.alpha * g * (self.n_accepted + 1), 0.5), self.n_hyp)
        return self.hyps[hkey]

    def open(self, kind, add_key, rem_key, hkey, eps, t):
        self.k_started += 1
        ex = Experiment(kind, add_key, rem_key, self.hypothesis(hkey, eps), t)
        self.slots.append(ex)
        return ex

    def free_slots(self):
        return self.m_slots - len(self.slots)

    def review(self):
        """[(decision, experiment, log_e)] with decision in {'accepted', 'closed'}; keeps the others running."""
        out, keep = [], []
        for ex in self.slots:
            le = ex.hyp.log_e(); self.fp += 3 * len(ex.hyp.lam)
            if le >= ex.hyp.log_thr:
                out.append(("accepted", ex, le))
            elif ex.n >= self.T_max or (ex.n >= self.n_min and ex.S <= 0.0):   # scan rule: not ahead -> free the slot
                out.append(("closed", ex, le))
            else:
                keep.append(ex)
        self.slots = keep
        return out

    def consume(self, ex):
        """an accepted change: count the discovery and retire its hypothesis (a new instance if ever retested)."""
        self.n_accepted += 1
        self.hyps = {k: h for k, h in self.hyps.items() if h is not ex.hyp}

    def busy_keys(self):
        return {k for ex in self.slots for k in (ex.add_key, ex.rem_key) if k}
