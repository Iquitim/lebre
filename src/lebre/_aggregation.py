"""Aggregation of two forecasters, as used by layers M1 and M2 of LEBRE v0.53 (transcribed from the promoted research
prototype, commit 4a2620e, keeping the order of every floating-point operation).

- AdaHedge (de Rooij, van Erven, Grünwald and Koolen, 2014, JMLR 15:1281-1316, Figure 1), invariant to translation and
  scale of the losses, so it runs on raw squared errors.
- One-way switch: posterior over "always R", "always L", "R until some time, L after" (switch distribution of van Erven,
  Grünwald and de Rooij, 2012, restricted to one switch from the first to the second forecaster; hazard 1/(n+1)).
- Horizon-free (A,B)-Prod (Sani, Neu and Lazaric, 2014) for losses in [0, 1].

Operation counts follow the rule of the research code (1 FP per elementary operation, exp and log included)."""
import math

import numpy as np

FP_ADAHEDGE_STEP = 48      # K = 2: weights (14), weighted output (3), squared losses (4), update (27)
FP_SWITCH_STEP = 37        # one-way switch: likelihoods, posterior, switch mass, output
FP_PROD_STEP = 31          # (A,B)-Prod: learning rate, weight update, mixing fraction


def _mix(eta, L):
    """Weights and cumulative mix loss for learning rate eta and cumulative losses L (Figure 1, function mix)."""
    mn = L.min()
    if math.isinf(eta):                                       # limit: follow the leader, ties with equal weight
        w = (L == mn).astype(float)
        s = w.sum()
        return w / s, mn
    w = np.exp(-eta * (L - mn))
    s = w.sum()
    return w / s, mn - math.log(s / len(L)) / eta


class AdaHedge:
    """AdaHedge for K experts."""

    def __init__(self, K=2):
        self.K = int(K)
        self.L = np.zeros(self.K)
        self.Delta = 0.0
        self.n = 0

    def eta(self):
        if self.Delta == 0.0:
            return math.inf
        return math.log(self.K) / self.Delta

    def weights(self):
        return _mix(self.eta(), self.L)[0]

    def update(self, losses):
        l = np.asarray(losses, float)
        eta = self.eta()
        w, M0 = _mix(eta, self.L)
        h = float(w @ l)
        self.L = self.L + l
        _, M1 = _mix(eta, self.L)
        delta = max(0.0, h - (M1 - M0))                       # max removes numerical violations of Jensen
        self.Delta += delta
        self.n += 1


class OneWaySwitch:
    """Posterior over "always R", "always L" and "R until t, L after"; update() takes negative log-likelihoods (R, L)."""

    def __init__(self):
        self.w = np.array([1.0, 1.0, 1.0, 0.0]) / 3.0          # [always R, always L, R with pending switch, switched]
        self.n = 0
        self.log_scale_updates = 0

    def weights(self):
        w = self.w
        p = np.array([w[0] + w[2], w[1] + w[3]])
        return p / p.sum()

    def update(self, losses):
        lr, ll = float(losses[0]), float(losses[1])
        m = min(lr, ll)
        vr, vl = math.exp(-(lr - m)), math.exp(-(ll - m))
        w = self.w * np.array([vr, vl, vr, vl])
        tot = w.sum()
        if tot > 0 and math.isfinite(tot):
            w = w / tot
        else:                                                  # underflow (0/0): the same computation in log scale
            with np.errstate(divide="ignore"):
                lw = np.log(self.w) - np.array([lr - m, ll - m, lr - m, ll - m])
            w = np.exp(lw - lw.max())
            w = w / w.sum()
            self.log_scale_updates += 1
        self.n += 1
        h = 1.0 / (self.n + 1)                                 # chance that the switch happens at the next step
        mov = h * w[2]
        w[2] -= mov; w[3] += mov
        self.w = w


class ProdAB:
    """Horizon-free (A,B)-Prod for losses in [0, 1]; s() is the fraction given to A (B has fixed weight 1/2)."""

    def __init__(self):
        self.wA = 0.5; self.wB = 0.5; self.S2 = 0.0; self.n = 0

    def eta(self):
        return min(0.5, math.sqrt(1.0 / (1.0 + self.S2)))

    def s(self):
        e = self.eta()
        return e * self.wA / (e * self.wA + self.wB / 2.0)

    def update(self, fB, fA):
        d = float(fB) - float(fA)
        er = self.eta()
        self.S2 += d * d
        en = self.eta()
        self.wA *= (1.0 + er * d) ** (en / er)
        self.n += 1
