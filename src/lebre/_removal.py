"""Triggered removal tests (LEBRE v0.54, M4): removals with per-episode evidence, added to the original schedule.

In the original core the removal hypothesis of a unit is persistent: it accumulates the evidence of every test episode
since acceptance. In a series with a regime change, the evidence from before the change (when removing was clearly
worse) weighs forever and the removal never reaches its level.

M4 adds, without changing anything in the original schedule of removal tests, a **triggered** test: when the mean
contribution of the unit (contrib_ema, already kept by the core) falls below FRAC_REM x the reference (the peak since
acceptance, reset to the current value at each triggered test), a removal episode opens with a new hypothesis ("on the
steps of this episode, removing does not improve the clipped loss by more than eps"), an e-process from scratch and an
e-LOND level of its own sequence: alpha x gamma_k x (accepted triggered removals + 1), capped at 0.5. The other
hypotheses keep the original budget, without counting the triggered ones. An episode lasts up to T_MAX_REM steps and the
removal is accepted only after at least N_REM_MIN steps of evidence (accepting only after a minimum is a valid stopping
time). Between triggered tests of the same unit, at least REM_PERIOD steps. A triggered test takes priority over a
periodic removal test of the same unit, which is closed ("superseded"; its persistent hypothesis is kept).

While no triggered test opens, forecasts are identical to those of the original core.
Transcribed from the promoted research prototype (lebre-research, experiments/LEBRE-V0.54-PROTO-01/lebre054/removal.py,
commit f9906c4), keeping the order of every floating-point operation."""
from ._core import ALPHA, DECIDE_EVERY, EPS, N_MIN, REM_PERIOD, T_MAX, Core
from ._engine import ChangeEngine, Experiment, Hypothesis, gamma_k
from ._screening import CheapEquivalenceCore

FRAC_REM, T_MAX_REM, N_REM_MIN = 0.5, 10000, T_MAX


class EpisodeEngine(ChangeEngine):
    """review() is the original; only the maximum duration and the minimum to accept become the episode's, when it has
    them (lines marked M4)."""

    def review(self):
        """[(decision, experiment, log_e)] with decision in {'accepted', 'closed'}; keeps the others running."""
        out, keep = [], []
        for ex in self.slots:
            le = ex.hyp.log_e(); self.fp += 3 * len(ex.hyp.lam)
            if le >= ex.hyp.log_thr and ex.n >= getattr(ex, "n_min_acc", 0):                       # M4
                out.append(("accepted", ex, le))
            elif ex.n >= getattr(ex, "t_max", self.T_max) or (ex.n >= self.n_min and ex.S <= 0.0):   # M4
                out.append(("closed", ex, le))
            else:
                keep.append(ex)
        self.slots = keep
        return out


class TriggeredRemovalMixin:
    def __init__(self, d, season=None, season2=None):
        super().__init__(d, season=season, season2=season2)
        self.n_hyp_rem = 0; self.n_acc_rem = 0; self.ref_contrib = {}; self.last_trig = {}
        e = self.engine                                              # same initial state, review with per-episode limits
        self.engine = EpisodeEngine(e.alpha, e.P, e.m_slots, e.n_min, e.T_max)

    def _open_trig(self, a):
        self.n_hyp_rem += 1
        hyp = Hypothesis(EPS["rem"], min(ALPHA * gamma_k(self.n_hyp_rem) * (self.n_acc_rem + 1), 0.5), self.n_hyp_rem)
        ex = Experiment("rem", None, a, hyp, self.t); ex.t_max = T_MAX_REM; ex.n_min_acc = N_REM_MIN; ex.trig = True
        self.engine.k_started += 1
        self.engine.slots.append(ex)
        self.last_tested[a] = self.t; self.last_trig[a] = self.t; self.ref_contrib[a] = self.contrib_ema[a]

    def _accept(self, ex, le):
        super()._accept(ex, le)
        if getattr(ex, "trig", False):                               # a triggered removal does not count in the original budget
            self.engine.n_accepted -= 1; self.n_acc_rem += 1

    def _fill(self):
        for a in self.active:                                        # reference = peak since acceptance or the last test
            if a in self.contrib_ema:
                self.ref_contrib[a] = max(self.ref_contrib.get(a, 0.0), self.contrib_ema[a])
        for a in [a for a in self.ref_contrib if a not in self.active]:
            del self.ref_contrib[a]
        self.fp["control"] += 2 * len(self.active) / DECIDE_EVERY
        trig = [a for a in self.active if a in self.contrib_ema
                and self.t - self.last_trig.get(a, -10 ** 9) >= REM_PERIOD
                and self.t - self.last_tested.get(a, -10 ** 9) >= N_MIN
                and self.contrib_ema[a] < FRAC_REM * self.ref_contrib.get(a, float("inf"))]
        for a in trig:                                               # the triggered test takes priority over the schedule
            others = [ex for ex in self.engine.slots if a in {self.unit_of(k) for k in (ex.add_key, ex.rem_key) if k}]
            if others and not all(ex.kind == "rem" and not getattr(ex, "trig", False) for ex in others):
                continue                                             # busy with an addition, swap, split or triggered test
            for ex in others:                                        # closes the periodic test (the persistent hypothesis stays)
                self.engine.slots.remove(ex)
                self.events.append((self.t, "superseded", ex.kind, ex.add_key, ex.rem_key, None, ex.n))
            if self.engine.free_slots() > 0:
                self._open_trig(a)
        super()._fill()                                              # original schedule, no other change


class TriggeredRemovalCore(TriggeredRemovalMixin, Core):
    """structural core with triggered removal tests (M4)"""


class TriggeredRemovalCheapCore(TriggeredRemovalMixin, CheapEquivalenceCore):
    """structural core with triggered removal tests (M4) and the cheap report of equivalent inputs (M5b): LEBRE v0.54"""


def core_class(cheap_equivalents, triggered_removal):
    """the structural core class for the chosen v0.54 changes (named classes, so states can be pickled)"""
    if triggered_removal:
        return TriggeredRemovalCheapCore if cheap_equivalents else TriggeredRemovalCore
    return CheapEquivalenceCore if cheap_equivalents else Core
