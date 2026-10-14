"""M1 dormancy (LEBRE v0.54, M5a): the precision expert sleeps while it carries no weight and wakes by evidence.

Active mode = exactly the M1 of LEBRE v0.53 (`PrecisionExpert`, unchanged). After the shadow period, if the expert's
weight omega stays below OMEGA_MIN for D_SLEEP consecutive learned targets, M1 sleeps: the output is L' = L; E is computed
only on a sample step every J learned targets; the regression is updated every J_RLS learned targets (on a sample step);
screening and reselection go on as in v0.53; AdaHedge is frozen. It wakes when the exponential mean (rate RATE) of the
improvement of E over L on the samples, normalised by the exponential mean of (y - L)^2, exceeds G_MIN with at least
N_WAKE samples; on waking, AdaHedge restarts with a new shadow of N_MIN updates. A change of the chosen inputs while
asleep wakes M1 on trial (AdaHedge restarted with a shadow); on trial it sleeps again after D_TRIAL consecutive learned
targets with omega < OMEGA_MIN, and if omega reaches OMEGA_MIN after the shadow the trial ends and D_SLEEP applies again.
Everything uses the past only.
Transcribed from the promoted research prototype (lebre-research, experiments/LEBRE-V0.54-PROTO-01/lebre054/dormancy.py,
commit f9906c4), keeping the order of every floating-point operation."""
import math

from ._aggregation import AdaHedge
from ._core import EVERY, LAM, N_MIN, T_MAX
from ._precision import PrecisionExpert

OMEGA_MIN, D_SLEEP, J, J_RLS = 1e-3, 2000, 8, 64
D_TRIAL = 500                # sleep threshold while a newly selected expert is on trial
RATE, G_MIN, N_WAKE = 0.01, 0.05, 100
FP_WEIGHT_CHECK = 15        # weights after the update (mixture, 14) and comparison (1)
FP_WAKE_SAMPLE = 11         # loss of E (2), improvement (1), two exponential means (6), ratio and comparison (2)


class DormantPrecisionExpert(PrecisionExpert):
    def __init__(self, d):
        super().__init__(d)
        self.dormant = False
        self.trial = False
        self.trials = 0
        self.low = 0
        self.sleeps = 0; self.wakes = 0; self.dormant_steps = 0
        self._sampled = False; self._last_sample = -1
        self.ema_g = None; self.ema_s = None; self.n_samples = 0

    # ----------------------------------------------------------------- prediction
    def predict(self, x, L):
        if not self.dormant:
            self._sampled = False
            return super().predict(x, L)
        self.dormant_steps += 1
        sample = self.n_obs % J == J - 1 and self.n_obs != self._last_sample
        if sample:
            self._last_sample = self.n_obs
            self._sampled = True
            super().predict(x, L)                                   # computes z and E (clipped); the output ignores E
            return L
        self._sampled = False
        x = self._fill(x)                                           # keeps the band history up to date
        self._x = x
        self.buf.append(x)
        if self.sel is not None:
            self.bufz.append((x[self.sel] - self.mu) / self.sd)
            self.fp += 2 * len(self.sel)
        self.z = None; self.e = None
        return L

    # ----------------------------------------------------------------- learning
    def _wake(self):
        self.dormant = False
        self.ag = AdaHedge(2)                                       # restarts with a new shadow of N_MIN updates
        self.low = 0; self.wakes += 1

    def _select(self):
        before = None if self.sel is None else tuple(self.sel)
        super()._select()
        if before is not None and tuple(self.sel) != before:
            self.low = 0
            if self.dormant:                                        # wakes on trial
                self._wake()
                self.trial = True; self.trials += 1

    def observe(self, y, L_core, learn, floor):
        if not self.dormant:
            super().observe(y, L_core, learn, floor)
            if learn and self.e is not None and self.ag.n >= N_MIN and not self.dormant:
                self.fp += FP_WEIGHT_CHECK
                if self.ag.weights()[1] < OMEGA_MIN:
                    self.low += 1
                    if self.low >= (D_TRIAL if self.trial else D_SLEEP):
                        self.dormant = True; self.sleeps += 1; self.low = 0; self.trial = False
                        self.ema_g = None; self.ema_s = None; self.n_samples = 0
                else:
                    self.low = 0; self.trial = False
            return
        e_core = (y - L_core) ** 2
        if self._sampled and self.e is not None:
            le = (y - self.e) ** 2
            if math.isfinite(e_core) and math.isfinite(le):
                g = e_core - le
                if self.ema_g is None:
                    self.ema_g, self.ema_s = g, e_core
                else:
                    self.ema_g += RATE * (g - self.ema_g); self.ema_s += RATE * (e_core - self.ema_s)
                self.n_samples += 1
            self.fp += FP_WAKE_SAMPLE
        self.e2_core = e_core if self.e2_core is None else self.e2_core + (1 - LAM) * (e_core - self.e2_core)
        self._floor = floor
        if learn:
            self.n_obs += 1
            if self.n_obs % EVERY == 0:
                self._screen(self._x, y)
                if self._sampled and self.z is not None and self.n_obs % J_RLS == 0:
                    self._rls(self.z, y)
            if self.n_obs >= N_MIN and (self.sel is None or self.n_obs - self.n_sel >= T_MAX):
                self._select()
        self.yobs.append(float(y))
        if self.dormant and self.n_samples >= N_WAKE and self.ema_s and self.ema_g / self.ema_s > G_MIN:
            self._wake()
