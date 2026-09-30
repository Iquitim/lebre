"""Quickstart: forecast a target driven by one delayed input, with a missing stretch of the target.

The target depends on input x0 with a delay of 12 steps; x1 is irrelevant. LEBRE learns online, forecasts one step
ahead and logs every structural change with its evidence.
"""
import numpy as np

from lebre import Lebre

rng = np.random.default_rng(0)
T = 20000
X = np.zeros((T, 2))
for t in range(1, T):
    X[t] = 0.5 * X[t - 1] + np.sqrt(0.75) * rng.standard_normal(2)
y = 0.8 * np.r_[np.zeros(12), X[:-12, 0]] + 0.5 * rng.standard_normal(T)
y[8000:8100] = np.nan                                   # a gap in the target

model = Lebre(n_inputs=2)
err = []
for t in range(T):
    f = model.predict(X[t])                             # forecast before seeing y[t]
    model.observe(None if np.isnan(y[t]) else y[t])     # then reveal y[t] (None = missing)
    if t >= T // 2 and not np.isnan(y[t]):
        err.append((y[t] - f.value) ** 2)

print(f"MSE on the second half: {np.mean(err):.3f} (noise variance 0.25)")
print(f"90% interval coverage: {model.coverage:.3f}")
print(f"active structure: {model.structure}")
print("recovered response of x0 (lags, weight per lag):", [(seg, round(w, 3)) for seg, w in model.response("x0")])
print(f"analytic cost: {model.cost_per_step:.0f} floating-point operations per step")
for e in model.events:
    if e.outcome == "accepted":
        print(f"  t={e.t}: {e.change} {e.added} (log evidence {e.log_evidence}, {e.samples} samples)")
