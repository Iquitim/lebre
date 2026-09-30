# lebre

[![tests](https://github.com/Iquitim/lebre/actions/workflows/tests.yml/badge.svg)](https://github.com/Iquitim/lebre/actions/workflows/tests.yml)

Online one-step-ahead forecasting for a target driven by exogenous inputs, with **statistically tested structural changes**, within a budget of a few hundred floating-point operations per step.

`lebre 0.1.0` implements **LEBRE v0.52-r1**, the frozen research version documented and evaluated in the research record:
- repository: https://github.com/Iquitim/lebre-research
- DOI: [10.5281/zenodo.23049103](https://doi.org/10.5281/zenodo.23049103)

The package reproduces the research code **bit for bit**. This is checked by the regression tests against references generated from the frozen code, and on real evaluation series in the research repository.

```bash
pip install lebre      # not yet published; for now: pip install ./packages/lebre
```

## Usage

```python
from lebre import Lebre

model = Lebre(n_inputs=3, season=24)        # season: declared cycle in steps (None if there is none)
for x_t, y_t in stream:                     # x_t: current inputs (length 3), y_t: target (None/NaN if missing)
    f = model.predict(x_t)                  # Forecast(value, lower, upper, t), issued before y_t is known
    model.observe(y_t)
    # or: f = model.step(x_t, y_t)

model.structure                             # e.g. ['x0', 'own_past']
model.response("x0")                        # recovered lag response: [((lag_lo, lag_hi), weight per lag), ...]
model.events                                # audit log: Event(t, outcome, change, added, removed, log_evidence, samples)
model.equivalent_inputs("x0")               # inputs indistinguishable from x0 (causal correlation >= 0.95)
model.coverage, model.cost_per_step         # empirical 90%-interval coverage; analytic operations per step
model.save("state.pkl"); model = Lebre.load("state.pkl")   # pickle: load only files you trust
```

See `examples/quickstart.py`.

## What it does

- **Live model:**
  - a memory of the target's own past: level, declared cycle(s), per-phase profile;
  - a linear regression on the current inputs;
  - the two combined by dynamic model averaging.
- **Structural changes run in shadow as experiments:** adding, removing or swapping a whole input, or refining its lag response by hierarchical splits of octave bands.
- **Acceptance:** a change is accepted only when an anytime-valid e-process of predictive improvement crosses a level fixed by an online multiple-testing rule.
- **Output safeguards:** target gaps are held (never filled with the model's own forecasts), and forecasts stay within an envelope of the observed values.
- **Inputs:** they are standardised causally inside the model; set `standardize=False` only if you already do this with past-only statistics.

## Scope and limits

These are the limits of the evaluated version, stated honestly (details in the specification, §19):
- evaluated on hydrology and building-energy series with 1–5 inputs, one step ahead;
- it is **not** meant to be the most accurate forecaster: in the evaluations it tied a SARIMAX with inputs fitted per series and had ~25% higher error than Chronos-2 with covariates, at a tiny fraction of the cost;
- the testing mechanism does not improve accuracy over switching all inputs on; its role is an audited, evidence-backed structure;
- an accepted input means *it improved the forecast of the current model, with evidence*, not *it is the cause*: correlated substitutes can be accepted, and near-identical inputs are indistinguishable;
- the e-values are valid under the stated assumptions; false-change control with the adaptive reference was verified empirically;
- delays of very smooth inputs are often not discovered (information limit), though forecasts remain good;
- the declared cost contract of the research version (mean ≤ 400, peak ≤ 1,000 operations per step) was narrowly not met.

## Licence

Apache-2.0. If you use it, please cite the research record (`CITATION.cff`).
