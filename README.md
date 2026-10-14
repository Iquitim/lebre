# lebre

[![tests](https://github.com/Iquitim/lebre/actions/workflows/tests.yml/badge.svg)](https://github.com/Iquitim/lebre/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/lebre)](https://pypi.org/project/lebre/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23073983.svg)](https://doi.org/10.5281/zenodo.23073983)

Online one-step-ahead forecasting for a target driven by exogenous inputs, with **statistically tested structural changes**, a **precision expert** and a **trivial-reference starting point**, at a fixed cost of a few hundred to under two thousand floating-point operations per step.

`lebre 0.3.0` implements **LEBRE v0.54**, the research version promoted on 2026-10-13 by a pre-registered rule (with errata 01 of 2026-10-14). The research record (code, pre-registrations, evaluations, stress audit, self-contained specification) is https://doi.org/10.5281/zenodo.23376152 (repository: https://github.com/Iquitim/lebre-research) (specification: `docs/architecture/pdf/LEBRE_ARCHITECTURE_v0.54_SPEC_EN.pdf`).

The package reproduces the promoted research code **bit for bit in the reference environment** (Windows, CPython 3.11, NumPy 2.2.5), checked against references generated from that code. On other platforms and NumPy versions, floating-point rounding differs in the last bits: the regression tests then require identical decisions and forecasts equal within 1e-9 of their scale.

```bash
pip install lebre            # 0.3.0: LEBRE v0.54
pip install "lebre==0.2.0"   # previous releases stay available (0.2.0: LEBRE v0.53; 0.1.0: LEBRE v0.52-r1)
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
model.layer_weights                         # {'m1': weight of the precision expert, 'h', 'w', 's': weights of M2, ...}
model.dormancy_state                        # {'dormant': True/False, 'sleeps', 'wakes', 'trials', 'dormant_steps'}
model.equivalent_inputs("x0")               # inputs whose causal correlation with x0 is >= 0.95
model.coverage, model.cost_per_step         # empirical 90%-interval coverage; analytic operations per step
model.save("state.pkl"); model = Lebre.load("state.pkl")   # pickle: load only files you trust
```

**The trivial reference.** Layer M2 starts from a reference: `reference="seasonal"` (value one season ago; the default when a season is declared), `"persistence"` (last observed value; the default otherwise) or `"zero"` — declare `"zero"` only for targets that are changes, such as financial returns.

**The previous versions.** `Lebre(..., dormancy=False, cheap_equivalents=False, triggered_removal=False)` gives exactly the forecasts of `lebre 0.2.0` (LEBRE v0.53). `LebreCore`, the same class as in 0.1.0, gives exactly the forecasts of `lebre 0.1.0` (the structural core alone, at about half the cost); `Lebre(..., core_only=True)` is the structural core of v0.54 (with `cheap_equivalents=False, triggered_removal=False`, that of 0.1.0).

See `examples/quickstart.py`.

## What it does

- **Structural core** (LEBRE v0.52-r1, with two changes in v0.54):
  - a memory of the target's own past (level, declared cycle(s), per-phase profile) and a linear regression on the current inputs, combined by dynamic model averaging;
  - structural changes run in shadow as experiments (adding, removing or swapping a whole input, or refining its lag response by hierarchical splits of octave bands), accepted only when an anytime-valid e-process of predictive improvement crosses a level fixed by an online multiple-testing rule;
  - **triggered removal tests** (v0.54): when a unit's mean contribution falls below half its peak, a removal test opens with a new hypothesis, evidence from that episode only and its own error budget (the original periodic tests stay);
  - a **cheap report of equivalent inputs** (v0.54): updated by rows from one sample every 31 steps, with no effect on forecasts or decisions;
  - output safeguards: target gaps are held, and forecasts stay within an envelope of the observed values.
- **M1, precision expert:** a recursive least-squares regression on the target's last two values and lag-band means of the (up to five) most correlated inputs, combined with the core forecast by AdaHedge on squared errors; where it does not help, its weight goes to zero. **Dormancy** (v0.54): while its weight stays below 10⁻³ for 2,000 targets, M1 sleeps (computed only on sample steps) and wakes when the samples show evidence that it would help.
- **M2, trivial-reference starting point:** the output mixes the reference and the adaptive forecast through a conservative path (AdaHedge) and a one-way switch, combined by an (A,B)-Prod that guarantees not falling far behind the conservative path. An e-process audits the switches (logged as `change="gate"` events).
- **Inputs** are standardised causally inside the model; set `standardize=False` only if you already do this with past-only statistics.

## Scope and limits

Stated honestly; details in the specification (§22):
- one step ahead only; few inputs, approximately linear relations;
- in the pre-registered final evaluation of v0.54 (93 series reserved before any v0.54 code: basins, buildings, wind and solar plants, EIA price and US Treasury rate levels), the error was kept against v0.53 (ratio 0.989 to 1.000 per family, no harm in any family) and the median per-series cost was 0.905 of v0.53's (95% CI 0.857–0.932); there is no cost cut where M1 helps (basins). In the previous evaluation (104 series), LEBRE v0.53 had 4.5% lower mean squared error than the core alone (ratio 0.955, 95% CI 0.951–0.960), with the gain from basins and FX;
- a relation that disappears for more than about 5,200 steps and returns is removed and must be re-detected: +5% to +10% error on series with that pattern (declared limit of v0.54);
- spurious inputs in level series can still be accepted, and the cycle must be declared;
- it is **not** meant to be the most accurate forecaster: against Chronos-2 with covariates (a foundation model of ~10⁹ operations per forecast) the error ratio of v0.54 ranged from 0.84 to 1.34 per family;
- cost: median 870 and maximum 1,245 operations per step (mean per series) in the final evaluation of v0.54; per-step peak at most 1.054 times v0.53's in the stress audit; state memory estimated at ~85 KB (the core was measured on a simulated Cortex-M4F; M1, M2 and the v0.54 changes were not);
- the core's testing mechanism does not improve accuracy over switching all inputs on; an accepted input means *it improved the forecast of the current model, with evidence*, not *it is the cause*;
- no formal error guarantee in the original scale for M1 and M2; with large single-step shocks no online combiner has a useful guarantee;
- no component is original; the contribution is the integration under a declared cost and the pre-registered evidence.

## Licence

Apache-2.0.

## Citation

If you use the library, please cite the research record it implements and the library version you used (see `CITATION.cff`):

- research record of LEBRE v0.54: https://doi.org/10.5281/zenodo.23376152 (repository: https://github.com/Iquitim/lebre-research);
- `lebre 0.3.0`: `10.5281/zenodo.23073983` until its version DOI is minted; `lebre 0.2.0`: https://doi.org/10.5281/zenodo.23282624; `lebre 0.1.0`: https://doi.org/10.5281/zenodo.23073984; `10.5281/zenodo.23073983` resolves to the latest version of the library.
