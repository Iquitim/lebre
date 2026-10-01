# Changelog

## 0.1.0 — 2026-10-01

First release. Implements **LEBRE v0.52-r1** (canonical configuration) as frozen in the research record (https://doi.org/10.5281/zenodo.23049103).

**Fidelity.** In the reference environment (Windows, CPython 3.11, NumPy 2.2.5), forecasts, interval, events and cost accounting are bit-for-bit identical to the frozen research code. On other platforms and NumPy versions, rounding differs in the last bits; the tests then require identical structural decisions and forecasts within 1e-9 of their scale. This is checked on:
- seven synthetic scenarios (`tests/test_regression.py`);
- four real reserve-3 series, in the research repository (`scripts/check_lebre_package.py`).

**API:**
- `Lebre(n_inputs, season, season2)`;
- `predict` / `observe` / `step`;
- adaptive 90% interval;
- audit `events`, `structure`, `response`, `equivalent_inputs`;
- `coverage`, `cost_per_step`;
- `save` / `load`.

**Differences from the research code** (no effect on results):
- Branches used only by research ablations were removed.
- The single `step` was split into `predict` + `observe`.
- Interval hits are counted instead of stored.
- Missing input values (NaN) do not update the input standardiser; the research pipeline assumed complete inputs.
