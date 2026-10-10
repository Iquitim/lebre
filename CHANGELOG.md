# Changelog

## 0.2.0 — 2026-10-10

Implements **LEBRE v0.53** (canonical configuration), promoted on 2026-10-09 by a pre-registered rule in the research
record (https://doi.org/10.5281/zenodo.23282527; promoted prototype at commit 4a2620e).

**Breaking:** `Lebre` is now the complete LEBRE v0.53. The structural core alone (LEBRE v0.52-r1, the whole of 0.1.0)
is `LebreCore`, or `Lebre(..., core_only=True)`; both give exactly the forecasts of 0.1.0. `pip install "lebre==0.1.0"`
remains available.

**New:**
- layer M1 (precision expert combined with the core forecast by AdaHedge on squared errors) and layer M2 (output
  starting from a declared trivial reference, two paths combined by an (A,B)-Prod; gate audit by an e-process);
- `reference=` ("zero", "persistence", "seasonal"; default seasonal if a season is declared, else persistence);
- `layer_weights` (weights of M1 and M2); gate-audit events with `change="gate"`;
- state files of 0.1.0 (`LebreCore`) still load with `Lebre.load`.

**Fidelity.** In the reference environment (Windows, CPython 3.11, NumPy 2.2.5), forecasts, interval, events, cost
accounting, coverage and the M1 weight are bit-for-bit identical to the promoted research prototype on ten scenarios
(`tests/test_v053.py`: the seven of 0.1.0 plus joint dynamics where M1 takes over, returns with reference zero and a
shock, and a random-walk level with a target gap). The core files are byte-for-byte those of 0.1.0.

**Differences from the research prototype** (no effect on results): only the promoted configuration was ported (the
prototype's development variants were left out); names in English; gate-audit events are labelled "adaptive" /
"reference".

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
