# Changelog

## 0.3.0 — 2026-10-14

Implements **LEBRE v0.54** (canonical configuration), promoted on 2026-10-13 by a pre-registered rule in the research
record (https://doi.org/10.5281/zenodo.23376152; frozen prototype at commit f9906c4: promoted code cfe72ce + errata 01,
which keeps the same forecasts).

**Changed:** `Lebre` is now LEBRE v0.54. Three changes, each with its own switch, all on by default:
- `dormancy`: M1 sleeps while its weight stays below 1e-3 for 2,000 targets (computed only on a sample step every 8,
  regression updated every 64) and wakes when the samples show its improvement over the core (M5a);
- `cheap_equivalents`: the report of equivalent inputs is updated by rows from one sample every 31 steps; no effect on
  forecasts or decisions (M5b, revision 2 of errata 01);
- `triggered_removal`: when a unit's mean contribution falls below half its peak, a removal test opens with a new
  hypothesis, per-episode evidence, its own e-LOND budget, at most 10,000 steps and at least 5,000 before acceptance; it
  takes priority over a periodic removal test of the same unit, which is closed as "superseded" (M4).

With the three switches off, the forecasts are those of 0.2.0 (LEBRE v0.53), bit for bit in the reference environment.
`core_only=True` now gives the structural core of v0.54; with `cheap_equivalents=False, triggered_removal=False` it is
the core of 0.1.0, as is `LebreCore` (unchanged). `pip install "lebre==0.2.0"` remains available.

**New:** `dormancy_state`; state format 3. State files of 0.2.0 still load with `Lebre.load` and go on as LEBRE v0.53;
those of 0.1.0 (`LebreCore`) too.

**Fidelity.** In the reference environment (Windows, CPython 3.11, NumPy 2.2.5), forecasts, interval, events, cost
accounting, coverage, the M1 weight, the dormancy state, the accepted triggered removals and the reports of equivalent
inputs are bit-for-bit identical to the promoted research prototype on thirteen scenarios (`tests/test_v054.py`: the ten
of 0.2.0 plus a target where M1 sleeps while a later joint dynamics is absorbed by the core, a relation that vanishes and
is removed by a triggered test, and forty inputs in correlated groups). The three new modules have the same syntax tree
as the prototype's, apart from names, comments and docstrings. The modules of 0.2.0 other than `full.py` and
`__init__.py` are unchanged.

**Scope and limits** (research record, specification §22): the cost cut is about 10% at the median and none where M1
helps; a relation that disappears for more than about 5,200 steps and returns is removed and must be re-detected (+5% to
+10% error on series with that pattern); spurious inputs in level series can still be accepted and the cycle must still
be declared.

**Differences from the research prototype** (no effect on results): only the promoted configuration was ported (the
prototype's cycle-as-hypothesis draft, off in v0.54, was left out); names in English; the core classes are named (so
states can be pickled) instead of built at run time.

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
