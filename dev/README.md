# dev/

`make_fixtures.py` regenerates `tests/fixtures/` from the **frozen research implementation**. It needs a checkout of the research record (https://github.com/Iquitim/lebre-research) and is run from its root:

    python packages/lebre/dev/make_fixtures.py

The fixtures are synthetic (seeds 5301–5307) and cover acceptance of inputs and of the residual state, hierarchical splits, swaps, removal tests, target gaps, out-of-contract inputs, quarantine, daily and weekly cycles, and a structureless target. An accepted removal and a "superseded" experiment do not occur in them. The removal code path is shared with the accepted swap.

`make_fixtures_v053.py` regenerates `tests/fixtures_v053/` from the **promoted research implementation of LEBRE v0.53** (research repository at commit 4a2620e or later; the prototype is frozen). Set `LEBRE053_PROTO` to its folder `experiments/LEBRE-V0.53-PROTO-01` and run, from this repository's root:

    python dev/make_fixtures_v053.py

It reuses the seven scenarios above and adds three (seeds 5311–5313) where M1 and M2 act: joint dynamics where M1 takes over, returns with reference zero and a shock, and a random-walk level with a target gap.

`make_fixtures_v054.py` regenerates `tests/fixtures_v054/` from the **promoted research implementation of LEBRE v0.54** (research repository at commit f9906c4 or later; the prototype is frozen). Set `LEBRE054_PROTO` to its folder `experiments/LEBRE-V0.54-PROTO-01` and run, from this repository's root:

    python dev/make_fixtures_v054.py

It reuses the ten scenarios above and adds three (seeds 5461–5463): a target where M1 sleeps while a joint dynamics that appears later is absorbed by the core (M1 stays asleep), a relation that vanishes for good and is removed by a triggered test, and forty inputs in four-input correlated groups, with the reports of equivalent inputs queried in the middle of an update cycle and at the end. Across the thirteen scenarios M1 sleeps, wakes by evidence (`hourly_gaps`, `random_walk_level`) and wakes on trial after a reselection (`many_inputs`, `regime_change`, `many_inputs_equivalents`), and triggered removals are accepted (`effect_vanishes`, `regime_change`, `relation_vanishes`). A periodic removal test superseded by a triggered one does not occur in them.
