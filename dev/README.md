# dev/

`make_fixtures.py` regenerates `tests/fixtures/` from the **frozen research implementation**. It needs a checkout of the research record (https://github.com/Iquitim/lebre-research) and is run from its root:

    python packages/lebre/dev/make_fixtures.py

The fixtures are synthetic (seeds 5301–5307) and cover acceptance of inputs and of the residual state, hierarchical splits, swaps, removal tests, target gaps, out-of-contract inputs, quarantine, daily and weekly cycles, and a structureless target. An accepted removal and a "superseded" experiment do not occur in them. The removal code path is shared with the accepted swap.

`make_fixtures_v053.py` regenerates `tests/fixtures_v053/` from the **promoted research implementation of LEBRE v0.53** (research repository at commit 4a2620e or later; the prototype is frozen). Set `LEBRE053_PROTO` to its folder `experiments/LEBRE-V0.53-PROTO-01` and run, from this repository's root:

    python dev/make_fixtures_v053.py

It reuses the seven scenarios above and adds three (seeds 5311–5313) where M1 and M2 act: joint dynamics where M1 takes over, returns with reference zero and a shock, and a random-walk level with a target gap.
