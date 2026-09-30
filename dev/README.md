# dev/

`make_fixtures.py` regenerates `tests/fixtures/` from the **frozen research implementation**. It needs a checkout of the research record (https://github.com/Iquitim/lebre-research) and is run from its root:

    python packages/lebre/dev/make_fixtures.py

The fixtures are synthetic (seeds 5301–5307) and cover acceptance of inputs and of the residual state, hierarchical splits, swaps, removal tests, target gaps, out-of-contract inputs, quarantine, daily and weekly cycles, and a structureless target. An accepted removal and a "superseded" experiment do not occur in them. The removal code path is shared with the accepted swap.
