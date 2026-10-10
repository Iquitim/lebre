"""LEBRE — online one-step-ahead forecasting with statistically tested structural changes, a precision expert and a
trivial-reference starting point.

`Lebre` implements **LEBRE v0.53** (structural core + M1 + M2, canonical configuration), promoted by a pre-registered
rule in the research record (https://github.com/Iquitim/lebre-research). `LebreCore` is the structural core alone,
**LEBRE v0.52-r1**, unchanged from lebre 0.1.0 (also available as ``Lebre(..., core_only=True)``). In the reference
environment (Windows, CPython 3.11, NumPy 2.2.5) the forecasts are bit-for-bit identical to the frozen research code;
elsewhere they agree up to floating-point rounding, with identical structural decisions in the tests.
"""
from .full import Lebre
from .model import Event, Forecast
from .model import Lebre as LebreCore

__version__ = "0.2.0"
ALGORITHM_VERSION = "LEBRE v0.53"
CORE_ALGORITHM_VERSION = "LEBRE v0.52-r1"

__all__ = ["Lebre", "LebreCore", "Forecast", "Event", "__version__", "ALGORITHM_VERSION", "CORE_ALGORITHM_VERSION"]
