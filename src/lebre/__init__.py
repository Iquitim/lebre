"""LEBRE — online one-step-ahead forecasting with statistically tested structural changes, within a small
computational budget.

This package implements **LEBRE v0.52-r1** (canonical configuration) exactly as specified and evaluated in the research
record https://doi.org/10.5281/zenodo.23049103. The forecasts are bit-for-bit identical to the frozen research code.
"""
from .model import Event, Forecast, Lebre

__version__ = "0.1.0"
ALGORITHM_VERSION = "LEBRE v0.52-r1"

__all__ = ["Lebre", "Forecast", "Event", "__version__", "ALGORITHM_VERSION"]
