"""
Decline Curve Analysis (DCA) Models

Implements traditional oil & gas decline curve models:
- Exponential decline (Arps b=0)
- Hyperbolic decline (Arps 0<b<1)
- Harmonic decline (Arps b=1)
"""

from .arps import ArpsExponential, ArpsHyperbolic, ArpsHarmonic
from .exponential import ExponentialDecline
from .hyperbolic import HyperbolicDecline

__all__ = [
    "ArpsExponential",
    "ArpsHyperbolic",
    "ArpsHarmonic",
    "ExponentialDecline",
    "HyperbolicDecline",
]

