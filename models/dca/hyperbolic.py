#!/usr/bin/env python3
"""Hyperbolic decline curve model (alias for ArpsHyperbolic)."""

from .arps import ArpsHyperbolic

HyperbolicDecline = ArpsHyperbolic

__all__ = ["HyperbolicDecline"]

