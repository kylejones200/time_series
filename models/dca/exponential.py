#!/usr/bin/env python3
"""Exponential decline curve model (alias for ArpsExponential)."""

from .arps import ArpsExponential

ExponentialDecline = ArpsExponential

__all__ = ["ExponentialDecline"]

