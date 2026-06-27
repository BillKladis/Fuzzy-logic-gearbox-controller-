"""Fuzzy logic controller for automatic gear changing.

A compact, dependency-light (NumPy only) Mamdani fuzzy inference system that
decides when an automatic gearbox should shift up, hold, or shift down, given
engine rpm, throttle, brake, road speed and a live driving-style estimate.
Everything is parameterised by a configurable engine torque curve and
drivetrain, so the shift map adapts to the hardware.
"""

from .engine import (
    DEFAULT_DRIVETRAIN,
    DEFAULT_ENGINE,
    Drivetrain,
    EngineSpec,
    optimal_power_gear,
)
from .controller import (
    Decision,
    FuzzyGearController,
    Inputs,
    Transmission,
    estimate_style,
)
from .variables import MODES, resolve_anchors

__all__ = [
    "EngineSpec",
    "Drivetrain",
    "DEFAULT_ENGINE",
    "DEFAULT_DRIVETRAIN",
    "optimal_power_gear",
    "FuzzyGearController",
    "Transmission",
    "Inputs",
    "Decision",
    "estimate_style",
    "MODES",
    "resolve_anchors",
]

__version__ = "0.1.0"
