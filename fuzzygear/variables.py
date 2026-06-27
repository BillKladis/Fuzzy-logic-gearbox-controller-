"""Linguistic variables and fuzzy sets for the gear controller.

Each engine/driver metric is modelled as a *linguistic variable* partitioned
into overlapping fuzzy sets (e.g. RPM -> {low, medium, high}).  The RPM
partition is **engine-relative**: its anchors are derived from the engine's
torque-peak, power-peak and redline, and from the driving mode.  The same
crisp engine speed is therefore "high" much sooner in ECO than in SPORT, and
the whole map slides automatically when a different engine is plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .membership import triangular, trapezoidal
from .engine import EngineSpec


@dataclass(frozen=True)
class DriveMode:
    """A driving mode expressed *relative to engine landmarks*.

    Attributes
    ----------
    name : str
        eco / comfort / sport.
    up_blend : float
        Position of the relaxed-upshift anchor along
        ``[torque_peak_rpm .. redline]``: 0 sits at the torque peak (shift
        early), 1 at the redline (wring it out).
    down_frac : float
        Position of the downshift anchor along ``[idle .. torque_peak_rpm]``;
        higher keeps the engine spinning faster before it is allowed to lug.
    style_gain : float
        Fraction of the remaining range to redline that an aggressive driving
        style may push the upshift anchor upward (hold gears longer).
    kickdown_gain : float
        Extra eagerness to drop a gear under heavy throttle.
    """

    name: str
    up_blend: float
    down_frac: float
    style_gain: float
    kickdown_gain: float


MODES = {
    "eco": DriveMode("eco", up_blend=0.06, down_frac=0.25,
                     style_gain=0.20, kickdown_gain=0.15),
    "comfort": DriveMode("comfort", up_blend=0.32, down_frac=0.45,
                         style_gain=0.30, kickdown_gain=0.25),
    "sport": DriveMode("sport", up_blend=0.72, down_frac=0.72,
                       style_gain=0.35, kickdown_gain=0.40),
}


def resolve_anchors(mode: DriveMode, engine: EngineSpec, style: float = 0.0):
    """Map a (mode, engine, style) triple to crisp (down_rpm, up_rpm).

    This is where elasticity lives: the linguistic anchors are computed from
    *this* engine's landmarks, so the shift logic adapts to the hardware.
    """
    tq, redline, idle = engine.peak_torque_rpm, engine.redline, engine.idle_rpm
    up = tq + mode.up_blend * (redline - tq)
    up = up + style * mode.style_gain * (redline - up)
    up = min(up, redline * 0.97)
    down = idle + mode.down_frac * (tq - idle)
    return down, up


def rpm_sets(engine: EngineSpec, mode: DriveMode, style: float = 0.0):
    """Return the {low, medium, high, redline} RPM membership functions.

    ``up`` is the *light-throttle* upshift anchor (peak of ``medium``).  The
    ``high`` set is the wind-out / power band, and ``redline`` is the
    over-rev guard that forces an upshift under any throttle.
    """
    down, up = resolve_anchors(mode, engine, style)
    redline = engine.redline
    # The "low / lugging" band is a property of the engine, not of the shift
    # map: it reaches toward the upshift anchor for good partition coverage,
    # but is capped at the power peak so a kickdown can never over-rev.
    low_top = min(0.5 * (engine.peak_torque_rpm + up), engine.peak_power_rpm)
    low_top = max(low_top, down * 1.05)
    hi = 0.5 * (up + redline)
    return {
        "low": lambda x: trapezoidal(x, 0.0, 0.0, down, low_top),
        "medium": lambda x: triangular(x, down, up, hi),
        "high": lambda x: triangular(x, up, hi, redline),
        "redline": lambda x: trapezoidal(x, hi, redline, redline, redline),
    }


# Throttle pedal, 0-100 %.
THROTTLE_SETS = {
    "light": lambda x: trapezoidal(x, 0.0, 0.0, 10.0, 35.0),
    "moderate": lambda x: triangular(x, 20.0, 50.0, 80.0),
    "heavy": lambda x: trapezoidal(x, 60.0, 85.0, 100.0, 100.0),
}

# Brake pedal, 0-100 %.
BRAKE_SETS = {
    "none": lambda x: trapezoidal(x, 0.0, 0.0, 5.0, 20.0),
    "moderate": lambda x: triangular(x, 10.0, 40.0, 70.0),
    "hard": lambda x: trapezoidal(x, 50.0, 80.0, 100.0, 100.0),
}

# Vehicle speed, 0-200 km/h -- context for the "creep / launch" logic.
SPEED_SETS = {
    "low": lambda x: trapezoidal(x, 0.0, 0.0, 15.0, 45.0),
    "medium": lambda x: triangular(x, 30.0, 80.0, 130.0),
    "high": lambda x: trapezoidal(x, 110.0, 160.0, 200.0, 200.0),
}

# Output variable: shift tendency in [-1, +1].
#   -1 -> downshift,  0 -> hold,  +1 -> upshift.
SHIFT_DOMAIN = (-1.0, 1.0)
SHIFT_SETS = {
    "downshift": lambda x: trapezoidal(x, -1.0, -1.0, -0.6, -0.15),
    "hold": lambda x: triangular(x, -0.4, 0.0, 0.4),
    "upshift": lambda x: trapezoidal(x, 0.15, 0.6, 1.0, 1.0),
}
