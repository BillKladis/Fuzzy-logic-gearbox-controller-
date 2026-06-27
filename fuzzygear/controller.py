"""The fuzzy gear-shift controller and a small transmission model.

Pipeline (a textbook Mamdani inference system):

    crisp inputs --> [fuzzify] --> [rule evaluation] --> [aggregate]
                 --> [defuzzify, centroid] --> crisp shift tendency

A thin state machine (:class:`Transmission`) turns the continuous shift
tendency into discrete gear changes with hysteresis and a minimum dwell time,
so the gearbox does not "hunt" back and forth between two gears.

Both the controller and the transmission are parameterised by an
:class:`~fuzzygear.engine.EngineSpec` / :class:`~fuzzygear.engine.Drivetrain`,
so the very same logic adapts to whatever engine and gearing you hand it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .membership import fuzzy_and, fuzzy_or
from .engine import DEFAULT_DRIVETRAIN, DEFAULT_ENGINE, Drivetrain, EngineSpec
from .variables import (
    BRAKE_SETS,
    MODES,
    SHIFT_DOMAIN,
    SHIFT_SETS,
    SPEED_SETS,
    THROTTLE_SETS,
    rpm_sets,
)


@dataclass
class Inputs:
    """One snapshot of the crisp controller inputs."""

    rpm: float
    throttle: float          # 0-100 %
    brake: float             # 0-100 %
    speed: float             # km/h
    style: float = 0.0       # 0 (calm) .. 1 (aggressive)


@dataclass
class Decision:
    """The controller's verdict for one time step."""

    shift_value: float                     # defuzzified tendency in [-1, 1]
    memberships: dict = field(default_factory=dict)
    rule_activations: dict = field(default_factory=dict)


# Discretisation of the output universe used by the centroid integral.
_SHIFT_GRID = np.linspace(SHIFT_DOMAIN[0], SHIFT_DOMAIN[1], 401)


class FuzzyGearController:
    """Mamdani fuzzy controller mapping engine/driver state to a shift cue."""

    def __init__(self, mode: str = "comfort", engine: EngineSpec = DEFAULT_ENGINE):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {list(MODES)}")
        self.mode = MODES[mode]
        self.engine = engine

    # -- step 1: fuzzification --------------------------------------------
    def fuzzify(self, x: Inputs) -> dict:
        rpm_mf = rpm_sets(self.engine, self.mode, x.style)
        # Clamp rpm into the engine's range so the partition always covers it.
        rpm_clamped = min(max(x.rpm, 0.0), self.engine.redline)
        return {
            "rpm": {k: float(f(rpm_clamped)) for k, f in rpm_mf.items()},
            "throttle": {k: float(f(x.throttle)) for k, f in THROTTLE_SETS.items()},
            "brake": {k: float(f(x.brake)) for k, f in BRAKE_SETS.items()},
            "speed": {k: float(f(x.speed)) for k, f in SPEED_SETS.items()},
        }

    # -- step 2 + 3: rule evaluation and aggregation ----------------------
    def _rules(self, mu: dict) -> dict:
        """Evaluate the rule base, return firing strength per output set.

        Each rule is an IF-THEN statement.  The antecedent strength is a
        min (AND) / max (OR) combination of input memberships; the strongest
        rule firing each consequent wins (max-aggregation).
        """
        rpm, thr, brk = mu["rpm"], mu["throttle"], mu["brake"]
        kd = self.mode.kickdown_gain

        rules = [
            # --- upshift: revs are up, no demand for power, and not braking
            ("upshift", fuzzy_and(rpm["medium"], thr["light"], brk["none"])),
            ("upshift", fuzzy_and(rpm["high"], thr["light"], brk["none"])),
            ("upshift", fuzzy_and(rpm["high"], thr["moderate"], brk["none"])),
            # over-rev guard: near redline, shift up under *any* throttle
            ("upshift", rpm["redline"]),
            # --- hold: comfortable cruising band (and gentle braking)
            ("hold", fuzzy_and(rpm["medium"], thr["moderate"])),
            ("hold", fuzzy_and(rpm["low"], thr["light"])),
            ("hold", brk["moderate"]),
            # --- downshift: engine lugging / kickdown for power / engine brake
            # Kickdown fires only at *low* rpm, where dropping a gear genuinely
            # finds power without risking an over-rev; ``kd`` sets its eagerness.
            ("downshift", fuzzy_and(rpm["low"], thr["moderate"])),
            ("downshift", min(1.0, fuzzy_and(rpm["low"], thr["heavy"]) * (1.0 + kd))),
            ("downshift", brk["hard"]),
        ]

        agg = {"downshift": 0.0, "hold": 0.0, "upshift": 0.0}
        for consequent, strength in rules:
            agg[consequent] = fuzzy_or(agg[consequent], strength)
        return agg

    # -- step 4: defuzzification (centre of gravity) ----------------------
    def _defuzzify(self, agg: dict) -> float:
        # Clip each output set at its firing strength, take the union (max),
        # then return the centroid of the resulting area.
        clipped = np.zeros_like(_SHIFT_GRID)
        for name, strength in agg.items():
            if strength <= 0.0:
                continue
            mf = np.minimum(SHIFT_SETS[name](_SHIFT_GRID), strength)
            clipped = np.maximum(clipped, mf)
        area = clipped.sum()
        if area == 0.0:
            return 0.0
        return float((clipped * _SHIFT_GRID).sum() / area)

    def decide(self, x: Inputs) -> Decision:
        mu = self.fuzzify(x)
        agg = self._rules(mu)
        shift = self._defuzzify(agg)
        return Decision(shift_value=shift, memberships=mu, rule_activations=agg)


@dataclass
class Transmission:
    """Turns a continuous shift tendency into discrete, debounced gears."""

    drivetrain: Drivetrain = field(default_factory=lambda: DEFAULT_DRIVETRAIN)
    engine: EngineSpec = field(default_factory=lambda: DEFAULT_ENGINE)
    gear: int = 1
    up_threshold: float = 0.33
    down_threshold: float = -0.33
    protect_threshold: float = 0.6  # over-rev guard bypasses the dwell timer
    min_dwell: float = 0.6          # seconds to wait between shifts
    _time_in_gear: float = 0.0

    @property
    def n_gears(self) -> int:
        return self.drivetrain.n_gears

    def engine_rpm(self, speed_kmh: float, gear: int | None = None) -> float:
        """Engine speed implied by road speed and the engaged gear."""
        g = self.gear if gear is None else gear
        return self.drivetrain.engine_rpm(speed_kmh, g, self.engine.idle_rpm)

    def _would_over_rev(self, speed_kmh: float | None) -> bool:
        """True if dropping a gear at this speed would exceed the redline."""
        if speed_kmh is None:
            return False
        return self.engine_rpm(speed_kmh, self.gear - 1) > self.engine.redline * 0.98

    def settle_gear(self, speed_kmh: float) -> int:
        """Pick a sensible starting gear for a given road speed.

        The highest gear that still keeps the engine comfortably above idle --
        avoids the unrealistic "pull away in 1st from 100 km/h" startup spike.
        """
        self.gear = 1
        for g in range(self.n_gears, 0, -1):
            if self.engine_rpm(speed_kmh, g) >= self.engine.idle_rpm * 1.6:
                self.gear = g
                break
        return self.gear

    def update(self, shift_value: float, dt: float, speed_kmh: float | None = None) -> int:
        """Advance the gearbox by ``dt`` seconds given a shift tendency.

        ``speed_kmh`` (optional) enables over-rev protection: a downshift that
        would spin the engine past the redline is suppressed, exactly as a real
        automatic refuses an unsafe kickdown.
        """
        self._time_in_gear += dt
        # The over-rev guard (a near-redline upshift) overrides the dwell timer
        # so the engine bounces off the limiter rather than spinning past it.
        forced_up = shift_value >= self.protect_threshold and self.gear < self.n_gears
        if self._time_in_gear < self.min_dwell and not forced_up:
            return self.gear
        if shift_value >= self.up_threshold and self.gear < self.n_gears:
            self.gear += 1
            self._time_in_gear = 0.0
        elif (shift_value <= self.down_threshold and self.gear > 1
              and not self._would_over_rev(speed_kmh)):
            self.gear -= 1
            self._time_in_gear = 0.0
        return self.gear


def estimate_style(throttle_history, brake_history, alpha: float = 0.08) -> float:
    """Cheap driving-style estimate in [0, 1].

    An exponentially weighted blend of how hard the throttle and brake have
    been worked recently -- a proxy for "aggressiveness" that the controller
    feeds back into the RPM partition so a spirited driver holds gears longer.
    """
    style = 0.0
    for thr, brk in zip(throttle_history, brake_history):
        instantaneous = 0.7 * (thr / 100.0) + 0.3 * (brk / 100.0)
        style = (1 - alpha) * style + alpha * instantaneous
    return float(np.clip(style * 1.8, 0.0, 1.0))
