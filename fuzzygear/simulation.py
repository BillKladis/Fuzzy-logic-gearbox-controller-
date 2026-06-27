"""Synthetic driving cycles and a closed-loop simulator.

A *driving cycle* is a time series of road speed plus driver pedal inputs.
The simulator feeds each sample through the fuzzy controller, advances the
transmission, and recomputes engine rpm from the engaged gear -- closing the
loop so an upshift visibly drops the revs, exactly like a real car.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import FuzzyGearController, Inputs, Transmission, estimate_style
from .engine import (
    DEFAULT_DRIVETRAIN,
    DEFAULT_ENGINE,
    Drivetrain,
    EngineSpec,
    optimal_power_gear,
)


@dataclass
class Cycle:
    """A synthetic driving scenario sampled on a uniform time grid."""

    name: str
    t: np.ndarray
    speed: np.ndarray        # km/h
    throttle: np.ndarray     # 0-100 %
    brake: np.ndarray        # 0-100 %

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def _ramp(t, t0, t1, v0, v1):
    """Linear segment helper, flat outside [t0, t1]."""
    frac = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    return v0 + (v1 - v0) * frac


def urban_cycle(duration=60.0, dt=0.1) -> Cycle:
    """Stop-and-go city driving: gentle launches, cruising, braking to rest."""
    t = np.arange(0.0, duration, dt)
    speed = np.zeros_like(t)
    throttle = np.zeros_like(t)
    brake = np.zeros_like(t)
    speed += _ramp(t, 2, 12, 0, 50) * (t < 20)
    speed += (50 + 8 * np.sin(0.6 * (t - 20))) * ((t >= 20) & (t < 35))
    speed += _ramp(t, 35, 45, 55, 0) * (t >= 35)
    speed = np.clip(speed, 0, None)
    throttle = np.clip(35 + 25 * np.sin(0.5 * t), 0, 100) * (np.gradient(speed, dt) > -1)
    brake = np.clip(-12 * np.gradient(speed, dt), 0, 100)
    throttle = np.where(brake > 5, 0.0, throttle)
    return Cycle("Urban stop-and-go", t, speed, throttle, brake)


def highway_onramp(duration=45.0, dt=0.1) -> Cycle:
    """Hard acceleration onto a motorway, then a steady cruise."""
    t = np.arange(0.0, duration, dt)
    speed = _ramp(t, 1, 18, 10, 130) + _ramp(t, 18, 45, 0, 6)
    throttle = np.clip(_ramp(t, 0, 6, 60, 95) - _ramp(t, 16, 22, 0, 70), 12, 100)
    brake = np.zeros_like(t)
    return Cycle("Highway on-ramp", t, speed, throttle, brake)


def spirited_run(duration=40.0, dt=0.1) -> Cycle:
    """Brisk back-road driving: bursts of throttle and trail braking."""
    t = np.arange(0.0, duration, dt)
    speed = 60 + 35 * np.sin(0.25 * t) + 10 * np.sin(0.8 * t)
    speed = np.clip(speed, 20, None)
    accel = np.gradient(speed, dt)
    throttle = np.clip(45 + 6 * accel, 0, 100)
    brake = np.clip(-6 * accel, 0, 100)
    throttle = np.where(brake > 5, 0.0, throttle)
    return Cycle("Spirited back-road", t, speed, throttle, brake)


@dataclass
class SimResult:
    cycle: Cycle
    mode: str
    rpm: np.ndarray
    gear: np.ndarray
    shift: np.ndarray
    style: np.ndarray
    optimal_gear: np.ndarray


def run_simulation(cycle: Cycle, mode="comfort", engine: EngineSpec = DEFAULT_ENGINE,
                   drivetrain: Drivetrain = DEFAULT_DRIVETRAIN,
                   style_window=40) -> SimResult:
    """Run a full closed-loop simulation of one cycle under one drive mode."""
    ctrl = FuzzyGearController(mode=mode, engine=engine)
    trans = Transmission(drivetrain=drivetrain, engine=engine, gear=1)
    trans.settle_gear(cycle.speed[0])

    n = len(cycle.t)
    rpm = np.zeros(n)
    gear = np.zeros(n, dtype=int)
    shift = np.zeros(n)
    style_series = np.zeros(n)
    optimal = np.zeros(n, dtype=int)

    for i in range(n):
        lo = max(0, i - style_window)
        style = estimate_style(cycle.throttle[lo:i + 1], cycle.brake[lo:i + 1])
        cur_rpm = trans.engine_rpm(cycle.speed[i])
        x = Inputs(rpm=cur_rpm, throttle=cycle.throttle[i],
                   brake=cycle.brake[i], speed=cycle.speed[i], style=style)
        decision = ctrl.decide(x)
        trans.update(decision.shift_value, cycle.dt, cycle.speed[i])

        rpm[i] = cur_rpm
        gear[i] = trans.gear
        shift[i] = decision.shift_value
        style_series[i] = style
        optimal[i] = optimal_power_gear(engine, drivetrain, cycle.speed[i])

    return SimResult(cycle, mode, rpm, gear, shift, style_series, optimal)


CYCLES = {
    "urban": urban_cycle,
    "onramp": highway_onramp,
    "spirited": spirited_run,
}
