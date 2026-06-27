"""Synthetic tests for the fuzzy gear controller.

Runnable with ``pytest`` or directly with ``python tests/test_controller.py``.
The assertions encode the qualitative behaviour we expect from the physics:
membership functions are well-formed, the inference engine reacts sensibly to
clear-cut inputs, and the closed-loop behaviour orders the drive modes the way
a real adaptive gearbox would.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzygear.controller import FuzzyGearController, Inputs, Transmission
from fuzzygear.engine import (
    DEFAULT_DRIVETRAIN,
    DEFAULT_ENGINE,
    EngineSpec,
    optimal_power_gear,
)
from fuzzygear.membership import trapezoidal, triangular
from fuzzygear.simulation import highway_onramp, run_simulation, urban_cycle
from fuzzygear.variables import MODES, resolve_anchors, rpm_sets


# --- membership functions -------------------------------------------------

def test_triangular_peak_and_zero():
    assert triangular(5, 0, 5, 10) == 1.0
    assert triangular(0, 0, 5, 10) == 0.0
    assert triangular(10, 0, 5, 10) == 0.0
    assert abs(triangular(2.5, 0, 5, 10) - 0.5) < 1e-9


def test_trapezoidal_plateau_and_shoulders():
    assert trapezoidal(5, 2, 4, 6, 8) == 1.0          # on the plateau
    assert trapezoidal(0, 2, 4, 6, 8) == 0.0          # left of support
    assert trapezoidal(9, 2, 4, 6, 8) == 0.0          # right of support
    assert abs(trapezoidal(3, 2, 4, 6, 8) - 0.5) < 1e-9


def test_memberships_bounded():
    x = np.linspace(-1, 1, 50)
    vals = triangular(x, -0.4, 0, 0.4)
    assert np.all((vals >= 0) & (vals <= 1))


# --- fuzzification & inference -------------------------------------------

def test_rpm_partition_covers_domain():
    """At every rpm at least one set has non-trivial membership."""
    sets = rpm_sets(DEFAULT_ENGINE, MODES["comfort"])
    for rpm in np.linspace(0, DEFAULT_ENGINE.redline, 60):
        total = sum(mf(rpm) for mf in sets.values())
        assert total > 0.4, f"coverage gap at {rpm:.0f} rpm (sum={total:.2f})"


def test_high_rpm_light_throttle_upshifts():
    ctrl = FuzzyGearController("comfort")
    d = ctrl.decide(Inputs(rpm=5200, throttle=8, brake=0, speed=80))
    assert d.shift_value > 0.33, d.shift_value


def test_low_rpm_heavy_throttle_downshifts():
    ctrl = FuzzyGearController("comfort")
    d = ctrl.decide(Inputs(rpm=1200, throttle=95, brake=0, speed=40))
    assert d.shift_value < -0.33, d.shift_value


def test_hard_braking_downshifts():
    ctrl = FuzzyGearController("comfort")
    d = ctrl.decide(Inputs(rpm=3000, throttle=0, brake=90, speed=60))
    assert d.shift_value < -0.2, d.shift_value


def test_redline_forces_upshift_under_full_throttle():
    """Over-rev guard: at redline the box upshifts even at full throttle."""
    ctrl = FuzzyGearController("sport")
    d = ctrl.decide(Inputs(rpm=DEFAULT_ENGINE.redline, throttle=100, brake=0, speed=60))
    assert d.shift_value > 0.33, d.shift_value


# --- drive-mode ordering --------------------------------------------------

def test_mode_upshift_anchor_ordering():
    """Sport must hold gears to higher rpm than comfort, comfort than eco."""
    _, up_eco = resolve_anchors(MODES["eco"], DEFAULT_ENGINE)
    _, up_com = resolve_anchors(MODES["comfort"], DEFAULT_ENGINE)
    _, up_spt = resolve_anchors(MODES["sport"], DEFAULT_ENGINE)
    assert up_eco < up_com < up_spt


def test_sport_runs_higher_rpm_than_eco():
    cycle = highway_onramp()
    eco = run_simulation(cycle, mode="eco")
    sport = run_simulation(cycle, mode="sport")
    assert sport.rpm.mean() > eco.rpm.mean()


# --- closed-loop sanity ---------------------------------------------------

def test_no_sustained_over_rev():
    """The controller must keep the engine essentially within its range."""
    for mode in MODES:
        for cyc in (highway_onramp(), urban_cycle()):
            res = run_simulation(cyc, mode=mode)
            over = np.mean(res.rpm > DEFAULT_ENGINE.redline + 100)
            assert over < 0.02, f"{mode}: {over:.1%} of steps over-revving"


def test_gears_stay_in_range():
    res = run_simulation(highway_onramp(), mode="comfort")
    assert res.gear.min() >= 1
    assert res.gear.max() <= DEFAULT_DRIVETRAIN.n_gears


# --- elasticity: a different engine re-anchors the shift map ---------------

def test_controller_adapts_to_a_different_engine():
    """A high-revving engine should push every upshift anchor upward."""
    screamer = EngineSpec.from_peaks(
        "4.0L NA V8", idle_rpm=900, redline=9000,
        peak_torque_nm=400, torque_peak_rpm=4500, power_peak_rpm=8000)
    _, up_default = resolve_anchors(MODES["sport"], DEFAULT_ENGINE)
    _, up_screamer = resolve_anchors(MODES["sport"], screamer)
    assert up_screamer > up_default + 1000


def test_optimal_power_gear_is_monotone_in_speed():
    """Faster -> never an unnecessarily lower power-optimal gear."""
    gears = [optimal_power_gear(DEFAULT_ENGINE, DEFAULT_DRIVETRAIN, s)
             for s in range(10, 200, 10)]
    assert gears == sorted(gears)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
