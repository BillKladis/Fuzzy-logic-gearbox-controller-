"""Configurable engine and drivetrain specifications.

The controller is *elastic*: nothing about a particular engine is baked into
the fuzzy logic.  An :class:`EngineSpec` carries a torque curve (from which a
power curve, a torque-peak rpm and a power-peak rpm are derived), and a
:class:`Drivetrain` carries the gear ratios.  The fuzzy shift map is then
anchored to those engine landmarks, so swapping in a different engine
automatically re-positions every shift point where it belongs for *that*
engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EngineSpec:
    """An engine described by a torque curve sampled over rpm.

    Parameters
    ----------
    name : str
        Label for plots / reports.
    idle_rpm, redline : float
        Bounds of the usable rev range.
    rpm_samples, torque_samples : sequence of float
        Matched samples of the torque curve [Nm].  Linearly interpolated
        in between, so as few as four points describe a believable engine.
    """

    name: str
    idle_rpm: float
    redline: float
    rpm_samples: np.ndarray
    torque_samples: np.ndarray
    peak_torque_rpm: float = field(init=False)
    peak_power_rpm: float = field(init=False)
    peak_power_kw: float = field(init=False)

    def __post_init__(self):
        self.rpm_samples = np.asarray(self.rpm_samples, dtype=float)
        self.torque_samples = np.asarray(self.torque_samples, dtype=float)
        grid = np.linspace(self.idle_rpm, self.redline, 600)
        tq = self.torque(grid)
        pw = self.power_kw(grid)
        self.peak_torque_rpm = float(grid[int(np.argmax(tq))])
        i_pow = int(np.argmax(pw))
        self.peak_power_rpm = float(grid[i_pow])
        self.peak_power_kw = float(pw[i_pow])

    def torque(self, rpm):
        """Crankshaft torque [Nm] at one or many rpm values."""
        rpm = np.clip(rpm, self.idle_rpm, self.redline)
        return np.interp(rpm, self.rpm_samples, self.torque_samples)

    def power_kw(self, rpm):
        """Power [kW] = torque * angular velocity."""
        rpm_arr = np.clip(np.asarray(rpm, dtype=float), self.idle_rpm, self.redline)
        return self.torque(rpm_arr) * rpm_arr * (2.0 * np.pi / 60.0) / 1000.0

    @classmethod
    def from_peaks(cls, name, idle_rpm, redline, peak_torque_nm,
                   torque_peak_rpm, power_peak_rpm):
        """Build a believable curve from a handful of headline numbers.

        Most spec sheets quote "X Nm at Y rpm" and "Z kW at W rpm"; this
        constructor turns those into a smooth four-point torque curve whose
        torque and power peaks land where requested.
        """
        rpm = [idle_rpm, torque_peak_rpm, power_peak_rpm, redline]
        tq = [peak_torque_nm * 0.55, peak_torque_nm,
              peak_torque_nm * 0.86, peak_torque_nm * 0.60]
        return cls(name, idle_rpm, redline, rpm, tq)


@dataclass
class Drivetrain:
    """Gear ratios + final drive + tyre, giving rpm-per-km/h per gear."""

    gear_ratios: np.ndarray
    final_drive: float = 3.9
    wheel_radius_m: float = 0.31
    _rpm_per_kmh: np.ndarray = field(init=False)

    def __post_init__(self):
        self.gear_ratios = np.asarray(self.gear_ratios, dtype=float)
        # rpm = v[m/s]/(2*pi*r) * ratio * final_drive * 60, with v = kmh/3.6
        k = self.final_drive * 60.0 / (3.6 * 2.0 * np.pi * self.wheel_radius_m)
        self._rpm_per_kmh = self.gear_ratios * k

    @property
    def n_gears(self) -> int:
        return len(self.gear_ratios)

    def rpm_per_kmh(self, gear: int) -> float:
        return float(self._rpm_per_kmh[gear - 1])

    def engine_rpm(self, speed_kmh: float, gear: int, idle_rpm: float) -> float:
        return max(idle_rpm, speed_kmh * self.rpm_per_kmh(gear))


def optimal_power_gear(engine: EngineSpec, drivetrain: Drivetrain,
                       speed_kmh: float) -> int:
    """Analytic benchmark: the gear that maximises available wheel power.

    At a fixed road speed the tractive force is power / speed, so the gear
    putting the engine where it makes the most power is the one giving the
    strongest acceleration.  Gears that would over-rev past redline are
    excluded.  This is the yardstick the fuzzy controller is compared against.
    """
    best_gear, best_power = 1, -np.inf
    for gear in range(1, drivetrain.n_gears + 1):
        rpm = speed_kmh * drivetrain.rpm_per_kmh(gear)
        if rpm > engine.redline:
            continue
        rpm = max(rpm, engine.idle_rpm)
        power = float(engine.power_kw(rpm))
        if power > best_power:
            best_power, best_gear = power, gear
    return best_gear


# A torquey turbo-four as the default engine, and a six-speed behind it.
DEFAULT_ENGINE = EngineSpec.from_peaks(
    name="2.0L turbo I4",
    idle_rpm=800.0,
    redline=7000.0,
    peak_torque_nm=350.0,
    torque_peak_rpm=2200.0,
    power_peak_rpm=5500.0,
)

DEFAULT_DRIVETRAIN = Drivetrain(
    gear_ratios=[3.6, 2.1, 1.45, 1.05, 0.82, 0.66],
    final_drive=3.9,
    wheel_radius_m=0.31,
)
