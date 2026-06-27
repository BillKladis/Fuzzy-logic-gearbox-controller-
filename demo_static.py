"""Generate the static figures used in the README.

Run ``python demo_static.py`` to (re)create the PNGs under ``assets/``:

  * membership_functions.png -- the fuzzy partitions of every variable
  * engine_curve.png         -- torque / power curve and its landmarks
  * shift_map.png            -- part-throttle upshift points per drive mode
  * simulation.png           -- a full closed-loop run, three modes overlaid
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend -- no display needed
import matplotlib.pyplot as plt
import numpy as np

from fuzzygear.controller import FuzzyGearController, Inputs, Transmission
from fuzzygear.engine import DEFAULT_DRIVETRAIN, DEFAULT_ENGINE
from fuzzygear.simulation import highway_onramp, run_simulation
from fuzzygear.variables import (
    BRAKE_SETS,
    MODES,
    SHIFT_SETS,
    THROTTLE_SETS,
    rpm_sets,
)

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)
MODE_COLORS = {"eco": "#2e8b57", "comfort": "#1f77b4", "sport": "#d62728"}


def _plot_sets(ax, x, sets, title, xlabel):
    for name, mf in sets.items():
        ax.plot(x, [mf(xi) for xi in x], label=name, lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"membership $\mu(x)$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


def figure_membership():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rpm_x = np.linspace(0, DEFAULT_ENGINE.redline, 500)
    # RPM partition for COMFORT, with the other modes' upshift anchors marked.
    comfort_sets = rpm_sets(DEFAULT_ENGINE, MODES["comfort"])
    _plot_sets(axes[0, 0], rpm_x, comfort_sets,
               "Engine speed (COMFORT partition)", "rpm")
    for mode, c in MODE_COLORS.items():
        from fuzzygear.variables import resolve_anchors
        _, up = resolve_anchors(MODES[mode], DEFAULT_ENGINE)
        axes[0, 0].axvline(up, color=c, ls="--", lw=1.2, alpha=0.8,
                           label=f"{mode} upshift")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    _plot_sets(axes[0, 1], np.linspace(0, 100, 400), THROTTLE_SETS,
               "Throttle pedal", "throttle [%]")
    _plot_sets(axes[1, 0], np.linspace(0, 100, 400), BRAKE_SETS,
               "Brake pedal", "brake [%]")
    _plot_sets(axes[1, 1], np.linspace(-1, 1, 400), SHIFT_SETS,
               "Output: shift tendency", "shift value")
    fig.suptitle("Fuzzy membership functions", fontsize=14, weight="bold")
    fig.tight_layout()
    path = os.path.join(ASSETS, "membership_functions.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def figure_engine():
    e = DEFAULT_ENGINE
    rpm = np.linspace(e.idle_rpm, e.redline, 400)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(rpm, e.torque(rpm), color="#1f77b4", lw=2.4, label="torque [Nm]")
    l2, = ax2.plot(rpm, e.power_kw(rpm), color="#d62728", lw=2.4, label="power [kW]")
    ax1.axvline(e.peak_torque_rpm, color="#1f77b4", ls=":", alpha=0.7)
    ax2.axvline(e.peak_power_rpm, color="#d62728", ls=":", alpha=0.7)
    ax1.set_xlabel("engine speed [rpm]")
    ax1.set_ylabel("torque [Nm]", color="#1f77b4")
    ax2.set_ylabel("power [kW]", color="#d62728")
    ax1.set_title(f"{e.name}: torque peak {e.peak_torque_rpm:.0f} rpm, "
                  f"power peak {e.peak_power_rpm:.0f} rpm")
    ax1.grid(alpha=0.3)
    ax1.legend(handles=[l1, l2], loc="lower center")
    fig.tight_layout()
    path = os.path.join(ASSETS, "engine_curve.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _upshift_points(mode, throttle):
    ctrl = FuzzyGearController(mode, DEFAULT_ENGINE)
    tr = Transmission(gear=1)
    pts, prev = [], 1
    for k in range(1, 2500):
        spd = k * 0.1
        rpm = tr.engine_rpm(spd)
        d = ctrl.decide(Inputs(rpm, throttle, 0, spd, 0.0))
        tr.update(d.shift_value, 0.1)
        if tr.gear > prev:
            pts.append((spd, rpm))
            prev = tr.gear
        if tr.gear == DEFAULT_DRIVETRAIN.n_gears:
            break
    return pts


def figure_shift_map():
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, c in MODE_COLORS.items():
        pts = _upshift_points(mode, throttle=40)
        if pts:
            sp, rp = zip(*pts)
            ax.plot(sp, rp, "o-", color=c, lw=2, ms=7, label=f"{mode}")
    ax.axhline(DEFAULT_ENGINE.peak_power_rpm, color="grey", ls="--", alpha=0.6,
               label="power peak")
    ax.axhline(DEFAULT_ENGINE.peak_torque_rpm, color="grey", ls=":", alpha=0.6,
               label="torque peak")
    ax.set_xlabel("road speed [km/h]")
    ax.set_ylabel("engine rpm at upshift")
    ax.set_title("Part-throttle (40%) upshift points by drive mode")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(ASSETS, "shift_map.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def figure_simulation():
    cycle = highway_onramp()
    results = {m: run_simulation(cycle, mode=m) for m in MODES}
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(cycle.t, cycle.speed, color="black", lw=2)
    axes[0].set_ylabel("speed [km/h]")
    axes[0].set_title("Closed-loop simulation -- Highway on-ramp")
    axes[0].grid(alpha=0.3)
    for m, c in MODE_COLORS.items():
        axes[1].plot(cycle.t, results[m].rpm, color=c, lw=1.6, label=m)
        axes[2].step(cycle.t, results[m].gear, color=c, lw=1.8, where="post", label=m)
    axes[1].axhline(DEFAULT_ENGINE.redline, color="grey", ls="--", alpha=0.6)
    axes[1].set_ylabel("engine rpm")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("gear")
    axes[2].set_xlabel("time [s]")
    axes[2].set_yticks(range(1, DEFAULT_DRIVETRAIN.n_gears + 1))
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(ASSETS, "simulation.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    for fn in (figure_membership, figure_engine, figure_shift_map, figure_simulation):
        print("wrote", fn())


if __name__ == "__main__":
    main()
