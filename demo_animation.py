"""Animated dashboard of the fuzzy gear controller in action.

Plays a driving cycle through the controller and animates, frame by frame:

  * the road-speed, engine-rpm and gear time series as they unfold;
  * the live fuzzy memberships of rpm and the pedals;
  * the aggregated output fuzzy set with the defuzzified centroid (the crisp
    shift decision) marked on it.

Usage
-----
    python demo_animation.py [--mode eco|comfort|sport] [--cycle urban|onramp|spirited]
                             [--out assets/dashboard.gif]

Saves an animated GIF (no ffmpeg required -- uses Pillow).
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fuzzygear.controller import FuzzyGearController, Inputs, Transmission, estimate_style
from fuzzygear.engine import DEFAULT_DRIVETRAIN, DEFAULT_ENGINE
from fuzzygear.simulation import CYCLES
from fuzzygear.variables import SHIFT_SETS, SHIFT_DOMAIN

MODE_COLOR = {"eco": "#2e8b57", "comfort": "#1f77b4", "sport": "#d62728"}
_SHIFT_X = np.linspace(SHIFT_DOMAIN[0], SHIFT_DOMAIN[1], 200)


def trace(cycle, mode, style_window=40):
    """Run the cycle and record everything the animation needs per step."""
    ctrl = FuzzyGearController(mode, DEFAULT_ENGINE)
    tr = Transmission(gear=1)
    tr.settle_gear(cycle.speed[0])
    n = len(cycle.t)
    rec = {k: np.zeros(n) for k in ("rpm", "gear", "shift", "style")}
    memb, aggs = [], []
    for i in range(n):
        lo = max(0, i - style_window)
        style = estimate_style(cycle.throttle[lo:i + 1], cycle.brake[lo:i + 1])
        rpm = tr.engine_rpm(cycle.speed[i])
        x = Inputs(rpm, cycle.throttle[i], cycle.brake[i], cycle.speed[i], style)
        d = ctrl.decide(x)
        tr.update(d.shift_value, cycle.dt, cycle.speed[i])
        rec["rpm"][i] = rpm
        rec["gear"][i] = tr.gear
        rec["shift"][i] = d.shift_value
        rec["style"][i] = style
        memb.append(d.memberships)
        aggs.append(d.rule_activations)
    return rec, memb, aggs


def build(cycle, mode, out_path, fps=20, max_frames=160):
    rec, memb, aggs = trace(cycle, mode)
    color = MODE_COLOR[mode]
    n = len(cycle.t)
    step = max(1, n // max_frames)
    frames = range(0, n, step)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 1.5], hspace=0.45, wspace=0.3)
    ax_spd = fig.add_subplot(gs[0, :2])
    ax_rpm = fig.add_subplot(gs[1, :2], sharex=ax_spd)
    ax_gear = fig.add_subplot(gs[2, :2], sharex=ax_spd)
    ax_rmemb = fig.add_subplot(gs[0, 2])
    ax_pmemb = fig.add_subplot(gs[1, 2])
    ax_out = fig.add_subplot(gs[2, 2])

    t = cycle.t
    # static faint context curves
    ax_spd.plot(t, cycle.speed, color="lightgrey", lw=1)
    ax_rpm.plot(t, rec["rpm"], color="lightgrey", lw=1)
    ax_rpm.axhline(DEFAULT_ENGINE.redline, color="grey", ls="--", lw=1, alpha=0.6)
    ax_gear.plot(t, rec["gear"], color="lightgrey", lw=1, drawstyle="steps-post")

    (spd_line,) = ax_spd.plot([], [], color="black", lw=2)
    (rpm_line,) = ax_rpm.plot([], [], color=color, lw=2)
    (gear_line,) = ax_gear.plot([], [], color=color, lw=2, drawstyle="steps-post")
    spd_dot = ax_spd.scatter([], [], color="black", zorder=5)
    rpm_dot = ax_rpm.scatter([], [], color=color, zorder=5)

    ax_spd.set_ylabel("speed\n[km/h]")
    ax_spd.set_ylim(0, cycle.speed.max() * 1.15 + 5)
    ax_rpm.set_ylabel("rpm")
    ax_rpm.set_ylim(0, DEFAULT_ENGINE.redline * 1.08)
    ax_gear.set_ylabel("gear")
    ax_gear.set_ylim(0.5, DEFAULT_DRIVETRAIN.n_gears + 0.5)
    ax_gear.set_yticks(range(1, DEFAULT_DRIVETRAIN.n_gears + 1))
    ax_gear.set_xlabel("time [s]")
    for ax in (ax_spd, ax_rpm, ax_gear):
        ax.grid(alpha=0.3)

    # live bars
    rpm_labels = ["low", "medium", "high", "redline"]
    ped_labels = ["thr.light", "thr.mod", "thr.heavy", "brk.none", "brk.mod", "brk.hard"]
    rbars = ax_rmemb.barh(rpm_labels, [0] * 4, color="#888")
    pbars = ax_pmemb.barh(ped_labels, [0] * 6, color="#888")
    for ax, title in ((ax_rmemb, "rpm membership"), (ax_pmemb, "pedal membership")):
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)

    ax_out.set_title("output set + centroid", fontsize=9)
    ax_out.set_xlim(-1, 1)
    ax_out.set_ylim(0, 1.05)
    ax_out.tick_params(labelsize=8)
    (out_fill,) = ax_out.plot([], [], color=color, lw=2)
    out_centroid = ax_out.axvline(0, color="black", ls="--", lw=1.5)
    for name, mf in SHIFT_SETS.items():
        ax_out.plot(_SHIFT_X, [mf(xi) for xi in _SHIFT_X], color="lightgrey", lw=1)

    title = fig.suptitle("", fontsize=13, weight="bold")

    def aggregated_curve(agg):
        curve = np.zeros_like(_SHIFT_X)
        for name, strength in agg.items():
            if strength <= 0:
                continue
            mf = np.minimum([SHIFT_SETS[name](xi) for xi in _SHIFT_X], strength)
            curve = np.maximum(curve, mf)
        return curve

    def update(i):
        spd_line.set_data(t[:i + 1], cycle.speed[:i + 1])
        rpm_line.set_data(t[:i + 1], rec["rpm"][:i + 1])
        gear_line.set_data(t[:i + 1], rec["gear"][:i + 1])
        spd_dot.set_offsets([[t[i], cycle.speed[i]]])
        rpm_dot.set_offsets([[t[i], rec["rpm"][i]]])

        rmu = memb[i]["rpm"]
        for bar, lab in zip(rbars, rpm_labels):
            bar.set_width(rmu[lab])
            bar.set_color(color if rmu[lab] > 0.05 else "#bbb")
        tmu, bmu = memb[i]["throttle"], memb[i]["brake"]
        pvals = [tmu["light"], tmu["moderate"], tmu["heavy"],
                 bmu["none"], bmu["moderate"], bmu["hard"]]
        for bar, v in zip(pbars, pvals):
            bar.set_width(v)
            bar.set_color(color if v > 0.05 else "#bbb")

        curve = aggregated_curve(aggs[i])
        out_fill.set_data(_SHIFT_X, curve)
        out_centroid.set_xdata([rec["shift"][i], rec["shift"][i]])

        decision = ("UPSHIFT" if rec["shift"][i] > 0.33 else
                    "DOWNSHIFT" if rec["shift"][i] < -0.33 else "hold")
        title.set_text(f"{mode.upper()}  |  {cycle.name}  |  t = {t[i]:5.1f}s  "
                       f"|  gear {int(rec['gear'][i])}  |  {rec['rpm'][i]:.0f} rpm  "
                       f"|  shift = {rec['shift'][i]:+.2f} ({decision})")
        return [spd_line, rpm_line, gear_line, out_fill, out_centroid]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="comfort", choices=list(MODE_COLOR))
    p.add_argument("--cycle", default="spirited", choices=list(CYCLES))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cycle = CYCLES[args.cycle]()
    out = args.out or f"assets/dashboard_{args.cycle}_{args.mode}.gif"
    print("rendering", out, "...")
    print("wrote", build(cycle, args.mode, out))


if __name__ == "__main__":
    main()
