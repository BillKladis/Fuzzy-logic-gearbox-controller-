"""Membership-function primitives for the fuzzy gear controller.

A *membership function* mu(x) maps a crisp value x onto a degree of
membership in the closed interval [0, 1].  These are the building blocks of
every fuzzy set used by the controller.  All functions are vectorised so they
work both on Python floats and on NumPy arrays (handy for plotting the
surfaces in the demos).
"""

from __future__ import annotations

import numpy as np


def triangular(x, a: float, b: float, c: float):
    r"""Triangular membership function.

    Defined by three points a <= b <= c:

        mu(x) = 0                       x <= a
        mu(x) = (x - a) / (b - a)       a <  x <= b
        mu(x) = (c - x) / (c - b)       b <  x <  c
        mu(x) = 0                       x >= c

    The peak (mu = 1) sits at b; the support is the open interval (a, c).
    """
    x = np.asarray(x, dtype=float)
    left = np.divide(x - a, b - a, out=np.zeros_like(x), where=(b > a))
    right = np.divide(c - x, c - b, out=np.zeros_like(x), where=(c > b))
    mu = np.maximum(np.minimum(left, right), 0.0)
    # Handle the degenerate "spike" cases where two anchors coincide.
    if b == a:
        mu = np.where(x <= a, 0.0, mu)
        mu = np.where((x > a) & (x < c), (c - x) / (c - b) if c > b else 0.0, mu)
    return _scalarise(x, mu)


def trapezoidal(x, a: float, b: float, c: float, d: float):
    r"""Trapezoidal membership function.

    Defined by four points a <= b <= c <= d.  The plateau (mu = 1) spans
    [b, c]; the function ramps up on [a, b] and down on [c, d].  Setting
    a == b gives a left shoulder, c == d gives a right shoulder.
    """
    x = np.asarray(x, dtype=float)
    up = np.divide(x - a, b - a, out=np.ones_like(x), where=(b > a))
    down = np.divide(d - x, d - c, out=np.ones_like(x), where=(d > c))
    mu = np.maximum(np.minimum(np.minimum(up, down), 1.0), 0.0)
    mu = np.where(x < a, 0.0, mu)
    mu = np.where(x > d, 0.0, mu)
    return _scalarise(x, mu)


def _scalarise(x_in, mu):
    """Return a plain float when the caller passed a scalar."""
    if np.ndim(x_in) == 0:
        return float(mu)
    return mu


# --- fuzzy connectives (t-norms / t-conorms) ------------------------------

def fuzzy_and(*degrees: float) -> float:
    """Conjunction of antecedent clauses -- the minimum t-norm."""
    return float(min(degrees))


def fuzzy_or(*degrees: float) -> float:
    """Disjunction of antecedent clauses -- the maximum t-conorm."""
    return float(max(degrees))
