"""Analytic reference solutions used by the tests (SOLVER_SPEC.md §7).

SI units throughout: m, s, m/s, m²/s.
"""
import numpy as np


def tilted_plane_steady_q(x, r):
    """Kinematic steady-state unit discharge on a rained-on tilted plane (§7 T3).

    q(x) = r * x

    x : distance downslope from the upstream divide (m), scalar or array
    r : rainfall rate (m/s)
    returns q (m²/s)
    """
    return r * np.asarray(x, dtype=float)


def tilted_plane_steady_h(x, r, n, S0):
    """Kinematic steady-state depth on a rained-on tilted plane (§7 T3).

    h(x) = (n * r * x / sqrt(S0)) ** (3/5)

    x  : distance downslope from the upstream divide (m), scalar or array
    r  : rainfall rate (m/s)
    n  : Manning roughness (s·m^-1/3)
    S0 : bed slope (-)
    returns h (m)
    """
    return (n * r * np.asarray(x, dtype=float) / np.sqrt(S0)) ** (3.0 / 5.0)
