"""Israel thin-shell junction condition tests.

Ground truth: the static spherical shell with flat interior and
Schwarzschild exterior (Poisson, "A Relativist's Toolkit", sec. 3.9).
With f = 1 - 2M/a at shell radius a,

    sigma = (1 - sqrt(f)) / (4 pi a)
    p     = ((1 - M/a)/sqrt(f) - 1) / (8 pi a)
"""

import numpy as np
import pytest

from warpfactory.physics.junction import IsraelJunction


def _schwarzschild(r, M):
    f = 1 - 2 * M / r
    return {"g_tt": -f, "g_rr": 1 / f, "g_theta_theta": r**2}


def _minkowski_spherical(r):
    ones = np.ones_like(r)
    return {"g_tt": -ones, "g_rr": ones, "g_theta_theta": r**2}


def test_schwarzschild_extrinsic_curvature():
    """Exterior Schwarzschild: K^tau_tau = M/(a^2 sqrt(f)),
    K^theta_theta = sqrt(f)/a."""
    junction = IsraelJunction()
    M, a = 1.0, 5.0
    r = np.linspace(3.0, 10.0, 561)
    K = junction.extrinsic_curvature(_schwarzschild(r, M), {"r": r}, a)

    f = 1 - 2 * M / a
    assert np.isclose(K["K_tau_tau"], M / (a**2 * np.sqrt(f)), rtol=1e-6)
    assert np.isclose(K["K_theta_theta"], np.sqrt(f) / a, rtol=1e-6)
    assert np.isclose(K["trace"], K["K_tau_tau"] + 2 * K["K_theta_theta"], rtol=1e-12)


def test_static_dust_shell_matches_analytics():
    """Flat interior + Schwarzschild exterior must reproduce the closed-form
    surface density and pressure of the static shell."""
    junction = IsraelJunction()
    M, a = 1.0, 5.0
    r = np.linspace(3.0, 10.0, 561)

    result = junction.surface_stress_energy(
        _minkowski_spherical(r), {"r": r}, _schwarzschild(r, M), {"r": r}, a
    )

    f = 1 - 2 * M / a
    sigma_exact = (1 - np.sqrt(f)) / (4 * np.pi * a)
    p_exact = ((1 - M / a) / np.sqrt(f) - 1) / (8 * np.pi * a)
    assert np.isclose(result["surface_density"], sigma_exact, rtol=1e-6)
    assert np.isclose(result["surface_pressure"], p_exact, rtol=1e-6)

    # A shell holding itself up against gravity needs positive density
    # and positive tangential pressure.
    assert result["surface_density"] > 0
    assert result["surface_pressure"] > 0

    # Weak-field consistency: sigma ~ M / (4 pi a^2) to leading order.
    assert np.isclose(result["surface_density"], M / (4 * np.pi * a**2), rtol=0.15)


def test_no_shell_no_surface_stress():
    """Matching Schwarzschild to itself leaves no surface layer."""
    junction = IsraelJunction()
    M, a = 1.0, 6.0
    r = np.linspace(3.0, 10.0, 561)
    metric = _schwarzschild(r, M)

    result = junction.surface_stress_energy(metric, {"r": r}, metric, {"r": r}, a)
    assert np.isclose(result["surface_density"], 0.0, atol=1e-12)
    assert np.isclose(result["surface_pressure"], 0.0, atol=1e-12)
    assert np.isclose(result["K_jump_trace"], 0.0, atol=1e-12)


def test_first_junction_condition_enforced():
    """Mismatched areal radii across the shell must be rejected."""
    junction = IsraelJunction()
    r = np.linspace(3.0, 10.0, 561)
    inner = _minkowski_spherical(r)
    outer = _schwarzschild(r, 1.0)
    outer["g_theta_theta"] = 1.1 * r**2

    with pytest.raises(ValueError, match="first junction condition"):
        junction.surface_stress_energy(inner, {"r": r}, outer, {"r": r}, 5.0)


def test_shell_radius_outside_grid_rejected():
    junction = IsraelJunction()
    r = np.linspace(3.0, 10.0, 101)
    with pytest.raises(ValueError, match="outside the sampled grid"):
        junction.extrinsic_curvature(_schwarzschild(r, 1.0), {"r": r}, 20.0)
