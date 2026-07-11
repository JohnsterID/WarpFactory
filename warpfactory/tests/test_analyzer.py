import pytest
import numpy as np
from warpfactory.analyzer import (
    FrameTransformer,
    EnergyConditions,
    MomentumFlow,
    ScalarInvariants,
)

def test_frame_transformation():
    """Test coordinate frame transformations."""
    transformer = FrameTransformer()
    
    # Test Lorentz boost transformation
    v = 0.5  # velocity in units of c
    gamma = 1/np.sqrt(1 - v**2)
    
    # Test metric components in lab frame
    metric_lab = {
        "g_tt": -np.array([1.0]),
        "g_tx": np.array([0.0]),
        "g_xx": np.array([1.0])
    }
    
    # Transform to moving frame
    metric_moving = transformer.lorentz_boost(metric_lab, v, axis='x')
    
    # Check transformed components
    assert np.isclose(metric_moving["g_tt"][0], -1.0)  # Invariant
    assert np.isclose(metric_moving["g_tx"][0], -v*gamma)
    assert np.isclose(metric_moving["g_xx"][0], gamma**2)

def test_energy_conditions():
    """Test energy conditions for stress-energy tensor."""
    conditions = EnergyConditions()
    
    # Test with perfect fluid stress-energy tensor
    rho = np.array([1.0])  # energy density
    p = np.array([0.3])    # pressure
    
    T_munu = {
        "T_tt": rho,
        "T_xx": p,
        "T_yy": p,
        "T_zz": p,
        "T_tx": np.zeros_like(rho)
    }
    
    # Test weak energy condition
    assert conditions.check_weak(T_munu)
    
    # Test null energy condition
    assert conditions.check_null(T_munu)
    
    # Test strong energy condition
    assert conditions.check_strong(T_munu)
    
    # Test dominant energy condition
    assert conditions.check_dominant(T_munu)

def test_energy_conditions_anisotropic():
    """Anisotropic violations must be detected (perfect-fluid shortcuts miss these).

    Regression: the previous implementation reduced each condition to a
    scalar inequality on T_tt and T_xx only, so a negative transverse
    pressure or |p| > rho along y/z was invisible.
    """
    conditions = EnergyConditions()
    rho = np.array([1.0])
    p = np.array([0.3])

    aniso = {
        "T_tt": rho,
        "T_xx": p,
        "T_yy": np.array([-2.0]),  # strong tension along y
        "T_zz": p,
        "T_tx": np.zeros_like(rho)
    }
    assert not conditions.check_null(aniso)
    assert not conditions.check_weak(aniso)

    dominant_violator = {
        "T_tt": rho,
        "T_xx": p,
        "T_yy": np.array([1.5]),  # |p_y| > rho
        "T_zz": p,
        "T_tx": np.zeros_like(rho)
    }
    assert not conditions.check_dominant(dominant_violator)

    dark_energy = {
        "T_tt": rho,
        "T_xx": np.array([-0.5]),
        "T_yy": np.array([-0.5]),
        "T_zz": np.array([-0.5]),
        "T_tx": np.zeros_like(rho)
    }
    assert not conditions.check_strong(dark_energy)
    assert conditions.check_weak(dark_energy)

def test_momentum_flow():
    """Test momentum flow line calculations."""
    flow = MomentumFlow()

    # Grid must resolve the Gaussian bubble: the conservation residual is
    # pure discretization error (Bianchi identity), so it converges to
    # zero with resolution but is O(1e-2) on a coarse 11-point grid.
    x = np.linspace(-5, 5, 401)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = 0.0

    v_x = np.exp(-x**2)
    metric = {
        "g_tt": -(1 - v_x**2),
        "g_tx": -v_x,
        "g_xx": np.ones_like(x)
    }

    flow_lines = flow.calculate_flow_lines(metric, x, y, z, t)

    assert "positions" in flow_lines
    assert "velocities" in flow_lines
    assert len(flow_lines["positions"]) > 0

    # Flow is localized at the bubble: zero far away, nonzero at the wall.
    # Eulerian observers are only timelike where 2 v_x^2 < 1, so the
    # velocity peaks near the wall rather than at the bubble center.
    speeds = np.abs(flow_lines["velocities"][:, 0])
    assert np.allclose(speeds[:10], 0.0, atol=1e-8)
    assert np.allclose(speeds[-10:], 0.0, atol=1e-8)
    peak_x = np.abs(x[np.argmax(speeds)])
    assert 0.1 < peak_x < 3.0

    # Covariant divergence vanishes up to discretization error
    div = flow.check_conservation(flow_lines, metric)
    assert np.allclose(div[4:-4], 0.0, atol=1e-3)

def test_scalar_invariants():
    """Scalar invariants computed from the metric must match Schwarzschild analytics."""
    invariants = ScalarInvariants()

    # Radial grid outside the horizon; derivatives are numerical, so a
    # single-point grid is not differentiable.
    r = np.linspace(3.0, 10.0, 141)
    theta = np.full_like(r, np.pi/2)

    metric = {
        "g_tt": -(1 - 2/r),
        "g_rr": 1/(1 - 2/r),
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(theta)**2
    }
    coords = {"r": r, "theta": theta}

    # Kretschmann scalar: K = 48 M^2 / r^6 for Schwarzschild (M=1)
    K = invariants.kretschmann(metric, coords)
    idx = np.argmin(np.abs(r - 4.0))
    assert np.isclose(K[idx], 48/(4**6), rtol=1e-4)

    # Ricci scalar vanishes for the vacuum solution
    R = invariants.ricci_scalar(metric, coords)
    assert np.allclose(R[5:-5], 0.0, atol=1e-5)

