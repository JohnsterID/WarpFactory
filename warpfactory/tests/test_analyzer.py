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

def test_momentum_flow():
    """Test momentum flow line calculations."""
    flow = MomentumFlow()
    
    # Test with simple warp drive metric
    x = np.linspace(-5, 5, 11)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = 0.0
    
    # Simple velocity field
    v_x = np.exp(-x**2)  # Gaussian profile
    
    # Metric components
    g_tt = -(1 - v_x**2)
    g_tx = -v_x
    g_xx = np.ones_like(x)
    
    metric = {
        "g_tt": g_tt,
        "g_tx": g_tx,
        "g_xx": g_xx
    }
    
    # Calculate momentum flow lines
    flow_lines = flow.calculate_flow_lines(metric, x, y, z, t)
    
    # Basic checks
    assert "positions" in flow_lines
    assert "velocities" in flow_lines
    assert len(flow_lines["positions"]) > 0
    
    # Check conservation of energy-momentum
    div = flow.check_conservation(flow_lines, metric)
    assert np.allclose(div, 0.0, atol=1e-3)

def test_scalar_invariants():
    """Test scalar invariant calculations."""
    invariants = ScalarInvariants()
    
    # Test with Schwarzschild metric
    r = np.array([4.0])  # Test at r = 4M
    
    metric = {
        "g_tt": -(1 - 2/r),
        "g_rr": 1/(1 - 2/r),
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(np.pi/2)**2
    }
    
    coords = {"r": r}
    
    # Calculate Kretschmann scalar
    K = invariants.kretschmann(metric, coords)
    
    # For Schwarzschild: K = 48M²/r⁶
    # At r = 4M with M = 1: K = 48/4⁶ = 0.046875
    assert np.isclose(K[0], 48/(4**6))
    
    # Calculate Ricci scalar (should be zero for vacuum)
    R = invariants.ricci_scalar(metric, coords)
    assert np.allclose(R, 0.0, atol=1e-10)