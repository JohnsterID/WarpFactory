import pytest
import numpy as np
from warpfactory.metrics import (
    MinkowskiMetric,
    LentzMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
    ThreePlusOneDecomposition,
)

def test_minkowski_metric():
    """Test the basic Minkowski metric in various coordinates."""
    metric = MinkowskiMetric()
    
    # Test in Cartesian coordinates
    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])
    t = 0.0
    
    components = metric.calculate(x, y, z, t)
    
    # Check metric signature (-,+,+,+)
    assert np.isclose(components["g_tt"], -1.0)
    assert np.isclose(components["g_xx"], 1.0)
    assert np.isclose(components["g_yy"], 1.0)
    assert np.isclose(components["g_zz"], 1.0)
    
    # Off-diagonal components should be zero
    assert np.isclose(components["g_tx"], 0.0)
    assert np.isclose(components["g_ty"], 0.0)
    assert np.isclose(components["g_tz"], 0.0)
    assert np.isclose(components["g_xy"], 0.0)
    assert np.isclose(components["g_xz"], 0.0)
    assert np.isclose(components["g_yz"], 0.0)

def test_lentz_metric():
    """Test the Lentz warp drive metric."""
    metric = LentzMetric()
    
    # Test parameters
    v_s = 2.0  # ship velocity (in c)
    R = 1.0    # radius of warp bubble
    sigma = 0.5 # thickness parameter
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = 0.0
    
    components = metric.calculate(x, y, z, t, v_s=v_s, R=R, sigma=sigma)
    
    # Basic checks
    assert "g_tt" in components
    assert "g_tx" in components
    assert "g_xx" in components
    
    # Asymptotic flatness checks
    far_index = -1
    assert np.isclose(components["g_tt"][far_index], -1.0, atol=1e-3)
    assert np.isclose(components["g_tx"][far_index], 0.0, atol=1e-3)
    assert np.isclose(components["g_xx"][far_index], 1.0, atol=1e-3)

def test_van_den_broeck_metric():
    """Test the Van Den Broeck warp drive metric."""
    metric = VanDenBroeckMetric()
    
    # Test parameters
    v_s = 2.0  # ship velocity (in c)
    R = 1.0    # radius of warp bubble
    B = 2.0    # expansion factor
    sigma = 0.5 # thickness parameter
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = 0.0
    
    components = metric.calculate(x, y, z, t, v_s=v_s, R=R, B=B, sigma=sigma)
    
    # Basic checks
    assert "g_tt" in components
    assert "g_tx" in components
    assert "g_xx" in components
    
    # Asymptotic flatness checks
    far_index = -1
    assert np.isclose(components["g_tt"][far_index], -1.0, atol=1e-3)
    assert np.isclose(components["g_tx"][far_index], 0.0, atol=1e-3)
    assert np.isclose(components["g_xx"][far_index], 1.0, atol=1e-3)

def test_warp_shell_metric():
    """Test the Warp Shell metric."""
    metric = WarpShellMetric()
    
    # Test parameters
    v_s = 2.0  # ship velocity (in c)
    R = 1.0    # radius of warp bubble
    thickness = 0.2  # shell thickness
    sigma = 1.0 # increased thickness parameter for faster decay
    x = np.linspace(-10, 10, 200)  # increased range to test asymptotic behavior
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = 0.0
    
    components = metric.calculate(x, y, z, t, v_s=v_s, R=R, thickness=thickness, sigma=sigma)
    
    # Basic checks
    assert "g_tt" in components
    assert "g_tx" in components
    assert "g_xx" in components
    
    # Asymptotic flatness checks
    far_index = -1
    assert np.isclose(components["g_tt"][far_index], -1.0, atol=1e-3)
    assert np.isclose(components["g_tx"][far_index], 0.0, atol=1e-3)
    assert np.isclose(components["g_xx"][far_index], 1.0, atol=1e-3)

def test_three_plus_one_decomposition():
    """Test the 3+1 decomposition of spacetime metrics."""
    decomposer = ThreePlusOneDecomposition()
    
    # Create a simple test metric (Schwarzschild-like)
    def test_metric(r):
        g_tt = -(1 - 2/r)
        g_rr = 1/(1 - 2/r)
        return {
            "g_tt": g_tt,
            "g_rr": g_rr,
            "g_theta_theta": r**2,
            "g_phi_phi": r**2 * np.sin(np.pi/4)**2  # at θ = π/4
        }
    
    r = np.array([4.0])  # Test at r = 4M
    metric = test_metric(r)
    
    # Perform 3+1 decomposition
    result = decomposer.decompose(metric)
    
    # Check basic properties
    assert "alpha" in result  # lapse function
    assert "beta" in result   # shift vector
    assert "gamma" in result  # spatial metric
    
    # Lapse should be positive
    assert result["alpha"] > 0
    
    # For spherically symmetric metric, shift should be zero
    assert np.allclose(result["beta"], 0.0)