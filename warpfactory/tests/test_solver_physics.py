import pytest
import numpy as np
from warpfactory.solver import (
    ChristoffelSymbols,
    RicciTensor,
    RicciScalar,
    EnergyTensor,
    FiniteDifference,
)

def test_finite_difference():
    """Test finite difference calculations."""
    fd = FiniteDifference()
    
    # Test 1D first derivative
    # Use a simpler linear function for more stable testing
    x = np.linspace(-1, 1, 21)
    f = x  # Test function (f(x) = x)
    df_dx_exact = np.ones_like(x)  # Exact derivative (df/dx = 1)
    
    df_dx = fd.derivative1(f, x, axis=0)
    assert np.allclose(df_dx, df_dx_exact, atol=1e-2)
    
    # Test 1D second derivative
    # Use quadratic function for second derivative test
    f2 = x**2  # Test function (f(x) = x²)
    d2f_dx2_exact = 2*np.ones_like(x)  # Exact second derivative (d²f/dx² = 2)
    d2f_dx2 = fd.derivative2(f2, x, axis=0)
    assert np.allclose(d2f_dx2, d2f_dx2_exact, atol=1e-2)

def test_christoffel_symbols():
    """Test Christoffel symbol calculations."""
    christoffel = ChristoffelSymbols()
    
    # Test with Schwarzschild metric
    r = np.array([4.0])  # Test at r = 4M (M=1)
    theta = np.array([np.pi/2])  # Test at equator
    
    # Metric components
    g_tt = -(1 - 2/r)
    g_rr = 1/(1 - 2/r)
    g_theta_theta = r**2
    g_phi_phi = r**2 * np.sin(theta)**2
    
    metric = {
        "g_tt": g_tt,
        "g_rr": g_rr,
        "g_theta_theta": g_theta_theta,
        "g_phi_phi": g_phi_phi
    }
    
    coords = {
        "r": r,
        "theta": theta
    }
    
    gamma = christoffel.calculate(metric, coords)
    
    # Test specific non-zero components
    # For M=1, r=4:
    # Γ^r_tt = (r-2)/r^3 = 2/64 = 1/32
    assert np.isclose(gamma["r_tt"][0], 1/32)  # at r=4M
    
    # Γ^t_tr = 1/(r(r-2)) = 1/8
    assert np.isclose(gamma["t_tr"][0], 1/8)

def test_ricci_tensor():
    """Test Ricci tensor calculations."""
    ricci = RicciTensor()
    
    # Test with Minkowski metric (should be zero)
    x = np.array([0.0])
    metric = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x)
    }
    
    coords = {"x": x}
    
    R_munu = ricci.calculate(metric, coords)
    
    # All components should be zero for flat spacetime
    for key in R_munu:
        assert np.allclose(R_munu[key], 0.0, atol=1e-10)

def test_ricci_scalar():
    """Test Ricci scalar calculations."""
    ricci_scalar = RicciScalar()
    
    # Test with Schwarzschild metric
    r = np.array([4.0])  # Test at r = 4M
    metric = {
        "g_tt": -(1 - 2/r),
        "g_rr": 1/(1 - 2/r),
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(np.pi/2)**2
    }
    
    coords = {"r": r}
    
    R = ricci_scalar.calculate(metric, coords)
    
    # Ricci scalar should be zero for vacuum solution
    assert np.allclose(R, 0.0, atol=1e-10)

def test_energy_tensor():
    """Test energy-momentum tensor calculations."""
    energy = EnergyTensor()
    
    # Test with simple stress-energy configuration
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    
    # Create a Gaussian energy density profile
    rho = np.exp(-x**2)  # Energy density
    p = rho/3  # Pressure (radiation equation of state)
    
    # Calculate full stress-energy tensor
    T_munu = energy.calculate_perfect_fluid(rho, p, x)
    
    # Basic checks
    assert "T_tt" in T_munu
    assert "T_xx" in T_munu
    assert "T_yy" in T_munu
    assert "T_zz" in T_munu
    
    # Energy conditions
    assert np.all(T_munu["T_tt"] >= 0)  # Weak energy condition
    assert np.all(T_munu["T_tt"] + T_munu["T_xx"] >= 0)  # Null energy condition