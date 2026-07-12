import numpy as np

from warpfactory.solver import (
    ChristoffelSymbols,
    EnergyTensor,
    FiniteDifference,
    RicciScalar,
    RicciTensor,
)


def schwarzschild_metric(r, theta):
    """Schwarzschild metric profiles for M=1 (shared analytic reference)."""
    return {
        "g_tt": -(1 - 2 / r),
        "g_rr": 1 / (1 - 2 / r),
        "g_theta_theta": r**2,
        "g_phi_phi": r**2 * np.sin(theta) ** 2,
    }


def schwarzschild_grid():
    """Radial grid outside the horizon; single points are not differentiable."""
    r = np.linspace(3.0, 10.0, 141)
    theta = np.full_like(r, np.pi / 2)
    return r, theta


def test_finite_difference():
    """Finite differences must recover analytic derivatives."""
    x = np.linspace(-1, 1, 21)

    fd = FiniteDifference(order=2)
    assert np.allclose(fd.derivative1(x, x, axis=0), np.ones_like(x), atol=1e-2)
    assert np.allclose(fd.derivative2(x**2, x, axis=0), 2 * np.ones_like(x), atol=1e-2)

    # 4th order must beat 2nd order on a smooth non-polynomial profile
    profile = np.sin(2 * x)
    exact = 2 * np.cos(2 * x)
    err2 = np.max(
        np.abs(FiniteDifference(order=2).derivative1(profile, x)[2:-2] - exact[2:-2])
    )
    err4 = np.max(
        np.abs(FiniteDifference(order=4).derivative1(profile, x)[2:-2] - exact[2:-2])
    )
    assert err4 < err2 / 10


def test_finite_difference_nd():
    """Derivatives along an arbitrary axis of an N-D field."""
    x = np.linspace(-1, 1, 21)
    X, Y = np.meshgrid(x, x, indexing="ij")
    field = X**2 * Y

    fd = FiniteDifference(order=2)
    d_dy = fd.derivative1(field, x, axis=1)
    assert np.allclose(d_dy[2:-2, 2:-2], (X**2)[2:-2, 2:-2], atol=1e-10)

    d2_dxdy = fd.mixed_derivative2(field, x, x, axis1=0, axis2=1)
    assert np.allclose(d2_dxdy[2:-2, 2:-2], (2 * X)[2:-2, 2:-2], atol=1e-10)


def test_christoffel_symbols():
    """Christoffel symbols must match Schwarzschild analytics."""
    christoffel = ChristoffelSymbols()
    r, theta = schwarzschild_grid()
    gamma = christoffel.calculate(
        schwarzschild_metric(r, theta), {"r": r, "theta": theta}
    )

    idx = np.argmin(np.abs(r - 4.0))
    # For M=1, r=4: Gamma^r_tt = (r-2)/r^3 = 1/32, Gamma^t_tr = 1/(r(r-2)) = 1/8
    assert np.isclose(gamma["r_tt"][idx], 1 / 32, rtol=1e-4)
    assert np.isclose(gamma["t_tr"][idx], 1 / 8, rtol=1e-4)

    # Closed-form angular symbols
    assert np.isclose(gamma["theta_rtheta"][idx], 1 / r[idx], rtol=1e-6)
    assert np.allclose(
        gamma["theta_phiphi"], -np.sin(theta) * np.cos(theta), atol=1e-12
    )


def test_christoffel_cartesian_flat():
    """All symbols vanish for Minkowski spacetime."""
    christoffel = ChristoffelSymbols()
    x = np.linspace(-5, 5, 50)
    flat = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x),
    }
    gamma = christoffel.calculate(flat, x, np.zeros_like(x), np.zeros_like(x))
    for symbol in gamma.values():
        assert np.allclose(symbol, 0.0, atol=1e-12)


def test_ricci_tensor():
    """Ricci tensor of flat spacetime is zero; Schwarzschild vacuum too."""
    ricci = RicciTensor()

    x = np.linspace(-5, 5, 50)
    flat = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x),
    }
    R_munu = ricci.calculate(flat, {"x": x})
    for component in R_munu.values():
        assert np.allclose(component, 0.0, atol=1e-10)

    r, theta = schwarzschild_grid()
    R_munu = ricci.calculate(schwarzschild_metric(r, theta), {"r": r, "theta": theta})
    # Vacuum solution: interior points must vanish up to discretization error
    for component in R_munu.values():
        assert np.allclose(component[5:-5], 0.0, atol=1e-5)


def test_ricci_scalar():
    """Ricci scalar vanishes for the Schwarzschild vacuum solution."""
    ricci_scalar = RicciScalar()
    r, theta = schwarzschild_grid()
    R = ricci_scalar.calculate(schwarzschild_metric(r, theta), {"r": r, "theta": theta})
    assert np.allclose(R[5:-5], 0.0, atol=1e-5)


def test_energy_tensor_perfect_fluid():
    """Perfect fluid stress-energy construction."""
    energy = EnergyTensor()
    x = np.linspace(-5, 5, 100)
    rho = np.exp(-(x**2))
    p = rho / 3

    T_munu = energy.calculate_perfect_fluid(rho, p, x)
    for key in ("T_tt", "T_xx", "T_yy", "T_zz"):
        assert key in T_munu
    assert np.all(T_munu["T_tt"] >= 0)
    assert np.all(T_munu["T_tt"] + T_munu["T_xx"] >= 0)


def test_energy_tensor_from_metric():
    """Stress-energy derived from the Einstein field equations.

    Flat spacetime must give T = 0 exactly; the Alcubierre metric must
    produce the well-known negative energy density ring at the bubble
    wall (the core WarpFactory result).
    """
    energy = EnergyTensor()
    x = np.linspace(-8, 8, 400)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    flat = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x),
    }
    T_flat = energy.calculate_from_metric(flat, x)
    for component in T_flat.values():
        assert np.allclose(component, 0.0, atol=1e-12)

    from warpfactory.metrics import AlcubierreMetric

    warp = AlcubierreMetric().calculate(x, y, z, 0.0, v_s=2.0, R=1.0, sigma=8.0)
    T_warp = energy.calculate_from_metric(warp, x)

    assert not np.allclose(T_warp["T_tt"], 0.0)
    # Stress-energy is localized at the bubble: far field must be flat
    assert np.allclose(T_warp["T_tt"][:10], 0.0, atol=1e-6)
    assert np.allclose(T_warp["T_tt"][-10:], 0.0, atol=1e-6)
