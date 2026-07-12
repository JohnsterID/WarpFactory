"""Bondi-Sachs quadrupole flux tests.

Ground truth:
- Peters (1964): circular binary energy flux P = (32/5) mu^2 M^3 / d^5.
- Fitchett (1983): circular binary momentum flux magnitude
  |F| = (464/105) (dm/M) nu^2 (M/d)^{11/2}, in the orbital plane.
- Any constant-velocity source: all multipole derivatives vanish in
  the radiation formulas, so both fluxes are identically zero.
"""

import numpy as np
import pytest

from warpfactory.physics.bondi_sachs import BondiSachsFlux


def _circular_binary(m1, m2, d, n_orbits=2.0, n_samples=4001):
    M = m1 + m2
    omega = np.sqrt(M / d**3)
    times = np.linspace(0.0, n_orbits * 2 * np.pi / omega, n_samples)
    phi = omega * times
    r1, r2 = (m2 / M) * d, (m1 / M) * d
    x1 = np.stack([r1 * np.cos(phi), r1 * np.sin(phi), np.zeros_like(phi)], axis=-1)
    x2 = np.stack([-r2 * np.cos(phi), -r2 * np.sin(phi), np.zeros_like(phi)], axis=-1)
    return times, np.array([m1, m2]), np.stack([x1, x2], axis=1)


def test_constant_velocity_source_does_not_radiate():
    """A drive coasting at constant velocity has vanishing flux."""
    flux = BondiSachsFlux()
    times = np.linspace(0.0, 10.0, 501)
    trajectory = np.stack(
        [2.0 * times, np.zeros_like(times), np.zeros_like(times)], axis=-1
    )
    result = flux.trajectory_fluxes(times, mass=1.0, trajectory=trajectory)
    # One-sided np.gradient stencils pollute a few samples at each end;
    # everywhere else the flux vanishes identically.
    interior = slice(5, -5)
    assert np.allclose(result["energy_flux"][interior], 0.0, atol=1e-12)
    assert np.allclose(result["momentum_flux"][interior], 0.0, atol=1e-12)


def test_peters_circular_binary_energy_flux():
    """Equal-mass circular binary must reproduce Peters' luminosity."""
    flux = BondiSachsFlux()
    m1 = m2 = 1.0
    d = 40.0
    times, masses, positions = _circular_binary(m1, m2, d)
    result = flux.fluxes(times, masses, positions)

    M = m1 + m2
    mu = m1 * m2 / M
    P_exact = 32 / 5 * mu**2 * M**3 / d**5

    interior = slice(400, -400)
    assert np.allclose(result["energy_flux"][interior], P_exact, rtol=1e-3)
    # Symmetric binary: no net momentum flux.
    assert np.allclose(result["momentum_flux"][interior], 0.0, atol=1e-6 * P_exact)


def test_fitchett_momentum_flux_unequal_binary():
    """Unequal-mass circular binary must reproduce the Fitchett recoil flux."""
    flux = BondiSachsFlux()
    m1, m2, d = 1.0, 0.5, 40.0
    times, masses, positions = _circular_binary(m1, m2, d)
    result = flux.fluxes(times, masses, positions)

    M = m1 + m2
    nu = m1 * m2 / M**2
    F_exact = 464 / 105 * abs(m1 - m2) / M * nu**2 * (M / d) ** 5.5

    interior = slice(400, -400)
    F = result["momentum_flux"][interior]
    magnitudes = np.linalg.norm(F, axis=1)
    assert np.allclose(magnitudes, F_exact, rtol=1e-3)
    # Recoil stays in the orbital plane and rotates with the binary:
    # zero z component, near-zero average over exactly one orbit
    # (4001 samples span 2 orbits, so one orbit is 2000 samples).
    assert np.allclose(F[:, 2], 0.0, atol=1e-12)
    assert np.abs(F[:2000, :2].mean(axis=0)).max() < 1e-3 * F_exact


def test_oscillating_source_radiates_energy():
    """A linearly oscillating mass (accelerating drive prototype) radiates.

    For x(t) = a sin(omega t), d3(I_xx) = m a^2 omega^3 [ -8/3 sin(2wt) ]
    is oscillatory: the energy flux must be nonnegative and nonzero on
    average, with zero net momentum flux by symmetry about the origin.
    """
    flux = BondiSachsFlux()
    # 4 periods, 1000 samples each, so period-exact averages are easy.
    a, omega = 1.0, 1.0
    times = np.linspace(0.0, 8 * np.pi / omega, 4001)
    trajectory = np.stack(
        [a * np.sin(omega * times), np.zeros_like(times), np.zeros_like(times)],
        axis=-1,
    )
    result = flux.trajectory_fluxes(times, mass=1.0, trajectory=trajectory)

    interior = slice(1000, 3000)
    P = result["energy_flux"][interior]
    assert P.min() >= -1e-10
    assert P.mean() > 0
    assert np.abs(result["momentum_flux"][interior].mean(axis=0)).max() < 1e-3 * (
        P.mean()
    )


def test_input_validation():
    flux = BondiSachsFlux()
    times = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="positions"):
        flux.fluxes(times, np.array([1.0]), np.zeros((11, 3)))
    with pytest.raises(ValueError, match="masses"):
        flux.fluxes(times, np.array([1.0, 2.0]), np.zeros((11, 1, 3)))
    with pytest.raises(ValueError, match="times"):
        flux.fluxes(times[:-1], np.array([1.0]), np.zeros((11, 1, 3)))
    with pytest.raises(ValueError, match="trajectory"):
        flux.trajectory_fluxes(times, 1.0, np.zeros((11, 2)))
