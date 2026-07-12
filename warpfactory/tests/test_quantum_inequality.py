"""Tests for the Ford-Roman quantum inequality evaluator.

Ground truth comes from Pfenning & Ford, "The unphysical nature of
Warp Drive" (arXiv gr-qc/9702026): the flat-space Ford-Roman bound,
the Lorentzian-sampling contour integral (their Eq. 17), the wall
thickness bound (Eq. 20), and the total energy estimate (Eqs. 24-25).
"""

import numpy as np
import pytest

from warpfactory.grid.shape_functions import alcubierre_shape
from warpfactory.physics import FordRomanInequality

HBAR = 1.0545718e-34
C = 299792458.0


@pytest.fixture
def qi():
    return FordRomanInequality()


class TestSamplingBound:
    def test_si_value(self, qi):
        tau0 = 1.0e-6
        expected = -3 * HBAR / (32 * np.pi**2 * C**3 * tau0**4)
        assert qi.sampling_bound(tau0) == pytest.approx(expected, rel=1e-6)

    def test_scales_as_inverse_fourth_power(self, qi):
        assert qi.sampling_bound(2.0) == pytest.approx(
            qi.sampling_bound(1.0) / 16, rel=1e-12
        )

    def test_rejects_nonpositive_tau0(self, qi):
        with pytest.raises(ValueError):
            qi.sampling_bound(0.0)


class TestSampledEnergyDensity:
    def test_constant_density_is_recovered(self, qi):
        # The Lorentzian weight integrates to 1, so a constant density
        # samples to itself (up to the finite integration window).
        tau0 = 1.0
        tau = np.linspace(-1000, 1000, 200001)
        rho0 = -2.5
        sampled = qi.sampled_energy_density(np.full_like(tau, rho0), tau, tau0)
        assert sampled == pytest.approx(rho0, rel=1e-3)

    def test_lorentzian_pulse_analytic(self, qi):
        # For rho = rho0 b^2/(tau^2 + b^2) the contour integral of
        # Pfenning-Ford Eq. 17 gives sampled = rho0 b / (tau0 + b).
        tau0, b, rho0 = 1.0, 3.0, -7.0
        tau = np.linspace(-2000, 2000, 400001)
        rho = rho0 * b**2 / (tau**2 + b**2)
        sampled = qi.sampled_energy_density(rho, tau, tau0)
        assert sampled == pytest.approx(rho0 * b / (tau0 + b), rel=1e-3)

    def test_check_flags_violation_and_pass(self, qi):
        tau0 = 1.0
        tau = np.linspace(-1000, 1000, 200001)
        bound = qi.sampling_bound(tau0)

        weak = qi.check_sampled(np.full_like(tau, bound / 2), tau, tau0)
        assert weak["satisfied"]

        strong = qi.check_sampled(np.full_like(tau, 2 * bound), tau, tau0)
        assert not strong["satisfied"]
        assert strong["sampled"] < strong["bound"]


class TestWarpBubbleBounds:
    def test_wall_thickness_matches_tanh_slope(self, qi):
        # Delta is defined so the piecewise-linear wall has the same
        # slope at r = R as the Alcubierre tanh shape function.
        R, sigma = 100.0, 8.0
        delta = qi.wall_thickness(R, sigma)
        eps = 1e-6
        slope = (
            alcubierre_shape(R + eps, R, sigma) - alcubierre_shape(R - eps, R, sigma)
        ) / (2 * eps)
        assert slope == pytest.approx(-1.0 / delta, rel=1e-6)

    def test_wall_thickness_large_sigma_limit(self, qi):
        assert qi.wall_thickness(100.0, 50.0) == pytest.approx(2 / 50.0, rel=1e-9)

    def test_max_thickness_order_planck(self, qi):
        # Pfenning-Ford Eq. 20 with alpha = 1/10: Delta <= about 10^2
        # v_b Planck lengths (the exact prefactor is 75 sqrt(3/pi)).
        v_b = 1.0
        expected = 75 * np.sqrt(3 / np.pi) * qi.planck_length
        assert qi.max_wall_thickness(v_b, alpha=0.1) == pytest.approx(
            expected, rel=1e-12
        )
        assert qi.max_wall_thickness(v_b, alpha=0.1) < 100 * qi.planck_length

    def test_total_energy_macroscopic_bubble(self, qi):
        # Pfenning-Ford Section 4: a 100 m bubble at v_b = 1 with the
        # QI-limited wall needs |E| ~ 6.2e65 grams -- ten orders of
        # magnitude beyond the visible universe. Order-of-magnitude
        # claim, so factor-2 tolerance.
        v_b, R = 1.0, 100.0
        delta = qi.max_wall_thickness(v_b, alpha=0.1)
        mass_grams = qi.total_energy(v_b, R, delta) / C**2 * 1000
        assert mass_grams == pytest.approx(-6.2e65, rel=1.0)
        assert mass_grams < -1e65

    def test_total_energy_meter_wall(self, qi):
        # Same section: with a (QI-violating) 1 m wall the energy drops
        # to the order of a quarter solar mass (exact Eq. 24 evaluation
        # gives 0.56 M_sun; the paper's quote is order-of-magnitude).
        m_sun = 1.989e30
        mass = qi.total_energy(1.0, 100.0, 1.0) / C**2
        assert -m_sun < mass < -0.25 * m_sun

    def test_check_warp_bubble_verdicts(self, qi):
        macroscopic = qi.check_warp_bubble(v_b=2.0, R=100.0, sigma=8.0)
        assert not macroscopic["satisfied"]
        assert macroscopic["total_energy"] < 0
        assert macroscopic["delta"] > macroscopic["delta_max"]

        planck_wall = qi.check_warp_bubble(
            v_b=2.0, R=100.0, delta=qi.planck_length * 10
        )
        assert planck_wall["satisfied"]

    def test_check_warp_bubble_requires_one_wall_spec(self, qi):
        with pytest.raises(ValueError):
            qi.check_warp_bubble(v_b=1.0, R=100.0)
        with pytest.raises(ValueError):
            qi.check_warp_bubble(v_b=1.0, R=100.0, sigma=8.0, delta=1.0)


class TestEulerianWorldline:
    def test_alcubierre_wall_observer_violates_qi(self, qi):
        # An Eulerian observer sitting in the bubble wall of a
        # macroscopic Alcubierre drive sees the analytic energy density
        # rho = -(c^4/G) v^2 rho_cyl^2 (df/dr)^2 / (32 pi r^2) as the
        # bubble sweeps past; the QI sampled average violates the bound
        # by many orders of magnitude (the core Pfenning-Ford result).
        v_b, R, sigma = 0.5, 100.0, 0.5
        rho_cyl = R
        tau0 = 1e-10

        t = np.linspace(-3000, 3000, 600001) / (v_b * C)
        r_s = np.sqrt((v_b * C * t) ** 2 + rho_cyl**2)
        eps = 1e-6
        dfdr = (
            alcubierre_shape(r_s + eps, R, sigma)
            - alcubierre_shape(r_s - eps, R, sigma)
        ) / (2 * eps)
        si_factor = C**4 / 6.6743e-11
        rho = -si_factor * v_b**2 * rho_cyl**2 * dfdr**2 / (32 * np.pi * r_s**2)

        verdict = qi.check_sampled(rho, t, tau0)
        assert not verdict["satisfied"]
        assert verdict["sampled"] < 1e10 * verdict["bound"]

    def test_minkowski_observer_satisfies_qi(self, qi):
        tau = np.linspace(-1.0, 1.0, 20001)
        verdict = qi.check_sampled(np.zeros_like(tau), tau, 1e-3)
        assert verdict["satisfied"]
        assert verdict["sampled"] == 0.0
