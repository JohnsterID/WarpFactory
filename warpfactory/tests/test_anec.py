"""Tests for the ANEC evaluator (warpfactory.physics.anec).

Ground truths: a hand-built energy profile on Minkowski has a
closed-form ANEC integral; the Alcubierre on-axis null contraction is
analytically zero (the Eulerian density scales as y^2 + z^2, which
vanishes on the axis), so the numerical ANEC must converge to zero
with grid refinement; the Van den Broeck on-axis stress-energy is
nonzero and its ANEC violation must be direction-independent (the
t = 0 slice is symmetric under x -> -x).
"""

import numpy as np
import pytest

from warpfactory.metrics import AlcubierreMetric, MinkowskiMetric, VanDenBroeckMetric
from warpfactory.physics import AveragedNullEnergy
from warpfactory.solver import EnergyTensor


def sampled_axis(n=801):
    x = np.linspace(-5.0, 5.0, n)
    return x, np.zeros_like(x)


class TestFlatSpace:
    def test_zero_stress_energy_gives_zero(self):
        x, zeros = sampled_axis()
        metric = MinkowskiMetric().calculate(x, zeros, zeros, 0)
        stress_energy = dict.fromkeys(("T_tt", "T_tx", "T_xx", "T_yy", "T_zz"), zeros)
        result = AveragedNullEnergy().integrate(metric, stress_energy, -4.5, +1.0, x=x)
        assert result["anec"] == 0.0

    def test_gaussian_profile_matches_closed_form(self):
        # On Minkowski the ray is x = x0 + t with A = 1, so the ANEC
        # of T_tt = a exp(-x^2/s) is the Gaussian integral a sqrt(pi s).
        x, zeros = sampled_axis()
        metric = MinkowskiMetric().calculate(x, zeros, zeros, 0)
        amplitude, width = 0.3, 1.5
        stress_energy = {
            "T_tt": amplitude * np.exp(-(x**2) / width),
            "T_tx": zeros,
            "T_xx": zeros,
            "T_yy": zeros,
            "T_zz": zeros,
        }
        result = AveragedNullEnergy().integrate(metric, stress_energy, -4.5, +1.0, x=x)
        closed_form = amplitude * np.sqrt(np.pi * width)
        np.testing.assert_allclose(result["anec"], closed_form, rtol=1e-4)

    def test_launch_outside_boundary_raises(self):
        x, zeros = sampled_axis()
        metric = MinkowskiMetric().calculate(x, zeros, zeros, 0)
        stress_energy = {"T_tt": zeros}
        with pytest.raises(ValueError):
            AveragedNullEnergy().integrate(metric, stress_energy, 4.95, -1.0, x=x)


class TestWarpMetrics:
    def test_alcubierre_on_axis_converges_to_zero(self):
        # The analytic on-axis contraction is exactly zero; the FD
        # residual must shrink under refinement (~2nd order here).
        integrals = []
        for n in (401, 801):
            x, zeros = sampled_axis(n)
            metric = AlcubierreMetric().calculate(
                x, zeros, zeros, 0, v_s=0.5, R=2.0, sigma=4.0
            )
            stress_energy = EnergyTensor().calculate_from_metric(metric, x)
            result = AveragedNullEnergy().integrate(
                metric, stress_energy, -4.5, +1.0, x=x
            )
            integrals.append(abs(result["anec"]))
        assert integrals[0] < 1e-6
        assert integrals[1] < integrals[0] / 4

    def test_van_den_broeck_violates_anec(self):
        x, zeros = sampled_axis()
        metric = VanDenBroeckMetric().calculate(x, zeros, zeros, 0, v_s=0.5)
        stress_energy = EnergyTensor().calculate_from_metric(metric, x)
        evaluator = AveragedNullEnergy()
        forward = evaluator.integrate(metric, stress_energy, -4.5, +1.0, x=x)
        backward = evaluator.integrate(metric, stress_energy, 4.5, -1.0, x=x)
        assert forward["anec"] < -1e-3
        # The t = 0 slice is x -> -x symmetric, so the two directions
        # must agree; the residual is ray-sampling discretization.
        np.testing.assert_allclose(forward["anec"], backward["anec"], rtol=1e-3)

    def test_result_arrays_are_consistent(self):
        x, zeros = sampled_axis(401)
        metric = VanDenBroeckMetric().calculate(x, zeros, zeros, 0, v_s=0.5)
        stress_energy = EnergyTensor().calculate_from_metric(metric, x)
        result = AveragedNullEnergy().integrate(metric, stress_energy, -4.5, +1.0, x=x)
        assert (
            len(result["times"]) == len(result["positions"]) == len(result["integrand"])
        )
        trapezoid = float(
            np.sum(
                (result["integrand"][1:] + result["integrand"][:-1])
                / 2
                * np.diff(result["times"])
            )
        )
        np.testing.assert_allclose(result["anec"], trapezoid)
