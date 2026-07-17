"""Tests for hyper-dual exact derivatives (warpfactory.grid.hyperdual)
and the exact curvature pipeline (warpfactory.grid.exact).

Ground truths: closed-form derivatives of elementary functions, the
analytic Alcubierre Eulerian energy density (exact match, not the FD
tolerance band), Minkowski/Schwarzschild-type vacuum behavior, and the
direction symmetry of the ANEC integral.
"""

import numpy as np
import pytest

from warpfactory.grid import (
    ExactGridSolver,
    ExactNullGeodesicANEC,
    GridSolver,
    HyperDual,
    alcubierre_metric,
    alcubierre_metric_fn,
    alcubierre_shape,
    exact_metric_derivatives,
)

V, R, SIGMA = 2.0, 3.0, 2.0


def centered_world(grid_size, spacing=1.0):
    return (0.0,) + tuple((n - 1) * spacing / 2 for n in grid_size[1:])


def hyperdual_seed(values):
    """Seed both dual parts on the same variable: f1 = f2 = 1."""
    return HyperDual(values, 1.0, 1.0)


class TestHyperDual:
    def test_polynomial_derivatives_are_exact(self):
        x = np.array([0.5, 1.0, 2.0])
        h = hyperdual_seed(x)
        result = 3.0 * h**3 - 2.0 * h + 1.0
        np.testing.assert_allclose(result.f, 3 * x**3 - 2 * x + 1, rtol=1e-15)
        np.testing.assert_allclose(result.f1, 9 * x**2 - 2, rtol=1e-15)
        np.testing.assert_allclose(result.f12, 18 * x, rtol=1e-15)

    def test_quotient_and_reciprocal(self):
        x = np.array([0.5, 2.0, 3.0])
        result = 1.0 / hyperdual_seed(x)
        np.testing.assert_allclose(result.f1, -1.0 / x**2, rtol=1e-14)
        np.testing.assert_allclose(result.f12, 2.0 / x**3, rtol=1e-14)

    def test_elementary_functions_match_closed_forms(self):
        x = np.array([0.3, 0.9, 1.7])
        h = hyperdual_seed(x)
        cases = {
            np.tanh: (1 - np.tanh(x) ** 2, -2 * np.tanh(x) * (1 - np.tanh(x) ** 2)),
            np.sqrt: (0.5 / np.sqrt(x), -0.25 * x**-1.5),
            np.exp: (np.exp(x), np.exp(x)),
            np.log: (1.0 / x, -1.0 / x**2),
            np.sin: (np.cos(x), -np.sin(x)),
            np.cos: (-np.sin(x), -np.cos(x)),
        }
        for ufunc, (d1, d2) in cases.items():
            result = ufunc(h)
            np.testing.assert_allclose(result.f1, d1, rtol=1e-14)
            np.testing.assert_allclose(result.f12, d2, rtol=1e-13)

    def test_mixed_partials_via_independent_seeds(self):
        # f(x, y) = x^2 y^3: d2f/dxdy = 6 x y^2
        x, y = 1.5, 0.7
        hx = HyperDual(x, f1=1.0)
        hy = HyperDual(y, f2=1.0)
        result = hx**2 * hy**3
        np.testing.assert_allclose(result.f1, 2 * x * y**3, rtol=1e-15)
        np.testing.assert_allclose(result.f2, 3 * x**2 * y**2, rtol=1e-15)
        np.testing.assert_allclose(result.f12, 6 * x * y**2, rtol=1e-15)

    def test_alcubierre_shape_flows_through_unchanged(self):
        # The existing shape function must accept HyperDual as-is.
        r0 = np.linspace(0.5, 5.0, 9)
        result = alcubierre_shape(hyperdual_seed(r0), R, SIGMA)
        s, RR = SIGMA, R
        dfdr = (
            s * (1 - np.tanh(s * (RR + r0)) ** 2)
            - s * (1 - np.tanh(s * (RR - r0)) ** 2)
        ) / (2 * np.tanh(s * RR))
        np.testing.assert_allclose(result.f, alcubierre_shape(r0, R, SIGMA))
        np.testing.assert_allclose(result.f1, dfdr, rtol=1e-12)

    def test_ndarray_left_operand_defers(self):
        x = np.array([1.0, 2.0])
        h = HyperDual(x, 1.0)
        result = x * h
        assert isinstance(result, HyperDual)
        np.testing.assert_allclose(result.f, x**2)
        np.testing.assert_allclose(result.f1, x)


class TestExactMetricDerivatives:
    def test_alcubierre_derivatives_match_closed_form(self):
        fn = alcubierre_metric_fn(V, R, SIGMA)
        t, x, y, z = 0.0, 1.0, 0.5, -0.3
        g, dg, d2g = exact_metric_derivatives(fn, t, x, y, z)

        h = 1e-5

        def g01(xx):
            r = np.sqrt((xx - V * t) ** 2 + y**2 + z**2)
            return -V * alcubierre_shape(r, R, SIGMA)

        fd1 = (g01(x + h) - g01(x - h)) / (2 * h)
        fd2 = (g01(x + h) - 2 * g01(x) + g01(x - h)) / h**2
        np.testing.assert_allclose(dg[1, 0, 1], fd1, rtol=1e-8)
        np.testing.assert_allclose(d2g[1, 1, 0, 1], fd2, rtol=1e-4)
        # Constant components carry no derivatives.
        np.testing.assert_array_equal(dg[:, 2, 2], 0.0)
        np.testing.assert_array_equal(d2g[:, :, 3, 3], 0.0)

    def test_second_derivative_symmetry(self):
        fn = alcubierre_metric_fn(V, R, SIGMA)
        g, dg, d2g = exact_metric_derivatives(fn, 0.2, 1.3, 0.4, 0.1)
        np.testing.assert_array_equal(d2g, np.swapaxes(d2g, 0, 1))


class TestExactGridSolver:
    def test_metric_matches_grid_builder(self):
        grid_size = (1, 8, 8, 8)
        wc = centered_world(grid_size)
        solver = ExactGridSolver(alcubierre_metric_fn(V, R, SIGMA), name="Alcubierre")
        exact = solver.metric_on_grid(grid_size, wc)
        builder = alcubierre_metric(grid_size, wc, v=V, R=R, sigma=SIGMA)
        np.testing.assert_array_equal(exact.tensor, builder.tensor)

    def test_alcubierre_energy_density_is_exact(self):
        # Same analytic rho as the FD test in test_grid.py, but the
        # match must be machine precision, not the 5% FD band -- and it
        # must hold at grid BOUNDARY points, where FD stencils degrade.
        n, h = 16, 0.5
        grid_size = (1, n, n, n)
        wc = centered_world(grid_size, h)
        solver = ExactGridSolver(alcubierre_metric_fn(V, R, SIGMA))
        T = solver.solve(grid_size, wc, grid_scale=(1, h, h, h))

        ax = np.arange(n) * h - wc[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        s = SIGMA
        dfdr = (
            s * (1 - np.tanh(s * (R + r)) ** 2) - s * (1 - np.tanh(s * (R - r)) ** 2)
        ) / (2 * np.tanh(s * R))
        rho = -(V**2) * (Y**2 + Z**2) * dfdr**2 / (32 * np.pi * r**2)

        error = np.abs(T.tensor[0, 0, 0] - rho)
        assert error.max() / np.abs(rho).max() < 1e-12

    def test_far_field_is_vacuum_to_round_off(self):
        solver = ExactGridSolver(alcubierre_metric_fn(V, R, SIGMA))
        T = solver.stress_energy_at(0.0, 50.0, 40.0, 30.0)
        assert np.abs(T).max() < 1e-15

    def test_fd_solver_converges_to_exact(self):
        # The exact pipeline is the ground truth the FD margins inherit
        # O(h^4) error against; halving h must shrink the gap.
        solver = ExactGridSolver(alcubierre_metric_fn(V, R, SIGMA))
        gaps = []
        for n, h in ((32, 0.25), (64, 0.125)):
            grid_size = (1, n, n, n)
            wc = centered_world(grid_size, h)
            metric = alcubierre_metric(
                grid_size, wc, v=V, R=R, sigma=SIGMA, grid_scale=(1, h, h, h)
            )
            T_fd = GridSolver(order=4).solve(metric)
            T_exact = solver.solve(grid_size, wc, grid_scale=(1, h, h, h))
            interior = (slice(n // 8, -(n // 8)),) * 3
            gaps.append(
                np.abs(
                    T_fd.tensor[0, 0, 0][interior] - T_exact.tensor[0, 0, 0][interior]
                ).max()
            )
        assert gaps[1] < gaps[0] / 8.0

    def test_null_tangent_is_null(self):
        solver = ExactGridSolver(alcubierre_metric_fn(0.5, 2.0, 4.0))
        k = solver.null_tangent(0.0, -1.0, 0.7, 0.2, [1.0, 0.3, -0.1])
        g = solver.metric_at(0.0, -1.0, 0.7, 0.2)
        assert abs(k @ g @ k) < 1e-12
        assert k[0] > 0.0


@pytest.fixture(scope="module")
def anec_solver():
    return ExactGridSolver(alcubierre_metric_fn(0.5, 2.0, 4.0), name="Alcubierre")


class TestExactNullGeodesicANEC:
    def test_off_axis_ray_violates_anec(self, anec_solver):
        # The head-on Alcubierre on-axis contraction is analytically
        # zero, so the violation only shows up off axis -- the gap the
        # 1-D slice evaluator cannot reach.
        evaluator = ExactNullGeodesicANEC(anec_solver)
        result = evaluator.integrate(
            start=(0.0, -8.0, 1.0, 0.0),
            spatial_direction=(1.0, 0.0, 0.0),
            comoving_velocity=0.5,
        )
        assert result["anec"] < -1e-3
        assert result["null_residual"] < 1e-6

    def test_wall_grazing_ray_sees_positive_average(self, anec_solver):
        # At impact parameter ~ R the ray passes through the outer wall
        # where the dominant contribution is positive.
        evaluator = ExactNullGeodesicANEC(anec_solver)
        result = evaluator.integrate(
            start=(0.0, -8.0, 2.0, 0.0),
            spatial_direction=(1.0, 0.0, 0.0),
            comoving_velocity=0.5,
        )
        assert result["anec"] > 1e-3

    def test_transverse_symmetry(self, anec_solver):
        # +y and -y impact parameters are equivalent by the axial
        # symmetry of the bubble.
        evaluator = ExactNullGeodesicANEC(anec_solver, num_samples=400)
        up = evaluator.integrate(
            (0.0, -8.0, 1.0, 0.0), (1.0, 0.0, 0.0), comoving_velocity=0.5
        )
        down = evaluator.integrate(
            (0.0, -8.0, -1.0, 0.0), (1.0, 0.0, 0.0), comoving_velocity=0.5
        )
        np.testing.assert_allclose(up["anec"], down["anec"], rtol=1e-6)

    def test_flat_space_ray_is_zero(self):
        def minkowski_fn(t, x, y, z):
            return [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

        solver = ExactGridSolver(minkowski_fn, name="Minkowski")
        evaluator = ExactNullGeodesicANEC(solver, num_samples=200)
        result = evaluator.integrate((0.0, -8.0, 1.5, 0.0), (1.0, 0.0, 0.0))
        assert result["anec"] == 0.0
        assert result["null_residual"] < 1e-12
