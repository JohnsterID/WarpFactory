"""Tests for the 4-D grid pipeline (warpfactory.grid)."""

import numpy as np
import pytest

from warpfactory.grid import (
    GridSolver,
    SpacetimeTensor,
    alcubierre_comoving_metric,
    alcubierre_metric,
    alcubierre_shape,
    change_tensor_index,
    compact_sigmoid,
    get_slice_data,
    legendre_radial_interp,
    lentz_comoving_metric,
    lentz_metric,
    minkowski_metric,
    minkowski_three_plus_one,
    modified_time_comoving_metric,
    modified_time_metric,
    quadrilinear_interp,
    schwarzschild_metric,
    three_plus_one_builder,
    three_plus_one_decomposer,
    trilinear_interp,
    van_den_broeck_comoving_metric,
    van_den_broeck_metric,
    verify_tensor,
    warp_shell_comoving_metric,
)


def centered_world(grid_size, spacing=1.0):
    return (0.0,) + tuple((n - 1) * spacing / 2 for n in grid_size[1:])


class TestSpacetimeTensor:
    def test_verify_valid_metric(self):
        assert verify_tensor(minkowski_metric((1, 4, 4, 4)))

    def test_verify_rejects_bad_shape(self):
        bad = SpacetimeTensor(tensor=np.zeros((4, 4, 3, 3)))
        assert not verify_tensor(bad)

    def test_verify_rejects_bad_type(self):
        t = minkowski_metric((1, 4, 4, 4))
        t.type = "banana"
        assert not verify_tensor(t)

    def test_verify_rejects_bad_index(self):
        t = minkowski_metric((1, 4, 4, 4))
        t.index = "sideways"
        assert not verify_tensor(t)


class TestChangeTensorIndex:
    def test_metric_inverse_roundtrip(self):
        metric = alcubierre_metric(
            (1, 8, 8, 8), centered_world((1, 8, 8, 8)), v=0.5, R=2.0, sigma=2.0
        )
        up = change_tensor_index(metric, "contravariant")
        back = change_tensor_index(up, "covariant")
        assert up.index == "contravariant"
        np.testing.assert_allclose(back.tensor, metric.tensor, atol=1e-12)

    def test_metric_mixed_rejected(self):
        metric = minkowski_metric((1, 4, 4, 4))
        with pytest.raises(ValueError):
            change_tensor_index(metric, "mixedupdown")

    def test_stress_energy_requires_metric(self):
        T = SpacetimeTensor(
            tensor=np.zeros((4, 4, 1, 4, 4, 4)),
            type="stress-energy",
            index="contravariant",
        )
        with pytest.raises(ValueError):
            change_tensor_index(T, "covariant")

    def test_stress_energy_roundtrip_through_all_indices(self):
        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=0.5, R=2.0, sigma=2.0
        )
        T = GridSolver(order=4).solve(metric)

        chain = [
            "covariant",
            "mixedupdown",
            "contravariant",
            "mixeddownup",
            "contravariant",
        ]
        current = T
        for index in chain:
            current = change_tensor_index(current, index, metric)
        np.testing.assert_allclose(current.tensor, T.tensor, atol=1e-10)

    def test_mixed_trace_is_invariant(self):
        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=0.5, R=2.0, sigma=2.0
        )
        T = GridSolver(order=4).solve(metric)
        t_ud = change_tensor_index(T, "mixedupdown", metric)
        t_du = change_tensor_index(T, "mixeddownup", metric)
        trace_ud = np.einsum("aa...->...", t_ud.tensor)
        trace_du = np.einsum("aa...->...", t_du.tensor)
        np.testing.assert_allclose(trace_ud, trace_du, atol=1e-12)


class TestGridSolver:
    def test_minkowski_vacuum_is_exact(self):
        T = GridSolver(order=4).solve(minkowski_metric((1, 8, 8, 8)))
        assert T.type == "stress-energy"
        assert T.index == "contravariant"
        np.testing.assert_array_equal(T.tensor, 0.0)

    def test_second_order_minkowski_vacuum(self):
        T = GridSolver(order=2).solve(minkowski_metric((1, 8, 8, 8)))
        np.testing.assert_array_equal(T.tensor, 0.0)

    def test_alcubierre_matches_analytic_energy_density(self):
        # Eulerian energy density of the Alcubierre metric:
        # rho = -v^2 (y^2 + z^2) (df/dr)^2 / (32 pi r^2); with alpha = 1
        # and beta^y = beta^z = 0 it equals T^tt.
        n, h = 64, 0.25
        grid_size = (1, n, n, n)
        world_center = centered_world(grid_size, h)
        v, R, sigma = 2.0, 3.0, 2.0
        metric = alcubierre_metric(
            grid_size, world_center, v=v, R=R, sigma=sigma, grid_scale=(1, h, h, h)
        )
        T = GridSolver(order=4).solve(metric)

        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dfdr = (
            alcubierre_shape(r + 1e-6, R, sigma) - alcubierre_shape(r - 1e-6, R, sigma)
        ) / 2e-6
        rho = -(v**2) * (Y**2 + Z**2) * dfdr**2 / (32 * np.pi * (r**2 + 1e-12))

        interior = (slice(4, -4),) * 3
        num = T.tensor[0, 0, 0][interior]
        ana = rho[interior]
        assert np.abs(num - ana).max() / np.abs(ana).max() < 0.05

    def test_schwarzschild_is_vacuum(self):
        n, h = 48, 0.5
        grid_size = (1, n, n, n)
        world_center = centered_world(grid_size, h)
        metric = schwarzschild_metric(
            grid_size, world_center, rs=1.0, grid_scale=(1, h, h, h)
        )
        T = GridSolver(order=4).solve(metric)

        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        exterior = (r > 4.0) & (r < 9.0)
        worst = max(
            np.abs(T.tensor[mu, nu, 0][exterior]).max()
            for mu in range(4)
            for nu in range(4)
        )
        assert worst < 1e-4

    def test_time_dependent_grid(self):
        grid_size = (6, 24, 12, 12)
        world_center = (1.5, 5.75, 2.75, 2.75)
        metric = alcubierre_metric(
            grid_size,
            world_center,
            v=0.5,
            R=2.0,
            sigma=2.0,
            grid_scale=(0.5, 0.5, 0.5, 0.5),
        )
        T = GridSolver(order=4).solve(metric)
        assert np.all(np.isfinite(T.tensor))
        assert T.grid_shape == grid_size

    def test_covariant_output_option(self):
        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=0.5, R=2.0, sigma=2.0
        )
        T_cov = GridSolver(order=4).solve(metric, contravariant=False)
        assert T_cov.index == "covariant"
        T_up = change_tensor_index(T_cov, "contravariant", metric)
        T_direct = GridSolver(order=4).solve(metric)
        np.testing.assert_allclose(T_up.tensor, T_direct.tensor, atol=1e-12)

    def test_rejects_invalid_metric(self):
        bad = SpacetimeTensor(tensor=np.zeros((4, 4, 3, 3)))
        with pytest.raises(ValueError):
            GridSolver().solve(bad)


class TestThreePlusOne:
    def test_flat_space_roundtrip(self):
        alpha, beta, gamma = minkowski_three_plus_one((1, 4, 4, 4))
        g = three_plus_one_builder(alpha, beta, gamma)
        np.testing.assert_array_equal(g, minkowski_metric((1, 4, 4, 4)).tensor)

    def test_decomposer_recovers_adm_variables(self):
        grid_shape = (1, 6, 6, 6)
        alpha, beta, gamma = minkowski_three_plus_one(grid_shape)
        rng = np.random.default_rng(42)
        alpha += 0.1 * rng.random(grid_shape)
        beta[0] = 0.3 * rng.random(grid_shape)
        beta[2] = -0.2 * rng.random(grid_shape)
        gamma[0, 0] += 0.5 * rng.random(grid_shape)
        gamma[0, 1] = gamma[1, 0] = 0.1 * rng.random(grid_shape)

        metric = SpacetimeTensor(tensor=three_plus_one_builder(alpha, beta, gamma))
        alpha2, beta_down, gamma_down, beta_up, gamma_up = three_plus_one_decomposer(
            metric
        )
        np.testing.assert_allclose(alpha2, alpha, atol=1e-12)
        np.testing.assert_allclose(beta_down, beta, atol=1e-12)
        np.testing.assert_allclose(gamma_down, gamma, atol=1e-12)
        identity = np.einsum("ij...,jk...->ik...", gamma_up, gamma_down)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                np.testing.assert_allclose(identity[i, j], expected, atol=1e-12)


class TestShapeFunctions:
    def test_alcubierre_shape_limits(self):
        r = np.array([0.0, 10.0])
        f = alcubierre_shape(r, R=2.0, sigma=8.0)
        assert f[0] == pytest.approx(1.0, abs=1e-9)
        assert f[1] == pytest.approx(0.0, abs=1e-9)

    def test_compact_sigmoid_support(self):
        r = np.linspace(0, 12, 500)
        f = compact_sigmoid(r, R1=3.0, R2=9.0, sigma=2.0, Rbuff=0.5)
        assert np.all(f[r <= 3.5] == 1.0)
        assert np.all(f[r >= 8.5] == 0.0)
        wall = f[(r > 3.5) & (r < 8.5)]
        assert np.all((wall >= 0) & (wall <= 1))
        assert np.all(np.diff(wall) <= 1e-12)


class TestInterpolation:
    def test_trilinear_exact_on_linear_field(self):
        i, j, k = np.meshgrid(np.arange(5), np.arange(5), np.arange(5), indexing="ij")
        field = 2.0 * i - 3.0 * j + 0.5 * k + 1.0
        pos = (1.25, 3.5, 2.75)
        expected = 2.0 * pos[0] - 3.0 * pos[1] + 0.5 * pos[2] + 1.0
        assert trilinear_interp(field, pos) == pytest.approx(expected)

    def test_quadrilinear_exact_on_linear_field(self):
        t, i, j, k = np.meshgrid(
            np.arange(4), np.arange(4), np.arange(4), np.arange(4), indexing="ij"
        )
        field = 1.5 * t + 2.0 * i - 3.0 * j + 0.5 * k
        pos = (1.5, 1.25, 2.5, 0.75)
        expected = 1.5 * pos[0] + 2.0 * pos[1] - 3.0 * pos[2] + 0.5 * pos[3]
        assert quadrilinear_interp(field, pos) == pytest.approx(expected)

    def test_quadrilinear_integer_time_uses_single_slice(self):
        field = np.zeros((3, 4, 4, 4))
        field[1] = 7.0
        assert quadrilinear_interp(field, (1.0, 2.0, 2.0, 2.0)) == pytest.approx(7.0)

    def test_legendre_radial_exact_on_cubic(self):
        idx = np.arange(20, dtype=float)
        profile = idx**3 - 2 * idx**2 + 5
        r = 7.3
        assert legendre_radial_interp(profile, r) == pytest.approx(r**3 - 2 * r**2 + 5)

    def test_legendre_radial_vectorized(self):
        profile = np.linspace(0, 19, 20) ** 2
        r = np.array([2.5, 7.3, 11.9])
        result = legendre_radial_interp(profile, r)
        np.testing.assert_allclose(result, r**2, atol=1e-9)


class TestGridMetrics:
    def test_alcubierre_flat_far_from_bubble(self):
        grid_size = (1, 32, 16, 16)
        world_center = centered_world(grid_size)
        metric = alcubierre_metric(grid_size, world_center, v=2.0, R=2.0, sigma=4.0)
        corner = metric.tensor[:, :, 0, 0, 0, 0]
        np.testing.assert_allclose(corner, np.diag([-1.0, 1, 1, 1]), atol=1e-6)
        center = metric.tensor[:, :, 0, 16, 8, 8]
        assert center[0, 1] == pytest.approx(-2.0, abs=1e-3)

    def test_alcubierre_comoving_shift_vanishes_inside(self):
        grid_size = (1, 33, 17, 17)
        metric = alcubierre_comoving_metric(
            grid_size, centered_world(grid_size), v=2.0, R=2.0, sigma=4.0
        )
        center = metric.tensor[:, :, 0, 16, 8, 8]
        assert center[0, 1] == pytest.approx(0.0, abs=1e-3)
        corner = metric.tensor[:, :, 0, 0, 0, 0]
        assert corner[0, 1] == pytest.approx(2.0, abs=1e-6)

    def test_comoving_builders_reject_multiple_time_slices(self):
        grid_size = (2, 8, 8, 8)
        world_center = (0, 4, 4, 4)
        builders = [
            lambda: alcubierre_comoving_metric(grid_size, world_center, 1.0, 2.0, 2.0),
            lambda: lentz_comoving_metric(grid_size, world_center, 1.0),
            lambda: van_den_broeck_comoving_metric(
                grid_size, world_center, 1.0, 2.0, 2.0, 4.0, 2.0, 0.5
            ),
            lambda: modified_time_comoving_metric(
                grid_size, world_center, 1.0, 2.0, 2.0, 2.0
            ),
            lambda: schwarzschild_metric(grid_size, world_center, 1.0),
        ]
        for build in builders:
            with pytest.raises(ValueError):
                build()

    def test_lentz_has_x_and_y_shift(self):
        grid_size = (1, 29, 29, 29)
        metric = lentz_metric(grid_size, centered_world(grid_size), v=1.0)
        assert np.any(metric.tensor[0, 1] != 0)
        assert np.any(metric.tensor[0, 2] != 0)
        # Template drags along +y and -y symmetrically (WFY odd in y).
        beta_y = metric.tensor[0, 2, 0]
        np.testing.assert_allclose(beta_y, -beta_y[:, ::-1, :], atol=1e-12)

    def test_van_den_broeck_volume_expansion(self):
        grid_size = (1, 17, 17, 17)
        A = 0.5
        metric = van_den_broeck_metric(
            grid_size,
            centered_world(grid_size),
            v=1.0,
            R1=2.0,
            sigma1=4.0,
            R2=4.0,
            sigma2=4.0,
            A=A,
        )
        assert metric.tensor[1, 1, 0, 8, 8, 8] == pytest.approx((1 + A) ** 2, abs=1e-3)
        assert metric.tensor[1, 1, 0, 0, 0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_modified_time_lapse_modification(self):
        grid_size = (1, 17, 17, 17)
        A = 2.0
        metric = modified_time_metric(
            grid_size, centered_world(grid_size), v=0.0, R=2.0, sigma=4.0, A=A
        )
        # v = 0: g_tt = -((1-fs) + fs/A)^2 -> -1/A^2 inside, -1 outside.
        assert metric.tensor[0, 0, 0, 8, 8, 8] == pytest.approx(-1.0 / A**2, abs=1e-3)
        assert metric.tensor[0, 0, 0, 0, 0, 0] == pytest.approx(-1.0, abs=1e-4)

    def test_schwarzschild_reduces_to_radial_form_on_axis(self):
        # On the +x axis: g_xx = 1/(1 - rs/r), g_yy = g_zz = 1.
        grid_size = (1, 32, 9, 9)
        world_center = (0.0, 0.0, 4.0, 4.0)
        rs = 1.0
        metric = schwarzschild_metric(grid_size, world_center, rs)
        ix = 10
        r = float(ix)
        g = metric.tensor[:, :, 0, ix, 4, 4]
        assert g[0, 0] == pytest.approx(-(1 - rs / r), rel=1e-6)
        assert g[1, 1] == pytest.approx(1.0 / (1 - rs / r), rel=1e-6)
        assert g[2, 2] == pytest.approx(1.0, rel=1e-6)
        assert g[3, 3] == pytest.approx(1.0, rel=1e-6)


class TestWarpShell:
    def test_shell_metric_structure(self):
        grid_size = (1, 24, 24, 24)
        world_center = centered_world(grid_size)
        metric = warp_shell_comoving_metric(
            grid_size,
            world_center,
            m=0.1,
            R1=4.0,
            R2=8.0,
            Rbuff=0.5,
            sigma=2.0,
            smooth_factor=10,
            v_warp=0.5,
            do_warp=True,
            r_sample_res=10**4,
        )
        g = metric.tensor
        assert np.all(np.isfinite(g))
        # Interior is flat-ish with a constant lapse below 1 (redshift).
        interior_gtt = g[0, 0, 0, 11, 11, 11]
        assert -1.0 < interior_gtt < -0.9
        # Interior shift equals -v_warp when do_warp is on.
        assert g[0, 1, 0, 11, 11, 11] == pytest.approx(-0.5, abs=1e-6)
        # Exterior approaches Schwarzschild: g_tt ~ -(1 - 2m/r).
        corner_r = np.sqrt(3) * 11.5
        expected = -(1 - 2 * 0.1 / corner_r)
        assert g[0, 0, 0, 0, 0, 0] == pytest.approx(expected, abs=5e-3)

    def test_shell_without_warp_has_no_shift(self):
        grid_size = (1, 16, 16, 16)
        metric = warp_shell_comoving_metric(
            grid_size,
            centered_world(grid_size),
            m=0.05,
            R1=3.0,
            R2=6.0,
            r_sample_res=10**4,
        )
        np.testing.assert_array_equal(metric.tensor[0, 1], 0.0)

    def test_shell_energy_density_positive_in_shell(self):
        grid_size = (1, 24, 24, 24)
        world_center = centered_world(grid_size)
        metric = warp_shell_comoving_metric(
            grid_size,
            world_center,
            m=0.1,
            R1=4.0,
            R2=8.0,
            Rbuff=0.5,
            sigma=2.0,
            smooth_factor=20,
            r_sample_res=10**4,
        )
        T = GridSolver(order=4).solve(metric)
        assert np.all(np.isfinite(T.tensor))
        # Sample mid-shell along the x axis (r = 6).
        assert T.tensor[0, 0, 0, 17, 11, 11] > 0

    def test_tov_pressure_profile(self):
        from warpfactory.grid import tov_constant_density_pressure

        r = np.linspace(0, 12, 2000)
        R2, m = 8.0, 0.1
        rho = np.where(r < R2, m / (4.0 / 3.0 * np.pi * R2**3), 0.0)
        M = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    (4 * np.pi * rho * r**2)[1:] / 2 * np.diff(r)
                    + (4 * np.pi * rho * r**2)[:-1] / 2 * np.diff(r)
                ),
            )
        )
        P = tov_constant_density_pressure(R2, M, rho, r)
        assert np.all(P[r < R2] >= 0)
        assert np.all(P[r >= R2] == 0)
        # Pressure decreases monotonically to the surface.
        inside = P[(r > 0.1) & (r < R2)]
        assert np.all(np.diff(inside) <= 1e-12)


class TestGridPlotting:
    def test_get_slice_data(self):
        field = np.arange(2 * 4 * 5 * 6, dtype=float).reshape(2, 4, 5, 6)
        data = get_slice_data(field, (0, 3), (1, 2))
        np.testing.assert_array_equal(data, field[1, :, :, 2])

    def test_get_slice_data_validates(self):
        field = np.zeros((1, 4, 4, 4))
        with pytest.raises(ValueError):
            get_slice_data(field, (0, 0), (0, 0))
        with pytest.raises(ValueError):
            get_slice_data(field, (0, 3), (5, 0))

    def test_plot_tensor_counts(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from warpfactory.grid import plot_tensor

        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=1.0, R=2.0, sigma=2.0
        )
        figures = plot_tensor(metric)
        assert len(figures) == 10
        for fig in figures:
            plt.close(fig)

        T = GridSolver(order=4).solve(metric)
        mixed = change_tensor_index(T, "mixedupdown", metric)
        figures = plot_tensor(mixed)
        assert len(figures) == 16
        for fig in figures:
            plt.close(fig)

    def test_plot_three_plus_one_counts(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from warpfactory.grid import plot_three_plus_one

        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=1.0, R=2.0, sigma=2.0
        )
        figures = plot_three_plus_one(metric)
        assert len(figures) == 10  # alpha + 3 beta + 6 gamma
        for fig in figures:
            plt.close(fig)

    def test_plot_three_plus_one_rejects_stress_energy(self):
        from warpfactory.grid import plot_three_plus_one

        grid_size = (1, 8, 8, 8)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=1.0, R=2.0, sigma=2.0
        )
        T = GridSolver(order=4).solve(metric)
        with pytest.raises(ValueError):
            plot_three_plus_one(T)
