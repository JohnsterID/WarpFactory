"""Tests for the f(R) effective stress-energy solver."""

import numpy as np

from warpfactory.grid import (
    FofRModel,
    FofRSolver,
    GridSolver,
    SpacetimeTensor,
    alcubierre_metric,
    gr_model,
    minkowski_metric,
    starobinsky_model,
)


def centered_world(grid_size, h):
    return tuple((n - 1) / 2 * s for n, s in zip(grid_size, (1, h, h, h)))


def weak_field_metric(n=201, dx=0.05, k=1.0, eps=1e-4):
    """Static g_tt = -(1 + 2 eps sin(k x)), spatial part flat.

    Linearized Ricci scalar R = 2 eps k^2 sin(k x); with
    F = 1 + 2 alpha R the scalaron correction to the field equations is
    (g_munu Box - nabla_mu nabla_nu) F at leading order in eps:

        dT_tt = +alpha eps k^4 sin(k x) / (2 pi)
        dT_yy = dT_zz = -dT_tt
        dT_xx = 0   (the Box and Hessian pieces cancel on the x axis)
    """
    x = np.arange(n) * dx - (n - 1) * dx / 2
    g = np.zeros((4, 4, 1, n, 1, 1))
    g[0, 0, 0, :, 0, 0] = -(1.0 + 2.0 * eps * np.sin(k * x))
    for i in (1, 2, 3):
        g[i, i] = 1.0
    metric = SpacetimeTensor(
        tensor=g,
        type="metric",
        index="covariant",
        scaling=(1.0, dx, dx, dx),
        name="weak field",
    )
    return metric, x


class TestFofRSolver:
    def test_gr_model_reproduces_grid_solver(self):
        grid_size = (1, 20, 20, 20)
        h = 0.5
        metric = alcubierre_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.5,
            R=2.0,
            sigma=4.0,
            grid_scale=(1, h, h, h),
        )
        T_gr = GridSolver(order=4).solve(metric)
        T_fr = FofRSolver(gr_model(), order=4).solve(metric)
        np.testing.assert_allclose(T_fr.tensor, T_gr.tensor, atol=1e-14)

    def test_minkowski_is_vacuum_in_starobinsky(self):
        metric = minkowski_metric((1, 8, 8, 8))
        T = FofRSolver(starobinsky_model(alpha=0.1), order=4).solve(metric)
        assert np.abs(T.tensor).max() == 0.0

    def test_cosmological_constant_closed_form(self):
        # f(R) = R - 2 Lambda on flat space demands T_munu =
        # Lambda g_munu / (8 pi) exactly (dark-energy vacuum source).
        lam = 0.3
        model = FofRModel(
            f=lambda R: R - 2.0 * lam,
            f_prime=np.ones_like,
            name="GR+Lambda",
            params={"Lambda": lam},
        )
        metric = minkowski_metric((1, 8, 8, 8))
        T = FofRSolver(model, order=4).solve(metric, contravariant=False)
        expected = lam * np.asarray(metric.tensor) / (8.0 * np.pi)
        np.testing.assert_allclose(T.tensor, expected, atol=1e-15)

    def test_weak_field_scalaron_matches_linearized_theory(self):
        alpha, k, eps = 0.1, 1.0, 1e-4
        metric, x = weak_field_metric(k=k, eps=eps)
        T0 = FofRSolver(gr_model(), order=4).solve(metric, contravariant=False)
        Ta = FofRSolver(starobinsky_model(alpha), order=4).solve(
            metric, contravariant=False
        )
        dT = Ta.tensor - T0.tensor

        expected_tt = alpha * eps * k**4 * np.sin(k * x) / (2.0 * np.pi)
        scale = np.abs(expected_tt).max()
        interior = slice(20, -20)

        tt = dT[0, 0, 0, :, 0, 0][interior]
        xx = dT[1, 1, 0, :, 0, 0][interior]
        yy = dT[2, 2, 0, :, 0, 0][interior]
        zz = dT[3, 3, 0, :, 0, 0][interior]

        assert np.abs(tt - expected_tt[interior]).max() / scale < 1e-2
        assert np.abs(yy + expected_tt[interior]).max() / scale < 1e-2
        assert np.abs(zz + expected_tt[interior]).max() / scale < 1e-2
        assert np.abs(xx).max() / scale < 1e-3

    def test_scalaron_correction_linear_in_alpha(self):
        # At leading order in the perturbation the correction is linear
        # in alpha, so doubling alpha must double dT.
        metric, _ = weak_field_metric(n=101)
        T0 = FofRSolver(gr_model(), order=4).solve(metric, contravariant=False)
        T1 = FofRSolver(starobinsky_model(0.1), order=4).solve(
            metric, contravariant=False
        )
        T2 = FofRSolver(starobinsky_model(0.2), order=4).solve(
            metric, contravariant=False
        )
        d1 = T1.tensor - T0.tensor
        d2 = T2.tensor - T0.tensor
        np.testing.assert_allclose(d2, 2.0 * d1, atol=1e-12)

    def test_ricci_scalar_map_weak_field(self):
        k, eps = 1.0, 1e-4
        metric, x = weak_field_metric(k=k, eps=eps)
        R = FofRSolver(gr_model(), order=4).ricci_scalar(metric)
        expected = 2.0 * eps * k**2 * np.sin(k * x)
        interior = slice(20, -20)
        num = R[0, :, 0, 0][interior]
        assert np.abs(num - expected[interior]).max() / eps < 1e-3

    def test_params_record_model_provenance(self):
        metric = minkowski_metric((1, 6, 6, 6))
        T = FofRSolver(starobinsky_model(alpha=0.25), order=2).solve(metric)
        assert T.params["f_of_r_model"] == "Starobinsky"
        assert T.params["alpha"] == 0.25
        assert T.params["order"] == 2
        assert T.type == "stress-energy"
        assert T.index == "contravariant"

    def test_alcubierre_starobinsky_shifts_energy_density(self):
        # The warp wall has nonzero curvature gradients, so the
        # scalaron term must produce a nonzero correction there while
        # the far field (flat) stays uncorrected.
        grid_size = (1, 24, 24, 24)
        h = 0.5
        metric = alcubierre_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.5,
            R=2.0,
            sigma=4.0,
            grid_scale=(1, h, h, h),
        )
        T_gr = FofRSolver(gr_model(), order=4).solve(metric, contravariant=False)
        T_st = FofRSolver(starobinsky_model(0.05), order=4).solve(
            metric, contravariant=False
        )
        diff = np.abs(T_st.tensor - T_gr.tensor)
        assert diff.max() > 1e-6
        corner = diff[:, :, 0, 0, 0, 0]
        assert corner.max() < 1e-2 * diff.max()
