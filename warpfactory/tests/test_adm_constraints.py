"""Tests for the ADM constraint evaluator."""

import numpy as np

from warpfactory.grid import (
    ADMConstraints,
    GridSolver,
    alcubierre_metric,
    get_scalars,
    minkowski_metric,
    schwarzschild_metric,
)


def centered_world(grid_size, h):
    return tuple((n - 1) / 2 * s for n, s in zip(grid_size, (1, h, h, h)))


class TestADMConstraints:
    def test_minkowski_satisfies_constraints_exactly(self):
        metric = minkowski_metric((1, 8, 8, 8))
        res = ADMConstraints(order=4).evaluate(metric)
        assert np.abs(res.hamiltonian).max() == 0.0
        assert np.abs(res.momentum).max() == 0.0
        assert np.abs(res.extrinsic_curvature).max() == 0.0

    def test_alcubierre_with_own_stress_energy_closes_hamiltonian(self):
        # rho and R3 + K^2 - K_ij K^ij come from the same FD pass, so
        # the Hamiltonian residual is algebraically zero (round-off).
        n, h = 32, 0.25
        grid_size = (1, n, n, n)
        metric = alcubierre_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.5,
            R=2.0,
            sigma=4.0,
            grid_scale=(1, h, h, h),
        )
        T = GridSolver(order=4).solve(metric)
        res = ADMConstraints(order=4).evaluate(metric, T)
        interior = (slice(4, -4),) * 3
        assert np.abs(res.hamiltonian[0][interior]).max() < 1e-12

    def test_momentum_residual_converges_with_resolution(self):
        # D_j K^j_i differentiates FD output, so the momentum residual
        # is pure discretization error and must shrink under
        # refinement of the same physical bubble.
        ratios = []
        for n, h in [(24, 0.5), (48, 0.25)]:
            grid_size = (1, n, n, n)
            metric = alcubierre_metric(
                grid_size,
                centered_world(grid_size, h),
                v=0.5,
                R=2.0,
                sigma=2.0,
                grid_scale=(1, h, h, h),
            )
            T = GridSolver(order=4).solve(metric)
            res = ADMConstraints(order=4).evaluate(metric, T)
            border = int(2.0 / h)
            interior = (slice(None),) + (slice(border, -border),) * 3
            resid = np.abs(res.momentum[:, 0][interior]).max()
            scale = 8 * np.pi * np.abs(res.momentum_density[:, 0][interior]).max()
            ratios.append(resid / scale)
        assert ratios[1] < ratios[0] / 3.0
        assert ratios[1] < 0.1

    def test_vacuum_alcubierre_violates_hamiltonian(self):
        # The Alcubierre slice is NOT valid vacuum initial data: with
        # rho = 0 the Hamiltonian residual equals 16 pi rho_required.
        n, h = 24, 0.5
        grid_size = (1, n, n, n)
        metric = alcubierre_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.5,
            R=2.0,
            sigma=4.0,
            grid_scale=(1, h, h, h),
        )
        T = GridSolver(order=4).solve(metric)
        adm = ADMConstraints(order=4)
        res_vac = adm.evaluate(metric)
        res_matter = adm.evaluate(metric, T)
        interior = (slice(4, -4),) * 3
        vac = np.abs(res_vac.hamiltonian[0][interior]).max()
        closed = np.abs(res_matter.hamiltonian[0][interior]).max()
        assert vac > 0.1
        assert closed < 1e-6 * vac

    def test_mean_curvature_is_minus_expansion(self):
        # get_scalars returns theta of the Eulerian congruence;
        # K_ij = -nabla_i n_j means theta = -K.
        n, h = 24, 0.5
        grid_size = (1, n, n, n)
        metric = alcubierre_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.5,
            R=2.0,
            sigma=4.0,
            grid_scale=(1, h, h, h),
        )
        expansion, _, _ = get_scalars(metric)
        res = ADMConstraints(order=4).evaluate(metric)
        interior = (slice(4, -4),) * 3
        np.testing.assert_allclose(
            expansion[0][interior],
            -res.mean_curvature[0][interior],
            atol=1e-12,
        )

    def test_schwarzschild_slice_is_time_symmetric(self):
        # Static metric with zero shift: K_ij = 0 identically, and the
        # vacuum Hamiltonian constraint reduces to R3 = 0 to FD
        # tolerance in the exterior.
        n, h = 40, 0.5
        grid_size = (1, n, n, n)
        world_center = centered_world(grid_size, h)
        metric = schwarzschild_metric(
            grid_size, world_center, rs=1.0, grid_scale=(1, h, h, h)
        )
        res = ADMConstraints(order=4).evaluate(metric)
        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        exterior = (r > 4.0) & (r < 8.0)
        assert np.abs(res.extrinsic_curvature[:, :, 0][:, :, exterior]).max() == 0.0
        assert np.abs(res.hamiltonian[0][exterior]).max() < 1e-3
