"""Tests for grid energy-condition maps (warpfactory.grid.energy_conditions)."""

import numpy as np
import pytest

from warpfactory.grid import (
    GridSolver,
    alcubierre_metric,
    do_frame_transfer,
    eulerian_transformation_matrix,
    even_points_on_sphere,
    generate_uniform_field,
    get_energy_conditions,
    minkowski_metric,
    schwarzschild_metric,
)
from warpfactory.grid.shape_functions import alcubierre_shape

MINKOWSKI_ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def centered_world(grid_size, spacing=1.0):
    return (0.0,) + tuple((n - 1) * spacing / 2 for n in grid_size[1:])


class TestObserverFields:
    def test_sphere_points_have_requested_radius(self):
        pts = even_points_on_sphere(2.5, 64)
        assert pts.shape == (3, 64)
        np.testing.assert_allclose(np.linalg.norm(pts, axis=0), 2.5, rtol=1e-12)

    def test_sphere_points_are_spread_out(self):
        pts = even_points_on_sphere(1.0, 100)
        # Every octant of the sphere must be sampled (ignore points that
        # fall exactly on a coordinate plane).
        off_plane = pts[:, np.all(pts != 0.0, axis=0)]
        octants = set(map(tuple, np.sign(off_plane).T.astype(int)))
        assert len(octants) == 8

    def test_null_field_shape_and_norm(self):
        field = generate_uniform_field("nulllike", 50)
        assert field.shape == (4, 50)
        np.testing.assert_allclose(np.sum(field**2, axis=0), 1.0)
        # Unit-Euclidean-norm with |spatial| = t makes these null in eta.
        eta_norm = np.einsum(
            "m,mn,ni->i", *(field[:, :1].ravel(), MINKOWSKI_ETA, field)
        )[0]
        assert abs(eta_norm) < 1e-12

    def test_timelike_field_is_timelike(self):
        field = generate_uniform_field("timelike", 30, 6)
        assert field.shape == (4, 30, 6)
        eta_norm = np.einsum("mij,mn,nij->ij", field, MINKOWSKI_ETA, field)
        assert np.all(eta_norm < 1e-12)

    def test_unknown_field_type_raises(self):
        with pytest.raises(ValueError):
            generate_uniform_field("spacelike", 10)


class TestEulerianTransformation:
    def test_transforms_metric_to_eta(self):
        grid_size = (1, 12, 12, 12)
        metric = alcubierre_metric(
            grid_size, centered_world(grid_size), v=0.9, R=3.0, sigma=1.0
        )
        M = eulerian_transformation_matrix(metric.tensor)
        eta = np.einsum("am...,ab...,bn...->mn...", M, metric.tensor, M)
        target = MINKOWSKI_ETA.reshape(4, 4, 1, 1, 1, 1)
        np.testing.assert_allclose(
            eta, np.broadcast_to(eta * 0 + target, eta.shape), atol=1e-12
        )

    def test_identity_on_minkowski(self):
        metric = minkowski_metric((1, 3, 3, 3))
        M = eulerian_transformation_matrix(metric.tensor)
        identity = np.eye(4).reshape(4, 4, 1, 1, 1, 1)
        np.testing.assert_allclose(
            M, np.broadcast_to(M * 0 + identity, M.shape), atol=1e-12
        )


class TestFrameTransfer:
    def test_minkowski_energy_unchanged(self):
        metric = minkowski_metric((1, 8, 8, 8))
        T = GridSolver(order=2).solve(metric)
        local = do_frame_transfer(metric, T)
        assert local.frame == "eulerian"
        assert local.index == "contravariant"
        np.testing.assert_allclose(local.tensor, 0.0, atol=1e-12)

    def test_already_eulerian_is_noop(self):
        metric = minkowski_metric((1, 8, 8, 8))
        T = GridSolver(order=2).solve(metric)
        local = do_frame_transfer(metric, T)
        again = do_frame_transfer(metric, local)
        assert again is local

    def test_alcubierre_eulerian_density_matches_analytic(self):
        n, h = 64, 0.25
        grid_size = (1, n, n, n)
        world_center = centered_world(grid_size, h)
        v, R, sigma = 2.0, 3.0, 2.0
        metric = alcubierre_metric(
            grid_size, world_center, v=v, R=R, sigma=sigma, grid_scale=(1, h, h, h)
        )
        T = GridSolver(order=4).solve(metric)
        local = do_frame_transfer(metric, T)

        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dfdr = (
            alcubierre_shape(r + 1e-6, R, sigma) - alcubierre_shape(r - 1e-6, R, sigma)
        ) / 2e-6
        rho = -(v**2) * (Y**2 + Z**2) * dfdr**2 / (32 * np.pi * (r**2 + 1e-12))

        interior = (slice(4, -4),) * 3
        num = local.tensor[0, 0, 0][interior]
        ana = rho[interior]
        assert np.abs(num - ana).max() / np.abs(ana).max() < 0.05

    def test_unsupported_frame_raises(self):
        metric = minkowski_metric((1, 4, 4, 4))
        T = GridSolver(order=2).solve(metric)
        with pytest.raises(ValueError):
            do_frame_transfer(metric, T, frame="comoving")


@pytest.fixture(scope="module")
def alcubierre_setup():
    n, h = 32, 0.5
    grid_size = (1, n, n, n)
    metric = alcubierre_metric(
        grid_size,
        centered_world(grid_size, h),
        v=2.0,
        R=3.0,
        sigma=2.0,
        grid_scale=(1, h, h, h),
    )
    T = GridSolver(order=4).solve(metric)
    return metric, T


class TestEnergyConditionMaps:
    def test_minkowski_all_conditions_zero(self):
        metric = minkowski_metric((1, 8, 8, 8))
        T = GridSolver(order=2).solve(metric)
        for condition in ("Null", "Weak", "Dominant", "Strong"):
            violation = get_energy_conditions(
                T, metric, condition, num_angular_vec=20, num_time_vec=4
            )
            assert violation.shape == (1, 8, 8, 8)
            np.testing.assert_allclose(violation, 0.0, atol=1e-12, err_msg=condition)

    @pytest.mark.parametrize("condition", ["Null", "Weak", "Strong"])
    def test_alcubierre_violates(self, alcubierre_setup, condition):
        metric, T = alcubierre_setup
        violation = get_energy_conditions(
            T, metric, condition, num_angular_vec=30, num_time_vec=5
        )
        assert violation.min() < -1e-6

    def test_null_map_bounds_weak_map(self, alcubierre_setup):
        # WEC observers include the rest observer and approach null ones;
        # in the Eulerian frame min over timelike cannot violate more
        # than min over null plus rest-frame rho.
        metric, T = alcubierre_setup
        weak = get_energy_conditions(
            T, metric, "Weak", num_angular_vec=30, num_time_vec=5
        )
        rho = do_frame_transfer(metric, T).tensor[0, 0]
        assert np.all(weak <= rho + 1e-12)

    def test_return_vec_shapes(self, alcubierre_setup):
        metric, T = alcubierre_setup
        violation, vec, field = get_energy_conditions(
            T, metric, "Weak", num_angular_vec=12, num_time_vec=3, return_vec=True
        )
        assert violation.shape == metric.grid_shape
        assert vec.shape == metric.grid_shape + (12, 3)
        assert field.shape == (4, 12, 3)
        np.testing.assert_allclose(vec.min(axis=(-1, -2)), violation)

    def test_return_vec_shapes_null(self, alcubierre_setup):
        metric, T = alcubierre_setup
        violation, vec, field = get_energy_conditions(
            T, metric, "Null", num_angular_vec=12, return_vec=True
        )
        assert vec.shape == metric.grid_shape + (12,)
        assert field.shape == (4, 12)
        np.testing.assert_allclose(vec.min(axis=-1), violation)

    def test_schwarzschild_exterior_satisfies_conditions(self):
        # Vacuum: T ~ 0 up to FD error, so no significant violations.
        n, h = 32, 0.5
        grid_size = (1, n, n, n)
        world_center = centered_world(grid_size, h)
        metric = schwarzschild_metric(
            grid_size, world_center, rs=1.0, grid_scale=(1, h, h, h)
        )
        T = GridSolver(order=4).solve(metric)
        # Inside the horizon no Eulerian observer exists, so the frame
        # transfer warns about non-finite entries there; the assertion
        # only samples the exterior.
        with pytest.warns(UserWarning, match="non-finite"):
            violation = get_energy_conditions(T, metric, "Null", num_angular_vec=20)
        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        exterior = (r > 4.0) & (r < 7.0)
        assert np.abs(violation[0][exterior]).max() < 1e-3

    def test_bad_condition_raises(self, alcubierre_setup):
        metric, T = alcubierre_setup
        with pytest.raises(ValueError):
            get_energy_conditions(T, metric, "Fantastic")

    def test_condition_names_case_insensitive(self, alcubierre_setup):
        metric, T = alcubierre_setup
        lower = get_energy_conditions(T, metric, "null", num_angular_vec=10)
        upper = get_energy_conditions(T, metric, "Null", num_angular_vec=10)
        np.testing.assert_array_equal(lower, upper)
