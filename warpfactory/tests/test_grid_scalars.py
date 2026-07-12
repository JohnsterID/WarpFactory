"""Tests for kinematic scalars on grids (warpfactory.grid.scalars)."""

import numpy as np
import pytest

from warpfactory.grid import (
    alcubierre_metric,
    eulerian_velocity,
    get_scalars,
    minkowski_metric,
)
from warpfactory.grid.shape_functions import alcubierre_shape


def centered_world(grid_size, spacing=1.0):
    return (0.0,) + tuple((n - 1) * spacing / 2 for n in grid_size[1:])


@pytest.fixture(scope="module")
def alcubierre_setup():
    n, h = 64, 0.25
    grid_size = (1, n, n, n)
    world_center = centered_world(grid_size, h)
    v, R, sigma = 2.0, 3.0, 2.0
    metric = alcubierre_metric(
        grid_size, world_center, v=v, R=R, sigma=sigma, grid_scale=(1, h, h, h)
    )
    coords = np.arange(n) * h - world_center[1]
    return metric, coords, (v, R, sigma)


class TestEulerianVelocity:
    def test_minkowski_rest_observer(self):
        metric = minkowski_metric((1, 4, 4, 4))
        u_up, u_down = eulerian_velocity(metric)
        np.testing.assert_allclose(u_up[0], 1.0)
        np.testing.assert_allclose(u_up[1:], 0.0)
        np.testing.assert_allclose(u_down[0], -1.0)
        np.testing.assert_allclose(u_down[1:], 0.0)

    def test_normalized_on_alcubierre(self, alcubierre_setup):
        metric, _, _ = alcubierre_setup
        u_up, u_down = eulerian_velocity(metric)
        norm = np.einsum("m...,m...->...", u_up, u_down)
        np.testing.assert_allclose(norm, -1.0, atol=1e-12)

    def test_covariant_form_is_lapse_only(self, alcubierre_setup):
        # u_mu = (-alpha, 0, 0, 0) for any metric in ADM form.
        metric, _, _ = alcubierre_setup
        _, u_down = eulerian_velocity(metric)
        np.testing.assert_allclose(u_down[1:], 0.0, atol=1e-12)


class TestGetScalars:
    def test_minkowski_all_scalars_zero(self):
        metric = minkowski_metric((1, 8, 8, 8))
        expansion, shear, vorticity = get_scalars(metric)
        assert expansion.shape == (1, 8, 8, 8)
        np.testing.assert_allclose(expansion, 0.0, atol=1e-14)
        np.testing.assert_allclose(shear, 0.0, atol=1e-14)
        np.testing.assert_allclose(vorticity, 0.0, atol=1e-14)

    def test_alcubierre_expansion_matches_analytic(self, alcubierre_setup):
        """theta = v (x - xs)/r df/dr (Alcubierre 1994, eq. 12)."""
        metric, coords, (v, R, sigma) = alcubierre_setup
        expansion, _, _ = get_scalars(metric, order=4)

        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dfdr = (
            alcubierre_shape(r + 1e-6, R, sigma) - alcubierre_shape(r - 1e-6, R, sigma)
        ) / 2e-6
        theta_analytic = v * X / np.maximum(r, 1e-12) * dfdr

        interior = (slice(4, -4),) * 3
        num = expansion[0][interior]
        ana = theta_analytic[interior]
        assert np.abs(num - ana).max() / np.abs(ana).max() < 0.05

    def test_alcubierre_front_contracts_back_expands(self, alcubierre_setup):
        # Volume elements contract ahead of the bubble (x > 0 half,
        # where df/dr < 0 and (x-xs) > 0) and expand behind it.
        metric, coords, _ = alcubierre_setup
        expansion, _, _ = get_scalars(metric)
        mid = len(coords) // 2
        front = expansion[0, mid + 8 :, mid, mid]
        back = expansion[0, : mid - 8, mid, mid]
        assert front.min() < -1e-3
        assert back.max() > 1e-3

    def test_alcubierre_has_shear_but_no_vorticity(self, alcubierre_setup):
        metric, _, _ = alcubierre_setup
        _, shear, vorticity = get_scalars(metric)
        assert shear.max() > 1e-3
        # The Eulerian congruence is hypersurface-orthogonal, so the
        # vorticity vanishes identically.
        np.testing.assert_allclose(vorticity, 0.0, atol=1e-10)

    def test_shear_is_nonnegative(self, alcubierre_setup):
        # sigma_ij is spatial, so sigma^2 = sigma_ij sigma^ij / 2 >= 0.
        metric, _, _ = alcubierre_setup
        _, shear, _ = get_scalars(metric)
        assert shear.min() > -1e-12

    def test_second_order_agrees_with_fourth(self):
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
        e2, _, _ = get_scalars(metric, order=2)
        e4, _, _ = get_scalars(metric, order=4)
        interior = (slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
        scale = np.abs(e4[interior]).max()
        # The sigma = 2 wall is barely resolved at h = 0.5, so the two
        # FD orders agree only to truncation error (~10% here).
        assert np.abs(e2[interior] - e4[interior]).max() < 0.15 * scale
