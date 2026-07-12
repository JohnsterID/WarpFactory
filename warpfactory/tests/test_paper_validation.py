"""Quantitative validation against the WarpFactory paper.

Reproduces computational results from "Analyzing Warp Drive Spacetimes
with Warp Factory" (Helmerich et al., CQG 2024; arXiv 2404.03095v2)
using the grid pipeline plus the SI conversion layer.

The paper uses 1 m grid spacing on 1000+ point grids with 1000
observers; these tests use coarser grids and fewer observers so they
run in CI, with tolerances set accordingly. Each test states the paper
section it validates.
"""

import numpy as np
import pytest

from warpfactory.grid import (
    GridSolver,
    alcubierre_metric,
    do_frame_transfer,
    get_energy_conditions,
    modified_time_metric,
    van_den_broeck_metric,
)
from warpfactory.grid.shape_functions import alcubierre_shape
from warpfactory.grid.si_units import (
    si_energy_factor,
    stress_energy_to_geometric,
    stress_energy_to_si,
)

# Paper Section 4.1 (Figures 1-3): Alcubierre with v = 0.1c, R = 300 m,
# sigma = 0.015 1/m.
ALCUBIERRE_PAPER = {"v": 0.1, "R": 300.0, "sigma": 0.015}


def centered_world(grid_size, spacing):
    return (0.0,) + tuple((n - 1) * spacing / 2 for n in grid_size[1:])


def solve_alcubierre(n=64, h=12.5):
    grid_size = (1, n, n, n)
    world_center = centered_world(grid_size, h)
    metric = alcubierre_metric(
        grid_size, world_center, grid_scale=(1, h, h, h), **ALCUBIERRE_PAPER
    )
    return metric, GridSolver(order=4).solve(metric)


@pytest.fixture(scope="module")
def alcubierre_paper_setup():
    return solve_alcubierre()


class TestSIConversion:
    def test_si_factor_value(self):
        # c^4/G = (299792458)^4 / 6.6743e-11 ~ 1.2103e44 J/m^3 per 1/m^2
        assert si_energy_factor() == pytest.approx(1.2103e44, rel=1e-3)

    def test_roundtrip(self, alcubierre_paper_setup):
        _, T = alcubierre_paper_setup
        T_si = stress_energy_to_si(T)
        assert T_si.params["units"] == "si"
        back = stress_energy_to_geometric(T_si)
        np.testing.assert_allclose(back.tensor, T.tensor)
        assert "units" not in back.params

    def test_double_conversion_is_noop(self, alcubierre_paper_setup):
        _, T = alcubierre_paper_setup
        T_si = stress_energy_to_si(T)
        np.testing.assert_array_equal(stress_energy_to_si(T_si).tensor, T_si.tensor)

    def test_rejects_metric_tensors(self, alcubierre_paper_setup):
        metric, _ = alcubierre_paper_setup
        with pytest.raises(ValueError):
            stress_energy_to_si(metric)


class TestAlcubierrePaperSection41:
    def test_peak_energy_density_figure2(self, alcubierre_paper_setup):
        """Peak Eulerian energy density ~ -6.78e35 J/m^3.

        The analytic Eulerian density is
        rho = -(c^4/G) v^2 (y^2+z^2) (df/dr)^2 / (32 pi r^2); at the
        equatorial bubble wall with the paper parameters its extremum
        is -6.775e35 J/m^3 (consistent with the negative-energy shell
        of Figure 2, arXiv 2404.03095v2).
        """
        metric, T = alcubierre_paper_setup
        rho_si = stress_energy_to_si(do_frame_transfer(metric, T)).tensor[0, 0]

        r = np.linspace(200.0, 400.0, 20001)
        dfdr = np.gradient(
            alcubierre_shape(r, **{k: ALCUBIERRE_PAPER[k] for k in ("R", "sigma")}), r
        )
        analytic_peak = (
            -(ALCUBIERRE_PAPER["v"] ** 2) * dfdr**2 / (32.0 * np.pi)
        ).min() * si_energy_factor()

        assert analytic_peak == pytest.approx(-6.775e35, rel=1e-3)
        assert rho_si.min() == pytest.approx(analytic_peak, rel=0.02)

    def test_energy_density_negative_shell(self, alcubierre_paper_setup):
        """Figure 2: negative energy concentrated in the bubble wall."""
        metric, T = alcubierre_paper_setup
        rho = do_frame_transfer(metric, T).tensor[0, 0, 0]

        n, h = 64, 12.5
        world_center = centered_world((1, n, n, n), h)
        ax = np.arange(n) * h - world_center[1]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)

        wall = np.abs(r - ALCUBIERRE_PAPER["R"]) < 100.0
        deep_interior = r < 100.0
        peak_magnitude = -rho[wall].min()
        assert peak_magnitude > 1e-10
        # Density in the wall dominates the interior by orders of magnitude.
        assert np.abs(rho[deep_interior]).max() < 1e-3 * peak_magnitude

    @pytest.mark.parametrize("condition", ["Null", "Weak", "Dominant", "Strong"])
    def test_table1_all_conditions_violated(self, alcubierre_paper_setup, condition):
        """Table 1: the Alcubierre metric violates all four conditions."""
        metric, T = alcubierre_paper_setup
        violation = get_energy_conditions(
            T, metric, condition, num_angular_vec=50, num_time_vec=8
        )
        # Violations must be at the physical scale of the wall density
        # (~ -5.6e-9 1/m^2 geometric), far above FD noise.
        assert violation.min() < -1e-10


class TestVanDenBroeckPaperSection42:
    def test_table1_null_and_weak_violated(self):
        """Table 1: Van Den Broeck violates NEC and WEC."""
        n, h = 48, 10.0
        grid_size = (1, n, n, n)
        metric = van_den_broeck_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.1,
            R1=100.0,
            sigma1=0.06,
            R2=180.0,
            sigma2=0.06,
            A=2.0,
            grid_scale=(1, h, h, h),
        )
        T = GridSolver(order=4).solve(metric)
        for condition in ("Null", "Weak"):
            violation = get_energy_conditions(
                T, metric, condition, num_angular_vec=30, num_time_vec=5
            )
            assert violation.min() < -1e-8, condition


class TestModifiedTimePaperSection43:
    def test_table1_null_and_weak_violated(self):
        """Table 1: the Modified Time metric violates NEC and WEC."""
        n, h = 48, 12.5
        grid_size = (1, n, n, n)
        metric = modified_time_metric(
            grid_size,
            centered_world(grid_size, h),
            v=0.1,
            A=0.5,
            R=300.0,
            sigma=0.015,
            grid_scale=(1, h, h, h),
        )
        T = GridSolver(order=4).solve(metric)
        for condition in ("Null", "Weak"):
            violation = get_energy_conditions(
                T, metric, condition, num_angular_vec=30, num_time_vec=5
            )
            assert violation.min() < -1e-8, condition
