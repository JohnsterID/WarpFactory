"""CMB blueshift hazard map tests.

Ground truth for the Alcubierre bubble: the Eulerian-measured frequency
of a head-on photon (traveling in -x against the bubble motion) is
omega/omega_far = 1/(1 - v_x), where v_x = v_s f(r_s) is the local
shift velocity -- equal to 1/(1 - v_s) = 2 at the center of a
v_s = 0.5 bubble. Minkowski gives no shift anywhere.
"""

import numpy as np
import pytest

from warpfactory.metrics import AlcubierreMetric, MinkowskiMetric
from warpfactory.spacetime import CMBBlueshiftHazard


@pytest.fixture
def line_grid():
    x = np.linspace(-5, 5, 201)
    return x, np.zeros_like(x), np.zeros_like(x)


def test_minkowski_no_frequency_shift(line_grid):
    """Flat spacetime: the frequency ratio stays exactly 1 along the ray."""
    x, y, z = line_grid
    metric = MinkowskiMetric().calculate(x, y, z, 0.0)
    hazard = CMBBlueshiftHazard(t_max=8.0)

    result = hazard.trace_frequency_shift(
        metric, np.array([4.5, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    )
    assert np.allclose(result["blueshift"], 1.0, atol=1e-8)
    assert np.isclose(result["max_blueshift"], 1.0, atol=1e-8)


def test_alcubierre_head_on_blueshift(line_grid):
    """Head-on photon through a v_s = 0.5 bubble peaks at
    1/(1 - v_s) = 2 at the bubble center."""
    x, y, z = line_grid
    metric = AlcubierreMetric().calculate(x, y, z, 0.0, v_s=0.5, R=1.0, sigma=4.0)
    hazard = CMBBlueshiftHazard(t_max=12.0)

    result = hazard.trace_frequency_shift(
        metric, np.array([4.5, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    )
    assert np.isclose(result["max_blueshift"], 2.0, rtol=1e-3)
    assert abs(result["max_position"][0]) < 0.2

    # Far behind the bubble the photon returns to its emitted frequency.
    assert np.isclose(result["blueshift"][-1], 1.0, atol=1e-3)


def test_hazard_map_head_on_worst(line_grid):
    """The hazard sweep peaks for head-on incidence and decreases
    monotonically as the incidence angle opens up."""
    x, y, z = line_grid
    metric = AlcubierreMetric().calculate(x, y, z, 0.0, v_s=0.5, R=1.0, sigma=4.0)
    hazard = CMBBlueshiftHazard(t_max=12.0)

    report = hazard.hazard_map(metric, n_angles=5, max_angle=np.pi / 3)
    peaks = report["max_blueshift"]
    assert report["angles"].shape == peaks.shape == (5,)
    assert np.isclose(peaks[0], 2.0, rtol=1e-3)
    assert np.all(np.diff(peaks) < 0)
    assert peaks[-1] > 1.0


def test_hazard_map_rejects_bad_angle(line_grid):
    x, y, z = line_grid
    metric = MinkowskiMetric().calculate(x, y, z, 0.0)
    hazard = CMBBlueshiftHazard()
    with pytest.raises(ValueError, match="max_angle"):
        hazard.hazard_map(metric, max_angle=np.pi)
