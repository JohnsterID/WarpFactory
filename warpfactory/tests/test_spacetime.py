"""Test spacetime analysis features."""

import pytest
import numpy as np
from warpfactory.spacetime import (
    GeodesicSolver,
    HorizonFinder,
    SingularityDetector,
    GravitationalLensing
)
from warpfactory.metrics import AlcubierreMetric

@pytest.fixture
def alcubierre_setup(spatial_grid):
    """Set up spacetime for testing."""
    x, y, z = spatial_grid
    t = 0.0
    metric = AlcubierreMetric()
    components = metric.calculate(x, y, z, t, v_s=2.0, R=1.0, sigma=0.5)
    return components, (x, y, z)

def test_geodesic_solver(alcubierre_setup):
    """Test geodesic equation solver."""
    components, (x, y, z) = alcubierre_setup
    solver = GeodesicSolver()

    # Initial conditions: v0 must be timelike (|v| < 1 in flat regions);
    # the old value 1.0 was null and thus unphysical for a massive particle
    t0 = 0.0
    x0 = np.array([-3.0, 0.0, 0.0])
    v0 = np.array([0.5, 0.0, 0.0])

    times, positions, velocities = solver.solve(
        components, t0, x0, v0,
        t_max=10.0,
        dt=0.1
    )

    assert len(times) > 0
    assert positions.shape == (len(times), 3)
    assert velocities.shape == (len(times), 3)

    # 1. Time increases monotonically
    assert np.all(np.diff(times) > 0)

    # 2. Trajectory stays timelike throughout
    for i in range(len(times)):
        four_vel = np.concatenate([[1.0], velocities[i]])
        g = solver.interpolate_metric(components, positions[i])
        norm = four_vel @ g @ four_vel
        assert norm < 0, f"Four-velocity is not timelike at step {i}"

    # 3. Energy is conserved along the geodesic (stationary metric slice).
    # Tolerance is set by linear interpolation of the metric on the
    # coarse 50-point fixture grid, not by the integrator (rtol 1e-8).
    energies = np.array([
        solver.calculate_energy(components, positions[i], velocities[i])
        for i in range(len(times))
    ])
    assert np.allclose(energies, energies[0], rtol=1e-4)

    # 4. Unphysical (spacelike) initial data must be rejected.
    # Note +1.0 along x is actually timelike here due to frame dragging
    # inside the wide bubble; moving against the drag is spacelike.
    with pytest.raises(ValueError):
        solver.solve(components, t0, x0, np.array([-1.0, 0.0, 0.0]),
                     t_max=1.0, dt=0.1)


def test_geodesic_flat_spacetime():
    """Geodesics in Minkowski spacetime are straight lines."""
    from warpfactory.metrics import MinkowskiMetric
    x = np.linspace(-5, 5, 100)
    components = MinkowskiMetric().calculate(x, np.zeros_like(x),
                                             np.zeros_like(x), 0.0)
    solver = GeodesicSolver()

    x0 = np.array([-3.0, 0.0, 0.0])
    v0 = np.array([0.5, 0.2, 0.0])
    times, positions, velocities = solver.solve(components, 0.0, x0, v0,
                                                t_max=5.0, dt=0.1)

    expected = x0[np.newaxis, :] + times[:, np.newaxis] * v0[np.newaxis, :]
    assert np.allclose(positions, expected, atol=1e-9)
    assert np.allclose(velocities, v0, atol=1e-9)


def test_horizon_finder(alcubierre_setup):
    """Test event horizon finder."""
    components, (x, y, z) = alcubierre_setup
    finder = HorizonFinder()

    horizons = finder.find_horizons(components, x, y, z)

    assert "outer" in horizons
    assert "inner" in horizons
    assert "ergosphere" in horizons

    # The v_s=2 bubble is superluminal, so an ergo region (g_tt > 0)
    # must exist and its surface must be found
    assert len(horizons["ergosphere"]) > 0

    # In the standard slicing g_tt g_xx - g_tx^2 = -1 identically for
    # Alcubierre, so there is no coordinate horizon to detect
    assert len(horizons["outer"]) == 0

    # Every returned surface is a closed ring of 3-D points
    for surface in horizons.values():
        if len(surface) > 0:
            assert surface.shape[1] == 3
            assert np.allclose(surface[0], surface[-1])

    properties = finder.analyze_horizons(components, horizons)
    assert properties["area"] > 0
    assert properties["surface_gravity"] >= 0

    # Flat spacetime: no surfaces, zeroed properties
    from warpfactory.metrics import MinkowskiMetric
    flat = MinkowskiMetric().calculate(x, y, z, 0.0)
    flat_horizons = finder.find_horizons(flat, x, y, z)
    assert all(len(s) == 0 for s in flat_horizons.values())
    flat_props = finder.analyze_horizons(flat, flat_horizons)
    assert flat_props["area"] == 0.0

def test_singularity_detector_positive():
    """Detector must find a genuine curvature singularity.

    Regression: the previous implementation computed fake "invariants"
    from second derivatives of raw metric components and never detected
    anything real. A collapsing transverse area g_yy = g_zz = x^2 + eps
    produces a diverging Kretschmann scalar at x = 0.
    """
    x = np.linspace(-5, 5, 400)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    C = x**2 + 1e-4
    metric = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": C,
        "g_zz": C
    }

    detector = SingularityDetector()
    result = detector.find_singularities(metric, x, y, z)

    assert len(result["locations"]) == 1
    assert abs(result["locations"][0][0]) < 0.1
    assert result["strengths"][0] > 1e3

    # Smooth spacetimes must yield no detections
    flat = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x)
    }
    assert len(detector.find_singularities(flat, x, y, z)["locations"]) == 0

def test_singularity_detector(alcubierre_setup):
    """Test singularity detection."""
    components, (x, y, z) = alcubierre_setup
    detector = SingularityDetector()
    
    # Find singularities
    singularities = detector.find_singularities(components, x, y, z)
    
    # Check structure
    assert "locations" in singularities
    assert "types" in singularities
    assert "strengths" in singularities
    assert len(singularities["locations"]) == len(singularities["types"])
    
    # Analyze singularity properties (if any found)
    if len(singularities["locations"]) > 0:
        # Check if curvature invariants exist
        invariants = detector.calculate_invariants(
            components,
            singularities["locations"][0]
        )
        assert "kretschmann" in invariants
        assert "ricci_scalar" in invariants
        
        # Type classification should be valid
        assert singularities["types"][0] in ["spacelike", "timelike", "null"]
        
        # Strength should be positive
        assert singularities["strengths"][0] >= 0

