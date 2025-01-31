"""Test spacetime analysis features."""

import pytest
import numpy as np
import torch
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
    
    # Initial conditions for test particle
    t0 = 0.0
    x0 = np.array([-3.0, 0.0, 0.0])  # Starting position
    v0 = np.array([1.0, 0.0, 0.0])   # Initial velocity (normalized)
    
    # Solve geodesic equations
    times, positions, velocities = solver.solve(
        components, t0, x0, v0,
        t_max=10.0,
        dt=0.1
    )
    
    # Check output shapes
    assert len(times) > 0
    assert positions.shape == (len(times), 3)
    assert velocities.shape == (len(times), 3)
    
    # Physical checks
    # 1. Time should increase monotonically
    assert np.all(np.diff(times) > 0)
    
    # 2. Four-velocity should be timelike
    four_vel = np.column_stack([np.ones_like(times), velocities])
    for i in range(len(times)):
        g = solver.interpolate_metric(components, positions[i])
        norm = np.einsum('i,ij,j->', four_vel[i], g, four_vel[i])
        # Ensure timelike condition (norm < 0)
        assert norm < 0, f"Four-velocity is not timelike at step {i}"
    
    # 3. Maximum force location
    max_force_location = np.argmax(np.abs(velocities[:, 0]))
    # Find closest grid point
    x_max = positions[max_force_location, 0]
    idx = np.argmin(np.abs(x - x_max))
    assert 0.5 <= abs(x[idx]) <= 3.0  # Should be near bubble wall

def test_horizon_finder(alcubierre_setup):
    """Test event horizon finder."""
    components, (x, y, z) = alcubierre_setup
    finder = HorizonFinder()
    
    # Find horizons
    horizons = finder.find_horizons(components, x, y, z)
    
    # Check structure
    assert "outer" in horizons
    assert "inner" in horizons
    assert "ergosphere" in horizons
    
    # Each horizon should be a closed 2D surface
    for surface in horizons.values():
        if len(surface) > 0:  # Some metrics might not have all horizons
            assert surface.shape[1] == 3  # (x, y, z) coordinates
            # Check if surface is closed
            start = surface[0]
            end = surface[-1]
            assert np.allclose(start, end)
    
    # Test horizon properties
    properties = finder.analyze_horizons(components, horizons)
    
    # Check basic properties
    assert "area" in properties
    assert "surface_gravity" in properties
    assert "angular_velocity" in properties
    
    # Physical constraints (if horizons exist)
    if len(horizons["outer"]) > 0:
        assert properties["area"] >= 0
        assert properties["surface_gravity"] >= 0

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

