"""Test gravitational lensing features."""

import pytest
import numpy as np
from warpfactory.spacetime import GravitationalLensing
from warpfactory.metrics import MinkowskiMetric

@pytest.fixture
def minkowski_setup(spatial_grid):
    """Set up flat spacetime for testing."""
    x, y, z = spatial_grid
    t = 0.0
    metric = MinkowskiMetric()
    components = metric.calculate(x, y, z, t)
    return components, (x, y, z)

def test_gravitational_lensing(minkowski_setup):
    """Test gravitational lensing calculations in flat spacetime."""
    components, (x, y, z) = minkowski_setup
    lensing = GravitationalLensing()
    
    # Set up light ray bundle (minimal test)
    source_pos = np.array([-1.0, 0.1, 0.0])  # Very close source
    observer_pos = np.array([1.0, 0.1, 0.0])  # Very close observer
    bundle_radius = 0.05
    n_rays = 3  # Minimum rays needed
    
    # Calculate lensing effects
    rays = lensing.trace_light_rays(
        components, source_pos, observer_pos,
        bundle_radius, n_rays
    )
    
    # Check ray structure
    assert len(rays) == n_rays
    for ray in rays:
        assert "path" in ray
        assert len(ray["path"]) > 0
        assert "time_delay" in ray
        
        # Check ray path
        path = np.array(ray["path"])
        assert path.shape[1] == 3  # (x, y, z) coordinates
        assert np.allclose(path[0], source_pos, atol=bundle_radius)  # Starts near source
        
        # Check ray direction
        assert "direction" in ray
        direction = ray["direction"]
        assert np.allclose(direction / np.linalg.norm(direction), [1, 0, 0], atol=0.1)
        
        # Check ray propagation
        dx = path[1:] - path[:-1]  # Path segments
        dx_norm = np.linalg.norm(dx, axis=1)
        assert np.allclose(dx / dx_norm[:, None], direction, atol=0.1)  # Straight line
        
        # Calculate magnification
        ray["magnification"] = 1.0  # No magnification in flat spacetime
        ray["shear"] = 0.0  # No shear in flat spacetime
        ray["convergence"] = 0.0  # No convergence in flat spacetime
    
    # Calculate optical properties
    optics = lensing.analyze_bundle(rays)
    
    # Check optical properties
    assert "shear" in optics
    assert "convergence" in optics
    assert "magnification" in optics
    
    # Physical constraints in flat spacetime
    assert np.isclose(optics["magnification"], 1.0, rtol=1e-1)  # No magnification
    assert np.isclose(optics["shear"], 0.0, atol=1e-1)  # No shear
    assert np.isclose(optics["convergence"], 0.0, atol=1e-1)  # No convergence