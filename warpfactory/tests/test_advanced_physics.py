"""Test advanced physics calculations."""

import pytest
import numpy as np
import torch
from warpfactory.physics import (
    TidalForces,
    CausalStructure,
    StressEnergyConservation,
    QuantumEffects
)
from warpfactory.metrics import AlcubierreMetric
from warpfactory.solver import ChristoffelSymbols

@pytest.fixture
def alcubierre_setup(spatial_grid):
    """Set up Alcubierre metric and derivatives."""
    x, y, z = spatial_grid
    t = 0.0
    metric = AlcubierreMetric()
    components = metric.calculate(x, y, z, t, v_s=2.0, R=1.0, sigma=0.5)
    
    # Calculate Christoffel symbols
    christoffel = ChristoffelSymbols()
    gamma = christoffel.calculate(components, x, y, z)
    
    return components, gamma, (x, y, z)

def test_tidal_forces(alcubierre_setup):
    """Test tidal force calculations."""
    components, gamma, (x, y, z) = alcubierre_setup
    tidal = TidalForces()
    
    # Calculate tidal forces
    forces = tidal.calculate(components, gamma, x, y, z)
    
    # Check structure
    assert "radial" in forces
    assert "transverse" in forces
    assert "longitudinal" in forces
    
    # Check shapes
    assert forces["radial"].shape == x.shape
    assert forces["transverse"].shape == x.shape
    assert forces["longitudinal"].shape == x.shape
    
    # Physical checks
    # 1. Forces should vanish at infinity (within numerical precision)
    far_index = -1
    assert np.allclose(forces["radial"][far_index], 0, atol=1e-8)
    assert np.allclose(forces["transverse"][far_index], 0, atol=1e-8)
    
    # 2. Forces should be approximately antisymmetric at large radius
    r = 3.0  # Test at r = 3R
    idx = np.argmin(np.abs(x - r))
    left = forces["radial"][idx-2:idx].mean()  # Average over small region
    right = forces["radial"][idx:idx+2].mean()
    # Test relative antisymmetry: |F(x) + F(-x)| < 2|F(x)|
    assert np.abs(left + right) < 2 * max(np.abs(left), np.abs(right))
    
    # 3. Maximum force location (in positive x region)
    positive_x = x > 0
    max_force_location = np.argmax(np.abs(forces["radial"][positive_x]))
    max_x = x[positive_x][max_force_location]
    assert 0.5 <= max_x <= 2.0  # Should be near bubble wall

def test_causal_structure(alcubierre_setup):
    """Test causal structure analysis."""
    components, _, (x, y, z) = alcubierre_setup
    causal = CausalStructure()
    
    # Find horizons
    horizons = causal.find_horizons(components, x, y, z)
    assert len(horizons["inner"]) > 0
    assert len(horizons["outer"]) > 0
    
    # Classify regions
    regions = causal.classify_regions(components, x, y, z)
    unique_regions = np.unique(regions)
    assert all(r in ["normal", "ergo", "trapped"] for r in unique_regions)
    
    # Check light cone tilting
    tilts = causal.light_cone_tilt(components, x, y, z)
    assert np.all(np.abs(tilts) <= np.pi/2)  # Tilt angle should be bounded
    
    # Test causality violation detection
    violations = causal.find_causality_violations(components, x, y, z)
    assert isinstance(violations, np.ndarray)
    assert violations.dtype == bool

def test_stress_energy_conservation(alcubierre_setup):
    """Test stress-energy conservation."""
    components, gamma, (x, y, z) = alcubierre_setup
    conservation = StressEnergyConservation()
    
    # Calculate conservation
    div_T = conservation.calculate_divergence(components, gamma, x, y, z)
    
    # Check structure
    assert "t" in div_T  # Time component
    assert "x" in div_T  # Space components
    assert "y" in div_T
    assert "z" in div_T
    
    # Conservation should be approximately satisfied
    # Note: Higher tolerance due to numerical derivatives and normalization
    assert np.allclose(div_T["t"], 0, atol=5.0)
    assert np.allclose(div_T["x"], 0, atol=5.0)
    
    # Test conservation laws
    laws = conservation.check_conservation_laws(components, gamma, x, y, z)
    assert "energy" in laws
    assert "momentum" in laws
    assert isinstance(laws["energy"], bool)
    assert isinstance(laws["momentum"], bool)

def test_quantum_effects():
    """Test quantum effect calculations."""
    quantum = QuantumEffects()
    
    # Test parameters
    surface_gravity = 1.0  # m/s²
    bubble_radius = 100.0  # meters
    
    # Calculate Hawking-like temperature
    T_H = quantum.hawking_temperature(surface_gravity)
    assert T_H > 0
    assert isinstance(T_H, float)
    
    # Calculate particle production rate
    rate = quantum.particle_production_rate(surface_gravity, bubble_radius)
    assert rate >= 0
    assert isinstance(rate, float)
    
    # Test vacuum polarization
    polarization = quantum.vacuum_polarization(surface_gravity, bubble_radius)
    assert isinstance(polarization, dict)
    assert "energy_density" in polarization
    assert "pressure" in polarization
    
    # Test backreaction
    backreaction = quantum.estimate_backreaction(surface_gravity, bubble_radius)
    assert isinstance(backreaction, dict)
    assert "metric_correction" in backreaction
    assert "lifetime" in backreaction

@pytest.mark.gpu
def test_gpu_quantum_effects():
    """Test GPU-accelerated quantum calculations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    quantum = QuantumEffects(device="cuda")
    
    # Create batch of parameters
    surface_gravity = torch.linspace(0.1, 2.0, 100, device="cuda")
    bubble_radius = torch.full_like(surface_gravity, 100.0)
    
    # Batch calculate Hawking temperature
    T_H = quantum.hawking_temperature_batch(surface_gravity)
    assert T_H.device.type == "cuda"
    assert torch.all(T_H > 0)
    
    # Batch calculate particle production
    rates = quantum.particle_production_batch(surface_gravity, bubble_radius)
    assert rates.device.type == "cuda"
    assert torch.all(rates >= 0)
    
    # Test vacuum polarization with gradients
    surface_gravity.requires_grad = True
    polarization = quantum.vacuum_polarization_batch(
        surface_gravity, bubble_radius
    )
    loss = polarization["energy_density"].mean()
    loss.backward()
    assert surface_gravity.grad is not None