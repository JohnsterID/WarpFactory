"""Test advanced physics calculations."""

import subprocess
import sys

import pytest
import numpy as np
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

def test_tidal_forces():
    """Tidal forces from geodesic deviation on a well-resolved bubble.

    A sharp wall (sigma=4) is required so the tanh profile actually
    reaches vacuum inside the grid; the wide sigma=0.5 bubble from the
    shared fixture extends past x=5 and its far field is not zero.
    """
    x = np.linspace(-5, 5, 400)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    components = AlcubierreMetric().calculate(x, y, z, 0.0,
                                              v_s=0.5, R=1.0, sigma=4.0)
    gamma = ChristoffelSymbols().calculate(components, x, y, z)

    tidal = TidalForces()
    forces = tidal.calculate(components, gamma, x, y, z)

    assert "radial" in forces
    assert "transverse" in forces
    assert "longitudinal" in forces
    for component in forces.values():
        assert component.shape == x.shape

    # Flat far field: forces vanish away from the bubble
    assert np.allclose(forces["radial"][:20], 0, atol=1e-6)
    assert np.allclose(forces["radial"][-20:], 0, atol=1e-6)
    assert np.allclose(forces["transverse"][:20], 0, atol=1e-6)

    # Forces concentrate at the bubble wall (|x| ~ R)
    peak_x = np.abs(x[np.argmax(np.abs(forces["radial"]))])
    assert 0.5 <= peak_x <= 2.0

    # Flat spacetime yields exactly zero tidal forces
    flat = {
        "g_tt": -np.ones_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x)
    }
    flat_forces = tidal.calculate(flat, {}, x, y, z)
    for component in flat_forces.values():
        assert np.allclose(component, 0.0, atol=1e-12)

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

def test_stress_energy_conservation():
    """Covariant divergence of the EFE stress-energy must vanish.

    The Bianchi identity makes this exact in the continuum; on a
    well-resolved grid the residual is small discretization error, so a
    tight tolerance is meaningful (the old test used atol=5.0, which
    could never fail).
    """
    x = np.linspace(-5, 5, 400)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    components = AlcubierreMetric().calculate(x, y, z, 0.0,
                                              v_s=0.5, R=1.0, sigma=4.0)
    gamma = ChristoffelSymbols().calculate(components, x, y, z)
    conservation = StressEnergyConservation()

    div_T = conservation.calculate_divergence(components, gamma, x, y, z)

    assert set(div_T) == {"t", "x", "y", "z"}
    interior = slice(4, -4)
    for component in div_T.values():
        assert np.allclose(component[interior], 0, atol=1e-3)

    laws = conservation.check_conservation_laws(components, gamma, x, y, z)
    assert laws == {"energy": True, "momentum": True}

def test_quantum_effects():
    """Test quantum effect calculations."""
    quantum = QuantumEffects()
    
    # Test parameters
    surface_gravity = 1.0  # m/s^2
    bubble_radius = 100.0  # meters
    
    # Hawking temperature: T = hbar kappa / (2 pi c k_B)
    T_H = quantum.hawking_temperature(surface_gravity)
    assert T_H > 0
    assert isinstance(T_H, float)
    expected_T = (1.0545718e-34 * surface_gravity /
                  (2 * np.pi * 299792458.0 * 1.380649e-23))
    assert np.isclose(T_H, expected_T, rtol=1e-10)
    # T scales linearly with surface gravity
    assert np.isclose(quantum.hawking_temperature(2 * surface_gravity),
                      2 * T_H, rtol=1e-10)
    
    # Calculate particle production rate
    rate = quantum.particle_production_rate(surface_gravity, bubble_radius)
    assert rate >= 0
    assert isinstance(rate, float)
    
    # Vacuum polarization: thermal radiation with p = rho/3
    polarization = quantum.vacuum_polarization(surface_gravity, bubble_radius)
    assert isinstance(polarization, dict)
    assert polarization["energy_density"] > 0
    assert np.isclose(polarization["pressure"],
                      polarization["energy_density"] / 3, rtol=1e-12)
    # Radiation constant: rho = pi^2 k_B^4 T^4 / (15 hbar^3 c^3)
    hbar, c, k_B = 1.0545718e-34, 299792458.0, 1.380649e-23
    expected_rho = np.pi**2 * k_B**4 * T_H**4 / (15 * hbar**3 * c**3)
    assert np.isclose(polarization["energy_density"], expected_rho, rtol=1e-10)
    
    # Test backreaction
    backreaction = quantum.estimate_backreaction(surface_gravity, bubble_radius)
    assert isinstance(backreaction, dict)
    assert "metric_correction" in backreaction
    assert "lifetime" in backreaction

def test_physics_imports_without_torch():
    """warpfactory.physics must work when torch is not installed.

    torch is an optional heavy backend; only the *_batch methods of
    QuantumEffects use it. Blocking torch in a subprocess simulates an
    environment without it and exercises the scalar code path.
    """
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from warpfactory.physics import QuantumEffects\n"
        "q = QuantumEffects()\n"
        "T = q.hawking_temperature(1.0)\n"
        "assert T > 0\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

@pytest.mark.gpu
def test_gpu_quantum_effects():
    """Test GPU-accelerated quantum calculations."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
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