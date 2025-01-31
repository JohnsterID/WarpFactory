"""Test PyTorch-accelerated computations."""

import pytest
import numpy as np
import torch

from warpfactory.torch import (
    TorchMetricSolver,
    TorchEnergyTensor,
    TorchChristoffel,
    TorchRicci,
)

# Skip tests if CUDA is not available
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available. Install PyTorch with CUDA support."
    )
]

@pytest.fixture
def device():
    """Get PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def spatial_grid_torch(device):
    """Create spatial grid on GPU."""
    x = torch.linspace(-5, 5, 100, device=device)
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    return x, y, z

def test_torch_metric_solver(device, spatial_grid_torch, metric_params):
    """Test PyTorch-accelerated metric calculations."""
    solver = TorchMetricSolver(device=device)
    x, y, z = spatial_grid_torch
    t = 0.0
    
    # Calculate metric components
    components = solver.calculate_alcubierre_metric(
        x, y, z, t,
        v_s=metric_params["v_s"],
        R=metric_params["R"],
        sigma=metric_params["sigma"]
    )
    
    # Check device
    assert all(comp.device == device for comp in components.values())
    
    # Check metric signature
    assert torch.all(components["g_tt"] <= 0)  # Time-time component is negative
    assert torch.all(components["g_xx"] > 0)   # Space-space component is positive
    
    # Test asymptotic flatness
    far_index = -1
    assert torch.isclose(components["g_tt"][far_index], torch.tensor(-1.0, device=device))
    assert torch.isclose(components["g_tx"][far_index], torch.tensor(0.0, device=device))
    assert torch.isclose(components["g_xx"][far_index], torch.tensor(1.0, device=device))

def test_torch_energy_tensor(device, spatial_grid_torch, metric_params):
    """Test PyTorch-accelerated energy tensor calculations."""
    solver = TorchMetricSolver(device=device)
    energy = TorchEnergyTensor(device=device)
    x, y, z = spatial_grid_torch
    t = 0.0
    
    # Calculate metric and energy tensor
    metric = solver.calculate_alcubierre_metric(x, y, z, t, **metric_params)
    T_munu = energy.calculate(metric, x, y, z)
    
    # Check device
    assert all(comp.device == device for comp in T_munu.values())
    
    # Check energy conditions
    rho = T_munu["T_tt"]  # Energy density
    p = T_munu["T_xx"]    # Pressure
    
    # Weak energy condition: ρ ≥ 0
    assert torch.all(rho >= 0)
    
    # Null energy condition: ρ + p ≥ 0
    assert torch.all(rho + p >= -1e-10)  # Allow small numerical errors

def test_torch_christoffel(device):
    """Test PyTorch-accelerated Christoffel symbol calculations."""
    christoffel = TorchChristoffel(device=device)
    
    # Test with Schwarzschild metric
    r = torch.tensor([4.0], device=device)  # Test at r = 4M
    theta = torch.tensor([np.pi/2], device=device)  # Test at equator
    
    # Metric components
    g_tt = -(1 - 2/r)
    g_rr = 1/(1 - 2/r)
    g_theta_theta = r**2
    g_phi_phi = r**2 * torch.sin(theta)**2
    
    metric = {
        "g_tt": g_tt,
        "g_rr": g_rr,
        "g_theta_theta": g_theta_theta,
        "g_phi_phi": g_phi_phi
    }
    
    coords = {
        "r": r,
        "theta": theta
    }
    
    gamma = christoffel.calculate(metric, coords)
    
    # Check device
    assert all(comp.device == device for comp in gamma.values())
    
    # Test specific non-zero components
    # Γ^r_tt = (r-2)/(2r^3)
    assert torch.isclose(
        gamma["r_tt"][0],
        torch.tensor(1/32, device=device)
    )

def test_torch_ricci(device):
    """Test PyTorch-accelerated Ricci tensor calculations."""
    ricci = TorchRicci(device=device)
    
    # Test with Minkowski metric
    x = torch.tensor([0.0], device=device)
    metric = {
        "g_tt": -torch.ones_like(x),
        "g_xx": torch.ones_like(x),
        "g_yy": torch.ones_like(x),
        "g_zz": torch.ones_like(x)
    }
    
    coords = {"x": x}
    
    R_munu = ricci.calculate(metric, coords)
    
    # Check device
    assert all(comp.device == device for comp in R_munu.values())
    
    # All components should be zero for flat spacetime
    for comp in R_munu.values():
        assert torch.allclose(comp, torch.zeros_like(comp))

def test_device_transfer():
    """Test device transfer of tensors."""
    solver = TorchMetricSolver(device="cuda")
    
    # Create data on CPU
    x = torch.linspace(-5, 5, 100)
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    
    # Calculate metric (should automatically move to GPU)
    metric = solver.calculate_alcubierre_metric(
        x, y, z, t=0.0,
        v_s=2.0, R=1.0, sigma=0.5
    )
    
    # Check device
    assert all(comp.device.type == "cuda" for comp in metric.values())
    
    # Move back to CPU
    metric_cpu = {k: v.cpu() for k, v in metric.items()}
    assert all(comp.device.type == "cpu" for comp in metric_cpu.values())