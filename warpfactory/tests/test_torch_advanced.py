"""Test advanced PyTorch features."""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List

from warpfactory.torch import (
    TorchMetricBatch,
    TorchEnergyAnalyzer,
    TorchFieldVisualizer,
    TorchBenchmark,
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
def batch_params():
    """Create a batch of metric parameters."""
    return {
        "v_s": torch.tensor([1.0, 2.0, 3.0]),  # Different velocities
        "R": torch.tensor([0.5, 1.0, 1.5]),    # Different radii
        "sigma": torch.tensor([0.3, 0.5, 0.7])  # Different thicknesses
    }

def test_metric_batch_processing(device, spatial_grid_torch, batch_params):
    """Test batch processing of multiple metrics."""
    batch = TorchMetricBatch(device=device)
    x, y, z = spatial_grid_torch
    t = 0.0
    
    # Calculate metrics for all parameter combinations
    metrics = batch.calculate_metrics(x, y, z, t, batch_params)
    
    # Check batch size
    assert len(metrics) == 3  # Should match number of parameter sets
    
    # Check each metric
    for i, metric in enumerate(metrics):
        # Check device
        assert all(comp.device == device for comp in metric.values())
        
        # Check metric signature
        assert torch.all(metric["g_tt"] <= 0)
        assert torch.all(metric["g_xx"] > 0)
        
        # Test asymptotic flatness
        far_index = -1
        assert torch.isclose(metric["g_tt"][far_index], torch.tensor(-1.0, device=device))
        assert torch.isclose(metric["g_tx"][far_index], torch.tensor(0.0, device=device))
        assert torch.isclose(metric["g_xx"][far_index], torch.tensor(1.0, device=device))

def test_energy_analysis(device, spatial_grid_torch, batch_params):
    """Test GPU-accelerated energy condition analysis."""
    analyzer = TorchEnergyAnalyzer(device=device)
    batch = TorchMetricBatch(device=device)
    x, y, z = spatial_grid_torch
    t = 0.0
    
    # Calculate metrics
    metrics = batch.calculate_metrics(x, y, z, t, batch_params)
    
    # Analyze energy conditions for each metric
    results = analyzer.analyze_batch(metrics)
    
    # Check results structure
    assert len(results) == len(metrics)
    for result in results:
        assert "weak" in result
        assert "null" in result
        assert "strong" in result
        assert "dominant" in result
        
        # Check data types
        assert isinstance(result["weak"], bool)
        assert isinstance(result["null"], bool)
        assert isinstance(result["strong"], bool)
        assert isinstance(result["dominant"], bool)
    
    # Test violation regions
    violations = analyzer.find_violation_regions(metrics[0])
    assert isinstance(violations, torch.Tensor)
    assert violations.device == device
    assert violations.dtype == torch.bool

def test_field_visualization(device, spatial_grid_torch):
    """Test tensor field visualization with PyTorch."""
    visualizer = TorchFieldVisualizer(device=device)
    x, y, z = spatial_grid_torch
    
    # Create test field data
    field = torch.exp(-(x**2 + y**2 + z**2)/2).to(device)
    vectors = torch.stack([
        -x * field,
        -y * field,
        -z * field
    ], dim=0)
    
    # Test scalar field plot
    scalar_fig = visualizer.plot_scalar_field(field, x, y)
    assert hasattr(scalar_fig, "savefig")
    
    # Test vector field plot
    vector_fig = visualizer.plot_vector_field(vectors, x, y)
    assert hasattr(vector_fig, "savefig")
    
    # Test streamlines
    stream_fig = visualizer.plot_streamlines(vectors, x, y)
    assert hasattr(stream_fig, "savefig")
    
    # Test field animation
    anim = visualizer.animate_field_evolution(field, vectors, x, y, frames=10)
    assert hasattr(anim, "save")

def test_performance_benchmarks(device, spatial_grid_torch, batch_params):
    """Test performance benchmarking."""
    benchmark = TorchBenchmark(device=device)
    x, y, z = spatial_grid_torch
    t = 0.0
    
    # Test single metric calculation
    single_time = benchmark.measure_single_metric(x, y, z, t)
    assert isinstance(single_time, float)
    assert single_time > 0
    
    # Test batch processing
    batch_time = benchmark.measure_batch_metrics(x, y, z, t, batch_params)
    assert isinstance(batch_time, float)
    assert batch_time > 0
    
    # Test memory usage
    memory_stats = benchmark.measure_memory_usage(x, y, z, t, batch_params)
    assert "allocated" in memory_stats
    assert "cached" in memory_stats
    
    # Compare CPU vs GPU
    speedup = benchmark.compare_cpu_gpu(x, y, z, t)
    assert isinstance(speedup, float)
    assert speedup > 0  # GPU should be faster
    
    # Profile different components
    profile = benchmark.profile_components(x, y, z, t)
    assert "metric_calculation" in profile
    assert "energy_tensor" in profile
    assert "christoffel_symbols" in profile