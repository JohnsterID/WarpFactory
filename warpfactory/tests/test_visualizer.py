import pytest
import numpy as np
import matplotlib.pyplot as plt
from warpfactory.visualizer import (
    TensorPlotter,
    ThreePlusOnePlotter,
    SliceData,
    ColorMaps,
)

def test_tensor_plotter():
    """Test tensor component visualization."""
    plotter = TensorPlotter()
    
    # Create test metric tensor
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Simple warp bubble metric
    r = np.sqrt(X**2 + Y**2)
    f = np.exp(-r**2)  # Shape function
    v = 2.0  # Velocity
    
    metric = {
        "g_tt": -(1 - (v*f)**2),
        "g_tx": -v*f,
        "g_xx": np.ones_like(r)
    }
    
    # Test component plot
    fig = plotter.plot_component(metric, "g_tt", x, y)
    assert isinstance(fig, plt.Figure)
    
    # Test tensor heatmap
    fig = plotter.plot_heatmap(metric, x, y)
    assert isinstance(fig, plt.Figure)
    
    # Test eigenvalue plot
    fig = plotter.plot_eigenvalues(metric, x, y)
    assert isinstance(fig, plt.Figure)
    
    plt.close('all')

def test_three_plus_one_plotter():
    """Test 3+1 decomposition visualization."""
    plotter = ThreePlusOnePlotter()
    
    # Create test 3+1 data
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2 + Y**2)
    f = np.exp(-r**2)
    v = 2.0
    
    # 3+1 decomposition
    # Ensure lapse function is well-defined
    vsq = (v*f)**2
    vsq[vsq >= 1] = 0.99  # Cap velocity to avoid invalid sqrt
    
    decomp = {
        "alpha": 1/np.sqrt(1 - vsq),  # Lapse
        "beta": {"x": v*f, "y": np.zeros_like(r)},  # Shift
        "gamma": {  # Spatial metric
            "xx": np.ones_like(r),
            "xy": np.zeros_like(r),
            "yy": np.ones_like(r)
        }
    }
    
    # Test lapse function plot
    fig = plotter.plot_lapse(decomp, x, y)
    assert isinstance(fig, plt.Figure)
    
    # Test shift vector plot
    fig = plotter.plot_shift(decomp, x, y)
    assert isinstance(fig, plt.Figure)
    
    # Test spatial metric plot
    fig = plotter.plot_spatial_metric(decomp, x, y)
    assert isinstance(fig, plt.Figure)
    
    plt.close('all')

def test_slice_data():
    """Test slice data extraction and visualization."""
    slicer = SliceData()
    
    # Create 3D test data
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Simple 3D scalar field
    data = np.exp(-(X**2 + Y**2 + Z**2))
    
    # Test xy-slice
    slice_xy = slicer.get_slice(data, x, y, z, plane='xy', coord=0.0)
    assert slice_xy.shape == (20, 20)
    
    # Test xz-slice
    slice_xz = slicer.get_slice(data, x, y, z, plane='xz', coord=0.0)
    assert slice_xz.shape == (20, 20)
    
    # Test yz-slice
    slice_yz = slicer.get_slice(data, x, y, z, plane='yz', coord=0.0)
    assert slice_yz.shape == (20, 20)
    
    # Test invalid plane
    with pytest.raises(ValueError):
        slicer.get_slice(data, x, y, z, plane='invalid', coord=0.0)
    
    # Test visualization with different options
    # Basic plot
    fig = slicer.plot_slice(slice_xy, x, y, title='XY Slice')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Plot with custom title and axes labels
    fig = slicer.plot_slice(slice_xz, x, z, title='XZ Slice',
                           xlabel='X', ylabel='Z')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Plot with custom colormap
    fig = slicer.plot_slice(slice_yz, y, z, title='YZ Slice',
                           cmap='viridis')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    plt.close('all')

def test_colormaps():
    """Test custom colormaps."""
    cmaps = ColorMaps()
    
    # Test redblue diverging colormap
    cmap = cmaps.redblue()
    assert cmap.N == 256  # Default number of colors
    
    # Test warp colormap
    cmap = cmaps.warp()
    assert cmap.N == 256
    
    # Test data visualization with custom colormap
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    data = np.sin(X) * np.cos(Y)
    
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, data, cmap=cmap)
    assert isinstance(fig, plt.Figure)
    
    plt.close('all')