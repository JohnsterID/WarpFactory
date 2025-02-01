"""Test GUI components."""

import pytest
import numpy as np
from warpfactory.gui import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QMainWindow, QDoubleSpinBox,
    MetricExplorer, MetricPlotter, ParameterPanel,
    EnergyConditionViewer
)
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from warpfactory.gui import (
    MetricExplorer,
    MetricPlotter,
    ParameterPanel,
    EnergyConditionViewer,
)

@pytest.mark.gui
def test_metric_explorer(app, window, container, metric_params, spatial_grid):
    """Test the main metric explorer window."""
    explorer = MetricExplorer(container)
    
    # Check initial state
    assert not explorer.isVisible()
    
    # Test metric selection
    metric_combo = explorer.metric_selector
    assert metric_combo is not None
    assert "Alcubierre" in [metric_combo.itemText(i) for i in range(metric_combo.count())]
    assert "Lentz" in [metric_combo.itemText(i) for i in range(metric_combo.count())]
    
    # Test parameter panel
    param_panel = explorer.parameter_panel
    assert param_panel is not None
    assert param_panel.parameters == {}  # Initial state
    
    # Test metric change
    QTest.mouseClick(metric_combo, Qt.MouseButton.LeftButton)
    metric_combo.setCurrentText("Alcubierre")
    assert "v_s" in param_panel.parameters
    assert "R" in param_panel.parameters
    assert "sigma" in param_panel.parameters

@pytest.mark.gui
def test_metric_plotter(app, window, container, test_metric_components):
    """Test the metric visualization widget."""
    components, x = test_metric_components
    plotter = MetricPlotter()
    container.layout().addWidget(plotter)
    
    # Test initial state
    assert plotter.current_component == "g_tt"
    assert plotter.colormap == "redblue"
    
    # Test component selection
    plotter.components = components
    plotter.plot_component("g_tx")
    assert plotter.current_component == "g_tx"
    
    # Test colormap options
    plotter.set_colormap("warp")
    assert plotter.colormap == "warp"
    
    # Test plot data
    data = plotter.get_plot_data()
    assert isinstance(data, np.ndarray)
    assert data.shape == components["g_tx"].shape

@pytest.mark.gui
def test_parameter_panel(app, window, container, metric_params):
    """Test the parameter input panel."""
    panel = ParameterPanel()
    container.layout().addWidget(panel)
    
    # Test initial state
    assert panel.parameters == {}
    
    # Test parameter setup
    panel.set_parameters(metric_params)
    params = panel.get_all_parameters()
    for param, value in metric_params.items():
        assert params[param] == value
    
    # Test value changes
    panel.set_value("v_s", 3.0)
    assert panel.get_value("v_s") == 3.0
    
    # Test validation
    with pytest.raises(ValueError):
        panel.set_value("v_s", -1.0)  # Speed should be positive
    with pytest.raises(ValueError):
        panel.set_value("R", 0.0)  # Radius should be positive

@pytest.mark.gui
def test_energy_condition_viewer(app, window, container, test_energy_tensor):
    """Test the energy condition visualization."""
    tensor, x = test_energy_tensor
    viewer = EnergyConditionViewer()
    container.layout().addWidget(viewer)
    
    # Test initial state
    assert viewer.current_mode == "density"
    assert not viewer.violations_visible
    
    # Test tensor update
    viewer.tensor = tensor
    viewer.update_plot()
    
    # Test visualization modes
    viewer.set_mode("pressure")
    assert viewer.current_mode == "pressure"
    
    # Test region highlighting
    viewer.highlight_violations(True)
    assert viewer.violations_visible