"""Test GUI component interactions."""

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

from warpfactory.gui import MetricExplorer

@pytest.mark.gui
def test_metric_explorer_interactions(app, window, container, metric_params, spatial_grid):
    """Test interactions with the metric explorer window."""
    window = QMainWindow()
    explorer = MetricExplorer()
    container.layout().addWidget(explorer)
    window.setCentralWidget(explorer)
    
    # Test metric selection changes
    metric_combo = explorer.metric_selector
    assert metric_combo is not None
    
    # Test Alcubierre metric selection
    QTest.mouseClick(metric_combo, Qt.MouseButton.LeftButton)
    metric_combo.setCurrentText("Alcubierre")
    params = explorer.parameter_panel.get_all_parameters()
    assert "v_s" in params
    assert "R" in params
    assert "sigma" in params
    assert params["v_s"] == 2.0
    assert params["R"] == 1.0
    assert params["sigma"] == 0.5
    
    # Test Van Den Broeck metric selection
    QTest.mouseClick(metric_combo, Qt.MouseButton.LeftButton)
    metric_combo.setCurrentText("Van Den Broeck")
    params = explorer.parameter_panel.get_all_parameters()
    assert "B" in params  # Additional parameter for VDB metric
    assert params["B"] == 2.0
    
    # Test Warp Shell metric selection
    QTest.mouseClick(metric_combo, Qt.MouseButton.LeftButton)
    metric_combo.setCurrentText("Warp Shell")
    params = explorer.parameter_panel.get_all_parameters()
    assert "thickness" in params
    assert params["thickness"] == 0.2
    
    # Test Minkowski metric selection
    QTest.mouseClick(metric_combo, Qt.MouseButton.LeftButton)
    metric_combo.setCurrentText("Minkowski")
    params = explorer.parameter_panel.get_all_parameters()
    assert len(params) == 0  # Minkowski has no parameters

@pytest.mark.gui
def test_parameter_panel_validation(app, window, container):
    """Test parameter panel input validation."""
    window = QMainWindow()
    explorer = MetricExplorer()
    container.layout().addWidget(explorer)
    window.setCentralWidget(explorer)
    panel = explorer.parameter_panel
    
    # Set Alcubierre metric
    QTest.mouseClick(explorer.metric_selector, Qt.MouseButton.LeftButton)
    explorer.metric_selector.setCurrentText("Alcubierre")
    
    # Test invalid v_s (speed) values
    with pytest.raises(ValueError):
        panel.set_value("v_s", -1.0)  # Negative speed
    with pytest.raises(ValueError):
        panel.set_value("v_s", 0.0)  # Zero speed
        
    # Test invalid R (radius) values
    with pytest.raises(ValueError):
        panel.set_value("R", -1.0)  # Negative radius
    with pytest.raises(ValueError):
        panel.set_value("R", 0.0)  # Zero radius
        
    # Test invalid sigma (thickness) values
    with pytest.raises(ValueError):
        panel.set_value("sigma", -0.1)  # Negative thickness
    with pytest.raises(ValueError):
        panel.set_value("sigma", 0.0)  # Zero thickness

@pytest.mark.gui
def test_metric_visualization_update(app, window, container, spatial_grid):
    """Test metric visualization updates when parameters change."""
    window = QMainWindow()
    explorer = MetricExplorer()
    container.layout().addWidget(explorer)
    window.setCentralWidget(explorer)
    
    # Set Alcubierre metric
    QTest.mouseClick(explorer.metric_selector, Qt.MouseButton.LeftButton)
    explorer.metric_selector.setCurrentText("Alcubierre")
    
    # Get initial plot state
    initial_data = explorer.plotter.get_plot_data()
    assert isinstance(initial_data, np.ndarray)
    
    # Change v_s parameter
    explorer.parameter_panel.set_value("v_s", 3.0)
    new_data = explorer.plotter.get_plot_data()
    assert isinstance(new_data, np.ndarray)
    
    # Plot data should be different after parameter change
    if initial_data.size > 0 and new_data.size > 0:
        assert not np.array_equal(initial_data, new_data)
    
    # Change metric type
    QTest.mouseClick(explorer.metric_selector, Qt.MouseButton.LeftButton)
    explorer.metric_selector.setCurrentText("Minkowski")
    flat_data = explorer.plotter.get_plot_data()
    assert isinstance(flat_data, np.ndarray)
    
    # Minkowski metric should be different from Alcubierre
    if flat_data.size > 0 and new_data.size > 0:
        assert not np.array_equal(flat_data, new_data)