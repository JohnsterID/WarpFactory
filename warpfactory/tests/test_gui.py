"""Test GUI components."""

import pytest

# Try to import Qt, but don't fail if not available
try:
    from PyQt6.QtWidgets import QApplication, QComboBox
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False

# Skip all tests if Qt is not available
pytestmark = pytest.mark.skipif(
    not HAS_QT,
    reason="Qt6 is not available. Install system dependencies to run GUI tests."
)

from warpfactory.gui import (
    MetricExplorer,
    MetricPlotter,
    ParameterPanel,
    EnergyConditionViewer,
)

# Create QApplication instance for testing
@pytest.fixture(scope="session")
def app():
    """Create QApplication instance."""
    return QApplication([])

@pytest.mark.gui
def test_metric_explorer(app, metric_params, spatial_grid):
    """Test the main metric explorer window."""
    explorer = MetricExplorer()
    
    # Check initial state
    assert explorer.windowTitle() == "WarpFactory Metric Explorer"
    assert explorer.isVisible() == False
    
    # Test metric selection
    metric_combo = explorer.findChild(QComboBox, "metric_selector")
    assert "Alcubierre" in [metric_combo.itemText(i) for i in range(metric_combo.count())]
    assert "Lentz" in [metric_combo.itemText(i) for i in range(metric_combo.count())]
    
    # Test parameter panel
    param_panel = explorer.findChild(ParameterPanel, "parameter_panel")
    assert param_panel is not None
    assert "v_s" in param_panel.parameters
    assert "R" in param_panel.parameters
    assert "sigma" in param_panel.parameters

@pytest.mark.gui
def test_metric_plotter(app, test_metric_components):
    """Test the metric visualization widget."""
    components, x = test_metric_components
    plotter = MetricPlotter()
    
    # Test component selection
    plotter.set_metric(components)
    assert plotter.current_component == "g_tt"
    
    # Test plot update
    plotter.plot_component("g_tx")
    assert plotter.current_component == "g_tx"
    
    # Test colormap options
    assert plotter.colormap == "redblue"
    plotter.set_colormap("warp")
    assert plotter.colormap == "warp"

@pytest.mark.gui
def test_parameter_panel(app, metric_params):
    """Test the parameter input panel."""
    panel = ParameterPanel()
    
    # Test parameter setup
    panel.set_parameters(metric_params)
    for param, value in metric_params.items():
        assert panel.get_value(param) == value
    
    # Test value changes
    panel.set_value("v_s", 3.0)
    assert panel.get_value("v_s") == 3.0
    
    # Test validation
    with pytest.raises(ValueError):
        panel.set_value("v_s", -1.0)  # Speed should be positive
    with pytest.raises(ValueError):
        panel.set_value("R", 0.0)  # Radius should be positive

@pytest.mark.gui
def test_energy_condition_viewer(app, test_energy_tensor):
    """Test the energy condition visualization."""
    tensor, x = test_energy_tensor
    viewer = EnergyConditionViewer()
    
    # Test energy condition calculation
    viewer.set_tensor(tensor)
    conditions = viewer.check_conditions()
    assert "weak" in conditions
    assert "null" in conditions
    assert "strong" in conditions
    assert "dominant" in conditions
    
    # Test visualization modes
    assert viewer.current_mode == "density"
    viewer.set_mode("pressure")
    assert viewer.current_mode == "pressure"
    
    # Test region highlighting
    viewer.highlight_violations(True)
    assert viewer.violations_visible == True