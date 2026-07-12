"""Test GUI components."""

import numpy as np
import pytest

pytest.importorskip(
    "PyQt6.QtWidgets", reason="Qt6 is not available. Install PyQt6 to run GUI tests."
)

from warpfactory.gui import (
    EnergyConditionViewer,
    MetricExplorer,
    MetricPlotter,
    ParameterPanel,
)


@pytest.mark.gui
def test_metric_explorer(app, metric_params, spatial_grid):
    """Test the main metric explorer window."""
    explorer = MetricExplorer()

    assert explorer.windowTitle() == "WarpFactory Metric Explorer"
    assert explorer.isVisible() == False

    metric_combo = explorer.metric_selector
    item_texts = [metric_combo.itemText(i) for i in range(metric_combo.count())]
    assert "Alcubierre" in item_texts
    assert "Lentz" in item_texts

    # Default metric is Alcubierre, so its parameters must be populated
    param_panel = explorer.parameter_panel
    assert param_panel is not None
    assert "v_s" in param_panel.parameters
    assert "R" in param_panel.parameters
    assert "sigma" in param_panel.parameters

    # Selecting a metric rebuilds the parameter panel
    metric_combo.setCurrentText("Van Den Broeck")
    assert "B" in param_panel.parameters
    metric_combo.setCurrentText("Minkowski")
    assert param_panel.parameters == {}


@pytest.mark.gui
def test_metric_explorer_pipeline(app):
    """Parameter changes must propagate to the plot data."""
    explorer = MetricExplorer()
    explorer.metric_selector.setCurrentText("Alcubierre")

    initial = explorer.plotter.get_plot_data().copy()
    assert initial.size > 0

    explorer.parameter_panel.set_value("v_s", 3.0)
    updated = explorer.plotter.get_plot_data()
    assert not np.array_equal(initial, updated)

    # Energy viewer must hold the derived stress-energy tensor
    assert explorer.energy_viewer.tensor is not None
    assert "T_tt" in explorer.energy_viewer.tensor


@pytest.mark.gui
def test_metric_plotter(app, test_metric_components):
    """Test the metric visualization widget."""
    components, x = test_metric_components
    plotter = MetricPlotter()

    plotter.set_metric(components, x)
    assert plotter.current_component == "g_tt"

    plotter.plot_component("g_tx")
    assert plotter.current_component == "g_tx"

    assert plotter.colormap == "redblue"
    plotter.set_colormap("warp")
    assert plotter.colormap == "warp"

    plot_data = plotter.get_plot_data()
    assert isinstance(plot_data, np.ndarray)
    assert plot_data.shape == components["g_tx"].shape


@pytest.mark.gui
def test_parameter_panel(app, metric_params):
    """Test the parameter input panel."""
    panel = ParameterPanel()
    assert panel.parameters == {}

    panel.set_parameters(metric_params)
    for param, value in metric_params.items():
        assert panel.get_value(param) == value
    assert panel.get_all_parameters() == metric_params

    panel.set_value("v_s", 3.0)
    assert panel.get_value("v_s") == 3.0

    with pytest.raises(ValueError):
        panel.set_value("v_s", -1.0)
    with pytest.raises(ValueError):
        panel.set_value("R", 0.0)


@pytest.mark.gui
def test_energy_condition_viewer(app, test_energy_tensor):
    """Test the energy condition visualization."""
    tensor, x = test_energy_tensor
    viewer = EnergyConditionViewer()

    viewer.set_tensor(tensor, x)
    conditions = viewer.check_conditions()
    assert set(conditions) == {"weak", "null", "strong", "dominant"}

    assert viewer.current_mode == "density"
    viewer.set_mode("pressure")
    assert viewer.current_mode == "pressure"

    viewer.highlight_violations(True)
    assert viewer.violations_visible == True
