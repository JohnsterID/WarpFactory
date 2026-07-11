"""Main window for metric exploration."""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel
)

from ..metrics import (
    AlcubierreMetric,
    LentzMetric,
    VanDenBroeckMetric,
    WarpShellMetric,
    MinkowskiMetric,
)
from ..solver import EnergyTensor
from .plotter import MetricPlotter
from .parameters import ParameterPanel
from .energy import EnergyConditionViewer

METRIC_CATALOG = {
    "Alcubierre": (AlcubierreMetric, {"v_s": 2.0, "R": 1.0, "sigma": 0.5}),
    "Lentz": (LentzMetric, {"v_s": 2.0, "R": 1.0, "sigma": 0.5}),
    "Van Den Broeck": (VanDenBroeckMetric,
                       {"v_s": 2.0, "R": 1.0, "B": 2.0, "sigma": 0.5}),
    "Warp Shell": (WarpShellMetric,
                   {"v_s": 2.0, "R": 1.0, "thickness": 0.2, "sigma": 0.5}),
    "Minkowski": (MinkowskiMetric, {}),
}


class MetricExplorer(QMainWindow):
    """Main window for exploring warp drive metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WarpFactory Metric Explorer")
        self.grid_x = np.linspace(-8.0, 8.0, 200)
        self.energy_solver = EnergyTensor()
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("Select Metric:"))
        self.metric_selector = QComboBox()
        self.metric_selector.setObjectName("metric_selector")
        self.metric_selector.addItems(list(METRIC_CATALOG))
        left_layout.addWidget(self.metric_selector)

        self.parameter_panel = ParameterPanel()
        self.parameter_panel.setObjectName("parameter_panel")
        left_layout.addWidget(self.parameter_panel)

        layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plotter = MetricPlotter()
        self.plotter.setObjectName("metric_plotter")
        right_layout.addWidget(self.plotter)

        self.energy_viewer = EnergyConditionViewer()
        self.energy_viewer.setObjectName("energy_viewer")
        right_layout.addWidget(self.energy_viewer)

        layout.addWidget(right_panel)

        self.metric_selector.currentTextChanged.connect(self.on_metric_changed)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)

        self.on_metric_changed(self.metric_selector.currentText())

    def on_metric_changed(self, metric_name: str):
        """Rebuild the parameter panel and recompute for the new metric."""
        _, defaults = METRIC_CATALOG[metric_name]
        self.parameter_panel.set_parameters(defaults)
        self.recompute()

    def on_parameter_changed(self, param: str, value: float):
        self.recompute()

    def recompute(self):
        """Evaluate the selected metric and update both visualizations."""
        metric_name = self.metric_selector.currentText()
        metric_cls, _ = METRIC_CATALOG[metric_name]
        params = self.parameter_panel.get_all_parameters()

        x = self.grid_x
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        components = metric_cls().calculate(x, y, z, 0.0, **params)
        self.plotter.set_metric(components, x)

        stress_energy = self.energy_solver.calculate_from_metric(components, x)
        self.energy_viewer.set_tensor(stress_energy, x)
