"""Main window for metric exploration.

Maintenance-only: the Qt explorer is kept for compatibility, but new
interactive features land in warpfactory.interactive.JupyterExplorer.
Both front ends share warpfactory.interactive.ExplorerModel for the
catalog and the compute pipeline.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from ..interactive.model import METRIC_CATALOG, ExplorerModel
from .energy import EnergyConditionViewer
from .parameters import ParameterPanel
from .plotter import MetricPlotter

__all__ = ["METRIC_CATALOG", "MetricExplorer"]


class MetricExplorer(QMainWindow):
    """Main window for exploring warp drive metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WarpFactory Metric Explorer")
        self.model = ExplorerModel(x=np.linspace(-8.0, 8.0, 200))
        self.grid_x = self.model.x
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
        self.parameter_panel.set_parameters(self.model.defaults(metric_name))
        self.recompute()

    def on_parameter_changed(self, param: str, value: float):
        self.recompute()

    def recompute(self):
        """Evaluate the selected metric and update both visualizations."""
        result = self.model.evaluate(
            self.metric_selector.currentText(),
            self.parameter_panel.get_all_parameters(),
        )
        self.plotter.set_metric(result.metric, result.x)
        self.energy_viewer.set_tensor(result.stress_energy, result.x)
