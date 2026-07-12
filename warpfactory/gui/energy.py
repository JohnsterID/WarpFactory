"""Energy condition visualization widget."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QCheckBox, QComboBox, QLabel, QVBoxLayout, QWidget

from ..analyzer import EnergyConditions


class EnergyConditionViewer(QWidget):
    """Widget for visualizing energy conditions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = "density"
        self.violations_visible = False
        self.energy_conditions = EnergyConditions()
        self.tensor = None
        self.coordinates = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Display Mode:"))
        self.mode_selector = QComboBox(self)
        self.mode_selector.setObjectName("mode_selector")
        self.mode_selector.addItems(["density", "pressure", "violations"])
        layout.addWidget(self.mode_selector)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        self.highlight_check = QCheckBox("Highlight Violations", self)
        layout.addWidget(self.highlight_check)

        self.mode_selector.currentTextChanged.connect(self.set_mode)
        self.highlight_check.toggled.connect(self.highlight_violations)

    def set_tensor(self, tensor: dict, x: np.ndarray = None):
        """Set the stress-energy tensor to analyze.

        Parameters
        ----------
        tensor : dict
            Dictionary of tensor components
        x : np.ndarray, optional
            Coordinate array for the horizontal axis
        """
        self.tensor = tensor
        self.coordinates = x
        self.update_plot()

    def check_conditions(self) -> dict:
        """Check all energy conditions of the current tensor."""
        return {
            "weak": self.energy_conditions.check_weak(self.tensor),
            "null": self.energy_conditions.check_null(self.tensor),
            "strong": self.energy_conditions.check_strong(self.tensor),
            "dominant": self.energy_conditions.check_dominant(self.tensor),
        }

    def set_mode(self, mode: str):
        """Set the visualization mode ('density', 'pressure', 'violations')."""
        self.current_mode = mode
        self.update_plot()

    def highlight_violations(self, show: bool):
        """Toggle violation highlighting."""
        self.violations_visible = show
        self.update_plot()

    def update_plot(self):
        """Redraw the visualization for the current mode."""
        if self.tensor is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.current_mode == "density":
            values = np.asarray(self.tensor["T_tt"])
            title = "Energy Density"
        elif self.current_mode == "pressure":
            values = np.asarray(self.tensor["T_xx"])
            title = "Pressure (x)"
        else:
            conditions = self.check_conditions()
            violation_count = sum(
                1 for satisfied in conditions.values() if not satisfied
            )
            values = np.full_like(
                np.asarray(self.tensor["T_tt"]), float(violation_count)
            )
            title = "Energy Condition Violations"

        if values.ndim >= 2:
            image = ax.imshow(values)
            self.figure.colorbar(image)
        else:
            x = (
                self.coordinates
                if self.coordinates is not None
                else np.arange(len(values))
            )
            ax.plot(x, values)
        ax.set_title(title)

        if self.violations_visible:
            conditions = self.check_conditions()
            offset = 0.98
            for condition, satisfied in conditions.items():
                if not satisfied:
                    ax.text(
                        0.02,
                        offset,
                        f"{condition} violated",
                        transform=ax.transAxes,
                        verticalalignment="top",
                    )
                    offset -= 0.06

        self.canvas.draw_idle()
