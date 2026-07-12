"""Parameter input panel for metric configuration."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class ParameterPanel(QWidget):
    """Panel for inputting metric parameters."""

    parameter_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters = {}
        self.spinboxes = {}
        self._rows = []
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(QLabel("Parameters:"))

    def set_parameters(self, params: dict):
        """Replace the parameter inputs.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
        for row in self._rows:
            while row.count():
                item = row.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._layout.removeItem(row)
        self._rows.clear()
        self.spinboxes.clear()

        self.parameters = params.copy()
        for name, value in params.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))

            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 10.0)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(value)
            spinbox.valueChanged.connect(lambda v, n=name: self._on_value_changed(n, v))
            self.spinboxes[name] = spinbox
            row.addWidget(spinbox)

            self._layout.addLayout(row)
            self._rows.append(row)

    def get_value(self, param: str) -> float:
        """Current value of a parameter."""
        return self.spinboxes[param].value()

    def get_all_parameters(self) -> dict:
        """All current parameter values keyed by name."""
        return {name: self.get_value(name) for name in self.parameters}

    def set_value(self, param: str, value: float):
        """Set value of a parameter.

        Raises
        ------
        ValueError
            If value is not positive (all metric parameters here are
            physically positive quantities)
        """
        if value <= 0.0:
            raise ValueError(f"Parameter {param} must be positive")
        self.spinboxes[param].setValue(value)

    def _on_value_changed(self, param: str, value: float):
        self.parameters[param] = value
        self.parameter_changed.emit(param, value)
