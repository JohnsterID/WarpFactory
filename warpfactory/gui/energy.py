"""Energy condition visualization widget."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox,
    QLabel, QCheckBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from ..analyzer import EnergyConditions

class EnergyConditionViewer(QWidget):
    """Widget for visualizing energy conditions."""
    
    def __init__(self):
        """Initialize the viewer widget."""
        super().__init__()
        self.current_mode = "density"
        self.violations_visible = False
        self.energy_conditions = EnergyConditions()
        self.tensor = None
        self.mode_selector = QComboBox(self)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.highlight_check = QCheckBox("Highlight Violations", self)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Mode selector
        mode_label = QLabel("Display Mode:")
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "density",
            "pressure",
            "violations"
        ])
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_selector)
        
        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Violation highlighting
        self.highlight_check = QCheckBox("Highlight Violations")
        layout.addWidget(self.highlight_check)
        
        # Connect signals
        self.mode_selector.currentTextChanged.connect(self.set_mode)
        self.highlight_check.toggled.connect(self.highlight_violations)
    
    def set_tensor(self, tensor: dict):
        """Set the energy-momentum tensor to analyze.
        
        Parameters
        ----------
        tensor : dict
            Dictionary of tensor components
        """
        self.tensor = tensor
        self.update_plot()
    
    def check_conditions(self) -> dict:
        """Check all energy conditions.
        
        Returns
        -------
        dict
            Dictionary of condition results
        """
        return {
            "weak": self.energy_conditions.check_weak(self.tensor),
            "null": self.energy_conditions.check_null(self.tensor),
            "strong": self.energy_conditions.check_strong(self.tensor),
            "dominant": self.energy_conditions.check_dominant(self.tensor)
        }
    
    def set_mode(self, mode: str):
        """Set the visualization mode.
        
        Parameters
        ----------
        mode : str
            Display mode ('density', 'pressure', or 'violations')
        """
        self.current_mode = mode
        self.update_plot()
    
    def highlight_violations(self, show: bool):
        """Toggle violation highlighting.
        
        Parameters
        ----------
        show : bool
            Whether to show violations
        """
        self.violations_visible = show
        self.update_plot()
    
    def update_plot(self):
        """Update the visualization."""
        if not hasattr(self, 'tensor'):
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.current_mode == "density":
            data = self.tensor["T_tt"]
            title = "Energy Density"
        elif self.current_mode == "pressure":
            data = self.tensor["T_xx"]  # Using x-component pressure
            title = "Pressure"
        else:  # violations
            conditions = self.check_conditions()
            data = np.zeros_like(self.tensor["T_tt"])
            for condition, result in conditions.items():
                if not result:
                    data += 1
            title = "Energy Condition Violations"
        
        im = ax.imshow(data)
        self.figure.colorbar(im)
        ax.set_title(title)
        
        if self.violations_visible:
            conditions = self.check_conditions()
            for condition, result in conditions.items():
                if not result:
                    ax.text(0.02, 0.98, f"{condition} violated",
                           transform=ax.transAxes,
                           verticalalignment='top')
        
        self.canvas.draw()