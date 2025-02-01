"""Parameter input panel for metric configuration."""

from warpfactory.gui import *

class ParameterPanel(QWidget):
    """Panel for inputting metric parameters."""
    
    def __init__(self, parent=None):
        """Initialize the parameter panel."""
        super().__init__(parent)
        self.parameters = {}
        self.spinboxes = {}
        self.layout = QVBoxLayout(self)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        self.layout.addWidget(QLabel("Parameters:"))
    
    def set_parameters(self, params: dict):
        """Set up parameter inputs.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameter names and values
        """
        # Clear existing parameters
        for widget in self.spinboxes.values():
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self.spinboxes.clear()
        
        # Add new parameters
        self.parameters = params.copy()
        for name, value in params.items():
            # Create row layout
            row = QHBoxLayout()
            
            # Add label
            label = QLabel(f"{name}:")
            row.addWidget(label)
            
            # Add spinbox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 10.0)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(value)
            spinbox.valueChanged.connect(
                lambda v, n=name: self.on_value_changed(n, v)
            )
            self.spinboxes[name] = spinbox
            row.addWidget(spinbox)
            
            self.layout.addLayout(row)
    
    def get_value(self, param: str) -> float:
        """Get current value of a parameter.
        
        Parameters
        ----------
        param : str
            Parameter name
            
        Returns
        -------
        float
            Current parameter value
        """
        return self.spinboxes[param].value()
    
    def set_value(self, param: str, value: float):
        """Set value of a parameter.
        
        Parameters
        ----------
        param : str
            Parameter name
        value : float
            New parameter value
            
        Raises
        ------
        ValueError
            If value is invalid for the parameter
        """
        if value <= 0.0:
            raise ValueError(f"Parameter {param} must be positive")
        self.spinboxes[param].setValue(value)
    
    def on_value_changed(self, param: str, value: float):
        """Handle parameter value changes.
        
        Parameters
        ----------
        param : str
            Parameter name
        value : float
            New parameter value
        """
        try:
            self.parameters[param] = value
        except ValueError:
            # Restore previous value
            self.spinboxes[param].setValue(self.parameters[param])
            
    def get_all_parameters(self) -> dict:
        """Get all current parameter values.
        
        Returns
        -------
        dict
            Dictionary of parameter names and values
        """
        return {name: self.get_value(name) for name in self.parameters}
