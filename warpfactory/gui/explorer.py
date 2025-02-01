"""Main window for metric exploration."""

from warpfactory.gui import *

class MetricExplorer(QWidget):
    """Main window for exploring warp drive metrics."""
    
    def __init__(self, parent=None):
        """Initialize the main window."""
        super().__init__(parent)
        self.metric_selector = QComboBox(self)
        self.metric_selector.setObjectName("metric_selector")
        self.parameter_panel = ParameterPanel(self)
        self.parameter_panel.setObjectName("parameter_panel")
        self.plotter = MetricPlotter(self)
        self.plotter.setObjectName("metric_plotter")
        self.energy_viewer = EnergyConditionViewer(self)
        self.energy_viewer.setObjectName("energy_viewer")
        self.setup_ui()
        
    def findChild(self, cls, name):
        """Find a child widget by class and name."""
        for child in self.findChildren(cls):
            if child.objectName() == name:
                return child
        return None
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Metric selector
        metric_label = QLabel("Select Metric:")
        self.metric_selector = QComboBox()
        self.metric_selector.setObjectName("metric_selector")
        self.metric_selector.addItems([
            "Alcubierre",
            "Lentz",
            "Van Den Broeck",
            "Warp Shell",
            "Minkowski"
        ])
        left_layout.addWidget(metric_label)
        left_layout.addWidget(self.metric_selector)
        
        # Parameter panel
        self.parameter_panel = ParameterPanel()
        self.parameter_panel.setObjectName("parameter_panel")
        left_layout.addWidget(self.parameter_panel)
        
        # Add left panel to main layout
        layout.addWidget(left_panel)
        
        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Metric plotter
        self.plotter = MetricPlotter()
        self.plotter.setObjectName("metric_plotter")
        right_layout.addWidget(self.plotter)
        
        # Energy condition viewer
        self.energy_viewer = EnergyConditionViewer()
        self.energy_viewer.setObjectName("energy_viewer")
        right_layout.addWidget(self.energy_viewer)
        
        # Add right panel to main layout
        layout.addWidget(right_panel)
        
        # Set up connections
        self.metric_selector.currentTextChanged.connect(self.on_metric_changed)
        
        # Initialize with default metric
        self.on_metric_changed("Alcubierre")
        
    def on_metric_changed(self, metric_name: str):
        """Handle metric selection changes."""
        # Update parameter panel based on metric type
        if metric_name == "Alcubierre":
            params = {"v_s": 2.0, "R": 1.0, "sigma": 0.5}
        elif metric_name == "Lentz":
            params = {"v_s": 2.0, "R": 1.0, "sigma": 0.5}
        elif metric_name == "Van Den Broeck":
            params = {"v_s": 2.0, "R": 1.0, "B": 2.0, "sigma": 0.5}
        elif metric_name == "Warp Shell":
            params = {"v_s": 2.0, "R": 1.0, "thickness": 0.2, "sigma": 0.5}
        else:  # Minkowski
            params = {}
        
        self.parameter_panel.set_parameters(params)