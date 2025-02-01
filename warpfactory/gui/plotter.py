"""Interactive metric visualization widget."""

from warpfactory.gui import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

class MetricPlotter(QWidget):
    """Widget for interactive metric visualization."""
    
    def __init__(self, parent=None):
        """Initialize the plotter widget."""
        super().__init__(parent)
        self.current_component = "g_tt"
        self.colormap = "redblue"
        self.components = {}
        self.comp_selector = QComboBox(self)
        self.cmap_selector = QComboBox(self)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Component selector
        comp_label = QLabel("Component:")
        self.comp_selector = QComboBox()
        self.comp_selector.addItems([
            "g_tt", "g_tx", "g_xx", "g_yy", "g_zz"
        ])
        layout.addWidget(comp_label)
        layout.addWidget(self.comp_selector)
        
        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Colormap selector
        cmap_label = QLabel("Colormap:")
        self.cmap_selector = QComboBox()
        self.cmap_selector.addItems([
            "redblue", "warp", "viridis", "plasma"
        ])
        layout.addWidget(cmap_label)
        layout.addWidget(self.cmap_selector)
        
        # Connect signals
        self.comp_selector.currentTextChanged.connect(self.plot_component)
        self.cmap_selector.currentTextChanged.connect(self.set_colormap)
    
    def set_metric(self, components: dict):
        """Set the metric components to display.
        
        Parameters
        ----------
        components : dict
            Dictionary of metric components
        """
        self.components = components
        self.plot_component(self.current_component)
    
    def plot_component(self, component: str):
        """Plot a specific metric component.
        
        Parameters
        ----------
        component : str
            Name of component to plot
        """
        if not hasattr(self, 'components'):
            return
        
        self.current_component = component
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        data = self.components[component]
        im = ax.imshow(data, cmap=self.colormap)
        self.figure.colorbar(im)
        ax.set_title(f"Metric Component {component}")
        
        self.canvas.draw()
    
    def set_colormap(self, cmap: str):
        """Set the colormap for visualization.
        
        Parameters
        ----------
        cmap : str
            Name of colormap to use
        """
        self.colormap = cmap
        if hasattr(self, 'components'):
            self.plot_component(self.current_component)
            
    def get_plot_data(self) -> np.ndarray:
        """Get the current plot data.
        
        Returns
        -------
        np.ndarray
            Current component data being displayed
        """
        if not hasattr(self, 'components'):
            return np.array([])
        return self.components[self.current_component]
