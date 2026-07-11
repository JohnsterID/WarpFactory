"""Interactive metric visualization widget."""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..visualizer import ColorMaps


class MetricPlotter(QWidget):
    """Widget for interactive metric visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_component = "g_tt"
        self.colormap = "redblue"
        self.components = {}
        self.coordinates = None
        self.colormaps = ColorMaps()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Component:"))
        self.comp_selector = QComboBox(self)
        self.comp_selector.setObjectName("component_selector")
        self.comp_selector.addItems(["g_tt", "g_tx", "g_xx", "g_yy", "g_zz"])
        layout.addWidget(self.comp_selector)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        layout.addWidget(QLabel("Colormap:"))
        self.cmap_selector = QComboBox(self)
        self.cmap_selector.setObjectName("colormap_selector")
        self.cmap_selector.addItems(["redblue", "warp", "viridis", "plasma"])
        layout.addWidget(self.cmap_selector)

        self.comp_selector.currentTextChanged.connect(self.plot_component)
        self.cmap_selector.currentTextChanged.connect(self.set_colormap)

    def set_metric(self, components: dict, x: np.ndarray = None):
        """Set the metric components to display.

        Parameters
        ----------
        components : dict
            Dictionary of metric components
        x : np.ndarray, optional
            Coordinate array for the horizontal axis
        """
        self.components = components
        self.coordinates = x
        self.plot_component(self.current_component)

    def plot_component(self, component: str):
        """Plot a specific metric component."""
        if component not in self.components:
            self.current_component = component
            return

        self.current_component = component
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        values = np.asarray(self.components[component])
        cmap = self.colormaps.get(self.colormap)
        if values.ndim >= 2:
            image = ax.imshow(values, cmap=cmap)
            self.figure.colorbar(image)
        else:
            # 1-D slices cannot be imshow-n; draw a colormapped line plot
            x = self.coordinates if self.coordinates is not None else np.arange(len(values))
            ax.plot(x, values, color=cmap(0.8))
        ax.set_title(f"Metric Component {component}")
        self.canvas.draw_idle()

    def set_colormap(self, cmap: str):
        """Set the colormap for visualization."""
        self.colormap = cmap
        self.plot_component(self.current_component)

    def get_plot_data(self) -> np.ndarray:
        """Current component data being displayed."""
        if self.current_component not in self.components:
            return np.array([])
        return np.asarray(self.components[self.current_component])
