"""GUI components for WarpFactory."""

# Set matplotlib backend before any other imports
import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend with PyQt6

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
    QPushButton, QLabel, QMainWindow, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6 import sip

# Register our custom widget types with Qt's type system
def register_widget_type(cls):
    """Register a widget class with Qt's type system."""
    sip.setapi(cls.__name__, 2)
    return cls

from .explorer import MetricExplorer
from .plotter import MetricPlotter
from .parameters import ParameterPanel
from .energy import EnergyConditionViewer

# Register our widget types
register_widget_type(MetricExplorer)
register_widget_type(MetricPlotter)
register_widget_type(ParameterPanel)
register_widget_type(EnergyConditionViewer)

__all__ = [
    'MetricExplorer',
    'MetricPlotter',
    'ParameterPanel',
    'EnergyConditionViewer',
    'QWidget',
    'QVBoxLayout',
    'QHBoxLayout',
    'QComboBox',
    'QPushButton',
    'QLabel',
    'QMainWindow',
    'QDoubleSpinBox',
    'Qt',
]
