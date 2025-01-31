"""GUI components for WarpFactory."""

# Try to import Qt components, but don't fail if not available
try:
    from .explorer import MetricExplorer
    from .plotter import MetricPlotter
    from .parameters import ParameterPanel
    from .energy import EnergyConditionViewer
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    # Provide dummy classes for type hints
    class MetricExplorer: pass
    class MetricPlotter: pass
    class ParameterPanel: pass
    class EnergyConditionViewer: pass

__all__ = [
    'MetricExplorer',
    'MetricPlotter',
    'ParameterPanel',
    'EnergyConditionViewer',
    'HAS_GUI',
]