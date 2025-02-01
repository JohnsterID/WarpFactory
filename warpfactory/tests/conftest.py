"""Test configuration and shared fixtures."""

import os
import pytest
import numpy as np
from warpfactory.metrics import MinkowskiMetric
from warpfactory.units import Constants

# Set up virtual display for GUI tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow
    from PyQt6.QtCore import Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False

@pytest.fixture(scope="session")
def app():
    """Create QApplication instance for testing."""
    if not HAS_QT:
        pytest.skip("Qt6 is not available")
    app = QApplication([])
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    return app

@pytest.fixture
def window(app):
    """Create a QMainWindow for testing."""
    window = QMainWindow()
    window.resize(800, 600)
    return window

@pytest.fixture
def container(window):
    """Create a QWidget container for testing."""
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    window.setCentralWidget(container)
    return container


@pytest.fixture
def spatial_grid():
    """Create a standard spatial grid for testing."""
    x = np.linspace(-5, 5, 50)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return x, y, z


@pytest.fixture
def metric_params():
    """Standard metric parameters for testing."""
    return {
        "v_s": 2.0,    # ship velocity (in c)
        "R": 1.0,      # radius of warp bubble
        "sigma": 0.5,  # thickness parameter
    }


@pytest.fixture
def flat_metric():
    """Create a Minkowski metric instance for testing."""
    return MinkowskiMetric()


@pytest.fixture
def physical_constants():
    """Create a Constants instance for testing."""
    return Constants()


@pytest.fixture
def test_metric_components():
    """Create test metric components for a simple spacetime."""
    x = np.linspace(-1, 1, 10)
    components = {
        "g_tt": -np.ones_like(x),
        "g_tx": np.zeros_like(x),
        "g_xx": np.ones_like(x),
        "g_yy": np.ones_like(x),
        "g_zz": np.ones_like(x),
    }
    return components, x


@pytest.fixture
def test_energy_tensor():
    """Create test energy-momentum tensor components."""
    x = np.linspace(-1, 1, 10)
    rho = np.exp(-x**2)  # Energy density
    p = rho/3  # Pressure (radiation equation of state)
    
    return {
        "T_tt": rho,
        "T_xx": p,
        "T_yy": p,
        "T_zz": p,
        "T_tx": np.zeros_like(x),
    }, x