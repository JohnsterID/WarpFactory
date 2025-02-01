# Qt Integration Debugging Log

## Issue Overview
Integrating PyQt6 widgets with test framework is failing with multiple issues:
1. Widget type compatibility errors
2. Import issues
3. Parent-child widget relationship problems

## Attempted Solutions

### Attempt 1: Basic QWidget Inheritance
```python
class MetricExplorer(QWidget):
    def __init__(self):
        super().__init__()
```
**Result**: Failed - TypeError when adding to layout
**Issue**: Qt didn't recognize class as proper QWidget

### Attempt 2: Parent-Child Setup
```python
class MetricExplorer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
```
**Result**: Failed - Same TypeError
**Issue**: Still type recognition problems

### Attempt 3: __new__ Method
```python
class MetricExplorer(QWidget):
    def __new__(cls):
        instance = super().__new__(cls)
        instance.__init__()
        return instance
```
**Result**: Failed - Multiple initialization issues
**Issue**: Complicated Qt's own initialization

### Attempt 4: Import Reorganization
```python
# In __init__.py
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
```
**Result**: Failed - Still type recognition issues
**Issue**: Imports alone didn't solve type recognition

### Attempt 5: Qt Type Registration
```python
from PyQt6 import sip

def register_widget_type(cls):
    """Register a widget class with Qt's type system."""
    sip.setapi(cls.__name__, 2)
    return cls

register_widget_type(MetricExplorer)
```
**Result**: Failed - Missing dependencies and import issues
**Issues**: 
1. Missing matplotlib.backends.backend_qt6agg
2. Invalid import syntax in test files

### Attempt 6: Fix Dependencies and Imports
1. Add matplotlib dependency
2. Fix import syntax in test files
```python
# Before
from warpfactory.gui import *, QVBoxLayout

# After
from warpfactory.gui import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QMainWindow, QDoubleSpinBox,
    MetricExplorer, MetricPlotter, ParameterPanel,
    EnergyConditionViewer
)
```
**Result**: Failed - Still missing matplotlib backend
**Issue**: Qt6 backend not available

### Attempt 7: Use Qt5 Backend for Matplotlib
```python
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend instead of Qt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
```
**Result**: Failed - Need to update all matplotlib imports
**Issue**: Multiple files using Qt6 backend

### Attempt 8: Switch to PyQt6 Backend
1. Update matplotlib to use PyQt6 backend
2. Update all imports to use PyQt6
3. Move backend selection to __init__.py
```python
# In __init__.py
import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend with PyQt6
```
**Result**: In Progress
**Rationale**: Use consistent PyQt6 backend throughout the application

## Current Approach
1. Centralize Qt imports in `gui/__init__.py`
2. Use proper parent-child widget relationships
3. Ensure widget initialization happens in correct order

## Lessons Learned
1. Qt widgets must be properly initialized with parent-child relationships
2. Import organization is critical for Qt integration
3. Avoid multiple initialization methods (__new__, __init__)
4. Keep track of attempted solutions to avoid loops

## Next Steps
1. [ ] Fix remaining import issues in test files
2. [ ] Ensure proper widget hierarchy in all components
3. [ ] Add proper cleanup in test fixtures
4. [ ] Document working patterns for future reference

## Common Pitfalls
1. Multiple inheritance with Qt classes
2. Improper widget initialization order
3. Missing parent-child relationships
4. Circular imports with Qt classes

## Best Practices
1. Always use parent parameter in widget constructors
2. Initialize Qt application before creating widgets
3. Clean up widgets after tests
4. Use Qt's own layout management system

## References
- [Qt Documentation](https://doc.qt.io/)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [pytest-qt](https://pytest-qt.readthedocs.io/)
