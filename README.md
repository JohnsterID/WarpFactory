# WarpFactory (Python Port)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/warpfactory.svg)](https://badge.fury.io/py/warpfactory)

A Python port of [WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), a numerical toolkit for analyzing warp drive spacetimes using Einstein's theory of General Relativity. This implementation aims to make the toolkit more accessible to the Python scientific computing community while maintaining full compatibility with the original MATLAB implementation's functionality.

## Key Features

### Core Functionality
- 3D finite difference solver for the stress-energy tensor
- Energy condition evaluations (Null, Weak, Dominant, Strong)
- Metric scalar evaluation (shear, expansion, vorticity)
- Momentum flow visualizations

### GPU Acceleration
- PyTorch-based metric calculations
- CUDA-accelerated tensor operations
- GPU-optimized energy tensor computations
- Parallel Christoffel symbol calculations
- Device-agnostic code (CPU/GPU)

### Available Metrics
- Alcubierre warp drive
- Lentz positive-energy warp drive
- Van Den Broeck bubble with volume expansion
- Warp Shell configuration
- Minkowski (flat spacetime)
- Custom metric support via base class

### Analysis Tools
- Three-plus-one (ADM) decomposition
- Frame transformations and boosts
- Christoffel symbol calculations
- Ricci tensor and scalar computations
- Scalar invariant analysis
- Geodesic equation solver
- Event horizon finder
- Singularity detector
- Gravitational lensing

### Visualization
- Tensor component plots
- Energy density distributions
- Momentum flow lines
- Custom physics-oriented colormaps

## Installation

### Using pip
```bash
# Install from PyPI
pip install warpfactory

# Install from source
git clone https://github.com/YourUsername/WarpFactory.git
cd WarpFactory
pip install .
```

### Using poetry (recommended)
```bash
# Install poetry if you haven't already
pip install poetry

# Install dependencies and package
git clone https://github.com/YourUsername/WarpFactory.git
cd WarpFactory
poetry install
```

### Requirements

#### Python Dependencies
- Python 3.9 or higher
- NumPy
- SciPy
- PyTorch (optional, for GPU acceleration)
- Matplotlib (for visualization)
- PyQt6 (for GUI)

#### System Dependencies
For the GUI components, you need Qt6 system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install -y \
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon-x11-0 \
    libdbus-1-3
```

**Fedora/RHEL:**
```bash
sudo dnf install -y \
    mesa-libGL \
    mesa-libEGL \
    libxkbcommon-x11 \
    dbus-libs
```

**macOS:**
```bash
brew install qt@6
```

For GPU support:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### Basic Metric Calculation
```python
import numpy as np
from warpfactory.metrics import AlcubierreMetric
from warpfactory.solver import EnergyTensor
from warpfactory.visualizer import TensorPlotter

# Initialize metric
metric = AlcubierreMetric()

# Set parameters
v_s = 2.0  # Ship velocity (in units of c)
R = 1.0    # Characteristic radius
sigma = 0.5 # Thickness parameter

# Create spatial grid
x = np.linspace(-5, 5, 50)
y = np.zeros_like(x)
z = np.zeros_like(x)

# Calculate metric components
components = metric.calculate(x, y, z, t=0, v_s=v_s, R=R, sigma=sigma)

# Calculate stress-energy tensor
energy = EnergyTensor()
T_munu = energy.calculate(components, x, y, z)

# Visualize results
plotter = TensorPlotter()
fig = plotter.plot_component(T_munu, "T_tt", x, y)
fig.savefig("energy_density.png")
```

### Energy Conditions Analysis
```python
from warpfactory.analyzer import EnergyConditions

# Initialize analyzer
conditions = EnergyConditions()

# Check energy conditions
is_weak = conditions.check_weak(T_munu)
is_null = conditions.check_null(T_munu)
is_strong = conditions.check_strong(T_munu)
is_dominant = conditions.check_dominant(T_munu)
```

### Unit Conversions
```python
from warpfactory.units import Quantity, UnitSystem

# Create physical quantities
length = Quantity(1.0, "km")
time = Quantity(1.0, "hour")

# Convert to different units
length_m = length.to("m")
time_s = time.to("s")

# Calculate velocity
velocity = length / time
velocity_ms = velocity.to("m/s")

# Convert to geometric units
units = UnitSystem()
mass_kg = 1.989e30  # solar mass
mass_geometric = units.to_geometric_units("mass", mass_kg)
```

## Project Structure
```
warpfactory/
├── metrics/          # Spacetime metric implementations
├── solver/          # Numerical solvers and tensor calculations
├── analyzer/        # Physical analysis tools
├── units/           # Unit conversion and management
└── visualizer/      # Plotting and visualization tools
```

## Testing

### Running Tests
```bash
# Run all tests (excluding GUI)
poetry run pytest -m "not gui"

# Run tests with coverage
poetry run pytest --cov=warpfactory

# Run specific test suite
poetry run pytest warpfactory/tests/test_metrics.py -v

# Run GUI tests (requires Qt system dependencies)
poetry run pytest warpfactory/tests/test_gui.py -v
```

### Test Environment
- Core functionality tests can run in any environment
- GUI tests require Qt6 system libraries (see System Dependencies)
- For CI/CD, use the provided Docker configuration
- For local development without GUI, use `pytest -m "not gui"`

### Test Coverage
- Core functionality: ~98% coverage
- Spacetime analysis: ~87% coverage
- GUI components: Tested in CI environment
- Performance tests: GPU-dependent tests marked with `@pytest.mark.gpu`
- Overall: ~31% coverage (including optional components)

## GPU Acceleration

### Using CUDA
```python
import torch
from warpfactory.torch import TorchMetricSolver

# Initialize solver on GPU
solver = TorchMetricSolver(device="cuda")

# Create spatial grid on GPU
x = torch.linspace(-5, 5, 100, device="cuda")
y = torch.zeros_like(x)
z = torch.zeros_like(x)

# Calculate metric (automatically uses GPU)
metric = solver.calculate_alcubierre_metric(
    x, y, z, t=0.0,
    v_s=2.0, R=1.0, sigma=0.5
)

# Move results back to CPU if needed
metric_cpu = {k: v.cpu() for k, v in metric.items()}
```

### Performance Tips
- Use `device="cuda"` for GPU acceleration
- Keep data on GPU to avoid transfer overhead
- Use batch processing for multiple calculations
- Profile with `torch.cuda.profiler`
- Monitor memory with `torch.cuda.memory_summary()`

### Requirements
- CUDA-capable GPU
- PyTorch with CUDA support
- CUDA Toolkit 11.8 or higher
- cuDNN (recommended)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes using pytest
4. Implement your changes following the project structure
5. Run tests and ensure they pass
6. Update documentation if needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Create a Pull Request

## Development Guidelines

- Follow Test-Driven Development (TDD) practices
- Use type hints and docstrings for all functions
- Keep code modular and follow single responsibility principle
- Add tests for edge cases and error conditions
- Maintain backward compatibility when possible

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This Python implementation is based on the original MATLAB WarpFactory toolkit developed by:
- Christopher Helmerich
- Jared Fuchs

Original project contributors and reviewers:
- Alexey Bobrick
- Luke Sellers
- Brandon Melcher
- Justin Feng
- Gianni Martire

For more information about the original implementation, visit:
- [Original WarpFactory Repository](https://github.com/NerdsWithAttitudes/WarpFactory)
- [WarpFactory Documentation](https://applied-physics.gitbook.io/warp-factory)
- [CQG Paper](https://iopscience.iop.org/article/10.1088/1361-6382/ad2e42)
- [arXiv Preprint](https://arxiv.org/abs/2404.03095)

## References

- Alcubierre, M. (1994). The warp drive: hyper-fast travel within general relativity. Classical and Quantum Gravity, 11(5), L73.
- Lentz, E. W. (2021). Breaking the warp barrier: Hyper-fast solitons in Einstein-Maxwell-plasma theory. Classical and Quantum Gravity, 38(7), 075015.
- Van Den Broeck, C. (1999). A 'warp drive' with more reasonable total energy requirements. Classical and Quantum Gravity, 16(12), 3973.