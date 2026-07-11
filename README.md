# WarpFactory (Python Port)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/warpfactory.svg)](https://badge.fury.io/py/warpfactory)

A Python port of [WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory), a numerical toolkit for analyzing warp drive spacetimes using Einstein's theory of General Relativity. This implementation aims to make the toolkit more accessible to the Python scientific computing community. It reaches feature parity with the original MATLAB implementation (with a few deliberate physics-first fixes); see "Parity with the MATLAB original" below for details.

## Key Features

### Core Functionality
- Finite difference solver (2nd/4th order) for the stress-energy tensor
  via the Einstein field equations on full 4-D grids and on 1-D axial
  slices
- Christoffel symbol, Ricci tensor/scalar, and Kretschmann scalar
  computation for Cartesian slices and spherically symmetric metrics
- Tensor index management (covariant, contravariant, mixed) and ADM
  3+1 composition/decomposition on grids
- Energy condition evaluations (Null, Weak, Dominant, Strong) by
  pointwise observer sampling
- Momentum flow visualizations

### GPU Acceleration
- PyTorch-based metric calculations
- CUDA-accelerated tensor operations
- GPU-optimized energy tensor computations
- Parallel Christoffel symbol calculations
- Device-agnostic code (CPU/GPU)

### Available Metrics
- Alcubierre warp drive (lab and comoving frames)
- Lentz positive-energy warp drive (lab and comoving frames)
- Van Den Broeck bubble with volume expansion (lab and comoving frames)
- Modified Time warp drive (lab and comoving frames)
- TOV-based comoving Warp Shell (Helmerich & Fuchs, CQG 2024)
- Schwarzschild black hole
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
|- metrics/          # Spacetime metric implementations
|- solver/          # Numerical solvers and tensor calculations
|- analyzer/        # Physical analysis tools
|- units/           # Unit conversion and management
`- visualizer/      # Plotting and visualization tools
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

## Parity with the MATLAB original

The `warpfactory.grid` package implements the full MATLAB WarpFactory
grid pipeline:

- Metric builders on full (t, x, y, z) grids: Minkowski, Alcubierre,
  Lentz, Van Den Broeck, Modified Time (each with a Galilean comoving
  variant), Schwarzschild, and the TOV-based comoving Warp Shell
  (`metricGet_*` equivalents).
- Stress-energy solver on 4-D grids at 2nd or 4th finite-difference
  order (`met2den`/`met2den2`/`getEnergyTensor` equivalents), validated
  against the analytic Alcubierre Eulerian energy density and the
  Schwarzschild vacuum.
- Tensor index management (`verify_tensor`, `change_tensor_index`),
  ADM 3+1 composition/decomposition, and the interpolation utilities
  (trilinear, quadrilinear, Legendre radial).
- Energy-condition violation maps on grids (`get_energy_conditions`,
  the `getEnergyConditions.m` equivalent): the stress-energy tensor is
  transformed to the local Eulerian frame via the explicit Cholesky
  decomposition of the metric (`do_frame_transfer` /
  `eulerian_transformation_matrix`) and contracted against Fibonacci-
  lattice-sampled null or timelike observer fields
  (`generate_uniform_field`), returning the most-violating evaluation
  at every grid point.
- Kinematic scalars of the Eulerian congruence (`get_scalars`, the
  `getScalars.m` equivalent): expansion, shear, and vorticity computed
  from the finite-difference covariant derivative of the normal
  observer 4-velocity, validated against the analytic Alcubierre
  expansion theta = v (x - xs)/r df/dr.
- SI conversion at the API boundary (`stress_energy_to_si`,
  `si_energy_factor`): multiply geometric-unit stress-energy (1/m^2)
  by c^4/G to get J/m^3 for direct comparison with the MATLAB output
  and the published figures.
- Slice plotting (`plot_tensor`, `plot_three_plus_one`).

Quantitative validation against the WarpFactory paper (Helmerich et
al., CQG 2024; arXiv 2404.03095v2) is committed as
`warpfactory/tests/test_paper_validation.py`: the Section 4.1
Alcubierre configuration (v = 0.1c, R = 300 m, sigma = 0.015 1/m)
reproduces the analytic peak Eulerian energy density of
-6.775e35 J/m^3 to 2% on a 64^3 grid at 12.5 m spacing (4th-order FD),
the negative-energy shell localizes to the bubble wall (Figure 2), and
the Null/Weak/Dominant/Strong violations of Table 1 are reproduced for
the Alcubierre, Van Den Broeck, and Modified Time metrics.

This is a physics-first port, not a bug-for-bug one. Deliberate
differences from the MATLAB original:

- Geometric units (G = c = 1) throughout: the bubble center moves as
  `xs = v*t` and the solver returns `T = G_munu / 8 pi` (MATLAB scales
  time derivatives by 1/c and multiplies by c^4/8piG for J/m^3).
- The Modified Time metric fixes the indexing bugs present in
  `metricGet_ModifiedTime.m` (undefined `gridScale`, wrong axis scale
  indices, and a `(t,i,k,k)` symmetric-assignment typo), so it produces
  the intended construction rather than the MATLAB output.
- Grid indexing is zero-based; world centers are specified in physical
  coordinates.
- In `get_energy_conditions`, once components are in the local
  Eulerian (orthonormal) frame, indices are raised and lowered with the
  Minkowski metric for all four conditions. `getEnergyConditions.m`
  lowers with the full coordinate metric for Null/Weak but with
  Minkowski for Strong/Dominant; that inconsistency is resolved here in
  favor of the local-frame convention.

The older 1-D axial-slice API (`warpfactory.metrics`,
`warpfactory.solver`) remains available and unchanged.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
The license retains the copyright of the original MATLAB WarpFactory authors
(Christopher Helmerich & Jared Fuchs) alongside the Python port copyright.

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