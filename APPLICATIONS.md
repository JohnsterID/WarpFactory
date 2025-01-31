# WarpFactory Applications Guide

This guide demonstrates how to use WarpFactory for analyzing warp drive spacetimes. We'll cover common workflows and applications.

## Basic Workflow

### 1. Metric Calculation

```python
import numpy as np
from warpfactory.metrics import AlcubierreMetric
from warpfactory.visualizer import TensorPlotter

# Create spatial grid
x = np.linspace(-10, 10, 100)
y = np.zeros_like(x)
z = np.zeros_like(x)
t = 0.0

# Initialize metric
metric = AlcubierreMetric()

# Calculate metric components
components = metric.calculate(x, y, z, t,
    v_s=2.0,    # Ship velocity (in c)
    R=1.0,      # Bubble radius
    sigma=0.5   # Thickness parameter
)

# Visualize metric
plotter = TensorPlotter()
fig = plotter.plot_component(components, "g_tt", x, y)
fig.savefig("metric_tt.png")
```

### 2. Energy Conditions

```python
from warpfactory.solver import EnergyTensor
from warpfactory.analyzer import EnergyConditions

# Calculate energy-momentum tensor
energy = EnergyTensor()
T_munu = energy.calculate(components, x, y, z)

# Check energy conditions
conditions = EnergyConditions()
results = {
    "weak": conditions.check_weak(T_munu),
    "null": conditions.check_null(T_munu),
    "strong": conditions.check_strong(T_munu),
    "dominant": conditions.check_dominant(T_munu)
}

print("Energy Conditions:", results)
```

### 3. Curvature Analysis

```python
from warpfactory.solver import (
    ChristoffelSymbols,
    RicciTensor,
    RicciScalar
)

# Calculate Christoffel symbols
christoffel = ChristoffelSymbols()
gamma = christoffel.calculate(components, x, y, z)

# Calculate Ricci tensor
ricci = RicciTensor()
R_munu = ricci.calculate(components, gamma)

# Calculate Ricci scalar
scalar = RicciScalar()
R = scalar.calculate(components, R_munu)
```

## Advanced Applications

### 1. GPU-Accelerated Analysis

```python
import torch
from warpfactory.torch import (
    TorchMetricBatch,
    TorchEnergyAnalyzer
)

# Create parameters for multiple configurations
params = {
    "v_s": torch.tensor([1.0, 2.0, 3.0], device="cuda"),
    "R": torch.tensor([0.5, 1.0, 1.5], device="cuda"),
    "sigma": torch.tensor([0.3, 0.5, 0.7], device="cuda")
}

# Calculate metrics in parallel
batch = TorchMetricBatch(device="cuda")
metrics = batch.calculate_metrics_parallel(x, y, z, t, params)

# Analyze energy conditions
analyzer = TorchEnergyAnalyzer(device="cuda")
results = analyzer.analyze_batch(metrics)
```

### 2. Interactive Exploration

```python
from warpfactory.gui import MetricExplorer
import sys
from PyQt6.QtWidgets import QApplication

# Create application
app = QApplication(sys.argv)

# Create and show explorer
explorer = MetricExplorer()
explorer.show()

# Start event loop
sys.exit(app.exec())
```

### 3. Field Visualization

```python
from warpfactory.torch import TorchFieldVisualizer

# Initialize visualizer
visualizer = TorchFieldVisualizer(device="cuda")

# Create field data
field = torch.exp(-(x**2 + y**2 + z**2)/2).cuda()
vectors = torch.stack([
    -x * field,
    -y * field,
    -z * field
], dim=0)

# Create visualization
fig = visualizer.plot_scalar_field(field, x, y)
fig.savefig("field.png")

# Create animation
anim = visualizer.animate_field_evolution(
    field, vectors, x, y, frames=30
)
anim.save("evolution.gif")
```

## Common Analysis Tasks

### 1. Energy Density Distribution

```python
import matplotlib.pyplot as plt
from warpfactory.visualizer import TensorPlotter

# Calculate energy tensor
energy = EnergyTensor()
T_munu = energy.calculate(components, x, y, z)

# Plot energy density
plotter = TensorPlotter()
fig = plotter.plot_component(T_munu, "T_tt", x, y,
    title="Energy Density",
    cmap="RdBu_r"
)
plt.colorbar(label="ρ (c⁴/G)")
plt.savefig("energy_density.png")
```

### 2. Tidal Forces

```python
from warpfactory.analyzer import TidalForces

# Calculate tidal forces
tidal = TidalForces()
forces = tidal.calculate(components, gamma, x, y, z)

# Plot radial tidal force
plt.figure()
plt.plot(x, forces["radial"])
plt.xlabel("Distance (m)")
plt.ylabel("Tidal Force (m/s²)")
plt.savefig("tidal_forces.png")
```

### 3. Spacetime Analysis

```python
from warpfactory.spacetime import (
    GeodesicSolver,
    HorizonFinder,
    SingularityDetector,
    GravitationalLensing
)

# 1. Solve geodesic equations
solver = GeodesicSolver()
t0 = 0.0
x0 = np.array([-3.0, 0.0, 0.0])  # Starting position
v0 = np.array([1.0, 0.0, 0.0])   # Initial velocity
times, positions, velocities = solver.solve(
    components, t0, x0, v0,
    t_max=10.0,
    dt=0.1
)

# Plot geodesic trajectory
plt.figure()
plt.plot(positions[:, 0], positions[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Geodesic Path")
plt.savefig("geodesic.png")

# 2. Find event horizons
finder = HorizonFinder()
horizons = finder.find_horizons(components, x, y, z)

# Plot horizons
plt.figure()
for name, surface in horizons.items():
    if len(surface) > 0:
        plt.plot(surface[:, 0], surface[:, 1], label=name)
plt.legend()
plt.title("Event Horizons")
plt.savefig("horizons.png")

# 3. Detect singularities
detector = SingularityDetector()
singularities = detector.find_singularities(components, x, y, z)

# Print singularity information
for i, loc in enumerate(singularities["locations"]):
    print(f"Singularity {i+1}:")
    print(f"  Location: {loc}")
    print(f"  Type: {singularities['types'][i]}")
    print(f"  Strength: {singularities['strengths'][i]}")

# 4. Calculate gravitational lensing
lensing = GravitationalLensing()
source_pos = np.array([-2.0, 0.1, 0.0])
observer_pos = np.array([2.0, 0.1, 0.0])
bundle_radius = 0.1
n_rays = 4

rays = lensing.trace_light_rays(
    components, source_pos, observer_pos,
    bundle_radius, n_rays
)

# Plot ray paths
plt.figure()
for ray in rays:
    path = np.array(ray["path"])
    plt.plot(path[:, 0], path[:, 1])
plt.plot(source_pos[0], source_pos[1], 'ro', label='Source')
plt.plot(observer_pos[0], observer_pos[1], 'go', label='Observer')
plt.legend()
plt.title("Light Ray Paths")
plt.savefig("lensing.png")

# Calculate optical properties
optics = lensing.analyze_bundle(rays)
print("Optical Properties:")
print(f"  Magnification: {optics['magnification']:.2f}")
print(f"  Shear: {optics['shear']:.2f}")
print(f"  Convergence: {optics['convergence']:.2f}")
```

## Performance Tips

1. **GPU Acceleration**:
   - Use PyTorch for large-scale computations
   - Keep data on GPU to avoid transfers
   - Use batch processing for parameter studies

2. **Memory Management**:
   - Clear unused tensors with `torch.cuda.empty_cache()`
   - Monitor memory with `torch.cuda.memory_summary()`
   - Use `torch.cuda.synchronize()` for accurate timing

3. **Visualization**:
   - Move data to CPU for plotting
   - Use appropriate downsampling for large grids
   - Save figures to avoid memory leaks

## References

1. Alcubierre, M. (1994). The warp drive: hyper-fast travel within general relativity. Classical and Quantum Gravity, 11(5), L73.
2. Lentz, E. W. (2021). Breaking the warp barrier: Hyper-fast solitons in Einstein-Maxwell-plasma theory. Classical and Quantum Gravity, 38(7), 075015.
3. Van Den Broeck, C. (1999). A 'warp drive' with more reasonable total energy requirements. Classical and Quantum Gravity, 16(12), 3973.