# Quickstart

Two API surfaces are available:

- **`warpfactory.grid`** -- the full 4-D grid pipeline with MATLAB
  parity (recommended for quantitative work).
- **The 1-D axial-slice API** (`warpfactory.metrics`,
  `warpfactory.solver`, `warpfactory.analyzer`) -- lightweight
  exploration along the bubble axis.

## 4-D grid pipeline

From a metric to SI-unit energy density and energy-condition maps
(this is `examples/alcubierre_energy_conditions.py`, reproducing the
WarpFactory paper's Section 4.1 workflow):

```python
from warpfactory.grid import (
    GridSolver,
    alcubierre_metric,
    do_frame_transfer,
    get_energy_conditions,
    stress_energy_to_si,
)

N, SPACING = 64, 12.5
GRID_SIZE = (1, N, N, N)
WORLD_CENTER = (0.0,) + tuple((n - 1) * SPACING / 2 for n in GRID_SIZE[1:])

metric = alcubierre_metric(
    GRID_SIZE,
    WORLD_CENTER,
    v=0.1, R=300.0, sigma=0.015,
    grid_scale=(1, SPACING, SPACING, SPACING),
)
stress_energy = GridSolver(order=4).solve(metric)

# Eulerian-frame energy density in J/m^3
eulerian = do_frame_transfer(metric, stress_energy)
rho_si = stress_energy_to_si(eulerian).tensor[0, 0, 0]
print(f"peak energy density: {rho_si.min():.3e} J/m^3")  # ~ -6.8e35

# Pointwise energy-condition violation maps
null_map = get_energy_conditions(stress_energy, metric, "Null", num_angular_vec=50)
weak_map = get_energy_conditions(
    stress_energy, metric, "Weak", num_angular_vec=50, num_time_vec=8
)
```

## 1-D slice API

```python
import numpy as np

from warpfactory.metrics import AlcubierreMetric
from warpfactory.solver import EnergyTensor

x = np.linspace(-8, 8, 400)
y = np.zeros_like(x)
z = np.zeros_like(x)

metric = AlcubierreMetric()
components = metric.calculate(x, y, z, t=0, v_s=2.0, R=1.0, sigma=4.0)

# Stress-energy via the Einstein field equations (geometric units)
T_munu = EnergyTensor().calculate_from_metric(components, x)
print(T_munu["T_tt"].min())  # negative energy at the bubble wall
```

Check the pointwise energy conditions:

```python
from warpfactory.analyzer import EnergyConditions

conditions = EnergyConditions()
print(conditions.check_weak(T_munu))      # False for Alcubierre
print(conditions.check_null(T_munu))      # False
print(conditions.check_strong(T_munu))    # False
print(conditions.check_dominant(T_munu))  # False
```

Plot a component:

```python
from warpfactory.visualizer import TensorPlotter

fig = TensorPlotter().plot_component(components, "g_tx", x, y)
fig.savefig("g_tx.png")
```

## Quantum inequality bounds

```python
from warpfactory.physics import FordRomanInequality

qi = FordRomanInequality()

# Ford-Roman sampling bound in J/m^3 for a given sampling time
bound = qi.sampling_bound(tau0=1e-10)

# Pfenning-Ford warp bubble constraints: wall thickness vs the quantum
# inequality limit, and the total (negative) wall energy in Joules
bubble = qi.check_warp_bubble(v_b=2.0, R=100.0, sigma=8.0)
print(bubble["delta"], bubble["delta_max"], bubble["total_energy"])
```

## Unit conversions

```python
from warpfactory.units import Quantity, UnitSystem

velocity = (Quantity(1.0, "km") / Quantity(1.0, "hour")).to("m/s")

units = UnitSystem()
mass_geometric = units.to_geometric_units("mass", 1.989e30)  # solar mass
```

## Next steps

- Runnable scripts live in [`examples/`](https://github.com/JohnsterID/WarpFactory/tree/main/examples).
- The [Applications](applications.md) page covers curvature analysis,
  geodesics, horizons, lensing, and tidal forces.
- The [Interactive Explorer](interactive.md) page covers the Jupyter
  front end.
