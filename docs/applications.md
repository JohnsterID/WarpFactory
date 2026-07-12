# Applications

Workflows on the 1-D axial-slice API. All snippets below run as-is
after `pip install .`; start from this common setup:

```python
import numpy as np

from warpfactory.metrics import AlcubierreMetric

x = np.linspace(-8, 8, 400)
y = np.zeros_like(x)
z = np.zeros_like(x)
components = AlcubierreMetric().calculate(x, y, z, t=0, v_s=2.0, R=1.0, sigma=4.0)
```

## Curvature analysis

```python
from warpfactory.solver import ChristoffelSymbols, RicciScalar, RicciTensor

gamma = ChristoffelSymbols().calculate(components, x, y, z)
R_munu = RicciTensor().calculate_array(components, x)   # shape (4, 4, N)
R = RicciScalar().calculate(components, {"x": x})       # shape (N,)
```

Scalar invariants (Kretschmann, Ricci-squared, Gauss-Bonnet, and the
EFT-validity watchdog) are available for spherically symmetric metrics
in `warpfactory.analyzer.scalar_invariants`.

## Tidal forces

Geodesic deviation (-R^a_bcd u^b s^c u^d) for radial, transverse, and
longitudinal separation vectors:

```python
from warpfactory.physics import TidalForces

forces = TidalForces().calculate(components, gamma, x, y, z)
forces["radial"]        # tidal acceleration per unit separation
forces["transverse"]
forces["longitudinal"]
```

## Geodesics

```python
from warpfactory.spacetime import GeodesicSolver

times, positions, velocities = GeodesicSolver().solve(
    components,
    0.0,                          # start time
    np.array([-3.0, 0.0, 0.0]),   # start position
    np.array([1.0, 0.0, 0.0]),    # initial coordinate velocity
    t_max=5.0,
    dt=0.1,
)
```

## Horizons and singularities

```python
from warpfactory.spacetime import HorizonFinder, SingularityDetector

horizons = HorizonFinder().find_horizons(components, x, y, z)
horizons["outer"], horizons["inner"], horizons["ergosphere"]

sing = SingularityDetector().find_singularities(components, x, y, z)
for loc, kind, strength in zip(
    sing["locations"], sing["types"], sing["strengths"]
):
    print(loc, kind, strength)
```

!!! note
    In the standard Alcubierre slicing, g_tt g_xx - g_tx^2 = -1
    identically, so there is no t-x determinant horizon; only the
    ergosurface (g_tt = 0) exists for v_s > 1.

## Gravitational lensing

```python
from warpfactory.spacetime import GravitationalLensing

lensing = GravitationalLensing()
rays = lensing.trace_light_rays(
    components,
    np.array([-2.0, 0.1, 0.0]),  # source position
    np.array([2.0, 0.1, 0.0]),   # observer position
    0.1,                         # bundle radius
    4,                           # number of rays
)
for ray in rays:
    path = np.array(ray["path"])  # trajectory points
```

## Kinematic scalars on 4-D grids

Expansion, shear, and vorticity of the Eulerian congruence
(`getScalars.m` equivalent), validated against the analytic Alcubierre
expansion:

```python
from warpfactory.grid import alcubierre_metric, get_scalars

metric = alcubierre_metric(
    (1, 64, 64, 64), (0.0, 315.0, 315.0, 315.0),
    v=0.1, R=100.0, sigma=0.03, grid_scale=(1, 10.0, 10.0, 10.0),
)
expansion, shear, vorticity = get_scalars(metric)
```

## Parameter search and optimization

Wrap any metric builder into a searchable ansatz and minimize
exotic-matter requirements (see `examples/optimize_bubble.py` for the
full workflow):

```python
from warpfactory.optimize import CallableAnsatz
```

## Conservation checks

The covariant divergence of the solver's stress-energy is pure
discretization error (Bianchi identity); `warpfactory.physics.conservation`
evaluates it as a numerical sanity check on any metric.
