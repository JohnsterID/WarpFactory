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

## Observer-independent energy conditions (Hawking-Ellis)

Extension beyond the MATLAB original. `get_energy_conditions` samples
the energy conditions for a family of observers built from the
Eulerian frame; a violation seen only by some boosted observer can
slip through. The Hawking-Ellis classification instead works with the
eigenstructure of the mixed stress-energy T^a_b, which is
frame-independent: Type I points get exact all-observer margins in
closed form, Type II points additionally carry the Jordan-block
parameter f (T = f k k + Type-I part; every Type II condition needs
f >= 0, and a negative-amplitude null dust is invisible to the
eigenvalues alone), and Type III/IV points violate every condition
for every observer. It stays well-defined at superluminal warp
speeds.

```python
from warpfactory.grid import (
    GridSolver,
    alcubierre_metric,
    hawking_ellis_classify,
    invariant_energy_conditions,
)

metric = alcubierre_metric(
    (1, 32, 32, 32), (0.0, 7.75, 7.75, 7.75),
    v=2.0, R=3.0, sigma=2.0, grid_scale=(1, 0.5, 0.5, 0.5),
)
stress_energy = GridSolver(order=4).solve(metric)

classification = hawking_ellis_classify(stress_energy, metric)
classification.type_map          # Hawking-Ellis type (1-4) per point
classification.rho               # eigenframe energy density
classification.pressures         # principal pressures, shape (3,) + grid
classification.jordan_parameter  # Type II null-dust amplitude f

# All-observer margin maps (negative = violated by some observer)
nec = invariant_energy_conditions(
    stress_energy, metric, "Null", classification=classification
)
```

The Alcubierre bubble wall is Type IV dominated: no observer there
measures a well-defined energy density, and every pointwise energy
condition fails regardless of frame.

At Type I points the worst observer is available in closed form from
the same eigen-decomposition -- no optimizer involved:

```python
from warpfactory.grid import type_i_witnesses

observer, null_witness = type_i_witnesses(classification, metric)
# observer: u^a measuring the invariant rho (coordinate components)
# null_witness: k^a attaining the invariant null margin exactly,
#               NaN at non-Type-I points
```

## Averaged null energy condition (ANEC)

Extension beyond the MATLAB original. Pointwise violations can be
tolerated by quantum fields, but the averaged condition
`integral T_ab k^a k^b dlambda >= 0` along a complete null geodesic is
the sharper viability test. The evaluator integrates axial null rays
through the 1-D sampled metric (affine scale recovered from the
conserved Killing energy, as in the CMB hazard tracer):

```python
from warpfactory.physics import AveragedNullEnergy
from warpfactory.solver import EnergyTensor

stress_energy = EnergyTensor().calculate_from_metric(components, x)
result = AveragedNullEnergy().integrate(
    components, stress_energy, x_start=-4.5, direction=+1.0, x=x
)
result["anec"]       # negative = ANEC violated along this ray
result["integrand"]  # A T_ab w^a w^b per unit coordinate time
```

The affine normalization is dt/dlambda = 1 at launch; rescaling
multiplies the integral by a positive constant, so the sign of the
verdict is normalization-independent.

## Exact curvature and off-axis ANEC (hyper-dual autodiff)

Extension beyond the MATLAB original. For metrics available as
analytic functions of the coordinates, hyper-dual numbers (forward-mode
automatic differentiation, Fike & Alonso AIAA 2011-886) propagate exact
first and second metric derivatives through the metric function in pure
numpy -- machine-precision curvature with no stencil truncation error
and no new dependency:

```python
from warpfactory.grid import (
    ExactGridSolver,
    ExactNullGeodesicANEC,
    alcubierre_metric_fn,
)

solver = ExactGridSolver(alcubierre_metric_fn(v=0.5, R=2.0, sigma=4.0))

# Exact stress-energy at arbitrary points (no grid needed)
T = solver.stress_energy_at(0.0, 2.0, 1.0, 0.0)

# Or on a uniform grid, matching GridSolver.solve conventions
T_grid = solver.solve((1, 64, 64, 64), (0.0, 7.875, 7.875, 7.875),
                      grid_scale=(1, 0.25, 0.25, 0.25))
```

Because exact Christoffel symbols are available at arbitrary points,
ANEC rays no longer have to run down the x axis. The head-on Alcubierre
on-axis null contraction is analytically zero, so the violation lives
entirely off axis -- exactly where the 1-D evaluator cannot look:

```python
anec = ExactNullGeodesicANEC(solver)
result = anec.integrate(
    start=(0.0, -8.0, 1.0, 0.0),        # impact parameter y = 1
    spatial_direction=(1.0, 0.0, 0.0),
    comoving_velocity=0.5,              # bubble center xs = v t
)
result["anec"]           # negative = ANEC violated along this ray
result["null_residual"]  # max |g_ab k^a k^b|, solution-quality check
```

The FD `GridSolver` remains the right tool for metrics that exist only
as sampled data (TOV-built warp shells, piecewise profiles); autodiff
of non-smooth metrics is not meaningful in any backend.

## f(R) modified gravity

Extension beyond the MATLAB original. The standard pipeline answers
"what matter does general relativity require to sustain this metric?"
`FofRSolver` asks the same question in metric-formalism f(R) gravity
(Sotiriou & Faraoni, arXiv:0805.1726):

    F(R) R_munu - f(R)/2 g_munu + (g_munu Box - nabla_mu nabla_nu) F(R)
        = 8 pi T_munu,    F = df/dR

The scalaron terms turn curvature gradients into an effective
geometric source, so the matter budget of a warp metric shifts away
from the GR answer -- the frontier question being whether a modified
theory softens the exotic-matter requirement:

```python
from warpfactory.grid import (
    FofRSolver, GridSolver, alcubierre_metric,
    gr_model, starobinsky_model,
)

metric = alcubierre_metric(
    (1, 64, 64, 64), (0.0, 7.875, 7.875, 7.875),
    v=0.5, R=2.0, sigma=4.0, grid_scale=(1, 0.25, 0.25, 0.25),
)

T_gr = GridSolver(order=4).solve(metric)
T_st = FofRSolver(starobinsky_model(alpha=0.05), order=4).solve(metric)
delta = T_st.tensor - T_gr.tensor   # scalaron correction map
```

`gr_model()` (f(R) = R) reproduces `GridSolver` to round-off, and any
custom Lagrangian plugs in through `FofRModel(f=..., f_prime=...)` --
for example `f(R) = R - 2 Lambda` demands the exact dark-energy vacuum
source `T_munu = Lambda g_munu / 8 pi` on flat space. The output is a
regular stress-energy `SpacetimeTensor`, so the energy-condition and
Hawking-Ellis machinery applies unchanged.

`BransDickeSolver` extends the same contract to Jordan-frame
scalar-tensor gravity: supply the metric plus a scalar field profile
phi (and optionally a potential), get the demanded matter. Metric f(R)
gravity is its omega = 0 member with phi = F(R) and V = R F - f:

```python
from warpfactory.grid import BransDickeSolver

bd = BransDickeSolver(omega=100.0, order=4)
T_bd = bd.solve(metric, phi=1.0)   # phi = 1, V = 0 -> the GR answer
```

## Semiclassical trace anomaly

Extension beyond the MATLAB original. The conformal (trace) anomaly is
the state-independent piece of the renormalized quantum stress-energy
<T_munu>_ren: for massless conformally coupled fields the trace is a
closed-form curvature expression c C^2 - a E + xi Box R (Duff,
hep-th/9308075). Where it rivals the classical stress-energy scale,
semiclassical backreaction cannot be ignored and the classical warp
metric is not self-consistent:

```python
import numpy as np
from warpfactory.physics import ConformalTraceAnomaly

anomaly = ConformalTraceAnomaly(n_scalar=1)   # one conformal scalar

M = 1.0
r = np.linspace(3.0, 10.0, 141)
theta = np.full_like(r, np.pi / 2)
f = 1 - 2 * M / r
metric = {"g_tt": -f, "g_rr": 1 / f,
          "g_theta_theta": r**2, "g_phi_phi": r**2 * np.sin(theta)**2}
coords = {"r": r, "theta": theta}

trace = anomaly.trace(metric, coords)   # = M^2/(60 pi^2 r^6), published value
ratio = anomaly.backreaction_ratio(metric, coords, classical_trace_scale)
```

The state-dependent traceless part of <T_munu>_ren (choice of
Boulware/Unruh/Hartle-Hawking vacuum or their warp analogues) is a
genuinely open research problem and deliberately not implemented.

## ADM constraint residuals (initial-data validity)

Extension beyond the MATLAB original. Any 3+1 evolution starts from
initial data that must satisfy the Hamiltonian and momentum
constraints; `ADMConstraints` evaluates both residuals for a grid
metric slice against any matter source:

```python
from warpfactory.grid import ADMConstraints, GridSolver, alcubierre_metric

metric = alcubierre_metric(
    (1, 48, 48, 48), (0.0, 5.875, 5.875, 5.875),
    v=0.5, R=2.0, sigma=2.0, grid_scale=(1, 0.25, 0.25, 0.25),
)
adm = ADMConstraints(order=4)

# Against the slice's own EFE stress-energy: residuals are pure
# discretization error (Hamiltonian closes to round-off).
T = GridSolver(order=4).solve(metric)
res = adm.evaluate(metric, T)
res.hamiltonian, res.momentum

# Against vacuum: the nonzero Hamiltonian residual equals
# 16 pi rho_required -- the slice is not valid vacuum initial data.
res_vac = adm.evaluate(metric)
```

This is the initial-data half of the dynamic-evolution problem. The
evolution half (BSSN or generalized-harmonic reformulation, constraint
damping, gauge drivers, AMR) is what the Einstein Toolkit and GRChombo
exist for, and WarpFactory does not duplicate them.

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
