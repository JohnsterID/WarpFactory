# WarpFactory (Python)

A Python port of [WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory),
a numerical toolkit for analyzing warp drive spacetimes using
Einstein's theory of General Relativity. It reaches feature parity
with the original MATLAB implementation, with a few deliberate
physics-first fixes documented on the [MATLAB Parity](parity.md) page.

![Alcubierre Eulerian energy density and energy-condition violation maps](assets/alcubierre_energy_conditions.png)

*Eulerian energy density and Null/Weak energy-condition violation maps
for the Alcubierre metric with the WarpFactory paper's Section 4.1
parameters (v = 0.1c, R = 300 m, sigma = 0.015 1/m). Regenerate with
`python examples/alcubierre_energy_conditions.py`.*

## What it does

- Builds warp drive metrics on full 4-D grids and 1-D axial slices:
  Alcubierre, Lentz, Van Den Broeck, Modified Time (lab and comoving
  frames), TOV-based comoving Warp Shell, Schwarzschild, Minkowski,
  and custom metrics.
- Solves the Einstein field equations with 2nd/4th-order finite
  differences to obtain the stress-energy tensor.
- Evaluates the Null, Weak, Dominant, and Strong energy conditions by
  pointwise observer sampling in the local Eulerian frame.
- Computes curvature (Christoffel symbols, Ricci tensor/scalar,
  Kretschmann and Gauss-Bonnet invariants), kinematic scalars
  (expansion, shear, vorticity), geodesics, horizons, singularities,
  gravitational lensing, tidal forces, and the Ford-Roman quantum
  inequality bounds.
- Searches parameter spaces and minimizes exotic-matter requirements
  over metric ansatz families.
- Visualizes tensor components, energy densities, and momentum flows;
  includes an interactive Jupyter explorer.

## Validation

Quantitative validation against the WarpFactory method paper
(Helmerich et al., CQG 2024; [arXiv 2404.03095](https://arxiv.org/abs/2404.03095))
is committed as `warpfactory/tests/test_paper_validation.py`: the
Section 4.1 Alcubierre configuration reproduces the analytic peak
Eulerian energy density of -6.775e35 J/m^3 to 2% on a 64^3 grid at
12.5 m spacing (4th-order FD), and the energy-condition violations of
Table 1 are reproduced for the Alcubierre, Van Den Broeck, and
Modified Time metrics.

## Where to start

- [Installation](installation.md) -- pip, poetry, and the optional
  extras
- [Quickstart](quickstart.md) -- from a metric to energy-condition
  maps in a dozen lines
- [Interactive Explorer](interactive.md) -- the Jupyter front end,
  with Binder and Colab links
- [Applications](applications.md) -- curvature analysis, geodesics,
  horizons, lensing, tidal forces
- [MATLAB Parity](parity.md) -- exactly how this port relates to the
  original

## Citing

If you use WarpFactory in research, cite the method paper and this
port (see the repository `CITATION.cff`, or use the "Cite this
repository" button on GitHub):

```bibtex
@article{Helmerich2024WarpFactory,
  title   = {Analyzing warp drive spacetimes with {Warp Factory}},
  author  = {Helmerich, Christopher and Fuchs, Jared and Bobrick,
             Alexey and Sellers, Luke and Melcher, Brandon and
             Martire, Gianni},
  journal = {Classical and Quantum Gravity},
  volume  = {41},
  number  = {9},
  pages   = {095009},
  year    = {2024},
  doi     = {10.1088/1361-6382/ad2e42}
}
```
