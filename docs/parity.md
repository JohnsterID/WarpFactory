# Parity with the MATLAB original

This port is based on the upstream
[NerdsWithAttitudes/WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory)
MATLAB code as of upstream commit `03b10cb` (the last upstream commit
at the time of porting). This repository's git history starts at the
Python port; the MATLAB development history lives in the upstream
repository.

## What is ported

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

## Quantitative validation

Validation against the WarpFactory paper (Helmerich et al., CQG 2024;
[arXiv 2404.03095v2](https://arxiv.org/abs/2404.03095)) is committed
as `warpfactory/tests/test_paper_validation.py`: the Section 4.1
Alcubierre configuration (v = 0.1c, R = 300 m, sigma = 0.015 1/m)
reproduces the analytic peak Eulerian energy density of
-6.775e35 J/m^3 to 2% on a 64^3 grid at 12.5 m spacing (4th-order FD),
the negative-energy shell localizes to the bubble wall (Figure 2), and
the Null/Weak/Dominant/Strong violations of Table 1 are reproduced for
the Alcubierre, Van Den Broeck, and Modified Time metrics.

## Deliberate divergences

This is a physics-first port, not a bug-for-bug one. Differences from
the MATLAB original:

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

## Extensions beyond the original

Capabilities with no MATLAB counterpart, kept in separate modules so
the ported surface stays recognizable:

- Hawking-Ellis classification and observer-independent energy
  conditions (`hawking_ellis_classify`, `invariant_energy_conditions`):
  pointwise Type I-IV maps from the eigenstructure of the mixed
  stress-energy tensor in the local orthonormal frame, with exact
  all-observer energy-condition margins at Type I points and
  unconditional-violation detection at Type IV points. This
  complements the Eulerian-sampled `get_energy_conditions`, which can
  miss violations visible only to boosted observers.
- Closed-form worst-observer witnesses (`type_i_witnesses`): at Type I
  points the null vector attaining the invariant null margin and the
  timelike observer measuring the invariant energy density follow
  directly from the eigenframe, in coordinate components, with no
  optimizer involved.
- Averaged null energy condition (`AveragedNullEnergy` in
  `warpfactory.physics`): the ANEC integral along axial null geodesics
  of the 1-D sampled metric, with the affine scale recovered from the
  conserved Killing energy of the stationary slice.

The older 1-D axial-slice API (`warpfactory.metrics`,
`warpfactory.solver`) remains available and unchanged.
