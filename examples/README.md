# WarpFactory Examples

Runnable scripts demonstrating the 4-D grid pipeline. Each script
prints its results and saves any figures next to itself.

| Script | Demonstrates |
|--------|--------------|
| `alcubierre_energy_conditions.py` | Metric -> stress-energy -> Eulerian frame -> NEC/WEC violation maps, with SI-unit output matching the WarpFactory paper (arXiv 2404.03095) Section 4.1 parameters |
| `metric_scalars.py` | Expansion, shear, and vorticity of the Eulerian congruence for the Alcubierre bubble |
| `optimize_bubble.py` | Parameter scan and derivative-free exotic-matter minimization over an Alcubierre ansatz |

Install the package first (`pip install .`) or run from the
repository root with `PYTHONPATH=.`:

```bash
python examples/alcubierre_energy_conditions.py
python examples/metric_scalars.py
python examples/optimize_bubble.py
```

Only numpy, scipy, and matplotlib are required.
