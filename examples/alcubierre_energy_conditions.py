"""Alcubierre stress-energy and energy conditions in SI units.

Reproduces the workflow of Section 4.1 of "Analyzing Warp Drive
Spacetimes with Warp Factory" (arXiv 2404.03095): build the Alcubierre
metric with the paper parameters (v = 0.1c, R = 300 m,
sigma = 0.015 1/m), solve the Einstein field equations, transform to
the Eulerian frame, and map the Null and Weak energy-condition
violations. Grid resolution is reduced from the paper's 1 m spacing so
the script runs in under a minute.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
V, R, SIGMA = 0.1, 300.0, 0.015


def main():
    metric = alcubierre_metric(
        GRID_SIZE,
        WORLD_CENTER,
        v=V,
        R=R,
        sigma=SIGMA,
        grid_scale=(1, SPACING, SPACING, SPACING),
    )
    stress_energy = GridSolver(order=4).solve(metric)
    eulerian = do_frame_transfer(metric, stress_energy)
    rho_si = stress_energy_to_si(eulerian).tensor[0, 0, 0]

    print(f"Alcubierre v={V}c R={R} m sigma={SIGMA} 1/m on {N}^3 grid at {SPACING} m")
    print(
        f"peak Eulerian energy density: {rho_si.min():.3e} J/m^3 "
        "(paper Figure 2: ~ -6.8e35)"
    )

    null_map = get_energy_conditions(stress_energy, metric, "Null", num_angular_vec=50)
    weak_map = get_energy_conditions(
        stress_energy, metric, "Weak", num_angular_vec=50, num_time_vec=8
    )
    print(f"NEC most-violating value: {null_map.min():.3e} 1/m^2")
    print(f"WEC most-violating value: {weak_map.min():.3e} 1/m^2")

    mid = N // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (data, title) in zip(
        axes,
        [
            (rho_si, "Eulerian energy density [J/m^3]"),
            (null_map[0], "NEC map [1/m^2]"),
            (weak_map[0], "WEC map [1/m^2]"),
        ],
    ):
        im = ax.imshow(
            data[:, :, mid].T,
            origin="lower",
            cmap="RdBu",
            extent=[0, N * SPACING, 0, N * SPACING],
        )
        ax.set(title=title, xlabel="x [m]", ylabel="y [m]")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(__file__).with_name("alcubierre_energy_conditions.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
