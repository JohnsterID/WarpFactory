"""Kinematic scalars of the Alcubierre bubble.

Computes the expansion, shear, and vorticity of the Eulerian observer
congruence (the getScalars.m workflow). The expansion shows the
signature volume contraction ahead of the bubble and expansion behind
it; the vorticity vanishes because the Eulerian congruence is
hypersurface-orthogonal.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from warpfactory.grid import alcubierre_metric, get_scalars

N, SPACING = 64, 0.25
GRID_SIZE = (1, N, N, N)
WORLD_CENTER = (0.0,) + tuple((n - 1)*SPACING/2 for n in GRID_SIZE[1:])
V, R, SIGMA = 2.0, 3.0, 2.0


def main():
    metric = alcubierre_metric(GRID_SIZE, WORLD_CENTER, v=V, R=R,
                               sigma=SIGMA,
                               grid_scale=(1, SPACING, SPACING, SPACING))
    expansion, shear, vorticity = get_scalars(metric, order=4)

    print(f"Alcubierre v={V} R={R} sigma={SIGMA}")
    print(f"expansion range: [{expansion.min():.4f}, {expansion.max():.4f}]")
    print(f"peak shear sigma^2: {shear.max():.4f}")
    print(f"max |vorticity|: {np.abs(vorticity).max():.2e} "
          "(zero: congruence is hypersurface-orthogonal)")

    mid = N//2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (data, title) in zip(axes, [
            (expansion[0], "Expansion theta"),
            (shear[0], "Shear sigma^2")]):
        im = ax.imshow(data[:, :, mid].T, origin="lower", cmap="RdBu",
                       extent=[0, N*SPACING, 0, N*SPACING])
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path(__file__).with_name("metric_scalars.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
