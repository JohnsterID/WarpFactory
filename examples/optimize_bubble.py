"""Parameter scan and exotic-matter minimization for an Alcubierre ansatz.

Wraps the Alcubierre metric builder in a CallableAnsatz, sweeps the
bubble-wall thickness, and then runs a Nelder-Mead search over velocity
showing that the minimizer drives toward flat spacetime (exotic matter
scales as v^2).
"""

from functools import partial

from warpfactory.grid import alcubierre_metric
from warpfactory.optimize import (
    CallableAnsatz,
    minimize_exotic_matter,
    scan_parameters,
)

N, SPACING = 24, 0.5
GRID_SIZE = (1, N, N, N)
WORLD_CENTER = (0.0,) + tuple((n - 1)*SPACING/2 for n in GRID_SIZE[1:])


def main():
    builder = partial(alcubierre_metric, GRID_SIZE, WORLD_CENTER,
                      grid_scale=(1, SPACING, SPACING, SPACING))
    ansatz = CallableAnsatz(builder, ["v", "R", "sigma"],
                            name="alcubierre")

    print("== sigma scan (v = 0.5, R = 3.0) ==")
    results = scan_parameters(ansatz, {"sigma": [1.0, 2.0, 4.0]},
                              fixed_params={"v": 0.5, "R": 3.0},
                              num_angular_vec=15, num_time_vec=3)
    for r in results:
        print(f"sigma={r.params['sigma']:<4} exotic={r.exotic_matter:.4f} "
              f"total={r.total_energy:+.4f} NEC_ok={r.null_ok} "
              f"WEC_ok={r.weak_ok}")

    print("\n== minimize exotic matter over v (R = 3.0, sigma = 2.0) ==")
    opt = minimize_exotic_matter(
        ansatz, {"v": 0.5}, fixed_params={"R": 3.0, "sigma": 2.0},
        max_iterations=25, num_angular_vec=15, num_time_vec=3,
        callback=lambda p, val: print(f"  v={p['v']:+.4f} -> {val:.6f}"))
    print(f"best: v={opt.best_params['v']:+.6f} "
          f"exotic={opt.fun:.2e} (flat spacetime, as expected)")


if __name__ == "__main__":
    main()
