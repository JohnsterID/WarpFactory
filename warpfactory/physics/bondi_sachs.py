"""Bondi-Sachs energy and momentum flux in the quadrupole approximation.

Gravitational-wave energy and linear-momentum flux radiated to null
infinity by a slow-motion source, from the standard multipole expansion
of the Bondi-Sachs news (Thorne 1980, Rev. Mod. Phys. 52, 299):

    dE/dt  = (1/5) <d3I_ij d3I_ij>
    dP_i/dt = (2/63) <d4I_ijk d3I_jk> + (16/45) <eps_ijk d3I_jl d3J_kl>

with I_ij the symmetric trace-free (STF) mass quadrupole, I_ijk the STF
mass octupole, and J_ij the current quadrupole. Momentum is radiated
only through beating between multipoles of opposite parity, so both the
octupole and current-quadrupole couplings are kept -- this reproduces
the Fitchett (1983) momentum flux of an unequal-mass binary.

This linearized treatment is the tractable slice of the full
Bondi-Sachs formalism: it needs only source worldlines (an effective
point mass on the bubble trajectory, or several masses), not
null-infinity metric extraction. A bubble moving at constant velocity
has a static quadrupole in its rest frame -- all time derivatives of
the multipoles vanish and both fluxes are exactly zero; only
accelerating drives radiate. Geometric units G = c = 1.

Time derivatives of the multipoles are taken with second-order central
differences (np.gradient), so fluxes carry O(dt^2) discretization
error; edge samples use one-sided stencils and should be discarded by
the caller when accuracy matters.
"""

from typing import Dict

import numpy as np

_DELTA = np.eye(3)
_EPSILON = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPSILON[_i, _j, _k] = 1.0
    _EPSILON[_k, _j, _i] = -1.0


class BondiSachsFlux:
    """Quadrupole-order radiated energy and momentum flux."""

    @staticmethod
    def _validate(masses: np.ndarray, positions: np.ndarray) -> None:
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                "positions must have shape (n_times, n_bodies, 3), got "
                f"{positions.shape}"
            )
        if masses.shape != (positions.shape[1],):
            raise ValueError(
                f"masses shape {masses.shape} does not match "
                f"{positions.shape[1]} bodies"
            )

    @staticmethod
    def mass_quadrupole(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """STF mass quadrupole I_ij = sum m (x_i x_j - r^2 delta_ij / 3).

        Parameters
        ----------
        masses : np.ndarray
            Body masses, shape (n_bodies,)
        positions : np.ndarray
            Worldlines, shape (n_times, n_bodies, 3)

        Returns
        -------
        np.ndarray
            Shape (n_times, 3, 3)
        """
        xx = np.einsum("tai,taj->taij", positions, positions)
        r2 = np.einsum("tai,tai->ta", positions, positions)
        stf = xx - r2[..., None, None] * _DELTA / 3
        return np.einsum("a,taij->tij", masses, stf)

    @staticmethod
    def mass_octupole(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """STF mass octupole
        I_ijk = sum m (x_i x_j x_k - r^2 (x_i d_jk + x_j d_ik + x_k d_ij)/5).

        Returns
        -------
        np.ndarray
            Shape (n_times, 3, 3, 3)
        """
        xxx = np.einsum("tai,taj,tak->taijk", positions, positions, positions)
        r2 = np.einsum("tai,tai->ta", positions, positions)
        sym = (
            np.einsum("tai,jk->taijk", positions, _DELTA)
            + np.einsum("taj,ik->taijk", positions, _DELTA)
            + np.einsum("tak,ij->taijk", positions, _DELTA)
        )
        stf = xxx - r2[..., None, None, None] * sym / 5
        return np.einsum("a,taijk->tijk", masses, stf)

    @staticmethod
    def current_quadrupole(
        masses: np.ndarray, positions: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        """Current quadrupole J_ij = sum m [(x cross v)_(i x_j)] (symmetrized).

        Returns
        -------
        np.ndarray
            Shape (n_times, 3, 3)
        """
        angular = np.cross(positions, velocities)
        lx = np.einsum("tai,taj->taij", angular, positions)
        return np.einsum("a,taij->tij", masses, 0.5 * (lx + lx.swapaxes(2, 3)))

    def fluxes(
        self, times: np.ndarray, masses: np.ndarray, positions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Radiated energy and momentum flux along the source evolution.

        Parameters
        ----------
        times : np.ndarray
            Sample times, shape (n_times,); need not be uniform
        masses : np.ndarray
            Body masses, shape (n_bodies,)
        positions : np.ndarray
            Worldlines, shape (n_times, n_bodies, 3)

        Returns
        -------
        Dict[str, np.ndarray]
            "energy_flux" : dE/dt, shape (n_times,), always >= 0 up to
            discretization error
            "momentum_flux" : dP_i/dt, shape (n_times, 3); the source
            recoils opposite to this
        """
        times = np.asarray(times, dtype=float)
        masses = np.asarray(masses, dtype=float)
        positions = np.asarray(positions, dtype=float)
        self._validate(masses, positions)
        if times.shape != (positions.shape[0],):
            raise ValueError(
                f"times shape {times.shape} does not match {positions.shape[0]} samples"
            )

        def d_dt(f: np.ndarray) -> np.ndarray:
            return np.gradient(f, times, axis=0)

        velocities = d_dt(positions)
        quad3 = d_dt(d_dt(d_dt(self.mass_quadrupole(masses, positions))))
        oct4 = d_dt(d_dt(d_dt(d_dt(self.mass_octupole(masses, positions)))))
        curr3 = d_dt(d_dt(d_dt(self.current_quadrupole(masses, positions, velocities))))

        energy_flux = np.einsum("tij,tij->t", quad3, quad3) / 5
        momentum_flux = (2 / 63) * np.einsum("tijk,tjk->ti", oct4, quad3) + (
            16 / 45
        ) * np.einsum("ijk,tjl,tkl->ti", _EPSILON, quad3, curr3)
        return {"energy_flux": energy_flux, "momentum_flux": momentum_flux}

    def trajectory_fluxes(
        self, times: np.ndarray, mass: float, trajectory: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Fluxes from a single effective mass on a trajectory.

        Models a warp bubble as a point source of effective mass
        (energy) following the drive trajectory: constant-velocity
        motion radiates nothing, accelerating drives lose energy and
        momentum to gravitational waves.

        Parameters
        ----------
        times : np.ndarray
            Sample times, shape (n_times,)
        mass : float
            Effective source mass in geometric units
        trajectory : np.ndarray
            Positions, shape (n_times, 3)

        Returns
        -------
        Dict[str, np.ndarray]
            Same as fluxes()
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError(
                f"trajectory must have shape (n_times, 3), got {trajectory.shape}"
            )
        return self.fluxes(times, np.array([mass]), trajectory[:, None, :])
