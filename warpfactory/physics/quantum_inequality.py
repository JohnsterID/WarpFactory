"""Ford-Roman quantum inequality evaluator.

Implements the quantum inequality (QI) for a free massless scalar field
(Ford & Roman 1995) and its application to the Alcubierre warp drive by
Pfenning & Ford, "The unphysical nature of Warp Drive" (arXiv
gr-qc/9702026, CQG 14 (1997) 1743):

    (tau0/pi) Integral rho(tau) / (tau^2 + tau0^2) dtau
        >= -3 / (32 pi^2 tau0^4)        [hbar = c = 1]

where rho is the energy density along an inertial observer's worldline,
tau proper time, and tau0 the Lorentzian sampling time. The warp-drive
application yields the bubble wall thickness bound (their Eq. 20)

    Delta <= (3/4) sqrt(3/pi) v_b / alpha^2   [Planck lengths]

and the total-energy estimate (their Eq. 24)

    E = -(1/12) v_b^2 (R^2/Delta + Delta/12)  [geometric units, meters].

All public methods take and return SI quantities (seconds, meters,
J/m^3, J) except the dimensionless bubble speed v_b, which is in units
of c. The Planck length used by the paper is the reduced one,
sqrt(hbar G / c^3), derived here from the shared Constants (which
stores h, not hbar).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..grid.si_units import si_energy_factor
from ..units import Constants


class FordRomanInequality:
    """Quantum inequality checks for sampled energy densities and warp bubbles."""

    def __init__(self) -> None:
        const = Constants()
        self.c = const.c
        self.G = const.G
        self.hbar = const.h / (2 * np.pi)
        self.planck_length = np.sqrt(self.hbar * self.G / self.c**3)

    def sampling_bound(self, tau0: float) -> float:
        """Lower bound on the Lorentzian-sampled energy density, in J/m^3.

        -3 hbar / (32 pi^2 c^3 tau0^4), the SI form of the Ford-Roman
        bound -3/(32 pi^2 tau0^4).

        Parameters
        ----------
        tau0 : float
            Sampling time in seconds
        """
        if tau0 <= 0:
            raise ValueError("sampling time tau0 must be positive")
        return -3 * self.hbar / (32 * np.pi**2 * self.c**3 * tau0**4)

    def sampled_energy_density(
        self, rho: np.ndarray, tau: np.ndarray, tau0: float
    ) -> float:
        """Lorentzian-sampled average (tau0/pi) Integral rho/(tau^2+tau0^2) dtau.

        Parameters
        ----------
        rho : np.ndarray
            Energy density along the worldline in J/m^3
        tau : np.ndarray
            Proper time samples in seconds; must span several tau0 on
            both sides of the pulse for the sampling weight to be
            captured
        tau0 : float
            Sampling time in seconds

        Returns
        -------
        float
            Weighted average energy density in J/m^3
        """
        if tau0 <= 0:
            raise ValueError("sampling time tau0 must be positive")
        rho = np.asarray(rho, dtype=float)
        tau = np.asarray(tau, dtype=float)
        integrand = rho * tau0 / (np.pi * (tau**2 + tau0**2))
        # np.trapezoid needs numpy >= 2.0 and np.trapz is deprecated
        # there; the explicit trapezoid sum supports both.
        return float(np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(tau)))

    def check_sampled(
        self, rho: np.ndarray, tau: np.ndarray, tau0: float
    ) -> Dict[str, float]:
        """Evaluate the inequality for a sampled worldline energy density.

        Returns
        -------
        Dict[str, float]
            "sampled": Lorentzian-weighted average (J/m^3),
            "bound": Ford-Roman lower bound (J/m^3),
            "satisfied": whether sampled >= bound
        """
        sampled = self.sampled_energy_density(rho, tau, tau0)
        bound = self.sampling_bound(tau0)
        return {"sampled": sampled, "bound": bound, "satisfied": sampled >= bound}

    def wall_thickness(self, R: float, sigma: float) -> float:
        """Bubble wall thickness Delta of the Alcubierre shape function.

        Pfenning-Ford Eq. 8: the piecewise-linear wall whose slope at
        r = R matches the tanh shape function,

            Delta = (1 + tanh^2(sigma R))^2 / (2 sigma tanh(sigma R)),

        approaching 2/sigma for large sigma R.

        Parameters
        ----------
        R : float
            Bubble radius in meters
        sigma : float
            Alcubierre shape-function sharpness in 1/m
        """
        if R <= 0 or sigma <= 0:
            raise ValueError("bubble radius and sigma must be positive")
        t = np.tanh(sigma * R)
        return float((1 + t**2) ** 2 / (2 * sigma * t))

    def max_wall_thickness(self, v_b: float, alpha: float = 0.1) -> float:
        """Largest wall thickness allowed by the quantum inequality, meters.

        Pfenning-Ford Eq. 20: Delta <= (3/4) sqrt(3/pi) v_b / alpha^2
        Planck lengths, where alpha << 1 is the ratio of the sampling
        time to the minimal local radius of curvature (the paper
        quotes Delta <= 10^2 v_b L_Planck for alpha = 1/10).

        Parameters
        ----------
        v_b : float
            Bubble speed in units of c
        alpha : float
            Sampling-time fraction, 0 < alpha << 1
        """
        if v_b <= 0:
            raise ValueError("bubble speed must be positive")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        return float(0.75 * np.sqrt(3 / np.pi) * v_b / alpha**2 * self.planck_length)

    def total_energy(self, v_b: float, R: float, delta: float) -> float:
        """Total negative energy of the bubble wall, in Joules.

        Pfenning-Ford Eq. 24 for the piecewise-linear wall:
        E = -(1/12) v_b^2 (R^2/Delta + Delta/12) in geometric units of
        length, converted to Joules with c^4/G.

        Parameters
        ----------
        v_b : float
            Bubble speed in units of c
        R : float
            Bubble radius in meters
        delta : float
            Wall thickness in meters
        """
        if R <= 0 or delta <= 0:
            raise ValueError("bubble radius and wall thickness must be positive")
        energy_geometric = -(v_b**2) / 12 * (R**2 / delta + delta / 12)
        return energy_geometric * si_energy_factor()

    def check_warp_bubble(
        self,
        v_b: float,
        R: float,
        sigma: Optional[float] = None,
        delta: Optional[float] = None,
        alpha: float = 0.1,
    ) -> Dict[str, float]:
        """Quantum inequality verdict for an Alcubierre-type bubble.

        Provide either sigma (tanh sharpness, from which the effective
        wall thickness is derived) or delta (wall thickness directly).

        Returns
        -------
        Dict[str, float]
            "delta": wall thickness (m),
            "delta_max": QI upper bound on the thickness (m),
            "satisfied": whether delta <= delta_max,
            "total_energy": wall energy (J, negative)
        """
        if (sigma is None) == (delta is None):
            raise ValueError("provide exactly one of sigma or delta")
        if sigma is not None:
            delta = self.wall_thickness(R, sigma)
        assert delta is not None
        delta_max = self.max_wall_thickness(v_b, alpha)
        return {
            "delta": delta,
            "delta_max": delta_max,
            "satisfied": delta <= delta_max,
            "total_energy": self.total_energy(v_b, R, delta),
        }
