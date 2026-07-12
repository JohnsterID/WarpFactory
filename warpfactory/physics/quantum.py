"""Quantum effect calculations."""

from __future__ import annotations

from typing import Dict

import numpy as np

# torch is an optional heavy backend: only the *_batch methods consume
# torch.Tensor inputs, and they use plain arithmetic that works on any
# tensor passed in. Guard the import so the core numpy/scipy pipeline
# (hawking_temperature, vacuum_polarization, ...) works without torch.
try:
    import torch
except ImportError:
    torch = None


class QuantumEffects:
    """Calculate quantum effects in curved spacetime."""

    def __init__(self, device: str = "cpu"):
        """Initialize calculator.

        Parameters
        ----------
        device : str
            Device for computations ("cpu" or "cuda")
        """
        self.device = device
        self.hbar = 1.0545718e-34  # Reduced Planck constant
        self.c = 299792458.0  # Speed of light
        self.G = 6.67430e-11  # Gravitational constant
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.sigma_sb = 5.670374419e-8  # Stefan-Boltzmann constant

    def hawking_temperature(self, surface_gravity: float) -> float:
        """Calculate Hawking-like temperature.

        Parameters
        ----------
        surface_gravity : float
            Surface gravity at horizon

        Returns
        -------
        float
            Temperature in Kelvin
        """
        return surface_gravity * self.hbar / (2 * np.pi * self.c * self.k_B)

    def particle_production_rate(
        self, surface_gravity: float, bubble_radius: float
    ) -> float:
        """Calculate particle production rate.

        Parameters
        ----------
        surface_gravity : float
            Surface gravity at horizon
        bubble_radius : float
            Radius of warp bubble

        Returns
        -------
        float
            Production rate (particles/second)
        """
        T = self.hawking_temperature(surface_gravity)
        if T <= 0:
            return 0.0
        area = 4 * np.pi * bubble_radius**2

        # Radiated power (Stefan-Boltzmann) divided by the mean photon
        # energy of a thermal spectrum, <E> ~= 2.7 k_B T
        power = self.sigma_sb * area * T**4
        return power / (2.7 * self.k_B * T)

    def vacuum_polarization(
        self, surface_gravity: float, bubble_radius: float
    ) -> Dict[str, float]:
        """Calculate vacuum polarization effects.

        Parameters
        ----------
        surface_gravity : float
            Surface gravity at horizon
        bubble_radius : float
            Radius of warp bubble

        Returns
        -------
        Dict[str, float]
            Vacuum polarization effects
        """
        T = self.hawking_temperature(surface_gravity)

        # Thermal radiation energy density: a T^4 with
        # a = pi^2 k_B^4 / (15 hbar^3 c^3)
        energy_density = np.pi**2 * self.k_B**4 * T**4 / (15 * self.hbar**3 * self.c**3)

        # Radiation pressure
        pressure = energy_density / 3

        return {"energy_density": energy_density, "pressure": pressure}

    def estimate_backreaction(
        self, surface_gravity: float, bubble_radius: float
    ) -> Dict[str, float]:
        """Estimate quantum backreaction effects.

        Parameters
        ----------
        surface_gravity : float
            Surface gravity at horizon
        bubble_radius : float
            Radius of warp bubble

        Returns
        -------
        Dict[str, float]
            Backreaction estimates
        """
        effects = self.vacuum_polarization(surface_gravity, bubble_radius)

        # Dimensionless metric perturbation h ~ (G/c^4) rho R^2
        correction = (self.G / self.c**4) * effects["energy_density"] * bubble_radius**2

        # Lifetime = thermal energy content / radiated power
        T = self.hawking_temperature(surface_gravity)
        volume = 4 / 3 * np.pi * bubble_radius**3
        area = 4 * np.pi * bubble_radius**2
        power = self.sigma_sb * area * T**4
        lifetime = effects["energy_density"] * volume / power if power > 0 else np.inf

        return {"metric_correction": correction, "lifetime": lifetime}

    def hawking_temperature_batch(self, surface_gravity: torch.Tensor) -> torch.Tensor:
        """Batch calculate Hawking temperature.

        Parameters
        ----------
        surface_gravity : torch.Tensor
            Surface gravity values

        Returns
        -------
        torch.Tensor
            Temperature values
        """
        return surface_gravity * self.hbar / (2 * np.pi * self.c * self.k_B)

    def particle_production_batch(
        self, surface_gravity: torch.Tensor, bubble_radius: torch.Tensor
    ) -> torch.Tensor:
        """Batch calculate particle production rates.

        Parameters
        ----------
        surface_gravity : torch.Tensor
            Surface gravity values
        bubble_radius : torch.Tensor
            Bubble radius values

        Returns
        -------
        torch.Tensor
            Production rate values
        """
        T = self.hawking_temperature_batch(surface_gravity)
        area = 4 * np.pi * bubble_radius**2

        power = self.sigma_sb * area * T**4
        return power / (2.7 * self.k_B * T)

    def vacuum_polarization_batch(
        self, surface_gravity: torch.Tensor, bubble_radius: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Batch calculate vacuum polarization.

        Parameters
        ----------
        surface_gravity : torch.Tensor
            Surface gravity values
        bubble_radius : torch.Tensor
            Bubble radius values

        Returns
        -------
        Dict[str, torch.Tensor]
            Vacuum polarization effects
        """
        T = self.hawking_temperature_batch(surface_gravity)

        # Vacuum energy density
        energy_density = np.pi**2 * self.k_B**4 * T**4 / (15 * self.hbar**3 * self.c**3)

        # Radiation pressure
        pressure = energy_density / 3

        return {"energy_density": energy_density, "pressure": pressure}
