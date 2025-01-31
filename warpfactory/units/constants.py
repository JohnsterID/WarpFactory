import numpy as np

class Constants:
    """Physical constants in SI units."""
    
    def __init__(self):
        # Fundamental constants
        self._c = 299792458  # Speed of light in m/s
        self._G = 6.67430e-11  # Gravitational constant in m³/kg/s²
        self._h = 6.62607015e-34  # Planck constant in J⋅s
        
        # Derived constants
        self._planck_length = np.sqrt(self._h * self._G / self._c**3)
        self._planck_mass = np.sqrt(self._h * self._c / self._G)
        self._planck_time = np.sqrt(self._h * self._G / self._c**5)
    
    @property
    def c(self) -> float:
        """Speed of light in m/s."""
        return self._c
    
    @property
    def G(self) -> float:
        """Gravitational constant in m³/kg/s²."""
        return self._G
    
    @property
    def h(self) -> float:
        """Planck constant in J⋅s."""
        return self._h
    
    @property
    def planck_length(self) -> float:
        """Planck length in meters."""
        return self._planck_length
    
    @property
    def planck_mass(self) -> float:
        """Planck mass in kg."""
        return self._planck_mass
    
    @property
    def planck_time(self) -> float:
        """Planck time in seconds."""
        return self._planck_time