import numpy as np
from .constants import Constants

class UnitSystem:
    """Transform between different unit systems."""
    
    def __init__(self):
        self.const = Constants()
    
    def to_geometric_units(self, quantity_type: str, value: float) -> float:
        """Convert SI units to geometric units (G = c = 1).
        
        Parameters
        ----------
        quantity_type : str
            Type of quantity ('mass', 'length', 'time', etc.)
        value : float
            Value in SI units
            
        Returns
        -------
        float
            Value in geometric units
        """
        if quantity_type == "mass":
            return value * self.const.G / self.const.c**2
        elif quantity_type == "length":
            return value
        elif quantity_type == "time":
            return value * self.const.c
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
    
    def from_geometric_units(self, quantity_type: str, value: float) -> float:
        """Convert geometric units (G = c = 1) to SI units.
        
        Parameters
        ----------
        quantity_type : str
            Type of quantity ('mass', 'length', 'time', etc.)
        value : float
            Value in geometric units
            
        Returns
        -------
        float
            Value in SI units
        """
        if quantity_type == "mass":
            return value * self.const.c**2 / self.const.G
        elif quantity_type == "length":
            return value
        elif quantity_type == "time":
            return value / self.const.c
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
    
    def to_natural_units(self, quantity_type: str, value: float) -> float:
        """Convert SI units to natural units (ℏ = c = 1).
        
        Parameters
        ----------
        quantity_type : str
            Type of quantity ('energy', 'mass', 'length', etc.)
        value : float
            Value in SI units
            
        Returns
        -------
        float
            Value in natural units
        """
        hbar = self.const.h / (2 * np.pi)
        
        if quantity_type == "energy":
            return value / (hbar * self.const.c)
        elif quantity_type == "mass":
            return value * self.const.c**2 / (hbar * self.const.c)
        elif quantity_type == "length":
            return value / (hbar / (self.const.c))
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")
    
    def from_natural_units(self, quantity_type: str, value: float) -> float:
        """Convert natural units (ℏ = c = 1) to SI units.
        
        Parameters
        ----------
        quantity_type : str
            Type of quantity ('energy', 'mass', 'length', etc.)
        value : float
            Value in natural units
            
        Returns
        -------
        float
            Value in SI units
        """
        hbar = self.const.h / (2 * np.pi)
        
        if quantity_type == "energy":
            return value * (hbar * self.const.c)
        elif quantity_type == "mass":
            return value * (hbar * self.const.c) / self.const.c**2
        elif quantity_type == "length":
            return value * (hbar / self.const.c)
        else:
            raise ValueError(f"Unknown quantity type: {quantity_type}")