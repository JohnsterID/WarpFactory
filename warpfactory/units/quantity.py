import numpy as np
from typing import Dict, Tuple
from .converter import UnitConverter

class Quantity:
    """Physical quantity with units."""
    
    # Unit conversion factors to SI
    _UNIT_FACTORS = {
        "m": 1.0,
        "km": 1000.0,
        "s": 1.0,
        "hour": 3600.0,
        "kg": 1.0,
        "solar_mass": 1.989e30,
    }
    
    # Unit dimensions
    _UNIT_DIMENSIONS = {
        "m": {"length": 1},
        "km": {"length": 1},
        "s": {"time": 1},
        "hour": {"time": 1},
        "kg": {"mass": 1},
        "solar_mass": {"mass": 1},
    }
    
    def __init__(self, value: float, unit: str):
        """Initialize quantity.
        
        Parameters
        ----------
        value : float
            Numerical value
        unit : str
            Unit string (e.g., 'm', 'km/s', etc.)
        """
        self.value = value
        self.unit = unit
        self._parse_unit(unit)
    
    def _parse_unit(self, unit: str) -> None:
        """Parse unit string into numerator and denominator."""
        if "/" in unit:
            num, denom = unit.split("/")
            self._numerator = num
            self._denominator = denom
            # Add compound unit dimensions
            self._UNIT_DIMENSIONS[unit] = {
                **self._UNIT_DIMENSIONS[num],
                **{k: -v for k, v in self._UNIT_DIMENSIONS[denom].items()}
            }
            # Add compound unit conversion factor
            self._UNIT_FACTORS[unit] = self._UNIT_FACTORS[num] / self._UNIT_FACTORS[denom]
        else:
            self._numerator = unit
            self._denominator = None
    
    def to(self, new_unit: str) -> 'Quantity':
        """Convert to new unit.
        
        Parameters
        ----------
        new_unit : str
            Target unit
            
        Returns
        -------
        Quantity
            New quantity in target unit
        """
        # Check unit compatibility
        if not self._is_compatible(new_unit):
            raise ValueError(f"Incompatible units: {self.unit} and {new_unit}")
        
        # Convert value
        old_factor = self._UNIT_FACTORS[self.unit]
        new_factor = self._UNIT_FACTORS[new_unit]
        new_value = self.value * old_factor / new_factor
        
        return Quantity(new_value, new_unit)
    
    def _is_compatible(self, other_unit: str) -> bool:
        """Check if units have same dimensions."""
        # Parse other unit if it's a compound unit
        if "/" in other_unit and other_unit not in self._UNIT_DIMENSIONS:
            num, denom = other_unit.split("/")
            if num not in self._UNIT_DIMENSIONS or denom not in self._UNIT_DIMENSIONS:
                return False
            # Add compound unit dimensions
            self._UNIT_DIMENSIONS[other_unit] = {
                **self._UNIT_DIMENSIONS[num],
                **{k: -v for k, v in self._UNIT_DIMENSIONS[denom].items()}
            }
            # Add compound unit conversion factor
            self._UNIT_FACTORS[other_unit] = self._UNIT_FACTORS[num] / self._UNIT_FACTORS[denom]
        
        if other_unit not in self._UNIT_DIMENSIONS:
            return False
        
        # Compare dimensions
        this_dims = self._UNIT_DIMENSIONS[self.unit]
        other_dims = self._UNIT_DIMENSIONS[other_unit]
        
        # Check if both units have the same dimensions with same powers
        return all(
            this_dims.get(dim, 0) == other_dims.get(dim, 0)
            for dim in set(this_dims) | set(other_dims)
        )
    
    def __add__(self, other: 'Quantity') -> 'Quantity':
        """Add quantities with compatible units."""
        if not self._is_compatible(other.unit):
            raise ValueError(f"Cannot add quantities with units {self.unit} and {other.unit}")
        
        # Convert other to this unit
        other_converted = other.to(self.unit)
        return Quantity(self.value + other_converted.value, self.unit)
    
    def __truediv__(self, other: 'Quantity') -> 'Quantity':
        """Divide quantities."""
        if isinstance(other, Quantity):
            # Create new unit string
            if other._denominator:
                raise ValueError("Cannot divide by quantity with compound unit")
            new_unit = f"{self.unit}/{other.unit}"
            
            # Calculate new value
            return Quantity(self.value / other.value, new_unit)
        else:
            return Quantity(self.value / other, self.unit)