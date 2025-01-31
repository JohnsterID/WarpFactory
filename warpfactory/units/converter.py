import numpy as np
from .constants import Constants

class UnitConverter:
    """Convert between different physical units."""
    
    def __init__(self):
        self.const = Constants()
    
    def meters_to_km(self, meters: float) -> float:
        """Convert meters to kilometers."""
        return meters / 1000.0
    
    def km_to_meters(self, km: float) -> float:
        """Convert kilometers to meters."""
        return km * 1000.0
    
    def meters_to_light_years(self, meters: float) -> float:
        """Convert meters to light years."""
        return meters / (self.const.c * 31557600)  # c * (seconds in a year)
    
    def kg_to_solar_masses(self, kg: float) -> float:
        """Convert kilograms to solar masses."""
        return kg / 1.989e30
    
    def solar_masses_to_kg(self, solar_masses: float) -> float:
        """Convert solar masses to kilograms."""
        return solar_masses * 1.989e30
    
    def seconds_to_years(self, seconds: float) -> float:
        """Convert seconds to years."""
        return seconds / 31557600  # seconds in a year
    
    def years_to_seconds(self, years: float) -> float:
        """Convert years to seconds."""
        return years * 31557600  # seconds in a year