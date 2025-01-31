import pytest
import numpy as np
from warpfactory.units import (
    Constants,
    UnitConverter,
    UnitSystem,
    Quantity,
)

def test_physical_constants():
    """Test physical constants and their conversions."""
    const = Constants()
    
    # Test speed of light
    assert const.c == 299792458  # m/s
    
    # Test gravitational constant
    assert np.isclose(const.G, 6.67430e-11)  # m³/kg/s²
    
    # Test Planck constant
    assert np.isclose(const.h, 6.62607015e-34)  # J⋅s
    
    # Test derived constants
    assert np.isclose(const.planck_length, np.sqrt(const.h * const.G / const.c**3))
    assert np.isclose(const.planck_mass, np.sqrt(const.h * const.c / const.G))
    assert np.isclose(const.planck_time, np.sqrt(const.h * const.G / const.c**5))

def test_unit_converter():
    """Test unit conversions."""
    conv = UnitConverter()
    
    # Test length conversions
    assert np.isclose(conv.meters_to_km(1000), 1.0)
    assert np.isclose(conv.km_to_meters(1.0), 1000)
    assert np.isclose(conv.meters_to_light_years(9.461e15), 1.0, rtol=1e-3)
    
    # Test mass conversions
    assert np.isclose(conv.kg_to_solar_masses(1.989e30), 1.0, rtol=1e-3)
    assert np.isclose(conv.solar_masses_to_kg(1.0), 1.989e30, rtol=1e-3)
    
    # Test time conversions
    assert np.isclose(conv.seconds_to_years(31557600), 1.0)
    assert np.isclose(conv.years_to_seconds(1.0), 31557600)

def test_unit_system(physical_constants):
    """Test unit system transformations."""
    units = UnitSystem()
    const = physical_constants
    
    # Test geometric units (G = c = 1)
    # Test mass conversion
    mass_kg = 1.989e30  # 1 solar mass in kg
    mass_geometric = units.to_geometric_units("mass", mass_kg)
    mass_back = units.from_geometric_units("mass", mass_geometric)
    assert np.isclose(mass_back, mass_kg, rtol=1e-10)
    
    # Test length conversion
    length_m = 1.0e3  # 1 km in meters
    length_geometric = units.to_geometric_units("length", length_m)
    length_back = units.from_geometric_units("length", length_geometric)
    assert np.isclose(length_back, length_m, rtol=1e-10)
    
    # Test time conversion
    time_s = 3600.0  # 1 hour in seconds
    time_geometric = units.to_geometric_units("time", time_s)
    time_back = units.from_geometric_units("time", time_geometric)
    assert np.isclose(time_back, time_s, rtol=1e-10)
    
    # Test natural units (ℏ = c = 1)
    # Test energy conversion
    energy_joules = 1.0e6  # 1 MeV in Joules
    energy_natural = units.to_natural_units("energy", energy_joules)
    energy_back = units.from_natural_units("energy", energy_natural)
    assert np.isclose(energy_back, energy_joules, rtol=1e-10)
    
    # Test mass conversion in natural units
    mass_kg = 1.0e-27  # Proton mass
    mass_natural = units.to_natural_units("mass", mass_kg)
    mass_back = units.from_natural_units("mass", mass_natural)
    assert np.isclose(mass_back, mass_kg, rtol=1e-10)
    
    # Test length conversion in natural units
    length_m = const.h / (const.c * 1.0e6)  # Compton wavelength
    length_natural = units.to_natural_units("length", length_m)
    length_back = units.from_natural_units("length", length_natural)
    assert np.isclose(length_back, length_m, rtol=1e-10)
    
    # Test invalid quantity types
    with pytest.raises(ValueError):
        units.to_geometric_units("invalid", 1.0)
    with pytest.raises(ValueError):
        units.from_geometric_units("invalid", 1.0)
    with pytest.raises(ValueError):
        units.to_natural_units("invalid", 1.0)
    with pytest.raises(ValueError):
        units.from_natural_units("invalid", 1.0)

def test_quantity():
    """Test quantity class with units."""
    # Create quantities
    length = Quantity(1.0, "km")
    time = Quantity(1.0, "hour")
    mass = Quantity(1.0, "solar_mass")
    
    # Test conversions
    assert np.isclose(length.to("m").value, 1000.0)
    assert np.isclose(time.to("s").value, 3600.0)
    assert np.isclose(mass.to("kg").value, 1.989e30, rtol=1e-3)
    
    # Test arithmetic
    velocity = length / time
    assert velocity.unit == "km/hour"
    assert np.isclose(velocity.to("m/s").value, 0.277778, rtol=1e-5)
    
    # Test dimensionality checks
    with pytest.raises(ValueError):
        length + time  # Can't add length and time
    
    # Test unit compatibility
    length2 = Quantity(1000.0, "m")
    total_length = length + length2
    assert total_length.unit == "km"
    assert np.isclose(total_length.value, 2.0)