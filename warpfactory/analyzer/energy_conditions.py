import numpy as np
from typing import Dict

class EnergyConditions:
    """Check various energy conditions for stress-energy tensor."""
    
    def check_weak(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Check weak energy condition: T_μν t^μ t^ν ≥ 0 for timelike t^μ.
        
        Parameters
        ----------
        T_munu : Dict[str, np.ndarray]
            Stress-energy tensor components
            
        Returns
        -------
        bool
            True if weak energy condition is satisfied
        """
        # For perfect fluid: ρ ≥ 0
        return np.all(T_munu["T_tt"] >= 0)
    
    def check_null(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Check null energy condition: T_μν k^μ k^ν ≥ 0 for null k^μ.
        
        Parameters
        ----------
        T_munu : Dict[str, np.ndarray]
            Stress-energy tensor components
            
        Returns
        -------
        bool
            True if null energy condition is satisfied
        """
        # For perfect fluid: ρ + p ≥ 0
        return np.all(T_munu["T_tt"] + T_munu["T_xx"] >= 0)
    
    def check_strong(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Check strong energy condition: (T_μν - 1/2 T g_μν) t^μ t^ν ≥ 0.
        
        Parameters
        ----------
        T_munu : Dict[str, np.ndarray]
            Stress-energy tensor components
            
        Returns
        -------
        bool
            True if strong energy condition is satisfied
        """
        # For perfect fluid: ρ + 3p ≥ 0
        return np.all(T_munu["T_tt"] + 3*T_munu["T_xx"] >= 0)
    
    def check_dominant(self, T_munu: Dict[str, np.ndarray]) -> bool:
        """Check dominant energy condition: -T^μ_ν t^ν is future-directed timelike/null.
        
        Parameters
        ----------
        T_munu : Dict[str, np.ndarray]
            Stress-energy tensor components
            
        Returns
        -------
        bool
            True if dominant energy condition is satisfied
        """
        # For perfect fluid: ρ ≥ |p|
        return np.all(T_munu["T_tt"] >= np.abs(T_munu["T_xx"]))