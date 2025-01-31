import numpy as np
from typing import Dict

class EnergyTensor:
    """Calculate energy-momentum tensor components."""
    
    def calculate_perfect_fluid(self, rho: np.ndarray, p: np.ndarray, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate perfect fluid stress-energy tensor.
        
        Parameters
        ----------
        rho : np.ndarray
            Energy density
        p : np.ndarray
            Pressure
        x : np.ndarray
            Spatial coordinate
            
        Returns
        -------
        Dict[str, np.ndarray]
            Energy-momentum tensor components T_μν
        """
        # Perfect fluid stress-energy tensor
        # T_tt = ρ (energy density)
        # T_ij = p δ_ij (isotropic pressure)
        return {
            "T_tt": rho,
            "T_xx": p,
            "T_yy": p,
            "T_zz": p,
            "T_tx": np.zeros_like(x),
            "T_ty": np.zeros_like(x),
            "T_tz": np.zeros_like(x)
        }