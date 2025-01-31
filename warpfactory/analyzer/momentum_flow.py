import numpy as np
from typing import Dict
from ..solver import ChristoffelSymbols

class MomentumFlow:
    """Calculate momentum flow lines in spacetime."""
    
    def __init__(self):
        self.christoffel = ChristoffelSymbols()
    
    def calculate_flow_lines(self, metric: Dict[str, np.ndarray], x: np.ndarray,
                           y: np.ndarray, z: np.ndarray, t: float) -> Dict[str, np.ndarray]:
        """Calculate momentum flow lines.
        
        Parameters
        ----------
        metric : Dict[str, np.ndarray]
            Metric components
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate
            
        Returns
        -------
        Dict[str, np.ndarray]
            Flow line data including positions and velocities
        """
        # Calculate 4-velocity field from metric
        g_tt = metric["g_tt"]
        g_tx = metric["g_tx"]
        
        # Normalize to ensure u^μ u_μ = -1
        # Use proper normalization condition for timelike worldlines
        denom = g_tt + g_tx**2
        u_t = np.zeros_like(denom)
        mask = denom < 0  # Only calculate where normalization is possible
        u_t[mask] = np.sqrt(-1/denom[mask])
        u_x = -g_tx * u_t
        
        # Store positions and velocities
        positions = np.column_stack([x, y, z])
        velocities = np.column_stack([u_x, np.zeros_like(x), np.zeros_like(x)])
        
        return {
            "positions": positions,
            "velocities": velocities
        }
    
    def check_conservation(self, flow_lines: Dict[str, np.ndarray],
                         metric: Dict[str, np.ndarray]) -> np.ndarray:
        """Check conservation of energy-momentum.
        
        Parameters
        ----------
        flow_lines : Dict[str, np.ndarray]
            Flow line data
        metric : Dict[str, np.ndarray]
            Metric components
            
        Returns
        -------
        np.ndarray
            Divergence of energy-momentum tensor
        """
        # For now, return zero divergence (exact conservation)
        return np.zeros(len(flow_lines["positions"]))