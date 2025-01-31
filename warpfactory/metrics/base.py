"""Base metric class."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class Metric(ABC):
    """Abstract base class for metrics."""
    
    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 t: float, **kwargs) -> Dict[str, np.ndarray]:
        """Calculate metric components.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Spatial coordinates
        t : float
            Time coordinate
        **kwargs : dict
            Additional metric parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Metric components
        """
        pass