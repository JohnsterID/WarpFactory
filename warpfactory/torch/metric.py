"""PyTorch-accelerated metric calculations."""

import torch
from typing import Dict, Union

class TorchMetricSolver:
    """GPU-accelerated metric solver using PyTorch."""
    
    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """Initialize solver.
        
        Parameters
        ----------
        device : str or torch.device
            Device to use for computations
        """
        self.device = torch.device(device)
    
    def _ensure_tensor(self, x: Union[torch.Tensor, float]) -> torch.Tensor:
        """Ensure input is a tensor on the correct device.
        
        Parameters
        ----------
        x : torch.Tensor or float
            Input to convert
            
        Returns
        -------
        torch.Tensor
            Tensor on the correct device
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        return x
    
    def calculate_alcubierre_metric(self, x: torch.Tensor, y: torch.Tensor,
                                  z: torch.Tensor, t: float, v_s: float = 2.0,
                                  R: float = 1.0, sigma: float = 0.5) -> Dict[str, torch.Tensor]:
        """Calculate Alcubierre metric components.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
        v_s : float
            Ship velocity (in c)
        R : float
            Radius of warp bubble
        sigma : float
            Thickness parameter
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Metric components
        """
        # Convert inputs to tensors on device
        x = self._ensure_tensor(x)
        y = self._ensure_tensor(y)
        z = self._ensure_tensor(z)
        t = self._ensure_tensor(t)
        v_s = self._ensure_tensor(v_s)
        R = self._ensure_tensor(R)
        sigma = self._ensure_tensor(sigma)
        
        # Calculate ship position
        x_s = v_s * t
        
        # Calculate r (distance from ship center)
        r = torch.sqrt((x - x_s)**2 + y**2 + z**2)
        
        # Shape function
        f = torch.exp(-sigma * r**2 / R**2)
        
        # Add cutoff for numerical stability
        f = torch.where(r > 5*R, torch.zeros_like(f), f)
        
        # Calculate metric components
        v_x = v_s * f
        g_tt = -(1 - v_x**2)
        g_tx = -v_x
        g_xx = torch.ones_like(x)
        
        return {
            "g_tt": g_tt,
            "g_tx": g_tx,
            "g_xx": g_xx
        }