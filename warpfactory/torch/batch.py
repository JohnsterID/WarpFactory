"""Batch processing for multiple metrics using PyTorch."""

import torch
from typing import Dict, List, Union
from .metric import TorchMetricSolver

class TorchMetricBatch:
    """GPU-accelerated batch processing of metrics."""
    
    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """Initialize batch processor.
        
        Parameters
        ----------
        device : str or torch.device
            Device to use for computations
        """
        self.device = torch.device(device)
        self.solver = TorchMetricSolver(device=device)
    
    def _validate_batch_params(self, params: Dict[str, torch.Tensor]) -> int:
        """Validate batch parameters and get batch size.
        
        Parameters
        ----------
        params : Dict[str, torch.Tensor]
            Dictionary of parameter tensors
            
        Returns
        -------
        int
            Batch size
            
        Raises
        ------
        ValueError
            If parameters have inconsistent batch sizes
        """
        batch_sizes = {k: len(v) for k, v in params.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(
                f"Inconsistent batch sizes: {batch_sizes}"
            )
        return list(batch_sizes.values())[0]
    
    def calculate_metrics(self, x: torch.Tensor, y: torch.Tensor,
                        z: torch.Tensor, t: float,
                        params: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Calculate metrics for multiple parameter sets.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
        params : Dict[str, torch.Tensor]
            Dictionary of parameter tensors
            
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of metric components for each parameter set
        """
        batch_size = self._validate_batch_params(params)
        metrics = []
        
        # Process each parameter set
        for i in range(batch_size):
            batch_params = {
                k: v[i].to(self.device)
                for k, v in params.items()
            }
            
            metric = self.solver.calculate_alcubierre_metric(
                x, y, z, t, **batch_params
            )
            metrics.append(metric)
        
        return metrics
    
    def calculate_metrics_parallel(self, x: torch.Tensor, y: torch.Tensor,
                                z: torch.Tensor, t: float,
                                params: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Calculate metrics in parallel using vectorized operations.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
        params : Dict[str, torch.Tensor]
            Dictionary of parameter tensors
            
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of metric components for each parameter set
        """
        batch_size = self._validate_batch_params(params)
        
        # Move all parameters to device
        params = {
            k: v.to(self.device)
            for k, v in params.items()
        }
        
        # Expand coordinates for batch processing
        x = x.unsqueeze(0).expand(batch_size, -1)
        y = y.unsqueeze(0).expand(batch_size, -1)
        z = z.unsqueeze(0).expand(batch_size, -1)
        
        # Calculate ship positions
        x_s = params["v_s"] * t
        
        # Calculate r (distance from ship center)
        r = torch.sqrt((x - x_s.unsqueeze(1))**2 + 
                      y**2 + z**2)
        
        # Shape function
        f = torch.exp(-params["sigma"].unsqueeze(1) * 
                     r**2 / params["R"].unsqueeze(1)**2)
        
        # Add cutoff for numerical stability
        f = torch.where(r > 5*params["R"].unsqueeze(1),
                       torch.zeros_like(f), f)
        
        # Calculate metric components
        v_x = params["v_s"].unsqueeze(1) * f
        g_tt = -(1 - v_x**2)
        g_tx = -v_x
        g_xx = torch.ones_like(x)
        
        # Split into list of individual metrics
        return [
            {
                "g_tt": g_tt[i],
                "g_tx": g_tx[i],
                "g_xx": g_xx[i]
            }
            for i in range(batch_size)
        ]