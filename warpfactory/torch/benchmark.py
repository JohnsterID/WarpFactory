"""Performance benchmarking tools for PyTorch computations."""

import torch
import time
from typing import Dict, Union
from .batch import TorchMetricBatch
from .analyzer import TorchEnergyAnalyzer
from ..metrics import AlcubierreMetric  # CPU version for comparison

class TorchBenchmark:
    """GPU performance benchmarking tools."""
    
    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """Initialize benchmarking tools.
        
        Parameters
        ----------
        device : str or torch.device
            Device to use for GPU computations
        """
        self.device = torch.device(device)
        self.batch = TorchMetricBatch(device=device)
        self.analyzer = TorchEnergyAnalyzer(device=device)
    
    def measure_single_metric(self, x: torch.Tensor, y: torch.Tensor,
                            z: torch.Tensor, t: float) -> float:
        """Measure time for single metric calculation.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
            
        Returns
        -------
        float
            Computation time in seconds
        """
        # Warm up
        _ = self.batch.calculate_metrics(
            x, y, z, t,
            {"v_s": torch.tensor([2.0], device=self.device),
             "R": torch.tensor([1.0], device=self.device),
             "sigma": torch.tensor([0.5], device=self.device)}
        )
        
        # Synchronize and measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = self.batch.calculate_metrics(
            x, y, z, t,
            {"v_s": torch.tensor([2.0], device=self.device),
             "R": torch.tensor([1.0], device=self.device),
             "sigma": torch.tensor([0.5], device=self.device)}
        )
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return end - start
    
    def measure_batch_metrics(self, x: torch.Tensor, y: torch.Tensor,
                            z: torch.Tensor, t: float,
                            params: Dict[str, torch.Tensor]) -> float:
        """Measure time for batch metric calculation.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
        params : Dict[str, torch.Tensor]
            Batch parameters
            
        Returns
        -------
        float
            Computation time in seconds
        """
        # Warm up
        _ = self.batch.calculate_metrics_parallel(x, y, z, t, params)
        
        # Synchronize and measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = self.batch.calculate_metrics_parallel(x, y, z, t, params)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return end - start
    
    def measure_memory_usage(self, x: torch.Tensor, y: torch.Tensor,
                           z: torch.Tensor, t: float,
                           params: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Measure GPU memory usage.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
        params : Dict[str, torch.Tensor]
            Batch parameters
            
        Returns
        -------
        Dict[str, int]
            Memory statistics in bytes
        """
        torch.cuda.reset_peak_memory_stats()
        
        _ = self.batch.calculate_metrics_parallel(x, y, z, t, params)
        _ = self.analyzer.analyze_batch(_)
        
        return {
            "allocated": torch.cuda.max_memory_allocated(),
            "cached": torch.cuda.max_memory_reserved()
        }
    
    def compare_cpu_gpu(self, x: torch.Tensor, y: torch.Tensor,
                       z: torch.Tensor, t: float) -> float:
        """Compare CPU vs GPU performance.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
            
        Returns
        -------
        float
            Speedup factor (CPU time / GPU time)
        """
        # CPU computation
        cpu_metric = AlcubierreMetric()
        
        start = time.perf_counter()
        _ = cpu_metric.calculate(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            t,
            v_s=2.0, R=1.0, sigma=0.5
        )
        cpu_time = time.perf_counter() - start
        
        # GPU computation
        gpu_time = self.measure_single_metric(x, y, z, t)
        
        return cpu_time / gpu_time
    
    def profile_components(self, x: torch.Tensor, y: torch.Tensor,
                         z: torch.Tensor, t: float) -> Dict[str, float]:
        """Profile different computation components.
        
        Parameters
        ----------
        x, y, z : torch.Tensor
            Spatial coordinates
        t : float
            Time coordinate
            
        Returns
        -------
        Dict[str, float]
            Computation times for different components
        """
        times = {}
        
        # Metric calculation
        torch.cuda.synchronize()
        start = time.perf_counter()
        metric = self.batch.calculate_metrics(
            x, y, z, t,
            {"v_s": torch.tensor([2.0], device=self.device),
             "R": torch.tensor([1.0], device=self.device),
             "sigma": torch.tensor([0.5], device=self.device)}
        )[0]
        torch.cuda.synchronize()
        times["metric_calculation"] = time.perf_counter() - start
        
        # Energy tensor
        torch.cuda.synchronize()
        start = time.perf_counter()
        T_munu = {
            "T_tt": -torch.abs(metric["g_tx"]) / (8 * torch.pi),
            "T_xx": -torch.abs(metric["g_tx"]) / (24 * torch.pi),
        }
        torch.cuda.synchronize()
        times["energy_tensor"] = time.perf_counter() - start
        
        # Christoffel symbols (simplified)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.gradient(metric["g_tt"], dim=0)[0]  # Example calculation
        torch.cuda.synchronize()
        times["christoffel_symbols"] = time.perf_counter() - start
        
        return times