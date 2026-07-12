# GPU Acceleration

The optional `[torch]` extra provides PyTorch-based metric
calculations, CUDA-accelerated tensor operations, and GPU-optimized
energy tensor computations. The code is device-agnostic: everything
below also runs with `device="cpu"`.

```bash
pip install ".[torch]"
```

## Metric calculation on the GPU

```python
import torch

from warpfactory.torch import TorchMetricSolver

solver = TorchMetricSolver(device="cuda")

x = torch.linspace(-5, 5, 100, device="cuda")
y = torch.zeros_like(x)
z = torch.zeros_like(x)

metric = solver.calculate_alcubierre_metric(
    x, y, z, t=0.0, v_s=2.0, R=1.0, sigma=0.5
)

# Move results back to CPU if needed
metric_cpu = {k: v.cpu() for k, v in metric.items()}
```

## Batched parameter studies

`TorchMetricBatch` evaluates many parameter configurations in
parallel, and `TorchEnergyAnalyzer` checks energy conditions on the
whole batch:

```python
from warpfactory.torch import TorchEnergyAnalyzer, TorchMetricBatch

params = {
    "v_s": torch.tensor([1.0, 2.0, 3.0], device="cuda"),
    "R": torch.tensor([0.5, 1.0, 1.5], device="cuda"),
    "sigma": torch.tensor([0.3, 0.5, 0.7], device="cuda"),
}

batch = TorchMetricBatch(device="cuda")
metrics = batch.calculate_metrics_parallel(x, y, z, 0.0, params)

analyzer = TorchEnergyAnalyzer(device="cuda")
results = analyzer.analyze_batch(metrics)
```

## Performance tips

- Keep data on the GPU to avoid transfer overhead; move to CPU only
  for plotting.
- Use batch processing for parameter studies instead of Python loops.
- Monitor memory with `torch.cuda.memory_summary()` and clear unused
  tensors with `torch.cuda.empty_cache()`.
- Use `torch.cuda.synchronize()` around timing measurements.

## Requirements

- CUDA-capable GPU with a PyTorch CUDA build matching your driver
  (see the [PyTorch install selector](https://pytorch.org/get-started/locally/))
- CPU-only PyTorch runs the same code paths for development; the
  CUDA-marked tests skip without a GPU
