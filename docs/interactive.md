# Interactive Explorer

The primary interactive UI is the Jupyter/ipywidgets explorer in
`warpfactory.interactive`: live metric and stress-energy plots,
light-cone tilt with horizon and ergosphere markers, energy-condition
verdicts, and the Ford-Roman quantum inequality assessment updating as
sliders move.

## Run it in your browser

No local install needed:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JohnsterID/WarpFactory/main?labpath=examples%2Finteractive_explorer.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnsterID/WarpFactory/blob/main/examples/interactive_explorer.ipynb)

!!! note
    On Colab, install the package in the first cell before running the
    notebook:

    ```
    !pip install "warpfactory[jupyter] @ git+https://github.com/JohnsterID/WarpFactory"
    ```

## Run it locally

```bash
pip install ".[jupyter]"
```

```python
from warpfactory.interactive import JupyterExplorer

JupyterExplorer().display()
```

## Scripting the same pipeline

The explorer is a thin front end over `ExplorerModel`, which you can
drive directly for reproducible, scripted analysis:

```python
from warpfactory.interactive import ExplorerModel

model = ExplorerModel()
result = model.evaluate(
    "Alcubierre", {"v_s": 2.0, "R": 1.0, "sigma": 4.0}, diagnostics=True
)
print(result.conditions)           # energy condition verdicts
print(result.quantum_inequality)   # Ford-Roman wall-thickness check
```

Every catalog metric (Alcubierre, Lentz, Van Den Broeck, Modified
Time, ...) runs through the identical pipeline, so comparisons are
apples-to-apples. See
[`examples/interactive_explorer.ipynb`](https://github.com/JohnsterID/WarpFactory/blob/main/examples/interactive_explorer.ipynb)
for a full walkthrough including parameter sweeps.

## Desktop GUI (maintenance-only)

A PyQt6 desktop explorer is kept working in `warpfactory.gui` but no
longer gains features; both front ends share the same `ExplorerModel`
compute pipeline.

```bash
pip install ".[gui]"   # needs Qt6 system libraries, see Installation
```
