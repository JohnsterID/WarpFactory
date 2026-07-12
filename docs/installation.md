# Installation

## Using pip

```bash
# Install from source (not yet published on PyPI)
git clone https://github.com/JohnsterID/WarpFactory.git
cd WarpFactory

# Core install: numpy/scipy/matplotlib pipeline only
pip install .

# Optional extras (can be combined):
pip install ".[torch]"        # PyTorch backend (GPU acceleration)
pip install ".[jupyter]"      # ipywidgets interactive explorer (recommended)
pip install ".[gui]"          # PyQt6 desktop metric explorer (maintenance-only)
pip install ".[torch,jupyter,gui]"  # everything
```

## Using poetry

```bash
pip install poetry

git clone https://github.com/JohnsterID/WarpFactory.git
cd WarpFactory
poetry install                                # core only
poetry install --extras "torch jupyter gui"   # with optional backends
```

## Requirements

### Python dependencies

- Python 3.9 or higher
- NumPy
- SciPy
- Matplotlib (for visualization)
- PyTorch (optional `[torch]` extra, for GPU acceleration)
- ipywidgets (optional `[jupyter]` extra, for the interactive notebook
  explorer)
- PyQt6 (optional `[gui]` extra, for the maintenance-only desktop
  metric explorer)

### System dependencies

The Qt GUI extra needs Qt6 system libraries:

**Ubuntu/Debian:**

```bash
sudo apt-get install -y libgl1 libegl1 libxkbcommon-x11-0 libdbus-1-3
```

**Fedora/RHEL:**

```bash
sudo dnf install -y mesa-libGL mesa-libEGL libxkbcommon-x11 dbus-libs
```

**macOS:**

```bash
brew install qt@6
```

### GPU support

Install a CUDA build of PyTorch matching your driver (see the
[PyTorch install selector](https://pytorch.org/get-started/locally/)
for the current index URL), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

CPU-only PyTorch works for everything except the CUDA-marked tests:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running the tests

```bash
# Core suite (optional-backend tests skip automatically)
pytest warpfactory/tests -q --no-cov

# With coverage
pytest --cov=warpfactory

# GUI tests need the [gui] extra plus Qt system libraries
QT_QPA_PLATFORM=offscreen pytest warpfactory/tests/test_gui.py -v
```

Tests for optional backends skip (not fail) when the extra is not
installed; CUDA-only tests additionally need a GPU.
