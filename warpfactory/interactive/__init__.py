"""Interactive exploration: shared model plus the Jupyter front end.

The model (ExplorerModel, METRIC_CATALOG) has no UI dependencies and is
always importable. JupyterExplorer needs the optional ipywidgets extra;
like warpfactory.gui, its absence must not break the core package.
"""

from .model import METRIC_CATALOG, ExplorationResult, ExplorerModel

try:
    from .explorer import JupyterExplorer

    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False

    class JupyterExplorer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "ipywidgets is required for JupyterExplorer; "
                'install it with pip install "warpfactory[jupyter]"'
            )


__all__ = [
    "METRIC_CATALOG",
    "ExplorationResult",
    "ExplorerModel",
    "JupyterExplorer",
    "HAS_JUPYTER",
]
