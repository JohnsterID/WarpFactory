"""Parametrized metric families for search and optimization."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence

from ..grid.tensor import SpacetimeTensor


class Ansatz(ABC):
    """A named family of metrics parametrized by a flat dict.

    Subclasses implement build(params) -> SpacetimeTensor. The
    param_names attribute lists the keys build understands; search
    drivers use it to validate inputs before spending solver time.
    """

    name: str = "ansatz"
    param_names: Sequence[str] = ()

    @abstractmethod
    def build(self, params: Dict[str, float]) -> SpacetimeTensor:
        """Construct the metric for the given parameters."""

    def validate(self, params: Dict[str, float]) -> None:
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameters {sorted(unknown)}; "
                f"expected a subset of {list(self.param_names)}")


class CallableAnsatz(Ansatz):
    """Wrap any metric-builder callable as an Ansatz.

    Parameters
    ----------
    builder : Callable[..., SpacetimeTensor]
        Called as builder(**params); typically a warpfactory.grid
        metric builder with grid arguments bound via functools.partial
    param_names : Sequence[str]
        Keyword names the builder accepts as search parameters
    name : str, optional
        Display name; defaults to the builder's __name__
    """

    def __init__(self, builder: Callable[..., SpacetimeTensor],
                 param_names: Sequence[str],
                 name: Optional[str] = None):
        self.builder = builder
        self.param_names = tuple(param_names)
        self.name = name or getattr(builder, "__name__", "callable")

    def build(self, params: Dict[str, float]) -> SpacetimeTensor:
        self.validate(params)
        return self.builder(**params)
