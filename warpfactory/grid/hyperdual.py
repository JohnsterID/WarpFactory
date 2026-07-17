"""Hyper-dual numbers: exact first and second derivatives in pure numpy.

Forward-mode automatic differentiation after Fike & Alonso, "The
Development of Hyper-Dual Numbers for Exact Second-Derivative
Calculations" (AIAA 2011-886). A hyper-dual number

    x = f + f1 eps1 + f2 eps2 + f12 eps1 eps2,   eps1^2 = eps2^2 = 0

propagates exact first derivatives in the two dual parts and the exact
mixed second derivative in the cross part, with no truncation and no
subtractive-cancellation error (unlike finite differences, and unlike
the complex-step method for second derivatives).

Seeding eps1 on coordinate k and eps2 on coordinate n and evaluating an
analytic metric function once yields d_k g, d_n g and d_k d_n g to
machine precision; the ten (k, n) pairs supply everything the curvature
pipeline needs. This gives the accuracy of an autodiff backend (e.g.
JAX) for analytic metrics while staying a plain numpy dependency.

All parts are numpy arrays, so one hyper-dual evaluation covers a whole
grid of points at once.
"""

from typing import Any, Callable, Dict, Union

import numpy as np

Scalar = Union[float, int, np.ndarray]


class HyperDual:
    """A hyper-dual array: value, two dual parts, one cross part.

    Parameters
    ----------
    f : array_like
        Real (value) part
    f1, f2 : array_like
        Coefficients of eps1 and eps2 (first-derivative carriers)
    f12 : array_like
        Coefficient of eps1 eps2 (second-derivative carrier)
    """

    # Win binary ops against plain ndarrays so ndarray + HyperDual
    # defers to __radd__ instead of broadcasting over the object.
    __array_priority__ = 100.0

    __slots__ = ("f", "f1", "f2", "f12")

    def __init__(
        self, f: Scalar, f1: Scalar = 0.0, f2: Scalar = 0.0, f12: Scalar = 0.0
    ):
        self.f = np.asarray(f, dtype=float)
        shape = self.f.shape
        self.f1 = np.broadcast_to(np.asarray(f1, dtype=float), shape)
        self.f2 = np.broadcast_to(np.asarray(f2, dtype=float), shape)
        self.f12 = np.broadcast_to(np.asarray(f12, dtype=float), shape)

    @staticmethod
    def _lift(value: Union["HyperDual", Scalar]) -> "HyperDual":
        if isinstance(value, HyperDual):
            return value
        return HyperDual(value)

    def _map(self, value: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> "HyperDual":
        """Chain rule for a scalar function with derivatives d1, d2 at f."""
        return HyperDual(
            value,
            d1 * self.f1,
            d1 * self.f2,
            d1 * self.f12 + d2 * self.f1 * self.f2,
        )

    # -- arithmetic ------------------------------------------------------

    def __add__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        o = self._lift(other)
        return HyperDual(self.f + o.f, self.f1 + o.f1, self.f2 + o.f2, self.f12 + o.f12)

    __radd__ = __add__

    def __neg__(self) -> "HyperDual":
        return HyperDual(-self.f, -self.f1, -self.f2, -self.f12)

    def __sub__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        return self + (-self._lift(other))

    def __rsub__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        return self._lift(other) + (-self)

    def __mul__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        o = self._lift(other)
        return HyperDual(
            self.f * o.f,
            self.f1 * o.f + self.f * o.f1,
            self.f2 * o.f + self.f * o.f2,
            self.f12 * o.f + self.f1 * o.f2 + self.f2 * o.f1 + self.f * o.f12,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "HyperDual":
        inv = 1.0 / self.f
        return self._map(inv, -(inv**2), 2.0 * inv**3)

    def __truediv__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        return self * self._lift(other).reciprocal()

    def __rtruediv__(self, other: Union["HyperDual", Scalar]) -> "HyperDual":
        return self._lift(other) * self.reciprocal()

    def __pow__(self, exponent: float) -> "HyperDual":
        n = float(exponent)
        return self._map(
            self.f**n, n * self.f ** (n - 1.0), n * (n - 1.0) * self.f ** (n - 2.0)
        )

    # -- numpy ufunc dispatch --------------------------------------------

    _UNARY_TABLE: Dict[Any, Callable[["HyperDual"], "HyperDual"]] = {}

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Let np.tanh, np.sqrt, ... and binary ufuncs act on HyperDual.

        Existing metric/shape functions written for plain numpy arrays
        then work on hyper-dual inputs without modification.
        """
        if method != "__call__" or kwargs:
            return NotImplemented
        if ufunc in self._UNARY_TABLE:
            (target,) = inputs
            return self._UNARY_TABLE[ufunc](self._lift(target))
        binary = {
            np.add: HyperDual.__add__,
            np.subtract: HyperDual.__sub__,
            np.multiply: HyperDual.__mul__,
            np.true_divide: HyperDual.__truediv__,
        }
        if ufunc in binary:
            a, b = inputs
            return binary[ufunc](self._lift(a), b)
        if ufunc is np.power:
            base, exponent = inputs
            if isinstance(exponent, HyperDual):
                return NotImplemented
            return self._lift(base) ** float(exponent)
        if ufunc is np.negative:
            (target,) = inputs
            return -self._lift(target)
        return NotImplemented

    # -- elementary functions ----------------------------------------------

    def tanh(self) -> "HyperDual":
        t = np.tanh(self.f)
        sech2 = 1.0 - t**2
        return self._map(t, sech2, -2.0 * t * sech2)

    def sqrt(self) -> "HyperDual":
        s = np.sqrt(self.f)
        return self._map(s, 0.5 / s, -0.25 / (s * self.f))

    def exp(self) -> "HyperDual":
        e = np.exp(self.f)
        return self._map(e, e, e)

    def log(self) -> "HyperDual":
        return self._map(np.log(self.f), 1.0 / self.f, -1.0 / self.f**2)

    def sin(self) -> "HyperDual":
        s, c = np.sin(self.f), np.cos(self.f)
        return self._map(s, c, -s)

    def cos(self) -> "HyperDual":
        s, c = np.sin(self.f), np.cos(self.f)
        return self._map(c, -s, -c)

    def __repr__(self) -> str:
        return (
            f"HyperDual(f={self.f!r}, f1={self.f1!r}, f2={self.f2!r}, f12={self.f12!r})"
        )


HyperDual._UNARY_TABLE = {
    np.tanh: HyperDual.tanh,
    np.sqrt: HyperDual.sqrt,
    np.exp: HyperDual.exp,
    np.log: HyperDual.log,
    np.sin: HyperDual.sin,
    np.cos: HyperDual.cos,
}
