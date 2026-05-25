"""
Differentiable manifold with coordinate chart.
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp


class Manifold:
    """
    A smooth manifold with a single coordinate chart.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. 'S2', 'Schwarzschild').
    coords : list of sympy.Symbol
        Coordinate symbols (e.g. [r, theta, phi]).
    dim : int, optional
        Dimension; inferred from coords if omitted.

    Examples
    --------
    >>> import sympy as sp
    >>> from levipy import Manifold
    >>> r, th, ph = sp.symbols('r theta phi', positive=True)
    >>> M = Manifold('S2', coords=[th, ph])
    """

    def __init__(
        self,
        name: str,
        coords: List[sp.Symbol],
        dim: Optional[int] = None,
    ):
        self.name = name
        self.coords = list(coords)
        self.dim = dim if dim is not None else len(self.coords)
        if self.dim != len(self.coords):
            raise ValueError(
                f"dim={self.dim} but {len(self.coords)} coordinates given."
            )
        self._metric = None

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------

    def metric(self, components) -> "MetricTensor":
        """
        Attach a metric tensor to this manifold.

        Parameters
        ----------
        components : array-like (n×n)
            Covariant components g_{ij} as SymPy expressions.

        Returns
        -------
        MetricTensor
        """
        from levipy.tensors.metric import MetricTensor
        self._metric = MetricTensor(components, self.coords, manifold=self)
        return self._metric

    def __repr__(self) -> str:
        return (
            f"Manifold('{self.name}', coords={self.coords}, dim={self.dim})"
        )
