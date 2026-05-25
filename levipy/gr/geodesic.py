"""
Geodesic equations.

For an affinely parametrised curve x^i(λ):

    d²x^k/dλ² + Γ^k_{ij} (dx^i/dλ)(dx^j/dλ) = 0
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp
from sympy import Function, Symbol, diff, simplify


class GeodesicEquations:
    """
    Symbolic geodesic equations for a given metric.

    Parameters
    ----------
    metric : MetricTensor
    affine_param : Symbol, optional
        Affine parameter λ.  Defaults to a new symbol 'lambda'.

    Attributes
    ----------
    lam : Symbol        — the affine parameter
    x   : list[Function]   — coordinate functions x^i(λ)
    eqs : list[sp.Eq]   — geodesic equations (one per coordinate)

    Examples
    --------
    >>> geo = metric.geodesic_equations()
    >>> geo.display()
    """

    def __init__(self, metric, affine_param: Optional[Symbol] = None):
        self.metric = metric
        self.n = metric.n
        self.coords = metric.coords

        self.lam = affine_param or Symbol('lambda', real=True)
        # coordinate functions x^i(λ)
        self.x = [Function(str(c))(self.lam) for c in self.coords]

        self._eqs: Optional[List[sp.Eq]] = None

    @property
    def equations(self) -> List[sp.Eq]:
        if self._eqs is None:
            self._eqs = self._compute()
        return self._eqs

    def _compute(self) -> List[sp.Eq]:
        ch = self.metric.christoffel()
        n = self.n
        lam = self.lam
        x = self.x
        coords = self.coords

        # Build substitution map: coord symbol → function of λ
        subs_map = {coords[i]: x[i] for i in range(n)}

        eqs = []
        for k in range(n):
            acc = diff(x[k], lam, 2)
            for i in range(n):
                for j in range(n):
                    gamma_val = ch.Gamma[k][i][j].subs(subs_map)
                    acc += gamma_val * diff(x[i], lam) * diff(x[j], lam)
            eqs.append(sp.Eq(simplify(acc), 0))

        return eqs

    def display(self):
        """Print geodesic equations."""
        print(f"Geodesic equations  (affine parameter: {self.lam})\n")
        for k, eq in enumerate(self.equations):
            print(f"  [{self.coords[k]}]:  {eq}")

    def __repr__(self):
        return f"GeodesicEquations(dim={self.n}, param={self.lam})"
