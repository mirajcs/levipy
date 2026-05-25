"""
Metric tensor g_{ij} and its inverse g^{ij}.
All other tensors are derived from here.
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp
from sympy import Matrix, Symbol, latex, simplify


class MetricTensor:
    """
    Covariant metric tensor g_{ij}.

    Parameters
    ----------
    components : array-like (n×n)
        Sympy expressions for g_{ij}.
    coords : list of Symbol
        Coordinate symbols.
    manifold : Manifold, optional
        Parent manifold.

    Attributes
    ----------
    g   : sympy.Matrix  — covariant components g_{ij}
    g_inv : sympy.Matrix — contravariant components g^{ij}

    Examples
    --------
    Sphere S²:
    >>> import sympy as sp
    >>> from levipy import Manifold
    >>> r = sp.Symbol('r', positive=True)
    >>> th, ph = sp.symbols('theta phi', real=True)
    >>> M = Manifold('S2', [th, ph])
    >>> g = M.metric([[r**2, 0], [0, r**2 * sp.sin(th)**2]])
    >>> g.christoffel().display()
    """

    def __init__(self, components, coords: List[Symbol], manifold=None):
        self.coords = list(coords)
        self.n = len(coords)
        self.manifold = manifold
        self.g = Matrix(components).applyfunc(sp.sympify)
        if self.g.shape != (self.n, self.n):
            raise ValueError(
                f"Metric must be {self.n}×{self.n}, got {self.g.shape}."
            )
        self._g_inv: Optional[Matrix] = None
        self._christoffel = None
        self._riemann = None
        self._ricci_t = None
        self._ricci_s = None
        self._einstein = None

    # ------------------------------------------------------------------
    # Inverse metric
    # ------------------------------------------------------------------

    @property
    def g_inv(self) -> Matrix:
        if self._g_inv is None:
            self._g_inv = self.g.inv()
        return self._g_inv

    # ------------------------------------------------------------------
    # Derived tensors (lazy, cached)
    # ------------------------------------------------------------------

    def christoffel(self) -> "ChristoffelSymbols":
        if self._christoffel is None:
            from levipy.tensors.christoffel import ChristoffelSymbols
            self._christoffel = ChristoffelSymbols.from_metric(self)
        return self._christoffel

    def riemann(self) -> "RiemannTensor":
        if self._riemann is None:
            from levipy.tensors.riemann import RiemannTensor
            self._riemann = RiemannTensor.from_christoffel(self.christoffel())
        return self._riemann

    def ricci_tensor(self) -> "RicciTensor":
        if self._ricci_t is None:
            from levipy.tensors.ricci import RicciTensor
            self._ricci_t = RicciTensor.from_riemann(self.riemann())
        return self._ricci_t

    def ricci_scalar(self) -> sp.Expr:
        if self._ricci_s is None:
            from levipy.tensors.ricci import RicciScalar
            self._ricci_s = RicciScalar.from_ricci(self.ricci_tensor(), self)
        return self._ricci_s

    def einstein_tensor(self) -> "EinsteinTensor":
        if self._einstein is None:
            from levipy.tensors.einstein import EinsteinTensor
            self._einstein = EinsteinTensor.from_metric(self)
        return self._einstein

    def geodesic_equations(self, affine_param: Symbol = None):
        from levipy.gr.geodesic import GeodesicEquations
        return GeodesicEquations(self, affine_param=affine_param)

    def covariant_derivative(self):
        from levipy.geometry.covariant_derivative import CovariantDerivative
        return CovariantDerivative(self.christoffel())

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display(self, simplify_expr: bool = True):
        """Pretty-print the metric components."""
        print(f"Metric tensor g_ij  (coords: {self.coords})\n")
        for i in range(self.n):
            for j in range(i, self.n):
                val = simplify(self.g[i, j]) if simplify_expr else self.g[i, j]
                if val != 0:
                    print(f"  g_{{{self.coords[i]},{self.coords[j]}}} = {val}")

    def _latex_matrix(self) -> str:
        return latex(self.g)

    def __repr__(self) -> str:
        name = self.manifold.name if self.manifold else "?"
        return f"MetricTensor(manifold='{name}', dim={self.n})"
