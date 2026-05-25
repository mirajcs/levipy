"""
Einstein tensor  G_{ij} = R_{ij} − ½ g_{ij} R
"""
from __future__ import annotations

import sympy as sp
from sympy import simplify


class EinsteinTensor:
    """
    Einstein tensor G_{ij} = R_{ij} - (1/2) g_{ij} R.

    This is the left-hand side of Einstein's field equations:
        G_{ij} = 8π T_{ij}  (G = c = 1 units)

    Attributes
    ----------
    G : sympy.Matrix  (n×n)
    """

    def __init__(self, G: sp.Matrix, coords):
        self.G = G
        self.coords = list(coords)
        self.n = len(coords)

    @classmethod
    def from_metric(cls, metric) -> "EinsteinTensor":
        ricci_t = metric.ricci_tensor()
        ricci_s = metric.ricci_scalar()
        n = metric.n
        g = metric.g
        R = ricci_s.value

        G = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                G[i, j] = simplify(ricci_t.Ric[i, j] - sp.Rational(1, 2) * g[i, j] * R)

        return cls(G, metric.coords)

    def display(self, simplify_expr: bool = True):
        print(f"Einstein tensor G_{{ij}}  (coords: {self.coords})\n")
        found = False
        for i in range(self.n):
            for j in range(i, self.n):
                val = simplify(self.G[i, j]) if simplify_expr else self.G[i, j]
                if val != 0:
                    ci, cj = self.coords[i], self.coords[j]
                    print(f"  G_{{{ci},{cj}}} = {val}")
                    found = True
        if not found:
            print("  Einstein tensor vanishes.")

    def __repr__(self):
        return f"EinsteinTensor(dim={self.n})"
