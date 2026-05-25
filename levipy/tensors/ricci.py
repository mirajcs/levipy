"""
Ricci tensor  R_{ij} = R^k_{ikj}
Ricci scalar  R = g^{ij} R_{ij}
"""
from __future__ import annotations

import sympy as sp
from sympy import simplify


class RicciTensor:
    """
    Ricci tensor R_{ij} = R^k_{ikj} (contraction of Riemann on 1st and 3rd index).

    Attributes
    ----------
    Ric : sympy.Matrix  (n×n)
    coords : list
    """

    def __init__(self, Ric: sp.Matrix, coords):
        self.Ric = Ric
        self.coords = list(coords)
        self.n = len(coords)

    @classmethod
    def from_riemann(cls, riemann) -> "RicciTensor":
        n = riemann.n
        Ric = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                Ric[i, j] = simplify(
                    sum(riemann.R[k][i][k][j] for k in range(n))
                )
        return cls(Ric, riemann.coords)

    def display(self, simplify_expr: bool = True):
        print(f"Ricci tensor R_{{ij}}  (coords: {self.coords})\n")
        found = False
        for i in range(self.n):
            for j in range(i, self.n):
                val = simplify(self.Ric[i, j]) if simplify_expr else self.Ric[i, j]
                if val != 0:
                    ci, cj = self.coords[i], self.coords[j]
                    print(f"  R_{{{ci},{cj}}} = {val}")
                    found = True
        if not found:
            print("  Ricci tensor vanishes (Ricci-flat manifold).")

    def __getitem__(self, idx):
        i, j = idx
        return self.Ric[i, j]

    def __repr__(self):
        return f"RicciTensor(dim={self.n})"


class RicciScalar:
    """
    Ricci scalar R = g^{ij} R_{ij}.
    """

    def __init__(self, value: sp.Expr):
        self.value = value

    @classmethod
    def from_ricci(cls, ricci: RicciTensor, metric) -> "RicciScalar":
        n = ricci.n
        g_inv = metric.g_inv
        R = simplify(
            sum(g_inv[i, j] * ricci.Ric[i, j]
                for i in range(n) for j in range(n))
        )
        return cls(R)

    def display(self):
        print(f"Ricci scalar R = {self.value}")

    def __repr__(self):
        return f"RicciScalar({self.value})"
