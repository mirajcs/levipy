"""
Riemann curvature tensor R^l_{kij}

  R^l_{kij} = ∂_i Γ^l_{jk} − ∂_j Γ^l_{ik}
             + Γ^l_{im} Γ^m_{jk} − Γ^l_{jm} Γ^m_{ik}
"""
from __future__ import annotations

import sympy as sp
from sympy import simplify


class RiemannTensor:
    """
    Riemann curvature tensor R^l_{kij}.

    Index convention: R[ll][k][i][j]

    Attributes
    ----------
    R : 4-nested list of sp.Expr
    coords : list of symbols
    n : int
    """

    def __init__(self, R, coords):
        self.R = R          # R[ll][k][i][j]
        self.coords = list(coords)
        self.n = len(coords)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_christoffel(cls, ch) -> "RiemannTensor":
        """Compute R^l_{kij} from ChristoffelSymbols."""
        n = ch.n
        coords = ch.coords
        G = ch.Gamma

        R = [[[[sp.Integer(0)] * n for _ in range(n)]
               for _ in range(n)] for _ in range(n)]

        for ll in range(n):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        term1 = sp.diff(G[ll][j][k], coords[i])
                        term2 = sp.diff(G[ll][i][k], coords[j])
                        term3 = sum(G[ll][i][m] * G[m][j][k] for m in range(n))
                        term4 = sum(G[ll][j][m] * G[m][i][k] for m in range(n))
                        R[ll][k][i][j] = simplify(term1 - term2 + term3 - term4)

        return cls(R, coords)

    # ------------------------------------------------------------------
    # Covariant version R_{lkij} = g_{lm} R^m_{kij}
    # ------------------------------------------------------------------

    def lower(self, metric) -> "RiemannTensor":
        """Return fully covariant R_{lkij}."""
        n = self.n
        g = metric.g
        R_low = [[[[sp.Integer(0)] * n for _ in range(n)]
                   for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        val = sum(g[a, ll] * self.R[ll][k][i][j] for ll in range(n))
                        R_low[a][k][i][j] = simplify(val)
        lowered = RiemannTensor(R_low, self.coords)
        lowered._is_covariant = True
        return lowered

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            ll, k, i, j = idx
            return self.R[ll][k][i][j]
        return self.R[idx]

    def nonzero(self):
        result = []
        for ll in range(self.n):
            for k in range(self.n):
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        val = self.R[ll][k][i][j]
                        if val != 0:
                            result.append(((ll, k, i, j), val))
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, simplify_expr: bool = True):
        """Print nonzero independent components."""
        print(f"Riemann tensor R^l_{{kij}}  (coords: {self.coords})\n")
        found = False
        for (ll, k, i, j), val in self.nonzero():
            if simplify_expr:
                val = simplify(val)
            if val != 0:
                cl = self.coords[ll]
                ck, ci, cj = self.coords[k], self.coords[i], self.coords[j]
                print(f"  R^{cl}_{{{ck},{ci},{cj}}} = {val}")
                found = True
        if not found:
            print("  Riemann tensor vanishes (flat manifold).")

    def is_flat(self) -> bool:
        """Return True if all components vanish."""
        return all(v == 0 for _, v in self.nonzero())

    def __repr__(self) -> str:
        nz = len(self.nonzero())
        return f"RiemannTensor(dim={self.n}, nonzero_independent={nz})"
