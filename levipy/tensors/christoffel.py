"""
Christoffel symbols of the second kind: Γ^k_{ij}

  Γ^k_{ij} = (1/2) g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
"""
from __future__ import annotations

from typing import List

import sympy as sp
from sympy import latex, simplify


class ChristoffelSymbols:
    """
    Christoffel symbols Γ^k_{ij} of the Levi-Civita connection.

    Attributes
    ----------
    Gamma : list[list[list[sp.Expr]]]
        Γ[k][i][j]  (upper index first, consistent with ∇ notation).
    coords : list
        Coordinate symbols.
    n : int
        Dimension.
    """

    def __init__(self, gamma, coords):
        self.Gamma = gamma          # Gamma[k][i][j]
        self.coords = list(coords)
        self.n = len(coords)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_metric(cls, metric) -> "ChristoffelSymbols":
        """
        Compute Γ^k_{ij} from a MetricTensor.

        Uses the Koszul formula:
            Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})
        """
        g = metric.g
        g_inv = metric.g_inv
        coords = metric.coords
        n = metric.n

        gamma = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]

        for k in range(n):
            for i in range(n):
                for j in range(i, n):   # symmetry Γ^k_{ij} = Γ^k_{ji}
                    val = sp.Integer(0)
                    for ll in range(n):
                        val += g_inv[k, ll] * (
                            sp.diff(g[j, ll], coords[i])
                            + sp.diff(g[i, ll], coords[j])
                            - sp.diff(g[i, j], coords[ll])
                        )
                    val = simplify(val / 2)
                    gamma[k][i][j] = val
                    gamma[k][j][i] = val   # symmetry in lower indices

        return cls(gamma, coords)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Γ[k, i, j]  or  Γ[k][i][j]."""
        if isinstance(idx, tuple):
            k, i, j = idx
            return self.Gamma[k][i][j]
        return self.Gamma[idx]

    def nonzero(self):
        """Return list of ((k,i,j), value) for nonzero components."""
        result = []
        for k in range(self.n):
            for i in range(self.n):
                for j in range(i, self.n):
                    val = self.Gamma[k][i][j]
                    if val != 0:
                        result.append(((k, i, j), val))
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, simplify_expr: bool = True):
        """Print all nonzero Christoffel symbols."""
        print(f"Christoffel symbols Γ^k_{{ij}}  (coords: {self.coords})\n")
        found = False
        for (k, i, j), val in self.nonzero():
            if simplify_expr:
                val = simplify(val)
            if val != 0:
                ci, cj, ck = self.coords[i], self.coords[j], self.coords[k]
                print(f"  Γ^{ck}_{{{ci},{cj}}} = {val}")
                found = True
        if not found:
            print("  All Christoffel symbols vanish (flat connection).")

    def latex_display(self) -> List[str]:
        """Return list of LaTeX strings for nonzero components."""
        lines = []
        for (k, i, j), val in self.nonzero():
            ci = latex(self.coords[i])
            cj = latex(self.coords[j])
            ck = latex(self.coords[k])
            lhs = rf"\Gamma^{{{ck}}}_{{{ci}{cj}}}"
            lines.append(f"${lhs} = {latex(simplify(val))}$")
        return lines

    def __repr__(self) -> str:
        nz = len(self.nonzero())
        return f"ChristoffelSymbols(dim={self.n}, nonzero={nz})"
