"""
Parallel transport of a vector field V^k along a curve x^i(λ).

Equation:
    dV^k/dλ + Γ^k_{ij} (dx^i/dλ) V^j = 0
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp
from sympy import Function, Symbol, diff, simplify


class ParallelTransport:
    """
    Parallel transport equations for a vector V along a curve.

    Parameters
    ----------
    metric : MetricTensor
    affine_param : Symbol, optional

    Examples
    --------
    >>> pt = ParallelTransport(metric)
    >>> pt.equations(curve=[sp.cos(pt.lam), sp.sin(pt.lam)])
    """

    def __init__(self, metric, affine_param: Optional[Symbol] = None):
        self.metric = metric
        self.n = metric.n
        self.coords = metric.coords
        self.lam = affine_param or Symbol('lambda', real=True)

    def equations(self, curve: List[sp.Expr]) -> List[sp.Eq]:
        """
        Compute parallel transport equations for a vector V along `curve`.

        Parameters
        ----------
        curve : list of sp.Expr
            x^i(λ) — the coordinate functions of the curve,
            expressed in terms of self.lam.

        Returns
        -------
        list of sp.Eq
            One equation per component: DV^k/Dλ = 0.
        """
        n = self.n
        lam = self.lam
        ch = self.metric.christoffel()
        coords = self.coords

        # Vector components as functions of λ
        V = [Function(f'V{k}')(lam) for k in range(n)]

        # Substitution: replace coord symbols with curve values
        subs_map = {coords[i]: curve[i] for i in range(n)}
        # Also replace derivatives of coords
        curve_vel = [diff(curve[i], lam) for i in range(n)]

        eqs = []
        for k in range(n):
            transport = diff(V[k], lam)
            for i in range(n):
                for j in range(n):
                    gamma_val = ch.Gamma[k][i][j].subs(subs_map)
                    transport += gamma_val * curve_vel[i] * V[j]
            eqs.append(sp.Eq(simplify(transport), 0))

        return eqs

    def display(self, curve: List[sp.Expr]):
        print(f"Parallel transport equations  (λ = {self.lam})\n")
        for k, eq in enumerate(self.equations(curve)):
            print(f"  [V^{self.coords[k]}]:  {eq}")

    def __repr__(self):
        return f"ParallelTransport(dim={self.n})"
