"""
Covariant derivative ∇ using the Levi-Civita connection.

For a vector field X^k:
    (∇_i X)^k = ∂_i X^k + Γ^k_{ij} X^j

For a covector ω_k:
    (∇_i ω)_k = ∂_i ω_k − Γ^j_{ik} ω_j
"""
from __future__ import annotations

from typing import List

import sympy as sp
from sympy import diff, simplify


class CovariantDerivative:
    """
    Covariant derivative operator associated to the Levi-Civita connection.

    Parameters
    ----------
    christoffel : ChristoffelSymbols

    Examples
    --------
    >>> nabla = metric.covariant_derivative()
    >>> X = [sp.cos(th), sp.sin(th)]   # vector field components
    >>> nabla.of_vector(X)             # returns ∇_i X^k as a matrix
    """

    def __init__(self, christoffel):
        self.ch = christoffel
        self.coords = christoffel.coords
        self.n = christoffel.n

    def of_vector(self, X: List[sp.Expr]) -> sp.Matrix:
        """
        Compute ∇_i X^k.

        Returns
        -------
        sp.Matrix of shape (n, n)
            Entry [i, k]  =  (∇_i X)^k
        """
        n = self.n
        coords = self.coords
        result = sp.zeros(n, n)
        for i in range(n):
            for k in range(n):
                val = diff(X[k], coords[i])
                for j in range(n):
                    val += self.ch.Gamma[k][i][j] * X[j]
                result[i, k] = simplify(val)
        return result

    def of_covector(self, omega: List[sp.Expr]) -> sp.Matrix:
        """
        Compute ∇_i ω_k.

        Returns
        -------
        sp.Matrix of shape (n, n)
            Entry [i, k]  =  (∇_i ω)_k
        """
        n = self.n
        coords = self.coords
        result = sp.zeros(n, n)
        for i in range(n):
            for k in range(n):
                val = diff(omega[k], coords[i])
                for j in range(n):
                    val -= self.ch.Gamma[j][i][k] * omega[j]
                result[i, k] = simplify(val)
        return result

    def of_tensor_11(self, T: sp.Matrix) -> list:
        """
        Compute ∇_i T^k_j  for a (1,1)-tensor T.

        Returns
        -------
        3-nested list  result[i][k][j]
        """
        n = self.n
        coords = self.coords
        G = self.ch.Gamma
        result = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    val = diff(T[k, j], coords[i])
                    for m in range(n):
                        val += G[k][i][m] * T[m, j]
                        val -= G[m][i][j] * T[k, m]
                    result[i][k][j] = simplify(val)
        return result

    def display_vector(self, X: List[sp.Expr]):
        mat = self.of_vector(X)
        print(f"∇_i X^k  (coords: {self.coords})\n")
        for i in range(self.n):
            for k in range(self.n):
                val = mat[i, k]
                if val != 0:
                    print(f"  ∇_{self.coords[i]} X^{self.coords[k]} = {val}")

    def __repr__(self):
        return f"CovariantDerivative(dim={self.n})"

