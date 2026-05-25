"""
Lie bracket of two vector fields.

[X, Y]^k = X^i ∂_i Y^k − Y^i ∂_i X^k
"""

from __future__ import annotations

from typing import List

import sympy as sp
from sympy import diff, simplify


def lie_bracket(
    X: List[sp.Expr],
    Y: List[sp.Expr],
    coords: List[sp.Symbol],
) -> List[sp.Expr]:
    """
    Compute the Lie bracket [X, Y] of two vector fields.

    Parameters
    ----------
    X, Y : list of sp.Expr
        Components of the vector fields in the coordinate basis.
    coords : list of sp.Symbol
        Coordinate symbols.

    Returns
    -------
    list of sp.Expr
        Components of [X, Y].

    Notes
    -----
    For the Levi-Civita connection, the torsion-free condition implies:
        ∇_X Y − ∇_Y X = [X, Y]

    Examples
    --------
    >>> import sympy as sp
    >>> from levipy.geometry.lie_bracket import lie_bracket
    >>> x, y = sp.symbols('x y')
    >>> X = [y, 0]           # y ∂_x
    >>> Y = [0, x]           # x ∂_y
    >>> lie_bracket(X, Y, [x, y])
    [-x, -y]  # [y∂_x, x∂_y] = x∂_y(y)∂_x - y∂_x(x)∂_y  → ...
    """
    n = len(coords)
    result = []
    for k in range(n):
        val = sp.Integer(0)
        for i in range(n):
            val += X[i] * diff(Y[k], coords[i])
            val -= Y[i] * diff(X[k], coords[i])
        result.append(simplify(val))
    return result


def lie_derivative_vector(
    X: List[sp.Expr],
    Y: List[sp.Expr],
    coords: List[sp.Symbol],
) -> List[sp.Expr]:
    """Lie derivative L_X Y = [X, Y]."""
    return lie_bracket(X, Y, coords)


def lie_derivative_function(
    X: List[sp.Expr],
    f: sp.Expr,
    coords: List[sp.Symbol],
) -> sp.Expr:
    """
    Lie derivative of a scalar function f along X.

    L_X f = X^i ∂_i f
    """
    return simplify(sum(X[i] * diff(f, coords[i]) for i in range(len(coords))))
