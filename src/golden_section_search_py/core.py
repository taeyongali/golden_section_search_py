"""Core implementation of the golden section search algorithm.

The golden section search finds the minimum of a unimodal function on a
closed interval using only function evaluations.  No derivatives are
required.  Each iteration shrinks the interval by the inverse golden
ratio, which is the optimal reduction factor for a derivative-free
method that reuses one function value per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, List, Tuple


class GoldenSectionSearchError(ValueError):
    """Raised when the search is given arguments it cannot honour."""


@dataclass(frozen=True)
class GoldenSectionSearchResult:
    """The outcome of a golden section search.

    Attributes
    ----------
    x : float
        The estimated location of the minimum.
    fx : float
        The function value at ``x``.
    iterations : int
        Number of shrinking steps performed.
    evaluations : int
        Number of calls made to the objective function.
    """

    x: float
    fx: float
    iterations: int
    evaluations: int


# The inverse golden ratio.  Using a named constant keeps the arithmetic
# self-documenting and avoids recomputing sqrt(5) on every call.
_INV_PHI: float = (sqrt(5.0) - 1.0) / 2.0


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> GoldenSectionSearchResult:
    """Minimize a unimodal function ``f`` on ``[a, b]``.

    Parameters
    ----------
    f : callable
        A function of one float argument returning a float.  It must be
        unimodal on the interval: strictly decreasing then strictly
        increasing, or monotone.  Passing a non-unimodal function yields
        an arbitrary local minimum, not a global one.
    a, b : float
        Interval bounds.  ``a`` must be strictly less than ``b``.
    tol : float, optional
        Stop when the interval width falls below ``tol``.  Must be
        positive.  Defaults to ``1e-8``.
    max_iter : int, optional
        Hard cap on iterations.  Must be at least 1.  Defaults to 200.

    Returns
    -------
    GoldenSectionSearchResult
        The estimate, function value, and counters.

    Raises
    ------
    GoldenSectionSearchError
        If bounds are not ordered, ``tol`` is not positive, or
        ``max_iter`` is less than 1.

    Notes
    -----
    The algorithm maintains four interior points.  Because the golden
    ratio satisfies ``phi**2 = phi + 1``, one of the two interior points
    from the previous iteration becomes an interior point of the next
    interval, so only one new function evaluation is needed per step.
    The implementation evaluates the initial two interior points and
    then reuses one of them; ``evaluations`` therefore starts at 2.
    """
    if not (a < b):
        raise GoldenSectionSearchError(
            f"bounds must satisfy a < b, got a={a!r}, b={b!r}"
        )
    if not (tol > 0.0):
        raise GoldenSectionSearchError(f"tol must be positive, got {tol!r}")
    if max_iter < 1:
        raise GoldenSectionSearchError(
            f"max_iter must be at least 1, got {max_iter!r}"
        )

    inv_phi = _INV_PHI
    inv_phi_sq = inv_phi * inv_phi

    # Interior points.  c is left of d.
    c = a + inv_phi_sq * (b - a)
    d = a + inv_phi * (b - a)
    fc = f(c)
    fd = f(d)
    evaluations = 2

    iterations = 0
    while (b - a) > tol and iterations < max_iter:
        if fc < fd:
            # Minimum lies in [a, d]; d becomes the new right bound and
            # c is reused as the new d.
            b = d
            d = c
            fd = fc
            c = a + inv_phi_sq * (b - a)
            fc = f(c)
        else:
            # Minimum lies in [c, b]; c becomes the new left bound and
            # d is reused as the new c.
            a = c
            c = d
            fc = fd
            d = a + inv_phi * (b - a)
            fd = f(d)
        evaluations += 1
        iterations += 1

    # With fc < fd the minimum is bracketed by [a, d]; otherwise by
    # [c, b].  Returning the better of the two evaluated points is more
    # accurate than returning the midpoint, which may not have been
    # evaluated at all.
    if fc < fd:
        x, fx = c, fc
    else:
        x, fx = d, fd

    return GoldenSectionSearchResult(
        x=x, fx=fx, iterations=iterations, evaluations=evaluations
    )
