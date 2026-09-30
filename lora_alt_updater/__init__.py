"""Alternating low-rank linear-fit updater (NumPy, CPU).

The trainable increment to a frozen base matrix ``W0`` is represented as the
product of two factors::

    W(x) = W0 + alpha * A @ B          (effective matrix)
    Y    = W @ X

with ``A`` shape (m, r) the *output-side* factor and ``B`` shape (r, d) the
*input-side* factor. Per step exactly one factor changes; the first step
updates the input side (``B``), the next step the output side (``A``), and so
on. After the input-side step, loss/gradient are recomputed from the new state
before the output-side step is taken.

The step is *not* naive factor SGD. With the other factor frozen, the feasible
set of effective matrix updates is a linear subspace. The applied update is
the Euclidean (Frobenius) projection of the full effective-matrix gradient
onto that feasible subspace, scaled by the learning rate. Because the
projection uses the *pseudo-inverse* of the frozen factor it is invariant to
invertible gauge transforms ``(A, B) -> (A R^{-1}, R B)``.
"""

from .updater import (
    LowRankAlternatingUpdater,
    UpdateSide,
    make_random_problem,
)

__all__ = [
    "LowRankAlternatingUpdater",
    "UpdateSide",
    "make_random_problem",
]
