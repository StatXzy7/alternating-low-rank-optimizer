"""Core implementation of the alternating low-rank linear-fit updater.

Notation
--------
W0 : (m, d) frozen base matrix
A  : (m, r) output-side factor
B  : (r, d) input-side factor
X  : (d, n) input samples (one column per sample)
T  : (m, n) targets

Effective matrix::

    W      = W0 + alpha * A @ B
    Y      = W @ X = W0 @ X + alpha * A @ (B @ X)
    loss   = mean over (m*n) entries of (Y - T)^2

Full gradient of the loss w.r.t. the effective matrix W::

    G = (2 / (m * n)) * (W @ X - T) @ X.T
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Union

import numpy as np

__all__ = [
    "LowRankAlternatingUpdater",
    "UpdateSide",
    "make_random_problem",
]


class UpdateSide(enum.Enum):
    """Which factor is updated on the next step."""

    INPUT = "input"   # B (r, d)
    OUTPUT = "output"  # A (m, r)


PathLike = Union[str, Path]

def _as_2d_float(name: str, arr: Any, dtype=np.float64) -> np.ndarray:
    """Convert to a finite 2-D float64 ndarray; raise ValueError on problems."""
    if isinstance(arr, np.matrix):
        arr = np.asarray(arr)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    out = arr.astype(dtype, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf")
    return out


def _as_1d_float(name: str, value: Any) -> float:
    """Validate a finite real scalar (bool rejected)."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar number, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real scalar number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _full_row_rank(mat: np.ndarray, tol: float) -> bool:
    """True iff ``mat`` (p x q, p <= q) has full row rank."""
    p, q = mat.shape
    if p == 0:
        return False
    s = np.linalg.svd(mat, compute_uv=False)
    s_max = float(s[0])
    if s_max == 0.0:
        return False
    return bool(s[-1] > tol * s_max)


def _full_col_rank(mat: np.ndarray, tol: float) -> bool:
    """True iff ``mat`` (p x q, q <= p) has full column rank."""
    p, q = mat.shape
    if q == 0:
        return False
    s = np.linalg.svd(mat, compute_uv=False)
    s_max = float(s[0])
    if s_max == 0.0:
        return False
    return bool(s[-1] > tol * s_max)

class LowRankAlternatingUpdater:
    """Alternating projected-gradient updater for ``W0 + alpha * A @ B``.

    Parameters
    ----------
    W0, X, T : array_like
        Frozen base matrix (m, d), inputs (d, n), targets (m, n).
    A : array_like, shape (m, r)
        Initial output-side (full column rank) factor.
    B : array_like, shape (r, d)
        Initial input-side (full row rank) factor.
    alpha : non-zero finite float
        Increment scale. It must not be zero (the rank of the increment would
        collapse); negative values are allowed.
    lr : float, strictly positive
        Learning rate applied to the projected update.
    rank_tol : float, strictly positive, default 1e-10
        Relative singular-value tolerance: a factor with ``p`` rows is treated
        as rank-deficient when ``sigma_min <= rank_tol * sigma_max``.

    Raises
    ------
    ValueError
        On invalid scale/learning rate, dimension conflicts, non-finite data
        or rank-deficient factors. Constructor failures leave no state.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        W0,
        X,
        T,
        A,
        B,
        alpha,
        lr,
        rank_tol: float = 1e-10,
    ) -> None:
        W0 = _as_2d_float("W0", W0)
        X = _as_2d_float("X", X)
        T = _as_2d_float("T", T)
        A = _as_2d_float("A", A)
        B = _as_2d_float("B", B)

        m, d = W0.shape
        dx, n = X.shape
        mt, nt = T.shape
        ma, r = A.shape
        rb, db = B.shape

        if dx != d:
            raise ValueError(f"X rows {dx} != W0 cols {d}")
        if mt != m:
            raise ValueError(f"T rows {mt} != W0 rows {m}")
        if nt != n:
            raise ValueError(f"T cols {nt} != X cols {n}")
        if ma != m:
            raise ValueError(f"A rows {ma} != W0 rows {m}")
        if rb != r:
            raise ValueError(f"B rows {rb} != A cols {r}")
        if db != d:
            raise ValueError(f"B cols {db} != W0 cols {d}")
        if r <= 0:
            raise ValueError(f"rank r must be positive, got {r}")
        if n <= 0:
            raise ValueError(f"need at least one sample, got n={n}")

        alpha = _as_1d_float("alpha", alpha)
        if alpha == 0.0:
            raise ValueError("alpha must be non-zero")
        lr = _as_1d_float("lr", lr)
        if lr <= 0.0:
            raise ValueError(f"lr must be strictly positive, got {lr}")
        rank_tol = _as_1d_float("rank_tol", rank_tol)
        if rank_tol <= 0.0:
            raise ValueError(
                f"rank_tol must be strictly positive, got {rank_tol}"
            )

        if not _full_col_rank(A, rank_tol):
            raise ValueError("A is not full column rank (output factor)")
        if not _full_row_rank(B, rank_tol):
            raise ValueError("B is not full row rank (input factor)")

        self._W0 = W0
        self._X = X
        self._T = T
        self._A = A
        self._B = B
        self._alpha = alpha
        self._lr = lr
        self._rank_tol = rank_tol
        self._step = 0

    # ------------------------------------------------------------------ #
    # Read-only views of state
    # ------------------------------------------------------------------ #
    @property
    def step_count(self) -> int:
        """Number of completed update steps (the step number)."""
        return self._step

    @property
    def next_side(self) -> UpdateSide:
        """Factor updated by the next ``step()``: INPUT first, then OUTPUT."""
        return UpdateSide.INPUT if self._step % 2 == 0 else UpdateSide.OUTPUT

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def rank_tol(self) -> float:
        return self._rank_tol

    @property
    def W0(self) -> np.ndarray:
        return self._W0

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def T(self) -> np.ndarray:
        return self._T

    @property
    def A(self) -> np.ndarray:
        """Output factor (m, r), read-only view."""
        out = self._A.view()
        out.setflags(write=False)
        return out

    @property
    def B(self) -> np.ndarray:
        """Input factor (r, d), read-only view."""
        out = self._B.view()
        out.setflags(write=False)
        return out

    # ------------------------------------------------------------------ #
    # Forward queries
    # ------------------------------------------------------------------ #
    def effective_delta(self) -> np.ndarray:
        """Current trainable increment ``alpha * A @ B``, shape (m, d)."""
        return self._alpha * (self._A @ self._B)

    def predict(self) -> np.ndarray:
        """Current outputs ``(W0 + alpha*A@B) @ X``, shape (m, n)."""
        return self._W0 @ self._X + self._alpha * (
            self._A @ (self._B @ self._X)
        )

    def residual(self) -> np.ndarray:
        """Prediction minus targets, shape (m, n)."""
        return self.predict() - self._T

    def loss(self) -> float:
        """Mean squared output error ``mean((Y - T)**2)``."""
        e = self.residual()
        m, n = e.shape
        return float(np.einsum("ij,ij->", e, e) / (m * n))

    def full_gradient(self) -> np.ndarray:
        """Full gradient ``d loss / d W`` at the current state, shape (m, d)."""
        e = self.residual()
        m, n = e.shape
        return (2.0 / (m * n)) * (e @ self._X.T)

    def _check_factors_rank(self) -> None:
        """Raise ValueError if either factor lost full rank mid-training."""
        if not _full_col_rank(self._A, self._rank_tol):
            raise ValueError(
                "output factor A is rank deficient; alternating projection "
                "requires full column rank"
            )
        if not _full_row_rank(self._B, self._rank_tol):
            raise ValueError(
                "input factor B is rank deficient; alternating projection "
                "requires full row rank"
            )

    # ------------------------------------------------------------------ #
    # Alternating projected steps
    # ------------------------------------------------------------------ #
    def _input_step(self, G):
        """Input-side (B) step: column-space projection through frozen A."""
        A = self._A
        B = self._B
        try:
            dB = (-self._lr / self._alpha) * np.linalg.solve(A.T @ A, A.T @ G)
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular A^T A while projecting (rank loss?)") from exc
        return A, B + dB

    def _output_step(self, G):
        """Output-side (A) step: row-space projection through frozen B."""
        A = self._A
        B = self._B
        try:
            dA = (-self._lr / self._alpha) * np.linalg.solve(
                B @ B.T, (G @ B.T).T
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("singular B B^T while projecting (rank loss?)") from exc
        return A + dA, B

    # ------------------------------------------------------------------ #
    # Single / multi-step advance
    # ------------------------------------------------------------------ #
    def step(self):
        """Advance one alternating update; one factor changes at most.

        Returns a dict with keys ``step`` (completed step count after the
        update), ``side`` ("input"/"output"), ``loss_before``,
        ``loss_after`` and ``delta_norm`` (Frobenius norm of the effective
        matrix change applied in this step; 0.0 when the gradient vanishes).

        A failing step (rank-deficient factor or non-finite arithmetic) raises
        ValueError and leaves both factors and the step counter untouched.
        """
        self._check_factors_rank()
        side = self.next_side
        loss_before = self.loss()
        G = self.full_gradient()
        if not np.all(np.isfinite(G)):
            raise ValueError("non-finite gradient encountered")

        old_product = self._A @ self._B
        if side is UpdateSide.INPUT:
            A_new, B_new = self._input_step(G)
        else:
            A_new, B_new = self._output_step(G)

        if not (np.all(np.isfinite(A_new)) and np.all(np.isfinite(B_new))):
            raise ValueError("non-finite factor after update; step rejected")

        # Commit only after the whole candidate update is valid.
        self._A = A_new
        self._B = B_new
        self._step += 1
        loss_after = self.loss()
        if not np.isfinite(loss_after):
            raise ValueError("non-finite loss after update; step rejected")
        delta_eff = self._alpha * ((A_new @ B_new) - old_product)
        return {
            "step": self._step,
            "side": side.value,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta_norm": float(np.linalg.norm(delta_eff)),
        }


    def run(self, n: int) -> list[dict]:
        """Run ``n`` alternating steps and return their step records.

        ``n`` must be a positive int. Each step commits independently; if a
        step fails, already committed steps remain and the failing one leaves
        factors and step count unchanged (the ValueError propagates).
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise ValueError(f"n must be a positive int, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        records = []
        for _ in range(int(n)):
            records.append(self.step())
        return records

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: PathLike) -> None:
        """Save the complete training state to a ``.npz`` file.

        Stored: W0, X, T, A, B, alpha, lr, rank_tol and step. The next update
        side is determined by ``step`` (even -> input, odd -> output), so it
        is not stored separately.
        """
        path = Path(path)
        np.savez(
            path,
            version=np.int64(self.STATE_VERSION),
            W0=self._W0,
            X=self._X,
            T=self._T,
            A=self._A,
            B=self._B,
            alpha=np.float64(self._alpha),
            lr=np.float64(self._lr),
            rank_tol=np.float64(self._rank_tol),
            step=np.int64(self._step),
        )

    @classmethod
    def load(cls, path: PathLike) -> "LowRankAlternatingUpdater":
        """Restore an updater previously written with :meth:`save`.

        Construction runs the same validation (shapes, finiteness, ranks) and
        then restores the step counter, so the resumed updater continues with
        exactly the same factor and next-side state as an uninterrupted run.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            required = {
                "version", "W0", "X", "T", "A", "B",
                "alpha", "lr", "rank_tol", "step",
            }
            missing = required - keys
            if missing:
                raise ValueError(f"state file missing entries: {sorted(missing)}")
            version = int(data["version"])
            if version != cls.STATE_VERSION:
                raise ValueError(f"unsupported state version {version}")
            obj = cls(
                data["W0"], data["X"], data["T"],
                data["A"], data["B"],
                float(data["alpha"]), float(data["lr"]),
                rank_tol=float(data["rank_tol"]),
            )
            step = int(data["step"])
            if step < 0:
                raise ValueError(f"invalid step count in state: {step}")
            obj._step = step
        return obj

def make_random_problem(
    m: int = 6,
    d: int = 5,
    n: int = 12,
    r: int = 2,
    alpha: float = 0.5,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[LowRankAlternatingUpdater, dict]:
    """Build a small random valid problem for tests/demos.

    Returns ``(updater, raw)`` where ``raw`` holds the underlying arrays
    (W0, X, T, A, B, alpha, lr) so tests can build independent references.
    Initial factors are well conditioned and targets contain an independent
    random component so the gradient is generally non-zero.
    """
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(m, d))
    X = rng.normal(size=(d, n))
    A0 = rng.normal(size=(m, r))
    B0 = rng.normal(size=(r, d))
    noise = 0.1 * rng.normal(size=(m, n))
    T = W0 @ X + alpha * A0 @ B0 @ X + noise
    A = rng.normal(scale=0.3, size=(m, r))
    B = rng.normal(scale=0.3, size=(r, d))
    updater = LowRankAlternatingUpdater(W0, X, T, A, B, alpha, lr)
    raw = {"W0": W0, "X": X, "T": T, "A": A, "B": B, "alpha": alpha, "lr": lr}
    return updater, raw
