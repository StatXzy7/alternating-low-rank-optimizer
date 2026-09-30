"""Alternating low-rank linear-model updates with projected gradients."""

from __future__ import annotations

import numbers
import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import numpy as np


class LowRankUpdateError(ValueError):
    """Base class for invalid low-rank updater states or arguments."""


class DimensionError(LowRankUpdateError):
    """Raised when matrix or sample dimensions are incompatible."""


class RankError(LowRankUpdateError):
    """Raised when a factor is not full rank."""


@dataclass(frozen=True)
class StepResult:
    """Information from one completed alternating update."""

    step: int
    updated_side: str
    loss_before: float
    loss_after: float
    projected_gradient: np.ndarray
    matrix_update: np.ndarray


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise DimensionError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise LowRankUpdateError(f"{name} must contain only finite values")
    return array


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise LowRankUpdateError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise LowRankUpdateError(f"{name} must be finite and positive")
    return result


def _full_rank(matrix: np.ndarray) -> bool:
    rank = min(matrix.shape)
    try:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    tolerance = max(matrix.shape) * singular_values[0] * np.finfo(matrix.dtype).eps
    return singular_values[rank - 1] > tolerance


class AlternatingLowRankUpdater:
    """Fit ``W + alpha * B @ A`` by alternating exact projected-gradient steps.

    The input-side factor is ``A`` with shape ``(rank, input_dim)`` and the
    output-side factor is ``B`` with shape ``(output_dim, rank)``.  Each step
    updates one factor so that, in effective-weight space, the update is
    ``-learning_rate * P(gradient)``; ``P`` is the Frobenius-orthogonal
    projection onto all matrices expressible while the other factor is frozen.

    The loss is the elementwise mean squared output error over all samples and
    output coordinates.
    """

    INPUT_SIDE = "input"
    OUTPUT_SIDE = "output"

    def __init__(
        self,
        frozen_weight: np.ndarray,
        inputs: np.ndarray,
        targets: np.ndarray,
        input_factor: np.ndarray,
        output_factor: np.ndarray,
        alpha: float,
        learning_rate: float,
        *,
        next_side: str = INPUT_SIDE,
        step: int = 0,
    ) -> None:
        weight = _as_2d_float("frozen_weight", frozen_weight)
        x = _as_2d_float("inputs", inputs)
        y = _as_2d_float("targets", targets)
        factor_a = _as_2d_float("input_factor", input_factor)
        factor_b = _as_2d_float("output_factor", output_factor)

        output_dim, input_dim = weight.shape
        sample_count = x.shape[0]

        if sample_count == 0:
            raise DimensionError("inputs must contain at least one sample")
        if x.shape != (sample_count, input_dim):
            raise DimensionError(
                "inputs shape must be (sample_count, input_dim); "
                f"got {x.shape} for frozen_weight {weight.shape}"
            )
        if y.shape != (sample_count, output_dim):
            raise DimensionError(
                "targets shape must be (sample_count, output_dim); "
                f"got {y.shape}, expected ({sample_count}, {output_dim})"
            )
        if 0 in weight.shape:
            raise DimensionError("frozen_weight must have non-zero dimensions")
        if factor_a.ndim != 2 or factor_a.shape[1] != input_dim:
            raise DimensionError(
                f"input_factor shape must be (rank, {input_dim}); "
                f"got {factor_a.shape}"
            )
        if factor_b.ndim != 2 or factor_b.shape[0] != output_dim:
            raise DimensionError(
                f"output_factor shape must be ({output_dim}, rank); "
                f"got {factor_b.shape}"
            )

        rank = factor_a.shape[0]
        if rank == 0 or factor_b.shape[1] != rank:
            raise DimensionError(
                "the two factors must share the same positive rank; "
                f"got ranks {rank} and {factor_b.shape[1]}"
            )

        if isinstance(step, bool) or not isinstance(step, numbers.Integral) or step < 0:
            raise LowRankUpdateError("step must be a non-negative integer")
        if next_side not in (self.INPUT_SIDE, self.OUTPUT_SIDE):
            raise LowRankUpdateError("next_side must be 'input' or 'output'")

        self._w = weight.copy()
        self._x = x.copy()
        self._y = y.copy()
        self._a = factor_a.copy()
        self._b = factor_b.copy()
        self._alpha = _positive_float("alpha", alpha)
        self._learning_rate = _positive_float("learning_rate", learning_rate)
        self._step = int(step)
        self._next_side = next_side

        if not _full_rank(self._a):
            raise RankError("input_factor must have full row rank")
        if not _full_rank(self._b):
            raise RankError("output_factor must have full column rank")

    @property
    def frozen_weight(self) -> np.ndarray:
        return self._w.copy()

    @property
    def inputs(self) -> np.ndarray:
        return self._x.copy()

    @property
    def targets(self) -> np.ndarray:
        return self._y.copy()

    @property
    def input_factor(self) -> np.ndarray:
        return self._a.copy()

    @property
    def output_factor(self) -> np.ndarray:
        return self._b.copy()

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def next_side(self) -> str:
        return self._next_side

    def effective_delta(self) -> np.ndarray:
        """Return the current trainable increment ``alpha * B @ A``."""
        return self._alpha * (self._b @ self._a)

    def effective_weight(self) -> np.ndarray:
        """Return the frozen weight plus the current trainable increment."""
        return self._w + self.effective_delta()

    def predict(self, inputs: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the current effective linear map."""
        x = self._x if inputs is None else _as_2d_float("inputs", inputs)
        if inputs is not None and x.shape[1] != self._w.shape[1]:
            raise DimensionError("prediction inputs have the wrong feature count")
        return x @ self.effective_weight().T

    def residual(self) -> np.ndarray:
        return self.predict() - self._y

    def current_loss(self) -> float:
        """Return the current elementwise mean squared error."""
        error = self.residual()
        return float(np.mean(error * error))

    def current_gradient(self) -> np.ndarray:
        """Return the full effective-weight gradient at the current state."""
        error = self.residual()
        return (error.T @ self._x) / float(error.size)

    def _project_input_side(self, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not (_full_rank(self._a) and _full_rank(self._b)):
            raise RankError("both factors must be full rank before an update")
        basis_gram = self._b.T @ self._b
        factor_change = np.linalg.solve(basis_gram, self._b.T @ gradient)
        projected = self._b @ factor_change
        candidate = self._a - (self._learning_rate / self._alpha) * factor_change
        return projected, candidate

    def _project_output_side(self, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not (_full_rank(self._a) and _full_rank(self._b)):
            raise RankError("both factors must be full rank before an update")
        basis_gram = self._a @ self._a.T
        factor_change = np.linalg.solve(
            basis_gram, self._a @ gradient.T
        ).T
        projected = factor_change @ self._a
        candidate = self._b - (self._learning_rate / self._alpha) * factor_change
        return projected, candidate

    def step(self) -> StepResult:
        """Advance one alternating update and return measurements for it."""
        updated_side = self._next_side
        old_a = self._a.copy()
        old_b = self._b.copy()
        loss_before = self.current_loss()
        gradient = self.current_gradient()

        try:
            if updated_side == self.INPUT_SIDE:
                projected, candidate = self._project_input_side(gradient)
                if not np.all(np.isfinite(candidate)) or not _full_rank(candidate):
                    raise RankError("updated input_factor would be rank deficient")
                candidate_b = old_b
            else:
                projected, candidate = self._project_output_side(gradient)
                if not np.all(np.isfinite(candidate)) or not _full_rank(candidate):
                    raise RankError("updated output_factor would be rank deficient")
                candidate_a = old_a

            matrix_update = -self._learning_rate * projected
            expected_residual = self._residual_for(
                candidate_a if updated_side == self.OUTPUT_SIDE else candidate,
                candidate_b if updated_side == self.INPUT_SIDE else candidate,
            )
            if not np.all(np.isfinite(expected_residual)):
                raise LowRankUpdateError("updated state contains non-finite values")
        except Exception:
            self._a = old_a
            self._b = old_b
            self._next_side = updated_side
            raise

        if updated_side == self.INPUT_SIDE:
            self._a = candidate
        else:
            self._b = candidate

        self._step += 1
        self._next_side = (
            self.OUTPUT_SIDE if updated_side == self.INPUT_SIDE else self.INPUT_SIDE
        )

        return StepResult(
            step=self._step,
            updated_side=updated_side,
            loss_before=loss_before,
            loss_after=self.current_loss(),
            projected_gradient=projected.copy(),
            matrix_update=matrix_update.copy(),
        )

    def _residual_for(self, factor_a: np.ndarray, factor_b: np.ndarray) -> np.ndarray:
        effective_weight = self._w + self._alpha * (factor_b @ factor_a)
        return self._x @ effective_weight.T - self._y

    def run(self, steps: int) -> list[StepResult]:
        """Run ``steps`` alternating updates, recomputing at every boundary."""
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 0:
            raise LowRankUpdateError("steps must be a non-negative integer")
        return [self.step() for _ in range(int(steps))]

    def save_state(self, path: str) -> None:
        """Atomically save all data needed to resume exactly."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with NamedTemporaryFile(mode="wb", suffix=".tmp", dir=directory, delete=False) as tmp:
            temporary_path = tmp.name
            try:
                np.savez(
                    tmp,
                    frozen_weight=self._w,
                    inputs=self._x,
                    targets=self._y,
                    input_factor=self._a,
                    output_factor=self._b,
                    alpha=np.asarray(self._alpha),
                    learning_rate=np.asarray(self._learning_rate),
                    step=np.asarray(self._step),
                    next_side=np.asarray(self._next_side),
                )
                tmp.flush()
                os.fsync(tmp.fileno())
                os.replace(temporary_path, path)
            except BaseException:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                raise

    @classmethod
    def load_state(cls, path: str) -> "AlternatingLowRankUpdater":
        """Load a state previously written by :meth:`save_state`."""
        with np.load(path, allow_pickle=False) as state:
            return cls(
                frozen_weight=state["frozen_weight"],
                inputs=state["inputs"],
                targets=state["targets"],
                input_factor=state["input_factor"],
                output_factor=state["output_factor"],
                alpha=float(state["alpha"]),
                learning_rate=float(state["learning_rate"]),
                next_side=str(state["next_side"]),
                step=int(state["step"]),
            )
