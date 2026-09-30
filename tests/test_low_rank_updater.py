import os
import tempfile
import unittest

import numpy as np

from low_rank_updater import (
    AlternatingLowRankUpdater,
    DimensionError,
    LowRankUpdateError,
    RankError,
)


TOL = 1e-10


def make_problem(m=4, n=5, sample_count=7, rank=2, seed=3, next_side="input"):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, n))
    x = rng.standard_normal((sample_count, n))
    a0 = rng.standard_normal((rank, n))
    b0 = rng.standard_normal((m, rank))
    y = x @ (w + 0.07 * b0 @ a0).T + 0.1 * rng.standard_normal((sample_count, m))
    a = rng.standard_normal((rank, n))
    b = rng.standard_normal((m, rank))
    updater = AlternatingLowRankUpdater(
        w, x, y, a, b, alpha=0.07, learning_rate=0.05, next_side=next_side
    )
    return w, x, y, a, b, updater


def reference_state(w, x, y, a, b, alpha):
    residual = x @ (w + alpha * b @ a).T - y
    loss = np.mean(residual**2)
    gradient = residual.T @ x / residual.size
    return residual, loss, gradient


def projected_delta(b, a, gradient, lr, alpha, side):
    if side == "input":
        coefficient = np.linalg.lstsq(b, gradient, rcond=None)[0]
        projection = b @ coefficient
        factor_delta = -lr * np.linalg.solve(b.T @ b, b.T @ gradient)
        return projection, factor_delta / alpha

    basis = a.T
    coefficient = np.linalg.lstsq(basis, gradient.T, rcond=None)[0].T
    projection = coefficient @ a
    factor_delta = -lr * (np.linalg.solve(a @ a.T, a @ gradient.T)).T
    return projection, factor_delta / alpha


class AlternatingLowRankUpdaterTest(unittest.TestCase):
    def assert_array_close(self, actual, expected, tol=TOL):
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)

    def test_projected_step_and_recomputed_output_side(self):
        w, x, y, a, b, updater = make_problem()

        _, loss0, gradient0 = reference_state(w, x, y, a, b, updater.alpha)
        projection, expected_a_delta = projected_delta(
            b, a, gradient0, updater.learning_rate, updater.alpha, "input"
        )
        result = updater.step()

        self.assertEqual(result.updated_side, "input")
        self.assertEqual(result.step, 1)
        self.assertAlmostEqual(result.loss_before, loss0)
        self.assert_array_close(result.projected_gradient, projection)
        self.assert_array_close(result.matrix_update, -updater.learning_rate * projection)
        self.assert_array_close(updater.input_factor - a, expected_a_delta)
        self.assert_array_close(updater.output_factor, b)
        self.assert_array_close(
            updater.effective_delta(),
            updater.alpha * b @ updater.input_factor,
        )

        a1 = updater.input_factor
        b1 = b
        _, loss1, gradient1 = reference_state(w, x, y, a1, b1, updater.alpha)
        projection1, expected_b_delta = projected_delta(
            b1, a1, gradient1, updater.learning_rate, updater.alpha, "output"
        )
        result1 = updater.step()

        self.assertEqual(result1.updated_side, "output")
        self.assertEqual(result1.step, 2)
        self.assertEqual(updater.next_side, "input")
        self.assertAlmostEqual(result1.loss_before, loss1)
        self.assert_array_close(result1.projected_gradient, projection1)
        self.assert_array_close(updater.input_factor, a1)
        self.assert_array_close(updater.output_factor - b1, expected_b_delta)
        self.assertAlmostEqual(result1.loss_after, updater.current_loss())

    def test_projection_is_not_ordinary_factor_gradient_descent(self):
        w, x, y, a, b, updater = make_problem(seed=17)
        gradient = updater.current_gradient()
        ordinary_a_delta = -updater.learning_rate * (gradient.T @ b).T
        projection = b @ np.linalg.solve(b.T @ b, b.T @ gradient)

        updater.step()
        actual_a_delta = updater.input_factor - a

        self.assertGreater(np.linalg.norm(actual_a_delta - ordinary_a_delta / 0.07), 1e-3)
        self.assert_array_close(
            actual_a_delta,
            -updater.learning_rate
            * np.linalg.solve(b.T @ b, b.T @ gradient)
            / updater.alpha,
        )
        self.assertGreater(
            np.linalg.norm(b.T @ (gradient - projection)) / np.linalg.norm(gradient),
            0.0,
        )

    def test_projection_is_orthogonal_to_complement(self):
        w, x, y, a, b, updater = make_problem(seed=23)
        gradient = updater.current_gradient()
        result = updater.step()
        residual_projection = gradient - result.projected_gradient

        self.assert_array_close(
            b.T @ residual_projection,
            np.zeros((b.shape[1], gradient.shape[1])),
            tol=1e-9,
        )

    def test_invertible_factor_transform_gives_equivalent_matrix_step(self):
        for side in ("input", "output"):
            with self.subTest(side=side):
                w, x, y, a, b, original = make_problem(seed=31, next_side=side)
                rank = a.shape[0]
                rng = np.random.default_rng(101 + int(side == "output"))
                transform = np.eye(rank) + 0.05 * rng.standard_normal((rank, rank))
                transformed = AlternatingLowRankUpdater(
                    w,
                    x,
                    y,
                    np.linalg.solve(transform, a),
                    b @ transform,
                    alpha=original.alpha,
                    learning_rate=original.learning_rate,
                    next_side=side,
                )

                self.assert_array_close(
                    transformed.effective_delta(), original.effective_delta()
                )
                original_delta = original.effective_delta()
                transformed_delta = transformed.effective_delta()
                original.step()
                transformed.step()

                self.assert_array_close(
                    transformed.effective_delta() - transformed_delta,
                    original.effective_delta() - original_delta,
                    tol=1e-9,
                )

    def test_save_and_restore_matches_uninterrupted_execution(self):
        for stop_at in (0, 1, 2, 3):
            with self.subTest(stop_at=stop_at):
                _, _, _, _, _, interrupted = make_problem(seed=41)
                _, _, _, _, _, continuous = make_problem(seed=41)

                interrupted.run(stop_at)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "state.npz")
                    interrupted.save_state(path)
                    restored = AlternatingLowRankUpdater.load_state(path)

                self.assertEqual(restored.step_count, stop_at)
                self.assertEqual(restored.next_side, interrupted.next_side)
                self.assert_array_close(restored.input_factor, interrupted.input_factor)
                self.assert_array_close(restored.output_factor, interrupted.output_factor)

                interrupted.run(3)
                restored.run(3)
                continuous.run(stop_at + 3)

                self.assertEqual(restored.step_count, continuous.step_count)
                self.assertEqual(restored.next_side, continuous.next_side)
                self.assert_array_close(restored.input_factor, continuous.input_factor)
                self.assert_array_close(restored.output_factor, continuous.output_factor)
                self.assert_array_close(
                    restored.effective_delta(), continuous.effective_delta()
                )

    def test_zero_gradient_leaves_factors_unchanged_but_advances_schedule(self):
        w, x, y, a, b, updater = make_problem(seed=4)
        y = x @ (w + updater.alpha * b @ a).T
        updater = AlternatingLowRankUpdater(
            w, x, y, a, b, alpha=0.07, learning_rate=0.05, next_side="input"
        )

        result = updater.step()

        self.assertAlmostEqual(updater.current_loss(), 0.0)
        self.assert_array_close(result.projected_gradient, np.zeros_like(w))
        self.assert_array_close(result.matrix_update, np.zeros_like(w))
        self.assert_array_close(updater.input_factor, a)
        self.assert_array_close(updater.output_factor, b)
        self.assertEqual(updater.step_count, 1)
        self.assertEqual(updater.next_side, "output")

    def test_invalid_inputs_are_rejected_without_state_changes(self):
        w, x, y, a, b, updater = make_problem()
        before_a = updater.input_factor
        before_b = updater.output_factor
        before_step = updater.step_count

        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(w, x[:, :-1], y, a, b, 0.07, 0.05)
        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(w, x, y[:, :-1], a, b, 0.07, 0.05)
        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(w, x, y, a[:-1, :], b, 0.07, 0.05)
        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(w, x, y, a, b[:, :-1], 0.07, 0.05)
        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(w, x, y, np.empty((0, w.shape[1])), b, 0.07, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, 0.0, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, -0.07, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, 0.07, 0.0)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, 0.07, -0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, np.nan, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, b, 0.07, np.inf)

        bad_x = x.copy()
        bad_x[0, 0] = np.inf
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, bad_x, y, a, b, 0.07, 0.05)
        bad_w = w.copy()
        bad_w[0, 0] = np.nan
        bad_y = y.copy()
        bad_y[0, 0] = np.nan
        bad_a = a.copy()
        bad_a[0, 0] = np.nan
        bad_b = b.copy()
        bad_b[0, 0] = np.nan
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, bad_y, a, b, 0.07, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(bad_w, x, y, a, b, 0.07, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, bad_a, b, 0.07, 0.05)
        with self.assertRaises(LowRankUpdateError):
            AlternatingLowRankUpdater(w, x, y, a, bad_b, 0.07, 0.05)
        with self.assertRaises(DimensionError):
            AlternatingLowRankUpdater(
                w,
                np.empty((0, w.shape[1])),
                np.empty((0, w.shape[0])),
                a,
                b,
                0.07,
                0.05,
            )

        self.assert_array_close(updater.input_factor, before_a)
        self.assert_array_close(updater.output_factor, before_b)
        self.assertEqual(updater.step_count, before_step)

    def test_rank_deficient_factors_are_rejected(self):
        w, x, y, a, b, _ = make_problem(seed=6)

        with self.assertRaises(RankError):
            AlternatingLowRankUpdater(w, x, y, a * 0.0, b, 0.07, 0.05)
        with self.assertRaises(RankError):
            AlternatingLowRankUpdater(w, x, y, a, b * 0.0, 0.07, 0.05)

    def test_rank_deficient_proposed_update_is_atomic(self):
        m, n, rank = 2, 2, 2
        w = np.zeros((m, n))
        b = np.eye(2)
        a = np.array([[2.0, 1.0], [1.0, 2.0]])
        x = 2.0 * np.eye(2)
        alpha = 1.0
        lr = 2.0
        y = a
        updater = AlternatingLowRankUpdater(
            w, x, y, a, b, alpha=alpha, learning_rate=lr, next_side="input"
        )
        before_a = updater.input_factor.copy()
        before_b = updater.output_factor.copy()
        before_step = updater.step_count

        with self.assertRaises(RankError):
            updater.step()

        self.assert_array_close(updater.input_factor, before_a)
        self.assert_array_close(updater.output_factor, before_b)
        self.assertEqual(updater.step_count, before_step)
        self.assertEqual(updater.next_side, "input")

    def test_rank_deficient_output_side_proposed_update_is_atomic(self):
        w = np.zeros((2, 2))
        a = np.eye(2)
        b = np.array([[2.0, 1.0], [1.0, 1.0]])
        x = 2.0 * np.eye(2)
        y = b.T
        alpha = 1.0
        lr = 2.0
        updater = AlternatingLowRankUpdater(
            w,
            x,
            y,
            a,
            b,
            alpha=alpha,
            learning_rate=lr,
            next_side="output",
        )

        with self.assertRaises(RankError):
            updater.step()

        self.assert_array_close(updater.input_factor, a)
        self.assert_array_close(updater.output_factor, b)
        self.assertEqual(updater.step_count, 0)
        self.assertEqual(updater.next_side, "output")

    def test_frozen_weight_remains_unchanged(self):
        w, x, y, _, _, updater = make_problem(seed=9)
        observed = updater.frozen_weight
        observed.fill(100.0)

        updater.run(4)

        self.assert_array_close(updater.frozen_weight, w)


if __name__ == "__main__":
    unittest.main()
