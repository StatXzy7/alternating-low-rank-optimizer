"""Unittests for the alternating low-rank linear-fit updater.

Numerical tolerances (all arithmetic is float64):
* factor / effective-matrix / loss equality: rtol=1e-9, atol=1e-8;
* projection orthogonality residuals: absolute tolerance 1e-10;
* "frozen / unchanged parameter" assertions use exact np.array_equal
  (those bytes must never be touched by a step);
* the exact zero-gradient fixture (X == 0) requires an exactly zero
  effective update, also checked with np.array_equal.
"""

import os
import tempfile
import unittest

import numpy as np

from lora_alt_updater import (
    LowRankAlternatingUpdater,
    UpdateSide,
    make_random_problem,
)

RTOL = 1e-9
ATOL = 1e-8
ORTH_ATOL = 1e-10


def reference_full_gradient(W0, X, T, A, B, alpha):
    """G = 2/(m*n) * (W0 X + alpha A B X - T) X.T (independent copy)."""
    m, n = T.shape
    E = W0 @ X + alpha * A @ (B @ X) - T
    return (2.0 / (m * n)) * E @ X.T


def reference_input_step(W0, X, T, A, B, alpha, lr):
    """Reference input-side (B) step using np.linalg.lstsq throughout."""
    G = reference_full_gradient(W0, X, T, A, B, alpha)
    C, *_ = np.linalg.lstsq(A, G, rcond=None)        # min_C ||G - A C||
    P = A @ C
    dB, *_ = np.linalg.lstsq(A, (-lr / alpha) * P, rcond=None)
    return G, P, A.copy(), B + dB


def reference_output_step(W0, X, T, A, B, alpha, lr):
    """Reference output-side (A) step using np.linalg.lstsq throughout."""
    G = reference_full_gradient(W0, X, T, A, B, alpha)
    Zt, *_ = np.linalg.lstsq(B.T, G.T, rcond=None)   # min_Z ||G.T - B.T Z.T||
    P = (B.T @ Zt).T
    dAt, *_ = np.linalg.lstsq(B.T, (-lr / alpha) * P.T, rcond=None)
    return G, P, A + dAt.T, B.copy()

class ProjectionReferenceTests(unittest.TestCase):
    """The implemented step matches an independent lstsq-based reference."""

    def test_input_step_matches_reference_and_is_best_projection(self):
        u, raw = make_random_problem(seed=11)
        G, P, A_ref, B_ref = reference_input_step(
            raw["W0"], raw["X"], raw["T"], raw["A"], raw["B"],
            raw["alpha"], raw["lr"],
        )
        np.testing.assert_allclose(u.full_gradient(), G, RTOL, ATOL)

        # Remainder is orthogonal to the feasible subspace col(A):
        # the projection is the (unique) best approximation in Frobenius norm.
        np.testing.assert_allclose(
            raw["A"].T @ (G - P), 0.0, atol=ORTH_ATOL
        )
        # Optimality: perturbing P within col(A) cannot reduce ||G - .||.
        rng = np.random.default_rng(0)
        resid_norm = np.linalg.norm(G - P)
        for _ in range(20):
            Cp = rng.normal(scale=0.5, size=P.shape)
            Q = raw["A"] @ (np.linalg.lstsq(raw["A"], Cp, rcond=None)[0])
            self.assertGreaterEqual(np.linalg.norm(G - P - Q), resid_norm - 1e-12)

        A_before = u.A.copy()
        rec = u.step()
        self.assertEqual(rec["side"], "input")
        np.testing.assert_allclose(u.B, B_ref, RTOL, ATOL)
        np.testing.assert_array_equal(u.A, A_before)      # frozen exactly
        np.testing.assert_array_equal(u.A, A_ref)
        # effective update equals -lr * P exactly
        eff = u.effective_delta() - raw["alpha"] * (A_before @ raw["B"])
        np.testing.assert_allclose(eff, -raw["lr"] * P, RTOL, ATOL)
        self.assertEqual(u.step_count, 1)

    def test_output_step_matches_reference_and_is_best_projection(self):
        u, raw = make_random_problem(seed=12)
        u.step()  # consume the input step first (alternation contract)
        A_mid, B_mid = u.A.copy(), u.B.copy()
        G, P, A_ref, B_ref = reference_output_step(
            raw["W0"], raw["X"], raw["T"], A_mid, B_mid,
            raw["alpha"], raw["lr"],
        )
        np.testing.assert_allclose(u.full_gradient(), G, RTOL, ATOL)
        # Remainder is orthogonal to the feasible row-space row(B).
        np.testing.assert_allclose(
            (G - P) @ B_mid.T, 0.0, atol=ORTH_ATOL
        )
        resid_norm = np.linalg.norm(G - P)
        rng = np.random.default_rng(1)
        for _ in range(20):
            # Any element of the feasible space has form Z B (Z: m x r).
            Q = rng.normal(
                scale=0.5, size=(P.shape[0], B_mid.shape[0])
            ) @ B_mid
            self.assertGreaterEqual(np.linalg.norm(G - P - Q), resid_norm - 1e-12)

        B_before = u.B.copy()
        rec = u.step()
        self.assertEqual(rec["side"], "output")
        np.testing.assert_allclose(u.A, A_ref, RTOL, ATOL)
        np.testing.assert_array_equal(u.B, B_before)
        np.testing.assert_array_equal(u.B, B_ref)
        eff = u.effective_delta() - raw["alpha"] * (A_mid @ B_mid)
        np.testing.assert_allclose(eff, -raw["lr"] * P, RTOL, ATOL)
        self.assertEqual(u.step_count, 2)

class AlternationOrderTests(unittest.TestCase):
    """First input, then output, and the second step uses fresh state."""

    def test_sides_alternate_input_first(self):
        u, _ = make_random_problem(seed=21)
        self.assertIs(u.next_side, UpdateSide.INPUT)
        seen = [u.step()["side"] for _ in range(6)]
        self.assertEqual(seen, ["input", "output"] * 3)
        self.assertEqual(u.step_count, 6)
        self.assertIs(u.next_side, UpdateSide.INPUT)

    def test_output_step_uses_state_after_input_step(self):
        """The output-step gradient must be the post-input-step gradient."""
        u, raw = make_random_problem(seed=22)
        u.step()
        G_after_input = reference_full_gradient(
            raw["W0"], raw["X"], raw["T"], u.A, u.B, raw["alpha"]
        )
        _, _, A_ref, B_ref = reference_output_step(
            raw["W0"], raw["X"], raw["T"], u.A, u.B,
            raw["alpha"], raw["lr"],
        )
        np.testing.assert_allclose(u.full_gradient(), G_after_input, RTOL, ATOL)
        u.step()
        np.testing.assert_allclose(u.A, A_ref, RTOL, ATOL)
        np.testing.assert_array_equal(u.B, B_ref)

    def test_only_one_factor_changes_per_step(self):
        u, _ = make_random_problem(seed=23)
        A0, B0 = u.A.copy(), u.B.copy()
        u.step()
        self.assertTrue(np.array_equal(u.A, A0))
        self.assertFalse(np.array_equal(u.B, B0))
        A1, B1 = u.A.copy(), u.B.copy()
        u.step()
        self.assertFalse(np.array_equal(u.A, A1))
        self.assertTrue(np.array_equal(u.B, B1))

    def test_loss_is_recomputed_and_small_lr_decreases_loss(self):
        # With a small learning rate the projected (downhill) update lowers
        # the quadratic loss on each alternating step.
        u, _ = make_random_problem(seed=24, lr=0.02)
        for _ in range(4):
            before = u.loss()
            rec = u.step()
            self.assertAlmostEqual(rec["loss_before"], before, places=12)
            self.assertLess(rec["loss_after"], before)

    def test_run_advances_multiple_steps(self):
        u, _ = make_random_problem(seed=25)
        recs = u.run(5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(u.step_count, 5)
        self.assertEqual([r["step"] for r in recs], [1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            u.run(0)
        with self.assertRaises(ValueError):
            u.run(-1)
        with self.assertRaises(ValueError):
            u.run(2.0)
        with self.assertRaises(ValueError):
            u.run(True)


class FrozenBaseTests(unittest.TestCase):
    """The frozen base matrix W0 never changes."""

    def test_w0_unchanged_through_many_steps(self):
        u, raw = make_random_problem(seed=31)
        u.run(6)
        np.testing.assert_array_equal(u.W0, raw["W0"])


class TransformEquivalenceTests(unittest.TestCase):
    """(A, B) -> (A R^{-1}, R B) with invertible R preserves dynamics."""

    def setUp(self):
        self.u1, self.raw = make_random_problem(seed=41, alpha=0.5, lr=0.05)
        rng = np.random.default_rng(7)
        r = self.raw["A"].shape[1]
        self.R = np.eye(r) + 0.3 * rng.normal(size=(r, r))

    def test_initial_effective_delta_identical(self):
        u2 = LowRankAlternatingUpdater(
            self.raw["W0"], self.raw["X"], self.raw["T"],
            self.raw["A"] @ np.linalg.inv(self.R), self.R @ self.raw["B"],
            self.raw["alpha"], self.raw["lr"],
        )
        np.testing.assert_allclose(
            u2.effective_delta(), self.u1.effective_delta(), RTOL, ATOL
        )

    def test_equivalent_next_matrix_update_on_both_sides(self):
        raw = self.raw
        u2 = LowRankAlternatingUpdater(
            raw["W0"], raw["X"], raw["T"],
            raw["A"] @ np.linalg.inv(self.R), self.R @ raw["B"],
            raw["alpha"], raw["lr"],
        )
        # input side (step 1)
        d1_0 = self.u1.effective_delta()
        d2_0 = u2.effective_delta()
        self.u1.step(); u2.step()
        upd1 = self.u1.effective_delta() - d1_0
        upd2 = u2.effective_delta() - d2_0
        np.testing.assert_allclose(upd1, upd2, RTOL, ATOL)
        self.assertAlmostEqual(self.u1.loss(), u2.loss(), places=14)
        # output side (step 2)
        d1_1 = self.u1.effective_delta()
        d2_1 = u2.effective_delta()
        self.u1.step(); u2.step()
        np.testing.assert_allclose(
            self.u1.effective_delta() - d1_1,
            u2.effective_delta() - d2_1,
            RTOL, ATOL,
        )
        # several more steps stay in lockstep
        self.u1.run(4); u2.run(4)
        np.testing.assert_allclose(
            self.u1.effective_delta(), u2.effective_delta(), RTOL, ATOL
        )
        self.assertAlmostEqual(self.u1.loss(), u2.loss(), places=14)

    def test_equivalence_under_negative_alpha_pairing(self):
        """(alpha, A, B) and (-alpha, -A, B) represent the same increment."""
        raw = self.raw
        ua, _ = make_random_problem(seed=42, alpha=0.8)
        Aa = np.asarray(ua.A)
        Bb = np.asarray(ua.B)
        ub = LowRankAlternatingUpdater(
            ua.W0, ua.X, ua.T, -Aa, Bb, -0.8, ua.lr
        )
        np.testing.assert_allclose(
            ua.effective_delta(), ub.effective_delta(), RTOL, ATOL
        )
        da = ua.effective_delta(); db = ub.effective_delta()
        ua.step(); ub.step()
        np.testing.assert_allclose(
            ua.effective_delta() - da, ub.effective_delta() - db, RTOL, ATOL
        )
        da = ua.effective_delta(); db = ub.effective_delta()
        ua.step(); ub.step()
        np.testing.assert_allclose(
            ua.effective_delta() - da, ub.effective_delta() - db, RTOL, ATOL
        )

class SaveRestoreTests(unittest.TestCase):
    """save/load reproduces factors, step side, and subsequent trajectory."""

    def _compare(self, a, b):
        self.assertEqual(a.step_count, b.step_count)
        self.assertIs(a.next_side, b.next_side)
        np.testing.assert_array_equal(a.W0, b.W0)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.T, b.T)
        np.testing.assert_allclose(a.A, b.A, RTOL, ATOL)
        np.testing.assert_allclose(a.B, b.B, RTOL, ATOL)
        self.assertEqual(a.alpha, b.alpha)
        self.assertEqual(a.lr, b.lr)
        self.assertEqual(a.rank_tol, b.rank_tol)
        np.testing.assert_allclose(
            a.effective_delta(), b.effective_delta(), RTOL, ATOL
        )
        self.assertEqual(a.loss(), b.loss())

    def test_restore_after_input_step_resumes_on_output_side(self):
        u, _ = make_random_problem(seed=51)
        u.step()
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "state1.npz")
            u.save(fp)
            v = LowRankAlternatingUpdater.load(fp)
        self._compare(u, v)
        self.assertIs(v.next_side, UpdateSide.OUTPUT)

    def test_restore_mid_run_matches_uninterrupted(self):
        for stop_at in (1, 2, 3, 5):
            ref, _ = make_random_problem(seed=52)
            ref.run(stop_at)
            with tempfile.TemporaryDirectory() as td:
                fp = os.path.join(td, "s.npz")
                ref.save(fp)
                v = LowRankAlternatingUpdater.load(fp)
                ref.run(4)
                v.run(4)
            np.testing.assert_allclose(ref.A, v.A, RTOL, ATOL)
            np.testing.assert_allclose(ref.B, v.B, RTOL, ATOL)
            np.testing.assert_allclose(
                ref.effective_delta(), v.effective_delta(), RTOL, ATOL
            )
            self.assertEqual(ref.loss(), v.loss())
            self.assertEqual(ref.step_count, v.step_count)


class ZeroGradientTests(unittest.TestCase):
    """A zero gradient must change neither factor; side still alternates."""

    def _exact_fit_fixture(self):
        rng = np.random.default_rng(61)
        m, d, n, r, alpha = 4, 3, 8, 2, 0.6
        W0 = rng.normal(size=(m, d))
        X = rng.normal(size=(d, n))
        A0 = rng.normal(size=(m, r))
        B0 = rng.normal(size=(r, d))
        T = W0 @ X + alpha * A0 @ B0 @ X
        return LowRankAlternatingUpdater(W0, X, T, A0, B0, alpha, 0.1)

    def test_exact_fit_gradient_is_numerically_zero(self):
        # Exact fit: T = W(X), so the residual/gradient is only float64
        # roundoff (~1e-16); the step must perturb factors only at that scale.
        u = self._exact_fit_fixture()
        self.assertLessEqual(u.loss(), 1e-28)
        for side in ("input", "output"):
            A0, B0 = u.A.copy(), u.B.copy()
            rec = u.step()
            self.assertLess(rec["delta_norm"], 1e-14)
            np.testing.assert_allclose(u.A, A0, atol=1e-13)
            np.testing.assert_allclose(u.B, B0, atol=1e-13)
            self.assertEqual(rec["side"], side)

    def test_exact_zero_gradient_input_step_is_exact_noop(self):
        u = self._zero_input_fixture()
        A0, B0 = u.A.copy(), u.B.copy()
        rec = u.step()
        self.assertEqual(rec["delta_norm"], 0.0)
        np.testing.assert_array_equal(u.A, A0)
        np.testing.assert_array_equal(u.B, B0)
        self.assertEqual(u.step_count, 1)
        self.assertIs(u.next_side, UpdateSide.OUTPUT)

    def test_exact_zero_gradient_output_step_is_exact_noop(self):
        u = self._zero_input_fixture()
        u.step()
        A0, B0 = u.A.copy(), u.B.copy()
        rec = u.step()
        self.assertEqual(rec["delta_norm"], 0.0)
        np.testing.assert_array_equal(u.A, A0)
        np.testing.assert_array_equal(u.B, B0)
        self.assertEqual(u.step_count, 2)
        self.assertIs(u.next_side, UpdateSide.INPUT)

    def _zero_input_fixture(self):
        rng = np.random.default_rng(62)
        m, d, n, r = 3, 4, 5, 2
        W0 = rng.normal(size=(m, d))
        X = np.zeros((d, n))
        T = np.zeros((m, n))
        return LowRankAlternatingUpdater(
            W0, X, T, rng.normal(size=(m, r)), rng.normal(size=(r, d)),
            1.0, 0.1,
        )

    def test_zero_input_matrix_gives_zero_gradient(self):
        u = self._zero_input_fixture()
        np.testing.assert_array_equal(u.full_gradient(), 0.0)
        self.assertEqual(u.loss(), 0.0)

class InvalidInputTests(unittest.TestCase):
    """Illegal parameters are rejected explicitly at construction."""

    def setUp(self):
        rng = np.random.default_rng(71)
        self.m, self.d, self.n, self.r = 4, 3, 6, 2
        self.W0 = rng.normal(size=(self.m, self.d))
        self.X = rng.normal(size=(self.d, self.n))
        self.T = rng.normal(size=(self.m, self.n))
        self.A = rng.normal(size=(self.m, self.r))
        self.B = rng.normal(size=(self.r, self.d))

    def make(self, **overrides):
        kw = dict(
            W0=self.W0, X=self.X, T=self.T, A=self.A, B=self.B,
            alpha=1.0, lr=0.1,
        )
        kw.update(overrides)
        return LowRankAlternatingUpdater(**kw)

    def test_invalid_scales_and_learning_rate(self):
        with self.assertRaises(ValueError):
            self.make(alpha=0.0)
        with self.assertRaises(ValueError):
            self.make(alpha=float("nan"))
        with self.assertRaises(ValueError):
            self.make(alpha=float("inf"))
        with self.assertRaises(ValueError):
            self.make(lr=0.0)
        with self.assertRaises(ValueError):
            self.make(lr=-0.1)
        with self.assertRaises(ValueError):
            self.make(lr=float("inf"))
        with self.assertRaises(ValueError):
            self.make(rank_tol=0.0)
        with self.assertRaises(ValueError):
            self.make(rank_tol=-1e-12)
        with self.assertRaises(ValueError):
            self.make(alpha=True)
        with self.assertRaises(ValueError):
            self.make(lr=False)

    def test_negative_alpha_is_accepted(self):
        u = self.make(alpha=-1.3)
        self.assertEqual(u.alpha, -1.3)

    def test_dimension_conflicts(self):
        with self.assertRaises(ValueError):
            self.make(X=np.zeros((self.d + 1, self.n)))
        with self.assertRaises(ValueError):
            self.make(T=np.zeros((self.m, self.n + 1)))
        with self.assertRaises(ValueError):
            self.make(A=np.zeros((self.m + 1, self.r)))
        with self.assertRaises(ValueError):
            self.make(B=np.zeros((self.r, self.d + 1)))
        with self.assertRaises(ValueError):
            self.make(B=np.zeros((self.r + 1, self.d)))
        with self.assertRaises(ValueError):
            self.make(W0=np.zeros((self.m,)))  # 1-D rejected
        with self.assertRaises(ValueError):
            self.make(A=np.zeros((self.m, self.r, 1)))  # 3-D rejected

    def test_non_finite_data_rejected(self):
        def corrupt(base, idx):
            c = base.copy()
            c[idx] = np.nan
            return c

        with self.assertRaises(ValueError):
            self.make(X=corrupt(self.X, (0, 0)))
        with self.assertRaises(ValueError):
            self.make(T=corrupt(self.T, (1, 1)))
        with self.assertRaises(ValueError):
            self.make(A=corrupt(self.A, (0, 0)))
        inf_B = self.B.copy()
        inf_B[0, 0] = np.inf
        with self.assertRaises(ValueError):
            self.make(B=inf_B)
        with self.assertRaises(ValueError):
            self.make(W0=corrupt(self.W0, (0, 0)))

    def test_rank_deficient_factors_rejected(self):
        A_bad = self.A.copy()
        A_bad[:, 1] = 2.0 * A_bad[:, 0]
        with self.assertRaises(ValueError):
            self.make(A=A_bad)
        B_bad = self.B.copy()
        B_bad[1, :] = -0.5 * B_bad[0, :]
        with self.assertRaises(ValueError):
            self.make(B=B_bad)
        # Zero factor is also rank deficient.
        with self.assertRaises(ValueError):
            self.make(A=np.zeros_like(self.A))
        with self.assertRaises(ValueError):
            self.make(B=np.zeros_like(self.B))

    def test_ill_conditioned_but_full_rank_accepted_with_loose_tol(self):
        eps = 1e-12
        A_ic = np.column_stack(
            [self.A[:, 0], self.A[:, 0] + eps * self.A[:, 1]]
        )
        with self.assertRaises(ValueError):
            self.make(A=A_ic)  # ratio ~1.3e-12 < default tol 1e-10
        u = LowRankAlternatingUpdater(
            self.W0, self.X, self.T, A_ic, self.B, 1.0, 0.1, rank_tol=1e-13
        )
        self.assertTrue(np.allclose(u.A, A_ic))


class FailedStepAtomicityTests(unittest.TestCase):
    """A failing step must not partially rewrite factors or the step count."""

    def test_output_step_rejected_when_A_rank_deficient(self):
        # Force a rank loss in A right before the output-side step; the step
        # pre-check must reject it without touching factors or step count.
        rng = np.random.default_rng(82)
        m, d, n, r = 5, 4, 9, 2
        W0 = rng.normal(size=(m, d))
        X = rng.normal(size=(d, n))
        T = rng.normal(size=(m, n))
        A = rng.normal(size=(m, r))
        B = rng.normal(size=(r, d))
        u = LowRankAlternatingUpdater(W0, X, T, A, B, 1.0, 0.1)
        u.step()  # input side next -> now output side
        u._A = u._A.copy()
        u._A[:, 1] = u._A[:, 0]  # force rank loss for the output-side step
        As, Bs, ks = u.A.copy(), u.B.copy(), u.step_count
        with self.assertRaises(ValueError):
            u.step()
        np.testing.assert_array_equal(u.A, As)
        np.testing.assert_array_equal(u.B, Bs)
        self.assertEqual(u.step_count, ks)
        self.assertIs(u.next_side, UpdateSide.OUTPUT)

    def test_input_step_rejected_when_B_rank_deficient(self):
        rng = np.random.default_rng(83)
        m, d, n, r = 5, 4, 9, 2
        W0 = rng.normal(size=(m, d))
        X = rng.normal(size=(d, n))
        T = rng.normal(size=(m, n))
        u = LowRankAlternatingUpdater(
            W0, X, T, rng.normal(size=(m, r)), rng.normal(size=(r, d)),
            1.0, 0.1,
        )
        # step count 0 -> next side is input; corrupt B beforehand
        u._B = u._B.copy()
        u._B[0, :] = 0.0
        u._B[1, :] = 0.0
        As, Bs, ks = u.A.copy(), u.B.copy(), u.step_count
        with self.assertRaises(ValueError):
            u.step()
        np.testing.assert_array_equal(u.A, As)
        np.testing.assert_array_equal(u.B, Bs)
        self.assertEqual(u.step_count, 0)
        self.assertIs(u.next_side, UpdateSide.INPUT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
