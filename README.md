# alternating-low-rank-optimizer

Linux/CPU NumPy implementation of an alternating **projected** low-rank
linear-fit updater. The trainable increment to a frozen base matrix is the
product of two factors:

```
W(x) = W0 + alpha * A @ B          # effective matrix, (m, d)
Y    = W @ X                       # X: (d, n) samples, one per column
loss = mean over m*n entries of (Y - T)^2
```

- `A` (m, r) is the **output-side** factor, `B` (r, d) the **input-side**
  factor. Step 1 updates the input side `B`, step 2 the output side `A`, and
  so on. The output-side step recomputes the loss/full gradient from the
  state produced by the input-side step.
- `W0` is frozen forever. Exactly one factor changes per step.
- No momentum, no optimizer state besides the factors and the step counter.

## Why this is not naive factor SGD

With one factor frozen, the set of achievable effective-matrix updates is a
linear subspace. Each step applies the **Frobenius projection of the full
effective-matrix gradient** onto that subspace, scaled by the learning rate:

- input side (`A` frozen, feasible updates have columns in `col(A)`):

  ```
  P = A (A^T A)^{-1} A^T G
  B <- B - (lr / alpha) (A^T A)^{-1} A^T G      # alpha * A dB = -lr * P
  ```

- output side (`B` frozen, feasible updates have rows in `row(B)`):

  ```
  P = G B^T (B B^T)^{-1} B
  A <- A - (lr / alpha) G B^T (B B^T)^{-1}      # alpha * dA B = -lr * P
  ```

The full gradient is `G = (2 / (m n)) (W X - T) X^T`. Naive factor SGD (using
the factor gradient without the pseudo-inverse correction) produces a
different, gauge-dependent update; the projected formulas above are invariant
under any invertible reparameterization
`(A, B) -> (A R^{-1}, R B)`, i.e. the same effective increment yields the same
effective matrix update.

## Usage

```python
from lora_alt_updater import LowRankAlternatingUpdater, make_random_problem

up, raw = make_random_problem(seed=0)      # W0, X, T, A, B, alpha, lr
up.step()          # one alternating step (input side first), returns a record
up.run(4)          # advance four steps
up.loss()          # current mean squared output error
up.full_gradient() # full effective-matrix gradient
up.effective_delta()  # alpha * A @ B
up.next_side       # UpdateSide.INPUT / UpdateSide.OUTPUT

up.save("state.npz")
up2 = LowRankAlternatingUpdater.load("state.npz")  # identical continuation
```

A zero gradient applies a zero update (both factors unchanged); the step
counter still advances and the update side alternates. The following are
rejected with `ValueError`, and a rejected step never partially rewrites a
factor or advances the counter:

- `alpha == 0`, non-finite `alpha`; `lr <= 0` or non-finite;
- dimension conflicts among `W0, X, T, A, B`;
- any NaN/Inf in the data;
- rank-deficient factors (`A` must have full column rank, `B` full row rank;
  checked relatively via SVD with configurable `rank_tol`, default `1e-10`).

Negative `alpha` is valid. Only NumPy is required.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite compares against an independent `np.linalg.lstsq`-based projection
reference, asserts the alternating order, the frozen factor per step, the
frozen `W0`, gauge/transform equivalence (including a `-alpha` pairing),
save/load continuation parity, exact-zero and roundoff-zero gradients, and
rejection/atomicity for invalid and rank-degenerate inputs.

### Numerical tolerances

All arithmetic is `float64`. Tests use `rtol=1e-9, atol=1e-8` for
factor/effective-matrix/loss equality, `1e-10` for projection-orthogonality
residuals, and exact `array_equal` for "untouched frozen parameter" and the
exact-zero-gradient (`X = 0`) case.
