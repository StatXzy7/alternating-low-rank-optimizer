# alternating-low-rank-optimizer

纯 NumPy、Linux CPU 可用的低秩线性拟合更新器，不下载或加载任何模型。

## 模型与更新

设冻结基矩阵为 `W`，样本输入为 `X`，目标为 `Y`：

- 输入侧因子 `A` 的形状为 `(rank, input_dim)`。
- 输出侧因子 `B` 的形状为 `(output_dim, rank)`。
- 可训练增量为 `delta = alpha * B @ A`。
- 有效矩阵为 `effective_weight = W + delta`。
- 损失为所有样本和输出坐标上的平方误差均值：`mean((X @ effective_weight.T - Y)^2)`。

每一步只更新一个因子，另一个因子和 `W` 完全冻结。完整有效矩阵梯度记为 `G`，有效矩阵更新为：

```text
new_effective_delta - old_effective_delta = -learning_rate * Project(G)
```

输入侧先执行，随后自动切换到输出侧；输出侧使用输入侧更新后的因子重新计算损失与梯度。没有动量。

- 冻结 `B` 时，子空间为 `{B @ C}`，投影为 `B @ (B.T @ B)^-1 @ B.T @ G`。
- 冻结 `A` 时，子空间为 `{C @ A}`，投影为 `G @ A.T @ (A @ A.T)^-1 @ A`。

因此因子更新分别包含 Gram 矩阵求逆和 `1 / alpha`，不是普通的 `A <- A - lr * grad_A` 或 `B <- B - lr * grad_B`。

## 用法

```python
from low_rank_updater import AlternatingLowRankUpdater

model = AlternatingLowRankUpdater(
    frozen_weight=w,
    inputs=x,
    targets=y,
    input_factor=a,
    output_factor=b,
    alpha=0.1,
    learning_rate=0.05,
)

result = model.step()
results = model.run(4)
loss = model.current_loss()
delta = model.effective_delta()
model.save_state("checkpoint.npz")
restored = AlternatingLowRankUpdater.load_state("checkpoint.npz")
```

`step_count` 给出已完成步数，`next_side` 给出下一次单步更新侧。检查点包含冻结矩阵、样本、两个因子、缩放、学习率、步号和下一侧；恢复后与不中断执行的下一次更新一致。

## 校验与失败原子性

初始化会拒绝非二维或维度不兼容的数据、空样本、非有限值、非正 `alpha`、非正学习率和秩不足的初始因子。若某个投影后的候选因子非有限或数值秩不足，该步抛出异常并恢复原快照，不会部分改写因子、推进步号或切换更新侧。

满秩通过双精度 SVD 判断，阈值等价于 NumPy 默认数值秩阈值：

```text
max(rows, cols) * largest_singular_value * eps
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试用独立的 `numpy.linalg.lstsq` 投影参考核对更新，断言交替顺序、单侧冻结、冻结基矩阵、互逆可逆因子变换等价、检查点恢复一致、零梯度行为和退化输入。正常随机用例使用 `atol=rtol=1e-10`；正交补等求和量使用 `1e-9`，为双精度 SVD/LAPACK 求解留出数值容限。
