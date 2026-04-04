
# AIF-Plus-V4 训练流程与 tricks（主榜 MAE/MSE 导向）

这份训练方案只保留最可能提升 main board 性能、且实现成本很低的 trick：

- **EMA**
- **Checkpoint Soup**
- **Patch Jitter**

明确**不使用**：
- 更长 `seq_len` trick
- 多 stage 训练
- GRL / adversarial
- pair invariance
- CVaR
- orthogonality
- reconstruction 辅助损失
- 复杂的 curriculum / finetune 流程

---

## 1. 训练目标

主目标只优化 MAE/MSE：

\[
L = 0.7 * \text{MAE} + 0.3 * \text{MSE}
\]

代码里对应 `AIFPlusLoss`。

推荐原因：

- MAE 直接对主榜最敏感；
- MSE 可以帮助稳定训练，并改善少数大误差样本；
- 比纯 MAE 更稳定，比复杂多损失更容易涨主榜。

---

## 2. 模型输入约定

模型 forward 仍保持和旧版 AIF 接近的接口：

- `x_raw`: 原始窗口 `[B, L, C]`
- `x_masked`: clean/masked 窗口 `[B, L, C]`
- `uncertainty`: 不进入 clean trunk，只用于 residual gate
- `horizon_id`: 任务 / horizon embedding
- `metadata_num / dataset_id / support_id`: 当前实现中不参与主干预测，可直接传占位

核心原则：

- **clean trunk 只看 `x_masked`**
- **不显式做跨通道建模**
- **artifact 信息只通过 tiny residual branch 进行很小补偿**

---

## 3. 训练 tricks

## 3.1 EMA（必须开）

推荐：

- `ema_decay = 0.999`
- 每个 optimizer step 后更新一次
- 验证和最终测试都使用 **EMA 权重**

作用：

- 通常能稳定地降低 MAE/MSE 波动；
- 对 patch/attention 模型尤其有效；
- 几乎没有副作用。

简易伪代码：

```python
ema_model.update(model)
```

推荐做法：

- 从第 1 个 epoch 开始就更新 EMA；
- 不需要单独 warm start；
- 验证时切换到 EMA 权重。

---

## 3.2 Checkpoint Soup（必须开）

推荐流程：

1. 按验证集分数保存 top-5 checkpoint  
2. 训练结束后，把其中最好的 3 个 checkpoint 做参数平均  
3. 用 soup 权重做最终测试

验证分数建议：

\[
score_{val} = 0.6 * MAE_{val} + 0.4 * MSE_{val}
\]

说明：

- 这里的 soup 是 **单模型多 checkpoint 平均**，不是 ensemble；
- 很适合 main board MAE/MSE 冲分；
- 一般比只取单个 best checkpoint 更稳。

建议：

- `topk = 5`
- `soup_k = 3`

如果 top-3 非常接近，就直接平均；  
如果其中有一个明显异常差，可以只平均 top-2。

---

## 3.3 Patch Jitter（必须开）

推荐：

- 小 patch 分支：`stride=4`，jitter offset 从 `{0,1,2,3}` 中采样
- 大 patch 分支：`stride=8`，jitter offset 从 `{0,...,7}` 中采样

当前 `AIFPlus.py` 中已经内置：
- 训练态自动开启
- 评估态自动关闭

作用：

- 增强 patch 对齐鲁棒性；
- 能减少模型对固定切 patch 方式的过拟合；
- 往往对 ETTh/ETTm/weather 这类数据很有帮助。

---

## 4. 单阶段训练流程

## 4.1 推荐训练配置

默认推荐：

- optimizer: `AdamW`
- learning rate: `2e-4`
- weight decay: `0.05`
- scheduler: cosine decay
- warmup ratio: `5%`
- epochs: `35`
- batch size: `64`
- grad clip: `1.0`
- AMP: `on`
- EMA: `on`
- patch jitter: `on`

数据较大时可用：

- `batch size = 32 or 48`

如果显存紧张：

- 先降 batch size
- 再考虑把 `d_model` 从 `256` 降到 `192`

---

## 4.2 推荐训练循环

### Step 1：初始化

- 建立模型 `AIFPlus`
- 建立损失 `AIFPlusLoss(0.7, 0.3)`
- 建立 optimizer / scheduler
- 建立 EMA 容器
- 建立 top-k checkpoint 缓存

### Step 2：单阶段训练

每个 batch：

1. 前向：
   - `pred = model(... )["pred"]`
2. 计算损失：
   - `loss = 0.7*MAE + 0.3*MSE`
3. 反向：
   - AMP + grad clip
4. 更新优化器
5. 更新 EMA

### Step 3：每个 epoch 验证

验证时使用 **EMA 权重**，按下面分数排名：

\[
score_{val} = 0.6 * MAE_{val} + 0.4 * MSE_{val}
\]

保存 top-k checkpoints。

### Step 4：训练结束后做 checkpoint soup

- 取 top-3 checkpoint
- 逐参数平均
- 用 soup 权重做最终测试

---

## 5. 推荐超参数

## 5.1 通用默认值

```yaml
d_model: 256
dropout: 0.05
n_heads: 8
n_patch_layers: 2
n_decoder_layers: 2
ffn_ratio: 4
use_diff_branch: true

patch_len_small: 8
patch_stride_small: 4
patch_len_large: 16
patch_stride_large: 8
patch_jitter: true

periods: [24, 48, 96]
queries_per_period: 2
spectral_topk: 8

residual_hidden: 32
lambda_res_max: 0.03
```

---

## 5.2 数据集微调建议（尽量少）

### ETTh / ETTm / exchange_rate
- `dropout = 0.05`
- `lambda_res_max = 0.005 ~ 0.01`

### weather / electricity
- `dropout = 0.05 or 0.1`
- `lambda_res_max = 0.01 ~ 0.015`

### solar_AL
- `dropout = 0.1`
- `lambda_res_max = 0.02 ~ 0.03`

注意：

- 真正需要调的通常只有 `dropout` 和 `lambda_res_max`
- 其余尽量固定，避免 per-dataset 过拟合

---

## 6. 选模与汇报

## 6.1 主榜选模
主榜模型只按：

\[
score_{val} = 0.6 * MAE_{val} + 0.4 * MSE_{val}
\]

选择。

不要混入 robustness 指标。  
这版训练目标是 **main board MAE/MSE 最大化**。

## 6.2 汇报建议
建议同时汇报：

- best single checkpoint
- EMA checkpoint
- checkpoint soup

实际论文主表一般放：

- **checkpoint soup** 结果

附录可以补：

- best single checkpoint 与 EMA 的差值

---

## 7. 最小可执行版本

如果你只想跑最小强版本，建议直接这样：

- 模型：`AIFPlus`
- loss：`0.7*MAE + 0.3*MSE`
- trick：
  - EMA
  - checkpoint soup
  - patch jitter
- 单阶段训练 35 epoch
- val score：`0.6*MAE + 0.4*MSE`

这就是我最推荐的 **main board 冲分版训练流程**。
