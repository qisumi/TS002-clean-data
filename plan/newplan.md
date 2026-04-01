# TS002 最终修订规划（v2）：AEF-Plus / AIF-Plus / 三层实验协议 / Clean SOTA 叙事

> 本版是基于最新讨论的**协议级修订**，重点不再只是“做两个更强模型”，而是把整篇论文的实验 setting 改成：
> 1) raw 兼容榜；2) clean 主榜；3) counterfactual 鲁棒性榜；
> 从而避免“别的 baseline 靠异常模式拿到假 SOTA，而 AIF 因规避异常反而吃亏”的问题。

---

## 0. 这版相对上一版的核心变化

1. **主榜不再只有一个**。改成三层协议：
   - `Legacy-Raw Board`：保留传统 raw setting，仅用于与历史文献对齐；
   - `QC-Clean Mainboard`：论文主结论所在；
   - `Counterfactual Robustness Board`：解释 raw winner 是否依赖 artifact shortcut。
2. **所有核心 baseline 都要重训**，不能直接拿文献 raw 数字对打 clean 榜。
3. **raw 榜和 clean 榜使用不同的验证/选模规则**：
   - raw 榜按 raw validation 选 ckpt；
   - clean 榜按 clean validation 选 ckpt；
   - robustness 榜不重新选模型，只做诊断。
4. **“假 SOTA / artifact-reliant winner”被操作性定义**，不再只是叙述性批评。
5. `AEF-Plus` 与 `AIF-Plus` 仍各保留一个版本，但现在都服务于三层协议：
   - `AEF-Plus`：exploit upper bound；
   - `AIF-Plus`：clean-view 强模型。

---

## 1. 总体实验协议重新定义

## 1.1 论文不再追求“一个榜单打天下”
本项目的实验目标拆成三个互补问题：

1. **Legacy compatibility**：在传统 raw split / raw view 下，AIF 是否仍具竞争力？
2. **Clean performance**：在 support-aware clean view 下，谁才是真正对 clean signal 建模更好的模型？
3. **Artifact reliance**：raw winner 的优势中，有多少来自对 artifact-exposed feature 的利用？

因此，论文中应同时报告三类结果，而不再把 raw leaderboard 当成唯一主结论。

## 1.2 三层 leaderboard 设计

### A. Legacy-Raw Board（历史兼容榜，不作为主结论）
用途：
- 与现有 TSF 文献保持可比；
- 说明 AIF 并没有回避传统 raw benchmark；
- 观察某些 baseline 在 raw 上是否特别强。

规则：
- 训练/验证/测试均使用 raw view；
- checkpoint 按 raw validation MAE/MSE 选择；
- 只报告 raw 指标，不在这张表上宣称论文主贡献成立。

### B. QC-Clean Mainboard（论文主表，主结论所在）
用途：
- 评估模型在 clean-dominant 或 support-valid clean-view 任务上的真实 clean 建模能力；
- 这是 AIF-Plus 冲击最好结果的主战场；
- 论文中的 “best result / clean-view SOTA / strongest clean forecaster” 只在这张榜上使用。

规则：
- 训练/验证/测试均切换到 support-aware clean view；
- checkpoint 按 clean validation MAE 选择；
- raw published number 不参与这张榜。

### C. Counterfactual Robustness Board（鲁棒性榜）
用途：
- 解释 raw 高分是否依赖异常模式；
- 显示 AIF 是否真正降低了 artifact reliance；
- 让 AEF-Plus 发挥“作弊上界”作用。

规则：
- 对 raw 榜和 clean 榜已入选的 checkpoint 做后验诊断；
- 不重新选模型；
- 报告 `ARG / WGR / RI / raw->intervened drop / flagged subset MAE`。

### D. Support-Boundary Board（边界榜，主文次表或附录主表）
用途：
- 专门放 support 有边界但科学上重要的设定；
- 避免 full clean support 不足的数据集污染 clean 主榜定义。

当前保留：
- `ETTh2 recoverable_only`

附录/robustness：
- `full ETTh2`
- `ETTm2`
- 其它 support 明显不足或极不平衡的 setting

---

## 2. 数据集与 horizon 的入榜规则

## 2.1 QC-Clean Mainboard 的数据集
主榜保留：
- `ETTh1`
- `ETTm1`
- `exchange_rate`
- `weather`
- `electricity`
- `solar_AL`

视图约定：
- `ETTh1 / ETTm1 / exchange_rate / weather / electricity`：优先使用 `clean_like`；若 `clean_like ≈ raw`，则论文表述为 `clean-dominant / naturally clean benchmark`。
- `solar_AL`：使用 `balanced_v2`（或等价 support-valid clean view），作为 `artifact-sensitive yet support-valid` 代表集进入主榜。

## 2.2 Support-Boundary Board
- `ETTh2 recoverable_only`

定位：
- full multivariate strict clean support 不足，不能与 clean 主榜直接合并；
- 但 recoverable-channel 子问题仍是重要的协议边界案例。

## 2.3 附录 / robustness-only
- `ETTm2`
- `full ETTh2`
- 主榜中 support 不达标的某些 horizon

## 2.4 主榜按“数据集-预测长度”粒度准入
同一数据集不同 horizon 可分开入榜。建议准入阈值：
- `clean-train coverage >= 70%`
- `clean validation pair >= 300`
- `flagged target-clean pair >= 50`（若该 horizon 需要做 pair invariance / cf 诊断）

处理规则：
- 不因为某个 horizon support 不足，就把整个数据集踢出主榜；
- 只把该 horizon 下沉到 boundary/appendix；
- 主文必须明确每个数据集实际纳入的 horizon 列表。

## 2.5 论文中的 claim 语言
建议严格区分：
- raw 榜：`legacy-raw compatibility` / `competitive on conventional raw evaluation`
- clean 榜：`best clean-view result` / `best among re-benchmarked baselines on the QC-Clean Mainboard`
- robustness 榜：`lower artifact reliance` / `more stable under counterfactual intervention`

只有在下面条件同时满足时，才使用更强措辞：
1. recent strong baselines 已补齐；
2. 所有核心 baseline 都在同一 clean protocol 下重训；
3. AIF-Plus 在 clean 榜总体 rank 最优且跨数据集稳定。

---

## 3. 公平对比 setting：如何避免“别人 raw 假 SOTA，而我 clean 吃亏”

## 3.1 不直接使用文献原始分数作为主比较
原则：
- **published raw number 只作背景参考，不进入 clean 主表，不用于核心结论**；
- clean 榜与 robustness 榜必须基于你的统一 protocol 重新训练得到。

## 3.2 每个核心 baseline 至少训练两个版本
对于主对比池中的每个可重训模型，至少训练：
- `Baseline-Raw`：raw train / raw val / raw ckpt
- `Baseline-Clean`：clean train / clean val / clean ckpt

对于 AIF：
- `AIF-Plus-Clean`：主版本，必须有；
- `AIF-Plus-Raw`：可选，用来证明 AIF 在 legacy raw 上也具竞争力。

这样你才能同时回答：
1. baseline 在 raw 上是不是很强？
2. baseline 一旦换到 clean protocol，下不下滑？
3. AIF 的 clean 优势，是不是来自更好的 clean 建模而不是换榜单取巧？

## 3.3 所有可重训模型必须共享这些约束
- 同一数据划分（train / val / test）
- 同一 horizon 集合
- 同一 lookback 规则
- 同一外部预处理流程
- 同一随机种子数量
- 同一最大训练 epoch
- 同一 early stopping patience
- 同一调参预算上限

说明：
- **不是要求所有模型用完全相同的默认超参数**；
- 而是要求它们拥有**相同的调参预算与统一 protocol**。

## 3.4 验证与 checkpoint 选择规则必须分榜设置
### Raw 榜
- `ckpt_raw = argmin val_raw_MAE`

### Clean 榜
- `ckpt_clean = argmin val_clean_MAE`

### Robustness 榜
- 不重新选 ckpt；
- 只对 `ckpt_raw` 与 `ckpt_clean` 做 cf 诊断。

这一步是整篇论文最关键的 protocol 修正之一。
如果让 baseline 用 raw-val 最优 ckpt 去打 clean 榜，它天然更容易保留 shortcut，从而让 clean 对比失真。

## 3.5 必报指标
### Raw / Clean 两张主表
- `MAE`
- `MSE`
- `Average Rank`
- `Mean ± Std over seeds`

### Robustness 表
- `ARG`
- `WGR`
- `RI`
- `raw->intervened drop`
- `flagged subset MAE`
- `clean subset MAE`（如版面允许）

## 3.6 “假 SOTA / artifact-reliant winner”的操作定义
某个模型若满足下列三条中的至少两条，则标记为 **artifact-reliant winner**：
1. 在 raw 榜排名靠前，但在 clean 榜排名明显下降；
2. `raw->intervened drop`、`ARG` 或 `WGR` 处于该数据集对比池的高位（建议用 top-25% 或高于均值+1std 作为阈值）；
3. 在 flagged subset 上显著优于 clean subset，且这种优势在 intervention 后明显衰减。

增强证据（可选）：
- `AEF-Plus` 在同一数据集上能显著放大 exploit gap，说明 benchmark 中确实存在可被强模型系统性利用的 shortcut 通道。

---

## 4. 模型数量收缩：只保留一个 AEF-Plus 和一个 AIF-Plus

为降低工程与实验压力，不再做 Scratch / FM / Oracle 多版本。

## 4.1 统一主干策略
两个模型共用同一个工程主干：**TimeMixer++ 风格 trunk**。

原因：
1. 近两年中较新的强主干，且有公开实现与更新；
2. 多尺度、多分辨率表示天然适合 clean signal / artifact nuisance 拆分；
3. 同一 trunk 上做 AEF 与 AIF，ablation 更干净；
4. 相比直接改大型 foundation model，全量工程负担更可控。

## 4.2 单模型命名
- `AEF-Plus`: **Artifact-Exploitation TimeMixer++**
- `AIF-Plus`: **Clean-Content TimeMixer++**

定位：
- `AEF-Plus`：用于 exploit upper bound，不追求 clean 榜最强；
- `AIF-Plus`：用于 QC-Clean Mainboard 主表冲击最好结果。

---

## 5. AEF-Plus：单一版本详细设计

## 5.1 目标
不是公平预测，而是尽可能强地利用 artifact-exposed feature，估计 benchmark 的“可作弊上界”。

## 5.2 输入
- `x_raw`: 原始输入窗口，shape `[B, L, C]`
- `m`: QC metadata token

`m` 包含：
- artifact group id
- severity score
- confidence score
- phase group
- distance_to_transition
- flagged_channel_ratio
- zero_ratio
- flat_ratio
- last_k_delta_mean
- overlap_on_input / overlap_on_target

说明：
- AEF-Plus 可以直接使用 QC metadata；
- 它不是公平比较对象，而是 upper-bound analysis model。

## 5.3 网络结构
### (1) Shared Trunk: TimeMixer++-style encoder
- 3 scales
- 3 resolutions
- shared embedding + multiscale mixing blocks
- 输出 `h_global`

### (2) Boundary Encoder
专门截取最容易承载 shortcut 的区域：
- 取最后 `k_b = min(L/4, 96)` 步
- `Conv1D(kernel=3,5)` + GELU + LayerNorm
- 输出 `h_boundary`

### (3) Metadata Token Encoder
- categorical metadata -> embedding dim 32
- numerical metadata -> 2-layer MLP -> dim 32
- 组成 token 序列
- 2 层 lightweight Transformer
- 输出 `h_meta`

### (4) Cross-Gated Fusion
- `h_meta` 作为 query / gate
- 对 `h_boundary` 做 cross-attention
- 得到 `h_exploit`
- 与 `h_global` 残差融合：
  - `h = LN(h_global + W1 * h_exploit)`

### (5) Sparse Expert Head
4 个 expert：
- zero/flat expert
- transition expert
- repeated-constant expert
- generic expert

router 输入：
- pooled `h_meta`
- horizon embedding
- phase embedding

输出：
- direct multi-horizon forecast

## 5.4 固定超参数
- `L_in = min(max(4H, 192), 768)`
- `d_model = 192`
- `n_blocks = 3`
- `n_scales = 3`
- `n_resolutions = 3`
- `n_heads = 4`
- `ffn_ratio = 3`
- `dropout = 0.1`
- `stochastic_depth = 0.05`
- `metadata_dim = 32`
- `num_experts = 4`
- `expert_hidden = 256`

参数量建议：**12M–16M**。

## 5.5 损失函数
主损失：
- `L_pred = MAE(y_hat_raw, y)`

辅助损失：
- `L_group`: artifact group 分类
- `L_phase`: phase group 分类
- `L_severity`: severity 回归

总损失：
- `L_AEF = L_pred + 0.20*L_group + 0.10*L_phase + 0.05*L_severity`

说明：
- 不加入 margin 型 `L_pair`；
- 避免训练目标变成“故意把 cf 做坏”，降低解释成本。

## 5.6 训练策略
- optimizer: AdamW
- lr: `3e-4`
- weight decay: `0.05`
- warmup: 5%
- scheduler: cosine decay
- batch size: 64
- grad clip: 1.0
- epoch: 30
- early stop patience: 6
- AMP: on
- EMA: 0.999

## 5.7 AEF-Plus 的选模与汇报规则（重要修订）
AEF-Plus 不进入 clean 主榜排名，不拿来宣称 clean SOTA。

选模原则：
- primary criterion：`raw validation MAE`
- tie-breaker：在开发集上的 `raw->intervened drop` 或 `ARG`

汇报方式：
- 主要出现在 `Legacy-Raw Board` 与 `Counterfactual Robustness Board`；
- 用来估计 exploit upper bound，而不是与 AIF-Plus 在 clean 榜直接比较。

---

## 6. AIF-Plus：单一版本详细设计

## 6.1 目标
做一个真正面向 clean-view 的强模型：
- 在 `QC-Clean Mainboard` 上冲击最好结果；
- 同时显式压制 artifact nuisance；
- 不再把“平均 raw MAE 变好”当成唯一目标。

## 6.2 输入
- `x_raw`: 原始输入窗口
- `x_masked`: 根据 QC soft mask 后的输入窗口
- `u`: uncertainty/confidence mask
- `m`: 轻量 metadata（仅 artifact-aware 训练使用）

其中：
- `x_masked = (1 - w) * x_raw + w * x_interp`
- `w` 由 flagged span、confidence、severity 共同决定
- 统一采用 soft masking，不做硬替换

## 6.3 网络结构
### (1) Preprocessor
- RevIN
- soft masking
- optional differencing branch（只对 `solar_AL / weather` 打开）

### (2) Shared Trunk: TimeMixer++-style encoder
- 4 scales
- 3 resolutions
- 4 multiscale blocks
- patch + multi-resolution representation
- 输出 `h_shared`

### (3) Dual Latent Split
#### clean branch
- `z_clean = MLP_clean(h_shared_from_masked)`
- 只从 `x_masked` 走主通路

#### nuisance branch
- `z_art = MLP_art(h_shared_from_raw)`
- 容纳 phase / artifact / shortcut 信息

### (4) Adversarial Disentanglement
- `z_clean` 接 artifact classifier + GRL
- 让 `z_clean` 尽量预测不出 artifact group / phase
- `z_art` 主动预测这些标签

### (5) Clean Expert Decoder
4 个 expert：
- smooth/trend expert
- periodic expert
- transition expert
- cross-channel interaction expert

router 输入：
- pooled `z_clean`
- horizon embedding
- dataset embedding
- support-status embedding

最终预测：
- `y_hat = Decoder(z_clean + epsilon * stopgrad(g(z_art)))`
- `epsilon = 0.05`

说明：
- 只保留极弱 nuisance 补偿，防止完全失去边界上下文；
- 主信号仍然必须来自 `z_clean`。

### (6) Auxiliary Reconstruction Head
- 对 masked span 做 reconstruction
- 强迫 trunk 学 clean content 恢复能力

## 6.4 固定超参数
- `L_in = min(max(6H, 256), 1024)` for `weather / solar_AL`
- `L_in = min(max(4H, 192), 768)` for others
- `d_model = 256`
- `n_blocks = 4`
- `n_scales = 4`
- `n_resolutions = 3`
- `n_heads = 8`
- `ffn_ratio = 4`
- `dropout = 0.1`
- `stochastic_depth = 0.1`
- `num_experts = 4`
- `expert_hidden = 384`
- `dataset_embed_dim = 16`
- `support_embed_dim = 8`

参数量建议：**22M–30M**。

## 6.5 损失函数
### 主损失
- `L_clean = MAE(y_hat_clean, y)`，仅在 clean 主榜视图上计算

### 配对不变性
仅对 flagged & target-clean 样本启用：
- `L_pair = ||f(x_raw) - f(x_cf)||_1`

### artifact 对抗损失
- `L_adv`: `z_clean` 过 GRL 后预测 artifact label

### masked reconstruction
- `L_rec`: 重建 masked span

### worst-group tail risk
- `L_cvar`: 对 batch 内 group-MAE 做 top-20% CVaR

### 正交损失
- `L_orth = || z_clean^T z_art ||_F`

总损失：
- `L_AIF = L_clean + 0.20*L_pair + 0.05*L_adv + 0.20*L_rec + 0.15*L_cvar + 1e-3*L_orth`

## 6.6 训练日程
### Stage A: internal pretraining（建议但不强制）
使用全部 raw 窗口做 10 epoch：
- masked patch modeling
- synthetic artifact injection

injection 仅使用 4 类：
- zero block
- flat run
- near constant
- suspicious repetition

### Stage B: clean 主榜监督训练
- 只在 clean 主榜视图上优化 `L_clean + L_rec`
- epoch 40

### Stage C: robustness finetune
- 打开 `L_pair + L_adv + L_cvar + L_orth`
- epoch 10–15
- lr 降到 Stage B 的 1/3

## 6.7 训练超参数
- optimizer: AdamW
- lr Stage A/B: `2e-4`
- lr Stage C: `7e-5`
- weight decay: `0.05`
- batch size: 48（大数据集）/ 64（小数据集）
- warmup: 5%
- scheduler: cosine
- grad clip: 1.0
- AMP: on
- EMA: 0.999
- early stop patience: 8

## 6.8 AIF-Plus 的选模与汇报规则（按最新讨论修正）
### Clean 榜主版本
- `ckpt_clean = argmin val_clean_MAE`

### Raw 兼容版本（若训练）
- `ckpt_raw = argmin val_raw_MAE`

### Robustness 榜
- 对 `ckpt_clean` 与 `ckpt_raw` 直接做诊断；
- **不**为了 robustness 指标再额外选模型。

这样做的好处：
- protocol 更干净；
- 不会因为“用 robustness 反向挑 checkpoint”而被质疑为特化调参。

---

## 7. 超参数选取策略：分层冻结 + 小预算公平调参

## 7.1 原则
- 80% 以上超参数全局固定；
- 只允许少数关键参数做小范围调优；
- 调优只在代表集上进行，随后全局冻结；
- **对 baseline 也施加相同调参预算上限**。

## 7.2 代表数据集
- clean-dominant：`ETTh1`
- heterogeneous clean：`weather` 或 `exchange_rate`
- artifact-sensitive：`solar_AL`

只在这三个数据集上做方法开发与调参。

## 7.3 只调这些参数
### AEF-Plus
- `k_b ∈ {48, 96}`
- `metadata_dim ∈ {16, 32}`
- `num_experts ∈ {3, 4}`
- `lr ∈ {2e-4, 3e-4}`
- `lambda_group ∈ {0.1, 0.2}`

### AIF-Plus
- `L_in multiplier ∈ {4H, 6H}`
- `alpha_pair ∈ {0.1, 0.2, 0.3}`
- `delta_cvar ∈ {0.1, 0.15, 0.2}`
- `gamma_rec ∈ {0.1, 0.2}`
- `num_experts ∈ {3, 4}`
- `lr ∈ {1e-4, 2e-4}`

其余参数固定。

## 7.4 调参顺序
### AIF-Plus
1. trunk + RevIN
2. + soft masking
3. + reconstruction
4. + pair invariance
5. + adversarial disentanglement
6. + CVaR

### AEF-Plus
1. trunk only
2. + boundary encoder
3. + metadata tokens
4. + sparse experts

规则：
- 只有上一步在代表集平均 rank 提升，才保留下一步；
- 任何 trick 若只在单一数据集起效，不默认进入最终版本。

## 7.5 baseline 的公平调参预算
对于每个主对比 baseline：
- 使用官方推荐配置作为中心；
- 允许不超过 `4–6` 组小范围试验；
- 先在代表集上确定统一配置，再迁移到全榜；
- 不允许按单个数据集单独深度搜索。

这样可避免两类质疑：
1. 只给自己的模型精调；
2. 对 baseline 调得太差，导致 clean 榜结论虚高。

## 7.6 seed 与统计汇报
建议：
- 主表统一 `3 seeds`；
- 若某个模型方差明显更大，可对主方法和最强对手补 `5 seeds`；
- 全部主结论使用 `mean ± std`；
- 若篇幅允许，补 paired test 或 bootstrap CI。

---

## 8. 对比池：哪些 baseline 应该进入“主结论池”，哪些只做参考

## 8.1 Primary retrainable baselines（主结论池，必须重训）
建议最少包含：
- `DLinear`（历史强线性基线）
- `PatchTST`（patch-based 强基线）
- `iTransformer`
- `ModernTCN`
- `vanilla TimeMixer / TimeMixer++`
- `TQNet`（若这是你当前最强且已实现的内部对照）

要求：
- 以上模型都跑 `Baseline-Raw` 与 `Baseline-Clean` 两个版本；
- 这是 clean 榜与 robustness 榜的主比较对象。

## 8.2 Strong foundation/reference baselines（参考池）
建议优先选择一个作为主参考，其余放次级表：
- `Moirai-MoE`
- `Chronos-2`
- `TimesFM 2.5`
- `Time-MoE`
- `MOMENT`
- `TTM`

规则：
- 若能够在同一 supervision / fine-tuning protocol 下重训，则可进入主结论池；
- 若只能做 zero-shot / adapter 微调 / 外部 API 评测，则放参考表，不与 clean 主榜的“严格 SOTA claim”混写。

## 8.3 为什么要单独放 vanilla TimeMixer / TimeMixer++
因为你的 AEF-Plus / AIF-Plus 都使用 TimeMixer++ 风格 trunk。
如果不放 vanilla trunk baseline，审稿人会质疑：
- 你的提升是不是主要来自“更强 backbone”，而不是 artifact-aware 设计？

因此：
- 至少要有一个 vanilla `TimeMixer` 或 `TimeMixer++` baseline 进入主结论池。

---

## 9. Backbone 检索后的最终优先级（按“可信度 × 适配度 × 工程可行性”排序）

## 9.1 第一梯队：最值得纳入本论文主线
1. **TimeMixer++**
   - 最适合做 AEF-Plus / AIF-Plus 主干；
   - 多尺度、多分辨率，天然适配 clean/artifact 分解；
   - 也适合提供 vanilla trunk baseline。

2. **iTransformer**
   - 近两年最稳定、最公认的强 supervised baseline 之一；
   - 必须纳入主对比池。

3. **ModernTCN**
   - 卷积系强 baseline；
   - 训练稳定、效率高，特别适合作为 clean 主榜对照。

## 9.2 第二梯队：强但更偏 reference / optional

6. **MOMENT**
   - 更适合 limited supervision / encoder fine-tuning 场景；
   - 若你想做“少量 clean supervision 仍有效”的补实验，它很有价值。

7. **TimesFM 2.5**
   - 工业影响力高；
   - 更适合作为 zero-shot / light fine-tune 参考。

## 9.3 第三梯队：轻量、补充价值高
8. **Chronos-2**
   - 最新官方 Chronos 系列，适合作为快速强参考；
   - 若 zero-shot / covariate/multivariate 接口顺手，可作为补充表。

9. **TTM (Tiny Time Mixers)**
   - 参数小、效率高；
   - 适合资源受限或 few-shot 补实验。

## 9.4 当前最现实的最小对比池
若算力与时间有限，最小可发表对比池建议为：
- `DLinear`
- `PatchTST`
- `iTransformer`
- `ModernTCN`
- `vanilla TimeMixer / TimeMixer++`
- `TQNet`

---

## 10. 最终落地路线

## P0（必须先完成）
1. 把论文实验协议正式改成三层 leaderboard；
2. 明确每个数据集-预测长度是否进入 clean 主榜；
3. 先完成 `AIF-Plus`；
4. 主对比池至少补齐：
   - `iTransformer`
   - `ModernTCN`
   - `vanilla TimeMixer / TimeMixer++`
   - `1 个 foundation/reference baseline`

## P1（随后完成）
1. 实现 `AEF-Plus`；
2. 重点在以下集合上展示 exploit upper bound：
   - `solar_AL`
   - `ETTh2 recoverable_only`
   - `weather`

## P2（有余力再做）
1. 为最强 FM baseline 打通 clean fine-tuning；
2. 做 unified leaderboard 页面或附录；
3. 补统计显著性与误差分布分析。

---

## 11. 论文最终叙事（修订版）

修订后，整篇论文的故事应写成：

1. **support-aware QC 与 counterfactual evaluation protocol**
   - 不是所有 raw leaderboards 都可靠；
   - 必须把 raw 兼容性、clean performance、artifact reliance 分开评估。

2. **QC-Clean Mainboard**
   - 证明 AIF-Plus 不仅对有 artifact 风险的数据有用；
   - 对 clean-dominant / naturally clean 数据集也能带来增益。

3. **AEF-Plus 作为 exploit upper bound**
   - 定量说明 benchmark 中 shortcut 的可利用程度；
   - 解释某些 raw winner 为何可能是 artifact-reliant。

4. **AIF-Plus 作为 clean-content forecaster**
   - 不只是“避免 shortcut”，而是在 clean 榜上给出真正强的 clean-view 结果。

一句话总结新的核心主张：

> 我们不是简单地提出一个新 backbone 去刷 raw leaderboard；
> 我们提出了一套 support-aware 的时序质控与评测协议，并在该协议下同时给出了 artifact-exploitation 上界（AEF-Plus）与 clean-content 强模型（AIF-Plus）。
