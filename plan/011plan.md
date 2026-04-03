先给结论：
新的 AIF-Plus 不该是“把你现在这版再做大”，而应该是“clean-first 主干 + very weak artifact residual”的 v2。
你之前的规划把 AIF-Plus 定位成 clean-content model，带 soft masking、双 latent、GRL、expert decoder 和 pair/CVaR 等约束；当前代码也确实基本按这个方向实现了：同一个 backbone 分别跑 x_raw 和 x_masked，再构造 z_clean / z_art，把 metadata 和 nuisance gate 接回主 decoder，并用 TopKRouter 聚合 experts。

但从你最新的 strong-support clean 子榜看，真正稳定赢的不是这种“强 artifact-aware + 重路由”结构，而主要是 PatchTST、TQNet、TimeMixer；你自己的 AIF-Plus 在 solar_AL 上已经很接近最优，但在 ETTh1 / ETTm1 上还明显落后，winner table 里这个趋势很清楚。也就是说，你现在的短板不是 clean-aware 不够，而是主干的 clean forecasting inductive bias 还不如 clean 榜冠军。

这也和这些强 backbone 的公开特点一致：PatchTST 的核心是 patching + channel-independence，能保留局部语义并更高效地吃长历史；iTransformer 把变量当 token 来建模跨变量相关；TimeMixer 用 multiscale decomposition + predictor mixing 做细/粗尺度的趋势与季节信息聚合；TQNet 用周期性、可学习的 queries 去抓更稳健的跨变量关系，而且结构非常轻；ModernTCN 则说明纯卷积在时间序列里只要有足够大的有效感受野也能很强。

所以我建议你把新模型改成这一版：

新模型：AIF-Plus v2 / AIF-Plus-PQT

名字我建议叫 AIF-Plus-PQT，意思是 Patch + Query + TimeMix。

它由四部分组成：

CI-Patch 主支
用 PatchTST 风格的 patch encoder 做主通路。
这是 clean 榜里最该优先保住的能力，因为它在 ETTh1 / ETTm1 上最稳。
TQ Cross-Variable 主支
在 patch token 上再接一个 TQNet 风格 的跨变量模块：
K/V 来自样本 patch 表征，Q 用周期性 learnable queries。
这样比你现在全量融合的 variate branch 更像 clean 榜赢家，尤其对 ETTm1 这类多变量 clean task 更有利。
Multi-Scale Mixing 主支
再接一个轻量的 TimeMixer 风格 multiscale 分支，只负责 multiscale / phase-sensitive 结构。
这条支路主要是为了保住你在 solar_AL 上已经接近最优的能力。
Weak Artifact Residual 辅支
只从 delta = x_raw - x_masked 和 uncertainty 里做一个很小的 residual head。
它只做微弱校正，不再像现在这样让 z_art 和 metadata 深度参与主 decoder。
最终预测写成
y_hat = y_clean + λ_res * y_art，其中 λ_res 限制在 0~0.05。
为什么这版比当前代码更有希望

你当前代码里最容易伤 clean 榜的三件事是：

heavy dual-backbone：x_raw 和 x_masked 各跑一遍 backbone；
metadata 直连主预测：meta_vec 直接进 clean_state，router 也吃 metadata；
nuisance 回流主路：z_art 通过 gate 回到 decode_state。

这套设计对 solar_AL 这种 structured-artifact 数据有帮助，但会让模型在 clean-dominant 任务上更容易“为了 robustness 牺牲 forecasting bias”。
所以 v2 的核心原则是：

主干只跑一次，而且只跑 x_masked；
x_raw 不再走完整 backbone，只进一个很小的 residual artifact branch；
metadata 不进主 trunk，最多只保留一个轻量的 horizon embedding；
不再用重型 TopKRouter + 4 experts 当主 decoder，最多保留 2 个 soft experts，或者直接双头解码。
我建议你直接改的地方

相对你现在的代码，我建议：

删掉 meta_vec -> clean_state
删掉 TopKRouter 主路由
删掉 full-sequence reconstruction，改成 只重建 masked span
把 z_art 从“主 decoder 的一部分”改成“弱 residual head”
heavy backbone 只保留给 x_masked
x_raw - x_masked 只喂小卷积 residual branch

也就是说，新的主干不再是：
masked backbone + raw backbone + clean/art split + router

而是：
masked backbone(main) + delta residual(small)

直接可跑的默认超参数

我建议第一版就用一个统一 config：

d_model = 256
patch_len = 16
patch_stride = 8
n_patch_layers = 3
n_heads = 8
scales = [1, 2, 4]
dropout = 0.05（solar/weather 可用 0.1）
seq_len = max(4H, 192)，solar/weather 用 max(6H, 256)
λ_res_init = 0.01, λ_res_max = 0.05

TQ 分支建议：

num_queries = 32
num_tq_layers = 1
query_period = 24 for ETT/exchange
query_period = 48 for solar/weather

总参数量控制在 18M–24M，比你当前 v1 略瘦，但更像 clean 榜强模型。

训练策略也要改

你原先计划里，checkpoint 选择会混入 WGR / ARG / RI 组合分数。对 robustness 论文这没问题，但如果你现在想尽量把 clean 榜打到最好，那就不该这样选。

我建议把训练改成三段：

Stage B-clean-first
只训 L_clean + 0.10 * L_mask
这里的 L_mask 只重建 masked span，不重建整段历史。

Stage C-light robustness
再加
0.05 * L_adv + 0.05 * L_pair + 0.05 * L_cvar

也就是把你之前的 pair / rec / cvar 权重全降下来。你原先给的是 0.20 pair + 0.20 rec + 0.15 cvar，这对冲 clean-SOTA 来说偏重了。

选模规则
主榜 checkpoint 只按 clean val MAE 选。
Stage C 如果让 clean val MAE 变差，就保留 Stage B checkpoint 当 clean 榜主模型；Stage C 的 ckpt 只拿去做 robustness 附表。

这一点很关键。因为你现在的目标是“尽量达到 clean-SOTA”，不是“同一个 ckpt 同时最鲁棒又最准”。

最该先做的版本

我建议第一版只做这个：

AIF-Plus-PQT = CI-Patch + TQ + TimeMix + Weak Residual

不要再加：

full dual-backbone
强 metadata 主注入
重 MoE 路由
full reconstruction
从 epoch 0 就开的强 GRL / pair / CVaR

等你把这个版本跑出来，再做三个 ablation：

去掉 metadata 主注入
用 CI-Patch + TQ 替换现有 backbone
再加 multiscale 分支看 solar_AL 是否回升

最有可能的结果是：

ETTh1 / ETTm1 明显上涨；
solar_AL 只小幅回落，甚至 H192/H336 有机会继续接近或超过当前最好；
整体 clean 榜比现在更接近前二梯队。

一句话概括这版新模型：

别把 AIF-Plus v2 做成“更复杂的 artifact-aware 模型”，而要做成“以 PatchTST/TQNet/TimeMixer 的 clean inductive bias 为主干，artifact-aware 只做弱辅助”的 clean-first forecaster。

# AIF-Plus v2 设计建议（基于当前 AIFPlus 代码 + clean 榜强模型特征）

## 1. 为什么需要改版

当前 `AIFPlus` 代码已经具备：
- `x_raw / x_masked` 双输入；
- `z_clean / z_art` 双分支；
- `GRL` 对抗解耦；
- metadata 编码；
- MoE decoder；
- reconstruction head。  
见当前实现代码：`Pasted code.py`。其核心是双次 backbone 编码、再用 `meta_vec + z_art` 进入主 decoder 路径，并让 router 依赖 metadata、scale/var state 和 clean latent 共同决定。  

但从 clean 主榜现状看，稳定强势的方法并不是“强 artifact-aware + 重路由”的模型，而主要是：
- `PatchTST`
- `TQNet`
- `TimeMixer`
- 少量任务上的 `iTransformer`

在 strong-support clean 子榜（9 个任务）上：
| task_key          | dataset_name   |   horizon | winner_method   |   winner_mae |   aif_mae |   aif_gap |   train_clean_ratio |   val_clean_windows |
|:------------------|:---------------|----------:|:----------------|-------------:|----------:|----------:|--------------------:|--------------------:|
| ETTh1|L96|H192    | ETTh1          |       192 | PatchTST        |     0.416109 |  0.443166 |  0.027057 |            0.705854 |                1629 |
| ETTh1|L96|H336    | ETTh1          |       336 | PatchTST        |     0.422299 |  0.464059 |  0.04176  |            0.560361 |                1061 |
| ETTh1|L96|H96     | ETTh1          |        96 | TQNet           |     0.388996 |  0.40958  |  0.020584 |            0.800095 |                2007 |
| ETTm1|L96|H192    | ETTm1          |       192 | TQNet           |     0.370721 |  0.382604 |  0.011883 |            0.894173 |                9481 |
| ETTm1|L96|H336    | ETTm1          |       336 | PatchTST        |     0.390975 |  0.403799 |  0.012824 |            0.863518 |                8909 |
| ETTm1|L96|H720    | ETTm1          |       720 | PatchTST        |     0.427084 |  0.438542 |  0.011458 |            0.78056  |                7382 |
| ETTm1|L96|H96     | ETTm1          |        96 | TQNet           |     0.343536 |  0.350678 |  0.007142 |            0.913818 |                9862 |
| solar_AL|L96|H192 | solar_AL       |       192 | TimeMixer       |     0.288152 |  0.291372 |  0.00322  |            0.69322  |                3496 |
| solar_AL|L96|H96  | solar_AL       |        96 | TimeMixer       |     0.255946 |  0.266394 |  0.010448 |            0.902161 |                4600 |

方法级汇总：
| method       |   n_tasks |   mean_mae |   average_rank |   wins |
|:-------------|----------:|-----------:|---------------:|-------:|
| PatchTST     |         9 |   0.371281 |        2.22222 |      4 |
| TQNet        |         9 |   0.372028 |        2.33333 |      3 |
| TimeMixer    |         9 |   0.374499 |        3       |      2 |
| ModernTCN    |         9 |   0.394189 |        4.77778 |      0 |
| AIF-Plus     |         9 |   0.383355 |        4.88889 |      0 |
| iTransformer |         9 |   0.381011 |        5       |      0 |
| DLinear      |         9 |   0.406682 |        5.77778 |      0 |
| TimeMixerPP  |         9 |   0.594196 |        8       |      0 |

这说明 AIFPlus v1 的主要问题不是“clean-aware 不够”，而是**主干的 clean forecasting inductive bias 还不如 clean 榜最强方法**。

---

## 2. 从当前结果读出的设计原则

### 原则 A：主干必须先像 clean 榜冠军
- `ETTh1 / ETTm1` 上，主胜者主要是 `PatchTST` 与 `TQNet`。
- `solar_AL` 上，主胜者主要是 `TimeMixer`。
- 因此 v2 不能再以“artifact split + metadata routing”作为主角，而应该：
  - 用 `PatchTST` 风格 patch/token 主通路；
  - 用 `TQNet` 风格 cross-variate query 补充跨变量关系；
  - 用 `TimeMixer` 风格 multiscale branch 处理多尺度与 phase-sensitive 结构。

### 原则 B：artifact 信息只能做**弱辅助**
当前代码里，`meta_vec` 直接进入 `clean_fusion`，`z_art` 通过 `nuisance_gate` 进入 `decode_state`，而 router 也强依赖 `meta_vec`。这对 solar 有利，但对 ETTh1/ETTm1 这种 clean-dominant 任务容易伤 accuracy。

所以 v2 应该把 artifact 相关信息降级为：
- preprocessor 的 soft mask；
- auxiliary adversarial head；
- optional residual correction（默认非常小）。

### 原则 C：去掉“可能破坏 clean accuracy 的复杂路径”
最该弱化的部分：
- dataset/support/horizon metadata 直连主干；
- 强 MoE top-k 路由；
- full-sequence reconstruction；
- 过早打开的对抗解耦与 pair loss。

---

## 3. 新模型：AIF-Plus v2（CleanSOTA-Oriented）

## 3.1 总体结构

AIF-Plus v2 采用“三主支 + 一弱辅助支”：

1. **CI-Patch Branch（主分支）**
   - PatchTST 风格 patch embedding + channel-independence
   - 专门负责 clean-dominant 任务的稳定 forecasting

2. **TQ Cross-Variable Branch（主分支）**
   - TQNet 风格 Temporal Query cross-variate module
   - 用 learnable periodic queries 建模更稳健的跨变量相关

3. **Multi-Scale Mixing Branch（主分支）**
   - TimeMixer 风格 multiscale decomposition + predictor mixing
   - 负责 solar_AL 一类 phase-sensitive / multiscale 任务

4. **Artifact Residual Branch（弱辅助）**
   - 仅从 `x_raw - x_masked` 和 uncertainty 中提取一个小 residual
   - 默认以很小系数加到主预测上
   - 目标不是“利用 artifact”，而是保留必要边界上下文

最终预测：
\[
\hat y = \hat y_{clean} + \lambda_{res} \cdot \hat y_{art}, \quad \lambda_{res} \in [0, 0.05]
\]

其中：
- `\hat y_clean` 来自前三个 clean 主支；
- `\hat y_art` 只做小校正；
- 若训练发现 clean 榜下降，可直接把 `\lambda_res` 置零。

---

## 3.2 详细网络

### A. Preprocessor
输入：
- `x_raw`
- `x_masked`
- `u` (uncertainty/confidence)
- `delta = x_raw - x_masked`

处理：
- `RevIN` on `x_masked`
- optional difference branch 只对 `solar_AL / weather` 打开
- 不再把 `dataset_id / support_id / horizon_id` 送入主 encoder，只留给轻量 gating

输出：
- `x_clean_in`
- `x_resid_in`

### B. CI-Patch Branch（PatchTST 主支）
- patch_len = 16
- stride = 8
- 每个变量独立 patch embedding，共享 encoder 权重
- 2~3 层 Transformer encoder
- 输出：
  - `z_patch_token`
  - `z_patch_global`

目的：
- 保留 PatchTST 的 patching + channel-independence 优势
- 优先服务 ETTh1 / ETTm1 clean tasks

### C. TQ Cross-Variable Branch（TQNet 主支）
输入：
- 使用 `z_patch_token` 或原始 patch summary 作为 K/V
- learnable periodic queries 作为 Q

结构：
- 单层或双层 attention 即可
- 周期 query 数 `W_q ∈ {24, 48}`
- 输出 `z_tq`

目的：
- 用“稳定 query + sample-level K/V”建模跨变量关系
- 避免 iTransformer 那种全量 token attention 的训练负担
- 针对 clean 榜里 TQNet 短中期表现强这一点

### D. Multi-Scale Mixing Branch（TimeMixer 主支）
输入：
- `x_clean_in`

结构：
- downsample scales: `[1, 2, 4]`
- seasonal/trend decomposition
- 轻量 PDM/FMM 风格 mixing
- 输出 `z_scale`

目的：
- 把 solar_AL 上明显重要的 phase-aware / multiscale 结构吃进去
- 同时保持参数量可控

### E. Clean Fusion Head
把三个主支融合：
\[
z_{clean} = \text{MLP}([z_{patch}, z_{tq}, z_{scale}])
\]

然后做两个头：
1. `head_local`: per-channel direct forecast
2. `head_global`: cross-channel residual refinement

最终：
\[
\hat y_{clean} = head_{local}(z_{clean}) + head_{global}(z_{clean})
\]

这里不再使用重 MoE top-k router。  
如果你仍想保留 expert 概念，建议只保留 **2 expert**：
- local-periodic expert
- multiscale-transition expert

用 soft gate，不要 top-k。

### F. Weak Artifact Residual Branch
输入：
- `delta = x_raw - x_masked`
- `u`

结构：
- 小型 Conv1D + GAP + MLP
- 输出 horizon-wise residual

约束：
- `stopgrad` 主干输入
- residual scale 可学习，但上限 clamp 到 `0.05`

目的：
- 只保留极少量边界/相位补偿
- 避免 v1 里 `z_art + meta_vec` 深度侵入主预测

---

## 3.3 你当前代码应删/改什么

### 应删
1. `meta_vec` 直接拼到 `clean_state`
2. `TopKRouter` 多 expert 主路由
3. full sequence `reconstruction_head`（改成 masked span only）
4. `z_art` 直接深度进入主 decoder

### 应弱化
1. `GRL`
   - 不要从 epoch 0 开始全强度开
   - 改成 warm-start + schedule
2. `pair invariance`
   - 只在 Stage 3 开启
   - 且只对 support 足够的数据集启用
3. metadata
   - 只保留 `horizon embedding`
   - `dataset/support` 只用于 very light adapter 或 calibration，不进主 trunk

### 应保留
1. `x_raw / x_masked / uncertainty`
2. RevIN
3. weak artifact branch
4. adversarial disentangle（后期开）
5. clean-view 主损失

---

## 4. 推荐超参数（直接可跑）

## 4.1 统一版本
- `seq_len`:
  - `ETTh1 / ETTm1 / exchange_rate`: `max(4H, 192)` capped at 512
  - `solar_AL / weather`: `max(6H, 256)` capped at 768
- `d_model = 256`
- `patch_len = 16`
- `patch_stride = 8`
- `n_patch_layers = 3`
- `n_heads = 8`
- `ffn_ratio = 4`
- `dropout = 0.1`
- `stochastic_depth = 0.05`

## 4.2 TQ branch
- `num_tq_layers = 1`
- `num_queries = 32`
- `query_period = 24` for ETT / exchange_rate
- `query_period = 48` for solar/weather
- `tq_dim = 128`

## 4.3 Multiscale branch
- `scales = [1, 2, 4]`
- `decomp_kernel = 25`
- `scale_hidden = 192`

## 4.4 Residual artifact branch
- `resid_hidden = 64`
- `lambda_res_init = 0.01`
- `lambda_res_max = 0.05`

## 4.5 参数规模
- 总参数量目标：`18M ~ 24M`
- 明显比 v1 更“主干导向”，比你原来的 22M~30M 更瘦一点

---

## 5. 损失函数（v2 推荐）

### Stage 1: clean-first
\[
L = L_{clean} + 0.10 L_{mask}
\]

### Stage 2: add weak robustness
\[
L = L_{clean} + 0.10 L_{mask} + 0.05 L_{adv}
\]

### Stage 3: targeted robustness finetune
\[
L = L_{clean} + 0.10 L_{mask} + 0.05 L_{adv} + 0.05 L_{pair} + 0.05 L_{cvar}
\]

不建议一上来就用：
- `0.20 pair`
- `0.20 rec`
- `0.15 CVaR`

这些更像 v1 里拉低 clean accuracy 的高风险项。

### 具体定义
- `L_clean`: MAE on clean board
- `L_mask`: 只重建 masked span，不重建整段历史
- `L_adv`: z_clean 难预测 artifact label
- `L_pair`: 只对 flagged & target-clean，且只约束中间 clean representation
- `L_cvar`: top-10% worst group，不要一开始就 top-20%

---

## 6. 训练流程

### Stage A（可选）
- masked patch pretraining
- 5 epoch 足够
- 不做复杂 synthetic artifact taxonomy，只保留：
  - zero block
  - flat run
  - near constant

### Stage B（主训练）
- 只开 `L_clean + 0.10 L_mask`
- epoch 30~40
- 用 clean val MAE 早停

### Stage C（轻量 robustness finetune）
- 冻结 patch branch 前 1/2 层
- 开 `L_adv + L_pair + L_cvar`
- epoch 8~10
- lr 降到 1/4

### checkpoint 选择
第一轮直接按：
\[
score = 0.85 \cdot z(MAE_{clean}) + 0.10 \cdot z(ARG) + 0.05 \cdot z(WGR)
\]
不建议当前就把 `RI` 放进选模分数。

---

## 7. 为什么这个 v2 更可能接近 clean-SOTA

1. **更像 clean 榜冠军**
   - PatchTST: patching + CI
   - TQNet: robust cross-variate correlation
   - TimeMixer: multiscale mixing
   三者正好对应你 strong-support 主榜里的主要赢家。

2. **artifact-aware 不再主导预测**
   - 避免 v1 那种“为了 robustness 把 clean accuracy 拉掉”的结构性副作用

3. **对 solar_AL 仍保留优势**
   - 因为 multiscale branch + weak residual branch 还在
   - 不会像纯 PatchTST 那样完全丢 phase-aware 信息

4. **工程上比 FM/adapter 更现实**
   - 不需要引入外部 foundation 权重
   - 直接在你现有工程里能落地

---

## 8. 我建议你先跑的三个 ablation

### Ablation 1: 去掉 metadata 主注入
- 目标：验证 `meta_vec` 是否真的在伤 clean 榜
- 预期：ETTh1 / ETTm1 明显变好

### Ablation 2: v1 backbone vs CI-Patch + TQ
- 目标：验证 clean 主干是否比双路 latent split 更重要
- 预期：ETTh1 / ETTm1 / exchange_rate 提升最大

### Ablation 3: + multiscale branch
- 目标：看 solar_AL 是否回升
- 预期：solar H96/H192 至少追平当前 AIFPlus，H336 有机会再超一点

---

## 9. 最后的单句结论

AIF-Plus v2 不应该再是“更复杂的 artifact-aware model”，而应该是：

> **以 PatchTST/TQNet/TimeMixer 三类 clean 榜强 inductive bias 为主干，artifact-aware 只做弱辅助约束的 clean-first forecasting model。**

这条路线比继续堆 `metadata + GRL + MoE + reconstruction` 更有机会把你推到 clean 榜前列。

# AIF-Plus v2 设计建议（基于当前 AIFPlus 代码 + clean 榜强模型特征）

## 1. 为什么需要改版

当前 `AIFPlus` 代码已经具备：
- `x_raw / x_masked` 双输入；
- `z_clean / z_art` 双分支；
- `GRL` 对抗解耦；
- metadata 编码；
- MoE decoder；
- reconstruction head。  
见当前实现代码：`Pasted code.py`。其核心是双次 backbone 编码、再用 `meta_vec + z_art` 进入主 decoder 路径，并让 router 依赖 metadata、scale/var state 和 clean latent 共同决定。  

但从 clean 主榜现状看，稳定强势的方法并不是“强 artifact-aware + 重路由”的模型，而主要是：
- `PatchTST`
- `TQNet`
- `TimeMixer`
- 少量任务上的 `iTransformer`

在 strong-support clean 子榜（9 个任务）上：
| task_key          | dataset_name   |   horizon | winner_method   |   winner_mae |   aif_mae |   aif_gap |   train_clean_ratio |   val_clean_windows |
|:------------------|:---------------|----------:|:----------------|-------------:|----------:|----------:|--------------------:|--------------------:|
| ETTh1|L96|H192    | ETTh1          |       192 | PatchTST        |     0.416109 |  0.443166 |  0.027057 |            0.705854 |                1629 |
| ETTh1|L96|H336    | ETTh1          |       336 | PatchTST        |     0.422299 |  0.464059 |  0.04176  |            0.560361 |                1061 |
| ETTh1|L96|H96     | ETTh1          |        96 | TQNet           |     0.388996 |  0.40958  |  0.020584 |            0.800095 |                2007 |
| ETTm1|L96|H192    | ETTm1          |       192 | TQNet           |     0.370721 |  0.382604 |  0.011883 |            0.894173 |                9481 |
| ETTm1|L96|H336    | ETTm1          |       336 | PatchTST        |     0.390975 |  0.403799 |  0.012824 |            0.863518 |                8909 |
| ETTm1|L96|H720    | ETTm1          |       720 | PatchTST        |     0.427084 |  0.438542 |  0.011458 |            0.78056  |                7382 |
| ETTm1|L96|H96     | ETTm1          |        96 | TQNet           |     0.343536 |  0.350678 |  0.007142 |            0.913818 |                9862 |
| solar_AL|L96|H192 | solar_AL       |       192 | TimeMixer       |     0.288152 |  0.291372 |  0.00322  |            0.69322  |                3496 |
| solar_AL|L96|H96  | solar_AL       |        96 | TimeMixer       |     0.255946 |  0.266394 |  0.010448 |            0.902161 |                4600 |

方法级汇总：
| method       |   n_tasks |   mean_mae |   average_rank |   wins |
|:-------------|----------:|-----------:|---------------:|-------:|
| PatchTST     |         9 |   0.371281 |        2.22222 |      4 |
| TQNet        |         9 |   0.372028 |        2.33333 |      3 |
| TimeMixer    |         9 |   0.374499 |        3       |      2 |
| ModernTCN    |         9 |   0.394189 |        4.77778 |      0 |
| AIF-Plus     |         9 |   0.383355 |        4.88889 |      0 |
| iTransformer |         9 |   0.381011 |        5       |      0 |
| DLinear      |         9 |   0.406682 |        5.77778 |      0 |
| TimeMixerPP  |         9 |   0.594196 |        8       |      0 |

这说明 AIFPlus v1 的主要问题不是“clean-aware 不够”，而是**主干的 clean forecasting inductive bias 还不如 clean 榜最强方法**。

---

## 2. 从当前结果读出的设计原则

### 原则 A：主干必须先像 clean 榜冠军
- `ETTh1 / ETTm1` 上，主胜者主要是 `PatchTST` 与 `TQNet`。
- `solar_AL` 上，主胜者主要是 `TimeMixer`。
- 因此 v2 不能再以“artifact split + metadata routing”作为主角，而应该：
  - 用 `PatchTST` 风格 patch/token 主通路；
  - 用 `TQNet` 风格 cross-variate query 补充跨变量关系；
  - 用 `TimeMixer` 风格 multiscale branch 处理多尺度与 phase-sensitive 结构。

### 原则 B：artifact 信息只能做**弱辅助**
当前代码里，`meta_vec` 直接进入 `clean_fusion`，`z_art` 通过 `nuisance_gate` 进入 `decode_state`，而 router 也强依赖 `meta_vec`。这对 solar 有利，但对 ETTh1/ETTm1 这种 clean-dominant 任务容易伤 accuracy。

所以 v2 应该把 artifact 相关信息降级为：
- preprocessor 的 soft mask；
- auxiliary adversarial head；
- optional residual correction（默认非常小）。

### 原则 C：去掉“可能破坏 clean accuracy 的复杂路径”
最该弱化的部分：
- dataset/support/horizon metadata 直连主干；
- 强 MoE top-k 路由；
- full-sequence reconstruction；
- 过早打开的对抗解耦与 pair loss。

---

## 3. 新模型：AIF-Plus v2（CleanSOTA-Oriented）

## 3.1 总体结构

AIF-Plus v2 采用“三主支 + 一弱辅助支”：

1. **CI-Patch Branch（主分支）**
   - PatchTST 风格 patch embedding + channel-independence
   - 专门负责 clean-dominant 任务的稳定 forecasting

2. **TQ Cross-Variable Branch（主分支）**
   - TQNet 风格 Temporal Query cross-variate module
   - 用 learnable periodic queries 建模更稳健的跨变量相关

3. **Multi-Scale Mixing Branch（主分支）**
   - TimeMixer 风格 multiscale decomposition + predictor mixing
   - 负责 solar_AL 一类 phase-sensitive / multiscale 任务

4. **Artifact Residual Branch（弱辅助）**
   - 仅从 `x_raw - x_masked` 和 uncertainty 中提取一个小 residual
   - 默认以很小系数加到主预测上
   - 目标不是“利用 artifact”，而是保留必要边界上下文

最终预测：
\[
\hat y = \hat y_{clean} + \lambda_{res} \cdot \hat y_{art}, \quad \lambda_{res} \in [0, 0.05]
\]

其中：
- `\hat y_clean` 来自前三个 clean 主支；
- `\hat y_art` 只做小校正；
- 若训练发现 clean 榜下降，可直接把 `\lambda_res` 置零。

---

## 3.2 详细网络

### A. Preprocessor
输入：
- `x_raw`
- `x_masked`
- `u` (uncertainty/confidence)
- `delta = x_raw - x_masked`

处理：
- `RevIN` on `x_masked`
- optional difference branch 只对 `solar_AL / weather` 打开
- 不再把 `dataset_id / support_id / horizon_id` 送入主 encoder，只留给轻量 gating

输出：
- `x_clean_in`
- `x_resid_in`

### B. CI-Patch Branch（PatchTST 主支）
- patch_len = 16
- stride = 8
- 每个变量独立 patch embedding，共享 encoder 权重
- 2~3 层 Transformer encoder
- 输出：
  - `z_patch_token`
  - `z_patch_global`

目的：
- 保留 PatchTST 的 patching + channel-independence 优势
- 优先服务 ETTh1 / ETTm1 clean tasks

### C. TQ Cross-Variable Branch（TQNet 主支）
输入：
- 使用 `z_patch_token` 或原始 patch summary 作为 K/V
- learnable periodic queries 作为 Q

结构：
- 单层或双层 attention 即可
- 周期 query 数 `W_q ∈ {24, 48}`
- 输出 `z_tq`

目的：
- 用“稳定 query + sample-level K/V”建模跨变量关系
- 避免 iTransformer 那种全量 token attention 的训练负担
- 针对 clean 榜里 TQNet 短中期表现强这一点

### D. Multi-Scale Mixing Branch（TimeMixer 主支）
输入：
- `x_clean_in`

结构：
- downsample scales: `[1, 2, 4]`
- seasonal/trend decomposition
- 轻量 PDM/FMM 风格 mixing
- 输出 `z_scale`

目的：
- 把 solar_AL 上明显重要的 phase-aware / multiscale 结构吃进去
- 同时保持参数量可控

### E. Clean Fusion Head
把三个主支融合：
\[
z_{clean} = \text{MLP}([z_{patch}, z_{tq}, z_{scale}])
\]

然后做两个头：
1. `head_local`: per-channel direct forecast
2. `head_global`: cross-channel residual refinement

最终：
\[
\hat y_{clean} = head_{local}(z_{clean}) + head_{global}(z_{clean})
\]

这里不再使用重 MoE top-k router。  
如果你仍想保留 expert 概念，建议只保留 **2 expert**：
- local-periodic expert
- multiscale-transition expert

用 soft gate，不要 top-k。

### F. Weak Artifact Residual Branch
输入：
- `delta = x_raw - x_masked`
- `u`

结构：
- 小型 Conv1D + GAP + MLP
- 输出 horizon-wise residual

约束：
- `stopgrad` 主干输入
- residual scale 可学习，但上限 clamp 到 `0.05`

目的：
- 只保留极少量边界/相位补偿
- 避免 v1 里 `z_art + meta_vec` 深度侵入主预测

---

## 3.3 你当前代码应删/改什么

### 应删
1. `meta_vec` 直接拼到 `clean_state`
2. `TopKRouter` 多 expert 主路由
3. full sequence `reconstruction_head`（改成 masked span only）
4. `z_art` 直接深度进入主 decoder

### 应弱化
1. `GRL`
   - 不要从 epoch 0 开始全强度开
   - 改成 warm-start + schedule
2. `pair invariance`
   - 只在 Stage 3 开启
   - 且只对 support 足够的数据集启用
3. metadata
   - 只保留 `horizon embedding`
   - `dataset/support` 只用于 very light adapter 或 calibration，不进主 trunk

### 应保留
1. `x_raw / x_masked / uncertainty`
2. RevIN
3. weak artifact branch
4. adversarial disentangle（后期开）
5. clean-view 主损失

---

## 4. 推荐超参数（直接可跑）

## 4.1 统一版本
- `seq_len`:
  - `ETTh1 / ETTm1 / exchange_rate`: `max(4H, 192)` capped at 512
  - `solar_AL / weather`: `max(6H, 256)` capped at 768
- `d_model = 256`
- `patch_len = 16`
- `patch_stride = 8`
- `n_patch_layers = 3`
- `n_heads = 8`
- `ffn_ratio = 4`
- `dropout = 0.1`
- `stochastic_depth = 0.05`

## 4.2 TQ branch
- `num_tq_layers = 1`
- `num_queries = 32`
- `query_period = 24` for ETT / exchange_rate
- `query_period = 48` for solar/weather
- `tq_dim = 128`

## 4.3 Multiscale branch
- `scales = [1, 2, 4]`
- `decomp_kernel = 25`
- `scale_hidden = 192`

## 4.4 Residual artifact branch
- `resid_hidden = 64`
- `lambda_res_init = 0.01`
- `lambda_res_max = 0.05`

## 4.5 参数规模
- 总参数量目标：`18M ~ 24M`
- 明显比 v1 更“主干导向”，比你原来的 22M~30M 更瘦一点

---

## 5. 损失函数（v2 推荐）

### Stage 1: clean-first
\[
L = L_{clean} + 0.10 L_{mask}
\]

### Stage 2: add weak robustness
\[
L = L_{clean} + 0.10 L_{mask} + 0.05 L_{adv}
\]

### Stage 3: targeted robustness finetune
\[
L = L_{clean} + 0.10 L_{mask} + 0.05 L_{adv} + 0.05 L_{pair} + 0.05 L_{cvar}
\]

不建议一上来就用：
- `0.20 pair`
- `0.20 rec`
- `0.15 CVaR`

这些更像 v1 里拉低 clean accuracy 的高风险项。

### 具体定义
- `L_clean`: MAE on clean board
- `L_mask`: 只重建 masked span，不重建整段历史
- `L_adv`: z_clean 难预测 artifact label
- `L_pair`: 只对 flagged & target-clean，且只约束中间 clean representation
- `L_cvar`: top-10% worst group，不要一开始就 top-20%

---

## 6. 训练流程

### Stage A（可选）
- masked patch pretraining
- 5 epoch 足够
- 不做复杂 synthetic artifact taxonomy，只保留：
  - zero block
  - flat run
  - near constant

### Stage B（主训练）
- 只开 `L_clean + 0.10 L_mask`
- epoch 30~40
- 用 clean val MAE 早停

### Stage C（轻量 robustness finetune）
- 冻结 patch branch 前 1/2 层
- 开 `L_adv + L_pair + L_cvar`
- epoch 8~10
- lr 降到 1/4

### checkpoint 选择
第一轮直接按：
\[
score = 0.85 \cdot z(MAE_{clean}) + 0.10 \cdot z(ARG) + 0.05 \cdot z(WGR)
\]
不建议当前就把 `RI` 放进选模分数。

---

## 7. 为什么这个 v2 更可能接近 clean-SOTA

1. **更像 clean 榜冠军**
   - PatchTST: patching + CI
   - TQNet: robust cross-variate correlation
   - TimeMixer: multiscale mixing
   三者正好对应你 strong-support 主榜里的主要赢家。

2. **artifact-aware 不再主导预测**
   - 避免 v1 那种“为了 robustness 把 clean accuracy 拉掉”的结构性副作用

3. **对 solar_AL 仍保留优势**
   - 因为 multiscale branch + weak residual branch 还在
   - 不会像纯 PatchTST 那样完全丢 phase-aware 信息

4. **工程上比 FM/adapter 更现实**
   - 不需要引入外部 foundation 权重
   - 直接在你现有工程里能落地

---

## 8. 我建议你先跑的三个 ablation

### Ablation 1: 去掉 metadata 主注入
- 目标：验证 `meta_vec` 是否真的在伤 clean 榜
- 预期：ETTh1 / ETTm1 明显变好

### Ablation 2: v1 backbone vs CI-Patch + TQ
- 目标：验证 clean 主干是否比双路 latent split 更重要
- 预期：ETTh1 / ETTm1 / exchange_rate 提升最大

### Ablation 3: + multiscale branch
- 目标：看 solar_AL 是否回升
- 预期：solar H96/H192 至少追平当前 AIFPlus，H336 有机会再超一点

---

## 9. 最后的单句结论

AIF-Plus v2 不应该再是“更复杂的 artifact-aware model”，而应该是：

> **以 PatchTST/TQNet/TimeMixer 三类 clean 榜强 inductive bias 为主干，artifact-aware 只做弱辅助约束的 clean-first forecasting model。**

这条路线比继续堆 `metadata + GRL + MoE + reconstruction` 更有机会把你推到 clean 榜前列。

