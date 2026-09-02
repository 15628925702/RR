# RR-GID_CN：P4 代码审查与修正 Guidance

**审查日期：2026-08-26**  
**审查对象：公开 GitHub `main` 分支上的 P4 主路径，以及 `RR_GID_CN.pdf` 和阶段阻塞报告**  
**建议放置位置：`docs/P4_CODE_AUDIT_AND_REPAIR_GUIDANCE_20260826.md`**

---

## 0. 审查范围、结论边界与使用方式

本次审查重点覆盖：

- `src/rr_gid_cn/s1_gate.py`
  - `prepare_s1_oracle`
  - `a_optimal_information`
  - `solve_pilot_beta`
  - `panel_information_cross`
  - `final_rr_estimator`
  - `run_replication`
- `src/rr_gid_cn/synthetic_oracle.py`
  - true GMM/warp sampling
  - tilted full/conditional sampling
  - rejection sampling
  - fixed/adaptive QMC conditional mean
  - `log_partition` 与 `tilted_moments`
- `src/rr_gid_cn/policies.py`
  - cost-aware rounding
  - Frank–Wolfe
- `scripts/p4_formal_run.py`
- `scripts/p4_validate.py`
- `scripts/p4_summary.py`
- `scripts/p4_formal_manifest.py`
- `scripts/p4_prepare_high_precision.py`
- `scripts/p4_formal_rejection_20260826.sh`
- P2/P3/P4 相关测试和项目规划文档。

这是一次**静态代码审查**。我没有在远端 8×A800 环境实际执行正式任务，因此：

- “确定性错误”指可直接由代码与数学定义对照确认的问题；
- “高风险问题”指很可能影响 P4 曲线、但仍需通过下文的 gold-standard diagnostics 量化的问题；
- “性能瓶颈”按代码路径和复杂度判断，最终优先级应以新增的 stage-level profiler 数据确认。

### 总结性判断

当前不宜把已有 P4 结果解释为“核心理论失败”。在继续正式五档预算之前，应先解决以下四类问题：

1. **理论对象未在一个一致的 \(Q_0\) 下实现。**  
   target/full conditional 使用解析 GMM 的真实 \(Q_0\)，但 \(\mu_\beta\)、\(A(\beta)\)、pilot inverse map 和部分 \(H\) 使用固定 empirical reference pool；当前 estimator 是一个 true-oracle 与 empirical-oracle 混合的 estimating equation。

2. **评估和 baseline 存在明确错误。**  
   当前 `design_ratio` 不是 design ratio；A-OSQD 的 panel information 使用了错误的矩阵块；formal validator 又把 risk ratio 当成 design ratio。

3. **“exact observed score”标签和实际 finite-\(L_U\) rejection 路径不一致。**  
   rejection 给出 exact conditional draws，但有限样本均值不是 exact conditional expectation；有限 \(L_U\) 应使用 sandwich benchmark，或单独采用经过验证的 high-precision QMC oracle。

4. **代码的大部分时间花在重复 conditional expectation 和重复 reference-pool 扫描上，而不是 Frank–Wolfe。**  
   当前还存在 shared pilot 重算、百万 reference feature 重算、GPU rejection 过量 proposals、两进程争一张 GPU 等明显浪费。

**建议立即动作：保留现有输出作为 diagnostic，但暂停新增“formal acceptance”批次。先完成本文的 Phase 0–2，再恢复 50-replication sweep。**

**操作上限（2026-08-31）：正式 replication 一律 50，禁止再按 200 排期或启动。政策、预算、指标不变。**

---

# 1. 需要先统一的实验对象

## 1.1 P4 应拆成两个不同门禁

当前项目规划将 P4 定义为 **oracle-allocation estimator gate**：Uniform、A-OSQD 和 oracle RR-GID，其中 oracle RR-GID 直接使用在 \(\beta^\star\) 处预先求出的 \(p^\star\)。

这本身是合理的，但它只能验证：

\[
p=p^\star
\quad\Longrightarrow\quad
B\,\mathbb E[\mathrm{KL}]
\to \frac12\Phi_{\beta^\star}(p^\star),
\]

以及 pilot + Fisher-scoring 的 estimator 是否在 oracle allocation 下有效。

它**不能**验证完整 Algorithm 2 中

\[
\widetilde\beta
\longrightarrow
\widehat I_S(\widetilde\beta)
\longrightarrow
\widehat p
\]

这一条 plug-in design 链。当前 `run_replication` 对 `"oracle RR-GID"` 直接读取 `prepared["designs"]["oracle RR-GID"]`；pilot estimate 不参与这条 policy 的设计。

因此应明确拆成：

- **P4-A：oracle-allocation estimator gate**
  - policy 名称改为 `Oracle allocation p*(beta*)`
  - 只验证 final estimator 和理论风险常数
  - 该 policy 的 main-design ratio 应恒为 1（数值容差内）

- **P4-B：plug-in RR-GID design diagnostic**
  - 使用 \(\widetilde\beta\) 计算 \(\widehat F(\widetilde\beta)\)、\(\widehat I_S(\widetilde\beta)\)
  - 求出 \(\widehat p\)
  - 报告真正的 design ratio
  - 在 P4-A 通过后再作为完整 Algorithm 2 的验证

这样能避免把“oracle estimator gate”与“one-shot plug-in design theorem”混成同一个结论。

---

# 2. Correctness 审查：必须先修的问题

下面按严重程度列出。每一项都给出位置、影响、修正方向和验收标准。

---

## C-01 [Blocker] 同一个理论 \(Q_0\) 被实现成了 true/empirical 混合对象

### 位置

- `prepare_s1_oracle`
- `run_replication`
- `final_rr_estimator`
- `panel_information_cross`
- `solve_pilot_beta`
- `log_partition`
- `tilted_moments`

### 当前路径

当前代码中：

- `target_full` 由解析 GMM 下的 exact tilted rejection 采样；
- conditional score 的 completions 由解析 GMM conditional + tilt 得到；
- 但 `solve_pilot_beta` 的 \(\nabla A(\beta)\) 用 fixed reference pool；
- update 中的 \(\mu_\beta\) 默认用 `ref_large` 的 self-normalized empirical tilt；
- `panel_information_cross` 的 outer tilted samples 来自 empirical reference resampling，但 conditional completion 又来自解析 GMM；
- KL 的 \(A(\beta)\) 和 \(\mu_{\beta^\star}\) 使用 empirical reference-pool Bregman。

理论要求这些对象都由**同一个** \(Q_0\) 定义：

\[
A(\beta),\quad
\mu_\beta,\quad
F(\beta),\quad
Q_\beta,\quad
Q_\beta(X_{S^c}\mid X_S),\quad
I_S(\beta).
\]

### 为什么会造成系统误差

在 \(\beta=\beta^\star\) 处，真实 conditional score 的期望应为零：

\[
\mathbb E_{\beta^\star}
\left[
\mathbb E_{\beta^\star}[\phi(X)\mid X_S]
-\mu_{\beta^\star}
\right]=0.
\]

但当前实现可能实际使用

\[
\mu_{\beta^\star}^{(R)}
\neq
\mu_{\beta^\star}^{\mathrm{true}},
\]

于是每个 observation 的 score 有固定偏移

\[
\Delta_\mu
=
\mu_{\beta^\star}^{\mathrm{true}}
-
\mu_{\beta^\star}^{(R)}.
\]

因为 \(U_B\) 是 \(B\) 个 score 的和，

\[
\mathbb E U_B \approx B\Delta_\mu,
\qquad
H_B\approx BM,
\]

一次 scoring 后会留下

\[
M^{-1}\Delta_\mu
\]

量级的固定偏移。reference pool 固定时，这个偏移不随 target budget \(B\) 自动消失。在 weak-information directions 中还会被 \(M^{-1}\) 放大。

这也可能造成“pilot error 看似不随 \(B\) 缩小”：pilot moment 来自 true target，inverse moment map 却来自 empirical \(A_R\)。

### 修正方向

建立一个统一的 `OracleMeasure` 概念，保证下列接口来自同一近似层：

- `A(beta)`
- `mu(beta)`
- `F(beta)`
- `tilt_full(beta)`
- `tilt_cond(beta, panel, x_s)`
- `panel_information(beta, panel)`
- `kl(beta_true, beta_hat)`

Synthetic P4 建议选择 **true-\(Q_0\) 路径**：

1. target/full conditional 继续使用解析 GMM；
2. \(A,\mu,F\) 使用独立的高精度 full-law QMC 或经过收敛验证的超大 integration pool；
3. pilot inverse、update score、information 和 KL 全部调用同一个 oracle object；
4. 不要一部分使用 `reference`，另一部分使用 `reference_large`，又一部分使用 analytic conditional，而没有精度关系的明确证书。

如果暂时仍用 reference pool，应令 pool 精度随最大 \(B\) 达到 negligible，并用 gold integration 验证，而不能只凭 `R=10^6` 宣称足够。

### 必须新增的 diagnostic

在不经过 estimator 的情况下，直接检验：

\[
\Delta_\mu
=
\widehat{\mathbb E}_{X\sim Q_{\beta^\star}}
\left[
\widehat m_{\beta^\star,S}(X_S)
\right]
-
\widehat\mu_{\beta^\star}.
\]

对多个 active panels 和总体 allocation 报告：

- Euclidean norm；
- Fisher-whitened norm；
- 由其诱导的风险偏差近似
  \[
  B_{\max}\,
  \Delta_\mu^\top
  M^{-1}FM^{-1}
  \Delta_\mu.
  \]

### 验收标准

在最大预算 \(B_{\max}=32000\) 下，上述诱导偏差必须远小于 oracle first-order constant；建议先以不超过该常数的 1% 作为 engineering gate。若达不到，不能进入 P4 asymptotic gate。

---

## C-02 [Blocker] 当前 `design_ratio` 计算的是 realized risk ratio

### 位置

`src/rr_gid_cn/s1_gate.py`，`run_replication` 末尾：

```text
design_ratio = kl / [rr_phi / (2B)]
```

`p4_validate.py` 和 `p4_summary.py` 又沿用了这个字段。

### 当前量

代码计算的是

\[
R_{\mathrm{risk}}
=
\frac{B\cdot \mathrm{KL}}
{\frac12\Phi(p^\star)}.
\]

它同时包含：

- allocation error；
- pilot error；
- scoring error；
- \(H\) approximation；
- conditional-score approximation；
- KL numerical error；
- target sampling noise。

### 理论定义

真正的 design ratio 是

\[
R_{\mathrm{design}}
=
\frac{\Phi_{\beta^\star}(\widehat p)}
{\Phi_{\beta^\star}(p^\star)}.
\]

### 修正方向

结果 schema 中至少拆出：

- `phi_oracle`
- `phi_main`
- `phi_total_counts`
- `design_ratio_main`
- `design_ratio_total_counts`
- `B_kl_raw`
- `risk_ratio_raw`

其中：

\[
p_{\text{total},S}
=
\frac{(n_S^0+n_S^1)c(S)}{B}.
\]

有限预算下建议同时报告 main design 和包含 pilot/rounding 的 total design。

### 验收标准

- `Oracle allocation p*(beta*)` 的 `design_ratio_main` 必须为 1（仅允许 FW/MC artifact 的冻结容差）；
- validator 不得再把 `mean_B_kl / oracle_half_phi` 命名为 design ratio；
- 所有历史 `design_ratio` 字段标记为 legacy，不可进入论文表格。

---

## C-03 [Blocker] A-OSQD 的 panel information 公式错误

### 位置

`a_optimal_information`：

1. 先计算 full covariance 的逆；
2. 再取 full precision 的 \((S,S)\) block。

### 正确公式

对

\[
Y\sim N(\mu,\Sigma),
\]

只观察 \(Y_S\) 时，关于 full mean parameter \(\mu\) 的 information 应为

\[
I_S^{A}
=
\tau_S^\top
\Sigma_{SS}^{-1}
\tau_S,
\]

即先取 observed covariance submatrix \(\Sigma_{SS}\)，再求逆并嵌回 full dimension。

一般情况下：

\[
(\Sigma^{-1})_{SS}
\neq
(\Sigma_{SS})^{-1}.
\]

当前实现因此不是 Jang et al. 的 A-OSQD baseline，可能显著改变 baseline allocation。

### 修正方向

每个 panel：

1. 取 `cov_panel = full_cov[S,S]`；
2. 对 `cov_panel` 做稳定的 2×2 Cholesky/inverse；
3. 将其嵌入 16×16 zero matrix；
4. 用该 information 进入 A-optimal design。

### 验收标准

新增 3 维 correlated Gaussian 单元测试，选取一个使

\[
(\Sigma^{-1})_{SS}
\]

与

\[
(\Sigma_{SS})^{-1}
\]

明显不同的 covariance；测试必须与解析公式一致。

---

## C-04 [Blocker] “exact observed score” gate 没有约束实际数值方法

### 位置

- `configs/p4_formal.yaml`
  - `exact_observed_score: true`
  - `conditional_method: rejection`
  - finite logarithmic `lu_schedule`
- `p4_formal_run.py`
- `p4_validate.py`

### 问题

rejection sampler 提供的是 exact conditional **draws**。有限 \(L_U\) 下，

\[
\widehat m_{\beta,S}(x_S)
=
\frac1{L_U}
\sum_{\ell=1}^{L_U}
\phi(X_\ell)
\]

不是 exact conditional expectation。

PDF 已明确区分：

- exact observed score，或
- \(L_{U,B}\to\infty\)；
- 固定/有限 \(L_U\) 时最终 covariance 是 sandwich form。

当前 formal runner 只检查配置中的 boolean，不检查该 boolean 与 `conditional_method`、\(L_U\) 和 benchmark 是否匹配；validator 反而默认 formal method 为 `rejection`。

### 修正方向

把实验模式硬拆成三种，禁止共用一个 “formal exact” 标签：

#### Mode A：`oracle_gold_qmc`

- high-precision conditional integration；
- 多 independent scrambles；
- 有显式 numerical error certificate；
- 只跑少量 reps 和选定 budgets；
- 用于验证 exact-score theory endpoint。

#### Mode B：`validated_fixed_qmc`

- 固定 order；
- 先与 Mode A 做 query-level 和 end-to-end 一致性验证；
- 用于大规模 50-replication sweep。

#### Mode C：`finite_lu_rejection`

- exact conditional draws + finite MC average；
- 报告实际 \(L_U\)；
- 对比 finite-\(L_U\) sandwich constant；
- 不能把 exact-score horizontal line 当唯一有限样本 benchmark。

### 验收标准

formal validator 应根据 `experiment_mode` 强制：

- Mode A/B：存在 QMC error diagnostics；
- Mode C：存在 \(L_U\) 和 sandwich constant；
- `exact_observed_score=true` 不能与未认证的 finite-\(L_U\) 路径同时通过 gate。

---

## C-05 [High] `tilted_conditional_mean_exact` 实际并不 adaptive

### 位置

`synthetic_oracle.py` 的 `tilted_conditional_mean_exact`。

### 问题

docstring 声称从 `start_order` 开始逐步 doubling，直到 successive estimates 收敛；实际实现只评估：

\[
\{\text{max\_order}-1,\ \text{max\_order}\}.
\]

因此：

- `start_order` 实际不影响正常计算；
- 不能 early stop；
- 每次都接近最大计算量；
- “最后两级差很小”不是可靠的 absolute error certificate；
- 当前配置中的 `qmc_start_order` 基本是无效参数。

### 修正方向

二选一：

1. 真正实现从 `start_order` 到 `max_order` 的 nested early stopping；
2. 如果坚持只比较末两级，就将函数重命名为 terminal two-level check，不再称 adaptive。

无论哪一种，都应增加多个 independent scrambled Sobol replicates，用 between-scramble variation 估计误差。单一 nested prefix 的相邻差可能偶然抵消。

### 验收标准

- 测试能证明低难度 query 会在较低 order 提前停止；
- 难 query 会增加 order；
- 输出每个 call/group 的 `final_order`、`max_abs_delta`、`scramble_se`；
- formal rows 汇总这些字段，而不是只在函数内部丢失。

---

## C-06 [High] oracle information、\(p^\star\) 和理论常数缺少精度证书

### 位置

- `exact_panel_information`
- `prepare_s1_oracle`
- `p4_prepare_high_precision.py`
- `policy_designs`
- `run_replication`
- `p4_validate.py`

### 问题

prepared artifact 中的所谓 oracle information 仍由有限：

- outer tilted samples；
- conditional completions；
- PSD floor

得到。

但 artifact 只存最终 matrices/designs，没有完整保存：

- raw information；
- PSD correction 大小；
- FW gap 和 iterations；
- \(\Phi(p^\star)\)；
- MC/QMC convergence across sizes；
- mixture/scale/beta/config/commit hash。

formal shell 只检查 pool size 和 shape。runner 对 `reference_size=200000` 的配置只要求 artifact 至少 50,000。validator 又硬编码一个 oracle constant。

### 修正方向

prepared artifact 必须成为有版本的 immutable object，至少包含：

- `schema_version`
- `code_commit`
- `config_hash`
- mixture seed/alpha/parameter hash
- scale hash
- beta_true/hash
- reference pool/hash
- integration method 和 sample sizes
- raw/symmetrized/projected \(F,I_S\)
- 每个 panel 的 PSD correction norm
- \(p^\star\)
- FW gap、iterations、objective
- \(\Phi(p^\star)\)、\(\frac12\Phi(p^\star)\)
- independent convergence replicate summary

### Gold constant 的线性代数要求

对理论常数不要默认使用 `pinv`。Gold route 应：

1. 检查 \(M(p^\star)\) 的最小 eigenvalue/condition number；
2. 使用 Cholesky solve；
3. 若不正定，明确失败，而不是用 pseudoinverse 静默定义一个不同问题。

### 验收标准

- validator 从 artifact 读取唯一 oracle constant，不接受命令行硬编码默认值；
- 增大 oracle integration size 后 \(\Phi(p^\star)\) 和 top panel ranking 稳定；
- 两个 independent oracle builds 的 constant 差异小于预设 numerical tolerance；
- artifact 与当前 config/hash 不一致时立即拒绝。

---

## C-07 [High] \(H\) 的“oracle”路径、PSD floor 和 pseudoinverse 会掩盖问题

### 位置

`final_rr_estimator` 和 `use_oracle_H`。

### 问题

`use_oracle_H=true` 实际使用的是：

\[
I_S(\beta^\star)
\]

在所有 scoring steps 冻结的 prepared matrices，不是当前 iterate \(\beta^{(j)}\) 下的 exact information。因此 “oracle H 反而更差”不能用于否定 current-\(\beta\) information theory。

此外：

- 每个 panel matrix 已经做 eigenvalue floor；
- \(H\) 再用 `pinv`；
- validator 只检查 `lambda_min_H > 0`。

由于 floor 本身会制造正 eigenvalue，这个 gate 很弱，且无法判断 \(H\) 是否接近理论矩阵。

### 修正方向

明确区分：

- `frozen_beta_star_information`
- `gold_current_beta_information`
- `estimated_current_beta_information`

Gold-standard scoring 必须使用当前 \(\beta^{(j)}\) 下的 high-precision \(I_S\)。

每步记录：

- raw \(H/B\) eigenvalues；
- projected \(H/B\) eigenvalues；
- condition number；
- effective rank；
- PSD correction norm；
- 相对 gold \(H\) 的 whitened operator error；
- Newton decrement；
- pre-projection step 和各约束的 active status。

### 验收标准

- gold route 不使用 `pinv`；
- practical route 若使用 ridge/pinv，必须显式标记为 regularized estimator；
- 不允许仅凭 PSD 后的正 `lambda_min` 通过 formal gate。

---

## C-08 [High] pilot schedule 配置没有进入 `run_replication`

### 位置

- YAML 有 `pilot_multiplier`
- `run_replication` 内部写死 `ceil(10 * B^(1/3))`
- validator 也写死相同公式

### 影响

- 修改 YAML 不会改变实际 pilot；
- 无法干净比较 \(B^{1/3}\)、\(B^{0.4}\)、\(B^{1/2}\)；
- result row 不足以重建 pilot schedule；
- 容易把 diagnostic 与 formal 配置混淆。

### 修正方向

定义结构化 schedule：

- kind
- exponent
- multiplier，或 anchor budget/anchor pilot
- max fraction
- min per support
- rounding rule

由 runner 计算一次并传给 `run_replication`。validator 从 row/config 验证，而不是重新写一份公式。

### 验收标准

- 修改 config 后 pilot count 确实变化；
- 每条 row 保存 schedule 与实际 per-support counts；
- old hard-coded formula 仅作为一个 named ablation。

---

## C-09 [High] paired target manifest 未被使用，且 seed 含义不一致

### 位置

- `p4_formal_manifest.py`
- `p4_formal_run.py`
- `run_replication`

### 当前行为

manifest 写入：

\[
202600000+B\cdot1000+\text{rep}.
\]

runner 把它当 replication base seed，但实际 target draw 使用：

\[
\text{seed}+3.
\]

同时 config 虽然指向 manifest，runner 没有读取该文件。

### 修正方向

manifest 每条应明确保存：

- `replication_seed`
- `target_draw_seed`
- `pilot_or_design_seed`
- `score_seed_root`
- `information_seed_root`

runner 必须从 manifest 读取并逐项核对，不能自行重建另一套公式。

### 验收标准

- row 中记录的 target seed 与 manifest 完全一致；
- missing/duplicate seed 直接失败；
- 选取固定 5 个 manifest entries 重跑，target full draw hash 一致。

---

## C-10 [High] `p4_validate.py` 的 formal gate 不完整

### 当前遗漏

1. 如果某个预算整个缺失，`missing` 只遍历已出现预算，可能不报错；
2. 不核对完整 expected Cartesian product：
   \[
   \text{budget}\times\text{replication}\times\text{policy};
   \]
3. 不验证 config/artifact/code hashes；
4. 不验证 FW gap；
5. 不验证 true design ratio；
6. 不验证 QMC/rejection approximation certificate；
7. 不验证 raw \(H\) condition/operator error；
8. 对 KL 只要求 clipped `kl >= 0`，没有对 `kl_raw` 的负值做容差 gate；
9. 硬编码 oracle constant；
10. 只报告各 policy 独立 CI，没有利用 paired draws 报告 paired differences；
11. J ablation 的完整性与 replication pairing 没有系统检查。

### 修正方向

validator 应读取：

- frozen config；
- expected budgets/policies/reps；
- prepared artifact manifest；
- target seed manifest。

然后生成 expected key set 并与 observed key set 做精确差集/重复集检查。

### 验收标准

任何以下情况都必须非零退出：

- 缺一个 budget；
- 缺一个 policy；
- 重复 row；
- artifact/config/hash 不一致；
- formal mode 与 numerical method 不一致；
- raw KL 负值超过 integration tolerance；
- oracle main-design ratio 不为 1；
- FW gap 超过 tolerance；
- gold H 不正定。

---

## C-11 [Medium/High] summary 使用 clipped KL，可能产生上偏

### 位置

`p4_summary.py` 使用 `B_kl`，而不是 `B_kl_raw`。

### 问题

如果 empirical integration 产生小的 negative raw Bregman estimate，逐 run 做：

\[
\max(0,\widehat{\mathrm{KL}})
\]

再取均值会产生上偏，并隐藏 evaluation inconsistency。

### 修正方向

- formal summary 只使用 `B_kl_raw`；
- 非负版本只能作为 visualization compatibility 字段；
- 若 raw KL 负值超出 gold integration error bar，应把 run 标记为 evaluation failure，而不是截断。

### 验收标准

summary 同时报告：

- raw mean；
- numerical SE；
- negative count；
- minimum raw KL；
- integration tolerance。

---

## C-12 [Medium] cost-aware 与 feature-support 代码目前只对 P4 等成本/固定 feature 有效

### 位置

- `round_cost_share`
- `run_replication` 的 main counts
- `balanced_pilot_counts`
- `pilot_ht_moment`

### 问题 A：rounding

`round_cost_share` 的 while condition 只看 remainder 排名第一的 panel cost。如果第一名当前不可负担，但其他更便宜 panel 可负担，循环会提前结束。

`run_replication` 又直接按 `remaining_budget * probabilities` 做 unit counts，完全假设 cost=1。

P4 所有 pair panel 等成本，因此当前 P4 不受影响；但这不是真正一般的 cost-aware implementation。

### 问题 B：feature supports

`balanced_pilot_counts` 和 `pilot_ht_moment` 写死：

- 12 维 feature；
- 六个 `(i,i+6)` supports；
- 内部直接调用固定 `feature_map`。

这会阻塞后续 feature dictionary/reuse 或 Gas `r=16` 的通用接口。

### 修正方向

- rounding 使用统一的 cost-share apportionment；
- main campaign 也调用同一个 rounding API；
- feature supports、feature evaluator、dimension 都由 experiment specification 传入；
- HT estimator 不应知道 synthetic feature 的具体索引模式。

### 验收标准

- 不等成本 toy case 能用满所有可用预算；
- custom feature_fn/support manifest 的 HT moment 与手算一致；
- P4 当前结果在通用化后保持逐 seed 一致。

---

# 3. 运行速度审查

## 3.1 先明确：Frank–Wolfe 不是当前主瓶颈

P4 oracle design 在 prepared artifact 中预先求一次。每次 FW 只处理 120 个 12×12 information matrices。相比 observation-level conditional integrations，它不是主要耗时来源。

因此，在完成 profiler 前，不要优先替换 FW solver。即使放宽 tolerance，也不会解决当前数量级的运行时间。

---

## P-01 [最高优先级] 给每个 stage 加精确计时和工作量计数

在优化前，每条 replication 必须记录：

- `time_target_sampling`
- `time_pilot_build`
- `time_pilot_solve`
- `time_design_information`
- `time_fw`
- 每步：
  - `time_mu`
  - `time_H`
  - `time_score`
  - `time_linear_solve`
- `time_kl`
- `time_total`

还应记录：

- active panel 数；
- requested conditional completions；
- proposed/accepted rejection samples；
- acceptance rate 的 min/median/max；
- QMC nodes、final order、scramble count；
- reference feature scans 次数；
- CPU/GPU device；
- peak CPU/GPU memory。

GPU 计时必须在阶段边界 synchronize，否则 wall time 不准确。

### 验收标准

对单个 \(B\in\{2000,8000,32000\}\) replication，至少 95% wall time 能被以上 stage 归因。

---

## P-02 [高收益] shared pilot 和固定常数移出 policy loop

### 当前重复

三种 P4 policies 使用同一：

- target pilot rows；
- pilot panel counts；
- pilot observations；
- HT moment；
- pilot \(\widetilde\beta\)。

但上述工作位于 policy loop 内，重复三次。

此外每个 policy 又重复计算固定的：

- `pilot_mu_true`
- `A(beta_true)`
- 部分 oracle constant。

### 修正方向

`run_replication` 结构改为：

1. 生成一次 full target draw；
2. 构造一次 shared pilot；
3. 求一次 \(\widetilde\beta\)；
4. 对每个 policy 仅构造 main allocation/data；
5. 从同一个 pilot state 进入 final estimator。

第一步 scoring 时三种 policy 的起点相同。若需要 current-\(\beta\) panel information，可对 union of active panels 计算一次，再由各 policy counts 组装不同 \(H\)。

### 预期收益

这不会改变统计对象，属于无争议的去重。尤其 pilot solve 和第一步 \(H\) 可以明显减少重复成本。

---

## P-03 [高收益] 缓存 reference feature matrices，分离 update pool 与 evaluation pool

### 当前重复

`feature_map(reference)` 在以下路径反复计算：

- 每次 pilot Newton iteration 的 `tilted_moments`
- 每步 `panel_information_cross`
- 每步 `tilted_moments`
- KL 的两次 `log_partition`
- pilot diagnostics

`ref_large` 达到 1,000,000×16。每次重新计算 tanh 和 interactions 都会产生明显 CPU memory/compute 成本。

### 修正方向

prepared artifact 或 runtime cache 保存：

- `phi_reference`
- `phi_reference_large`
- `A_beta_true`
- `mu_beta_true`
- `F_beta_true`
- 必要时 `logits_beta_true`

增加基于 feature matrix 的 API：

- `log_partition_from_features`
- `tilted_mean_from_features`
- `tilted_mean_and_fisher_from_features`

同时分离：

- `update_integration_pool`
- `final_kl_integration_pool`

先用 convergence study 确定 update pool 所需规模；不应默认每个 update 都扫描 1,000,000 rows。

### 注意

缓存只是性能优化，不能替代 C-01 的同一-\(Q_0\) consistency 修复。先定义 gold oracle，再决定缓存哪一种 integration object。

---

## P-04 [最高优先级] rejection CUDA 路径过量生成 proposals

### 当前机制

GPU path 设置：

\[
\text{batch\_floor}
=
\max(2048,16n).
\]

例如 \(n=32\) 的 H completion，每个 active row 第一轮至少生成 2048 proposals，即 requested count 的 64 倍。即使 acceptance 约 0.2，只需要约 160 proposals/row 才够，当前仍可能有数量级浪费。

在 score 路径 \(n=L_U\approx 351\!-\!479\) 时，floor 为约 5616–7664 proposals/row，也可能远高于实际所需。

此外当前 GPU path：

- 为所有 proposals 计算四个 mixture component 的 conditional transforms，再用 mask 选择；
- 即使只返回 feature mean，也先保存完整 accepted \(X\) tensor；
- 最后再次计算 feature map；
- posterior weights 仍有 CPU 计算和 host→device transfer。

### 修正方向

1. 用 acceptance estimate 选择 proposal count：
   \[
   n_{\mathrm{prop}}
   \approx
   \frac{n_{\mathrm{remaining}}}{\widehat a}
   (1+\text{safety margin}).
   \]
2. 初始 acceptance 可由 panel/beta 的历史 cache 给出；
3. 对低 acceptance rows 单独分桶；
4. 按 sampled component 分组计算 conditional transform，不要为每个 proposal计算全部 component；
5. `return_feature_mean=True` 时在线累加 accepted \(\phi(X)\) 的 sum/count，不保存所有 accepted full samples；
6. posterior 和 conditional transform 尽量全留在 GPU；
7. batch size 以 accepted completions/second 和 replications/hour 选择，而不是以 GPU utilization 选择。

### 验收标准

对固定 seed：

- 输出分布/均值与旧 exact rejection 在 MC SE 内一致；
- proposals-per-accepted 大幅下降；
- acceptance 记录完整；
- 不改变 rejection target law。

---

## P-05 [高收益] 大规模 formal 不要用未经分层的逐 observation high-precision oracle

按当前 finite-\(L_U\) 路径，单个 \(B=32000\)、三 policy、两步 scoring，仅 observed-score 部分请求的 accepted completions 约为：

\[
3\times2\times32000\times479
\approx9.2\times10^7.
\]

这还不包括 rejection proposals 和 \(H\) estimation。

### 推荐计算策略

- 少量 gold runs：multi-scramble high-precision QMC；
- 大规模 runs：经过 gold runs 验证的 fixed-order QMC；
- practical ablation：finite-\(L_U\) rejection + sandwich theory。

不建议让所有 200×5×3 runs 都走 adaptive-to-max-order 或高 \(L_U\) exact-draw average。

---

## P-06 [中高收益] observation 数据结构改为 panel-grouped arrays

### 当前行为

先创建长度 \(B\) 的 Python tuple list，每个 scoring step 再遍历并 `grouped.setdefault`。

### 修正方向

采集阶段直接保存：

- `observed_by_panel[panel] -> ndarray`
- `count_by_panel`
- `active_panels`

pilot/main 合并时按 panel concatenate。每步 estimator 直接读取，不再重建 Python objects。

### 收益

- 减少 Python allocation；
- 减少重复 grouping；
- 更容易批量送入 GPU；
- 更方便共享第一步 information。

---

## P-07 [中高收益] pilot solver 使用预计算 features 和明确收敛条件

### 当前行为

`solve_pilot_beta` 虽然先计算一次 features 用于 objective，但每次迭代调用 `tilted_moments` 又重新计算 feature map。

此外 `steps` 默认 20，却实际运行 `max(steps,100)`，参数语义不一致。

### 修正方向

- 全部 moment/gradient/Hessian 基于预计算 `phi_reference`；
- `max_iter` 与 `min_iter` 分开；
- 返回 objective、gradient norm、Newton decrement、iterations、line-search status；
- shared pilot 只求一次；
- 对 gold route 避免 silent pseudoinverse。

---

## P-08 [中高收益] 重新 benchmark 并行布局

formal shell 当前每张 A800 启两个 Python 进程，每进程又给 OMP/MKL/OpenBLAS 4 threads。

项目交接本身已观察到：16 workers 时 GPU utilization 高，但 throughput 没有相应提高。

### 必做 benchmark

相同固定 replications，比较：

1. 1 process/GPU，BLAS threads=1；
2. 2 processes/GPU，BLAS threads=1；
3. 当前 2 processes/GPU，BLAS threads=4。

指标只看：

- completed replications/hour；
- accepted completions/second；
- CPU memory bandwidth；
- GPU memory；
- failure/retry rate。

不要用 `nvidia-smi` utilization 作为唯一依据。

### Artifact 加载

pickle 会让每进程独立反序列化大数组。建议把大 reference/feature arrays 改为 read-only `.npy` memmap 或其他可共享 page-cache 的格式。

---

## P-09 [低优先级] FW 可改进，但应最后做

如果 plug-in RR-GID 后每 replication 都要求一次 FW，可考虑：

- warm start；
- 减少固定 40 次 golden-section evaluations；
- 缓存 matrix products；
- 使用 certificate-based early stop。

但在 120×12 规模下，这仍远小于 conditional information 计算。只有 profiler 显示 FW 占比显著时才做。

---

# 4. 推荐的修正实施顺序

## Phase 0：冻结、停止混跑、加入 profiler

### 指令

1. 从当前 `main` 新建 repair branch；
2. 记录当前 commit、config、prepared artifact SHA256；
3. 保留已有 formal/diagnostic 输出，禁止覆盖；
4. 新增 stage timers 和 proposal/QMC counters；
5. 跑一个固定 seed：
   - \(B=2000,8000,32000\)
   - 只跑 oracle allocation policy
   - \(J=2\)
6. 导出 stage-level runtime breakdown。

### 进入下一阶段条件

- 结果可重现；
- 95% wall time 可归因；
- 不再用 “formal” 命名未经校验的 output。

---

## Phase 1：先修明确的 correctness bugs

按以下顺序：

1. 修正 A-OSQD information；
2. 拆分 design ratio 与 risk ratio；
3. 修正 summary/validator 对 raw KL 的处理；
4. 参数化 pilot schedule；
5. 统一 seed manifest；
6. 增加 artifact/config/code hashes；
7. 修复 expected-grid validation；
8. 给 oracle constant 保存 FW certificate；
9. 重命名 `use_oracle_H`；
10. 增加 analytic unit tests。

### 进入下一阶段条件

- 新增单元测试全部通过；
- oracle main-design ratio 为 1；
- validator 能故意捕获缺 budget、错 artifact、错 method 和 duplicate row。

---

## Phase 2：建立同一 \(Q_0\) 下的 gold oracle

### 指令

1. 实现统一 oracle object；
2. 对 \(A,\mu,F\) 建 full-law high-precision integration；
3. conditional score 使用 multi-scramble QMC；
4. 在 \(\beta^\star\) 做 score-centering test；
5. 当前-\(\beta\) gold information 与 estimated information 对比；
6. 冻结唯一 oracle artifact 和唯一 theory constant。

### 进入下一阶段条件

- score-centering 诱导的最大预算风险偏差 < oracle constant 的 1%；
- independent oracle builds 的 constant 稳定；
- raw KL 非负到 numerical tolerance；
- gold \(M(p^\star)\) 正定且不依赖 pseudoinverse。

---

## Phase 3：按 ladder 隔离理论与近似

不要直接恢复三 policy×五预算×50 reps。按下面的梯子逐层加入复杂性。

### G0：oracle-start sanity check

\[
\beta^{(0)}=\beta^\star,\quad
p=p^\star,\quad
H=H_{\mathrm{gold}}(\beta^{(0)}),
\]

使用 gold conditional score。

**目的：**验证 score sign、normalization、KL、\(H\) 和 theory constant。

### G1：actual pilot + oracle allocation + gold current-\(\beta\) H/score

做 \(J=0,1,2,3,4\)。

**目的：**判断 pilot 是否进入 Newton basin，以及 error-squaring 是否出现。

### G2：actual pilot + oracle allocation + estimated H + gold score

**目的：**隔离 information approximation。

### G3：actual pilot + oracle allocation + estimated H + finite score

分别比较 fixed QMC 和 finite-\(L_U\) rejection。

**目的：**隔离 score approximation。

### G4：plug-in RR-GID design

加入 \(\widehat p(\widetilde\beta)\)，报告真正 design ratio。

**目的：**验证一次设计达到 oracle first-order efficiency 的完整链条。

### 结果判读

- G0 失败：实现/评估错误，不是 pilot 问题；
- G0 通过、G1 失败：pilot/Newton basin guidance；
- G1 通过、G2 失败：\(H\) approximation；
- G2 通过、G3 失败：conditional-score approximation；
- G3 通过、G4 design ratio 失败：plug-in information/design；
- G4 通过后，才恢复 full formal sweep。

---

## Phase 4：性能重构

按优先级：

1. shared pilot；
2. cached reference features；
3. first-step shared information；
4. panel-grouped observations；
5. rejection adaptive proposals + online feature sums；
6. fixed-QMC large-scale mode；
7. 1 process/GPU benchmark；
8. memmap/shared artifacts。

每个改动必须：

- 固定 seed 对比旧输出；
- 分别记录 correctness difference 和 wall-time；
- 一次只改一个主要机制；
- 不用性能改动掩盖 statistical change。

---

## Phase 5：再研究 pilot schedule

在 G0–G4 全部清楚后，再比较：

\[
\gamma\in\left\{
\frac13,\;0.4,\;\frac12
\right\}.
\]

推荐使用相同 anchor 而不是相同 multiplier，例如固定 \(B=8000\) 时 pilot=400，再改变 exponent，以区分 exponent effect 与单纯样本量 effect。

主 numerical 不必限制死 \(10B^{1/3}\)，但 \(B^{1/3}\) 应保留为理论对应的 aggressive-pilot ablation。

---

# 5. 必须新增的测试矩阵

## T-01：A-OSQD analytic test

- 3 维 correlated Gaussian；
- panel 为两个坐标；
- 验证嵌入的是 \(\Sigma_{SS}^{-1}\)，不是 \((\Sigma^{-1})_{SS}\)。

## T-02：true design ratio test

- 输入 oracle \(p^\star\)；
- `design_ratio_main == 1`；
- 改动 allocation 后 ratio 按 \(\Phi\) 改变；
- risk ratio 不参与该断言。

## T-03：exact-score mode guard

- formal exact mode + finite rejection mean 必须被拒绝，除非 mode 明确为 finite-\(L_U\) 并使用 sandwich benchmark。

## T-04：QMC adaptive/replicate test

- 简单 query 低 order 停止；
- 难 query 提高 order；
- independent scrambles 的 SE 被记录；
- terminal delta 不通过时明确失败。

## T-05：single-\(Q_0\) score-centering test

在 \(\beta^\star\)：

\[
\widehat{\mathbb E}[m_{\beta^\star,S}(X_S)-\mu_{\beta^\star}]
\]

必须在独立 numerical + sampling error 内为零。

## T-06：artifact mismatch test

人为改变：

- mixture seed；
- alpha；
- reference size；
- beta hash；
- information method；
- code commit；

formal runner 必须拒绝旧 artifact。

## T-07：manifest test

- runner 必须实际读取 manifest；
- target seed 与 row 完全一致；
- duplicate/missing entry 失败。

## T-08：expected-grid validator test

分别删掉：

- 一个预算；
- 一个 replication；
- 一个 policy；
- 加一个 duplicate；

validator 都必须失败。

## T-09：closed-form small model end-to-end test

构造一个低维 Gaussian reference + linear feature 的 toy model，使：

- conditional score；
- \(I_S\)；
- \(F\)；
- \(\Phi(p)\)；
- partial likelihood/MLE

都有解析结果。用它验证整个 P4 chain，而不是只检查 shape/PSD/nonnegative clipped KL。

## T-10：cost-aware rounding test

设置 top remainder panel 当前不可负担、另一个便宜 panel 可负担，确认算法不会提前停止。

---

# 6. 推荐的新结果 schema

每条 replication row 至少保存：

## Reproducibility

- `schema_version`
- `code_commit`
- `config_hash`
- `artifact_id`
- `artifact_sha256`
- `manifest_id`
- `replication`
- `replication_seed`
- `target_draw_seed`
- `device`
- `dtype`

## Experiment definition

- `experiment_mode`
- `policy`
- `budget`
- `pilot_schedule`
- `pilot_budget`
- `pilot_counts`
- `main_counts`
- `total_counts`
- `conditional_method`
- `lu`
- `qmc_order`
- `qmc_scrambles`
- `h_tilted`
- `h_cond`
- `scoring_steps`

## Design metrics

- `fw_gap`
- `fw_iterations`
- `phi_oracle`
- `phi_main`
- `phi_total_counts`
- `design_ratio_main`
- `design_ratio_total_counts`

## Estimation metrics

- `beta_pilot`
- `beta_hat`
- `pilot_fisher_error`
- `kl_raw`
- `B_kl_raw`
- `risk_ratio_raw`
- `finite_allocation_constant`
- `sandwich_constant`（finite-\(L_U\) mode）

## Per-step diagnostics

- `beta_before/after`
- `score_norm`
- `newton_decrement`
- `H_raw_lambda_min/max`
- `H_projected_lambda_min/max`
- `H_condition_number`
- `H_psd_correction_norm`
- `H_operator_error_to_gold`（diagnostic modes）
- `raw_step_norm`
- `applied_step_norm`
- `active_constraints`

## Runtime

- stage timers；
- active panels；
- proposals/accepted；
- acceptance summary；
- QMC node/order/error summary；
- peak memory。

---

# 7. 最小可执行的下一轮实验

在修完 Phase 1–2 后，先跑以下集合，不要直接跑 full sweep：

## Set A：correctness

- policy：仅 oracle allocation
- \(B\in\{2000,8000\}\)
- reps：20
- \(J\in\{0,1,2,3\}\)
- gold current-\(\beta\) H
- gold conditional score

## Set B：approximation isolation

固定 \(B=8000\)、20 paired reps：

1. gold H + gold score
2. estimated H + gold score
3. estimated H + fixed QMC
4. estimated H + finite-\(L_U\) rejection

## Set C：pilot schedule

只有 Set A/B 通过后：

- \(B\in\{2000,8000,32000\}\)
- anchored \(\gamma\in\{1/3,0.4,1/2\}\)
- 先 50 reps
- 报 Fisher-norm pilot error、Newton decrement 和最终 risk

## 恢复 full formal 的条件

同时满足：

1. G0 oracle-start 通过；
2. single-\(Q_0\) score-centering 通过；
3. oracle constant 冻结一致；
4. true design ratio 实现正确；
5. A-OSQD baseline 修正；
6. fixed-QMC 与 gold endpoint 无可检测系统差异；
7. validator 的完整性测试通过；
8. single-rep profiler 显示正式预算可承受。

---

# 8. 对理论与论文表述的对应修订

代码修复之外，论文/规格建议同步做两处澄清：

## 8.1 区分 theorem 和 finite numerical prescription

保留一般：

\[
b_B=B^\gamma,\qquad
2^J\gamma>1.
\]

不要把 \(10B^{1/3}\) 写成 universal finite-sample recommendation。numerical 主配置可以用更稳健 schedule，并把原配置作为 aggressive low-pilot ablation。

## 8.2 明确 approximate \(H\) 与 score 的速率条件

固定 \(J\) 的递推应显式包含：

\[
\|e_{j+1}\|
\lesssim
\|e_j\|^2
+
a_B\|e_j\|
+
B^{-1/2}
+
r_{U,B},
\]

其中：

- \(a_B\)：whitened information operator error；
- \(r_{U,B}\)：normalized score approximation error。

design consistency 所需的 \(o_p(1)\) 与固定-\(J\) exact rate 所需的定量速率应分开陈述。

---

# 9. 最终优先级

## 必须先做

1. 单一 \(Q_0\) consistency；
2. design ratio / risk ratio 拆分；
3. A-OSQD 公式；
4. exact-score mode 与 benchmark 对齐；
5. oracle artifact/constant 冻结；
6. validator 和 seed manifest。

## 随后做

7. gold ladder G0–G4；
8. shared pilot/reference feature cache；
9. rejection proposal redesign；
10. fixed-QMC large-scale 路径；
11. parallel layout benchmark。

## 最后做

12. pilot exponent/multiplier sweep；
13. FW 微优化；
14. full five-budget × 50-replication formal sweep。

---

# 10. 一句话结论

当前最重要的不是立刻增大 replication 或更换 Frank–Wolfe，而是先把 P4 改造成一个**同一 \(Q_0\)、指标定义正确、oracle constant 唯一、近似层逐级可隔离**的验证链；完成后再做性能重构。否则运行得越久，只会更精确地估计一个混合了实现偏差和指标错误的 numerical object。
