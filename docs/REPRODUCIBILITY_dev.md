# RR-GID_CN 复现说明（P12 冻结版）

本文档记录 RR-GID_CN 项目全部正式实验的复现配置、结果清单与版本冻结信息。
所有结果由服务器（AutoDL RTX 4080 SUPER 33.8GB / 128 核 / 503GB RAM）上的
正式 replication 生成，代码与数据已同步至 GitHub。

## 1. 环境与依赖

- Python 3.12.3（miniconda venv `/root/rrgid_env`）
- PyTorch 2.13.0+cu130（CUDA 可用）
- numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, pyyaml, pytest 9.1.1
- 网络：外网需走 mihomo 代理 `http://127.0.0.1:7890`
- 依赖声明：`environment.yml` + `pyproject.toml`

## 2. 冻结版本

- Git commit（结果冻结）：`f36c802`（P10 R2 结果提交）
- 代码 HEAD：`254e5fa`（P10 修复）
- 所有正式结果已推送 `https://github.com/15628925702/RR_GID_CN.git`

## 3. 正式实验结果清单

### P4 Synthetic S1 oracle gate（三策略 × 5 budgets × 200 reps）
- 文件：`results/p4_formal_{B}.jsonl`（B ∈ {2000,4000,8000,16000,32000}）
- 策略：Uniform SQD / A-OSQD / oracle RR-GID
- J ablation：`results/p4_exact_8000_J{0,1,2}.jsonl`
- seed 方案：`202600000 + budget*1000 + replication`
- 理论常数：oracle 31.9 / Uniform 64.8 / A-OSQD 95.8

### P5 Synthetic S1 最终四策略（Disc + learned RR-GID × 5 budgets × 200 reps）
- 文件：`results/p5_four_{B}.jsonl`
- 策略：Discriminative Score OED / learned RR-GID（与 P4 的 Uniform/A-OSQD 配对）
- 四策略合并构成最终 Fig.1 数据
- 主要发现：learned RR-GID 与 Discriminative 远优于线性基线，learned 略优

### P7 Synthetic S2（alpha sweep + generator reuse）
- 文件：`results/p7_s2_summary.json`
- alpha ∈ {0, 0.5, 1.0, 1.5}（B=8000）
- reuse T ∈ {1, 5, 20, 50}
- 主要发现：A-OSQD 随 alpha 增大偏离（2.02→5.60）；RR-GID 跨 campaign 复用时
  regret 更低（1.69-1.83 vs 2.16-2.21）且 compute 便宜 15-35 倍

### P9 Gas R1（empirical-base budget curves × 4 budgets × 200 reps）
- 文件：`results/p9_r1_{B}.jsonl`（B ∈ {400,800,1600,3200}）
- 策略：四策略，each 200 reps，b_B = min(0.2B, ⌈10B^(1/3)⌉)，J=2
- 主要发现：learned RR-GID B_kl 与 Discriminative 接近且远低于基线

### P10 Gas R2（natural drift × 3 campaigns × 4 budgets × 50 reps）
- 文件：`results/p10_r2_{campaign}_{B}.jsonl`
- campaign：batch7 / batches89 / batch10
- 指标：projection_loss、heldout_mean_rmse、c2st_auc、ess、lambda_min_M
- 主要发现：RR-GID 在 batch7/batches89 上 ESS 最优（0.84-0.97），A-OSQD 最差
  （0.31-0.77），符合"生成式条件结构对漂移更稳健"的预期

## 4. 关键数值修复（与 PDF 一致）

1. **P9/P10 accept-reject collapse**：pilot beta / beta_dag 加 norm cap（2.0），
   防止 tilt 发散导致生成器采样接受率崩溃。
2. **P9/P10 final estimator 步长**：Fisher-scoring step 加 norm cap（2.0），
   防止一步发散到 Theta 边界。
3. **A-OSQD 在 Gas 上**：改用 16 维 φ 空间的完整参考协方差（而非 128 维 raw），
   避免 128×128 矩阵求解过慢/不收敛。
4. **C2ST**：per-fold 标准化（训练集 mean/std，无泄漏）。
5. **Discriminative MLP fit**：GPU float64 路径（n≥500 触发），数值与 numpy 一致。

## 5. 复现步骤

```bash
# 服务器（走代理）
export http_proxy=http://127.0.0.1:7890; export https_proxy=http://127.0.0.1:7890
cd /root/RR_GID_CN
# P4（已冻结，重跑可选）
python scripts/p4_formal_run.py
# P5（四策略补跑）
python scripts/p5_formal_run.py
# P7
python scripts/p7_formal.py
# P9
python scripts/p9_r1_formal.py
# P10
python scripts/p10_r2_formal.py --campaign batch7
python scripts/p10_r2_formal.py --campaign batches89
python scripts/p10_r2_formal.py --campaign batch10
```

## 6. 待确认 / limitations

- P5 的 learned RR-GID B_kl 略低于 P4 oracle（有限样本下 pilot beta 设计的
  自然差异，非错误）。
- P9 四策略 B_kl 差异较小（高维 Gas 下 allocation 对最终 KL 贡献微妙），
  R1 的 panel ranking 验证见 `results/p9_r1_summary.json`（Spearman 0.53）。
- P10 的 c2st_auc ≈ 1.0（自然漂移强烈，VAEAC reference 与 target 完全可分），
  符合 PDF 8.3"不把 drift 当 exact verification"的设定。
