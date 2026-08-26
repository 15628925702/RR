# RR-GID_CN 项目交接文档

更新时间：2026-08-24

本文件用于科研执行和复现实验交接。研究设计的唯一依据是仓库根目录的 `RR_GID_CN.pdf`；本文件、旧 acceptance 文件、图表和已有结果都不能替代 PDF，也不能修改 PDF 中的研究问题、模型、算法、数据集、policy、预算、指标或理论目标。

## 1. 当前总状态

- 当前 Git 分支：`main`
- 当前代码提交：`fe5b84a` (`Optimize exact conditional Gaussian sampling`)
- 远程 `origin/main`：与当前提交一致
- 本地已有用户修改：`data/gas/processed/gas_data_card.json`，必须保留，不能回滚或覆盖
- 服务器未纳入 Git 的诊断文件：
  - `experiments/p4_prepared_medinfo.pkl`
  - `experiments/p4_prepared_oracle_fix1_debug.pkl`
- 服务器上尚未纳入 Git 的中间 summary：
  - `results/p4_formal_fix4_summary.json`
  - `results/p4_formal_fix5_summary_4000.json`
- 旧结果必须保留；正式修复结果使用独立 `p4_formal_fix4` / `p4_formal_fix5` 前缀，不能静默覆盖 `p4_formal_*` 或 `p4_exact_*` 旧文件。

当前结论：P0-P3 工程和基础验收已完成；P4 正式实验仍在进行，P4 理论/统计门禁尚未整体通过。因此不得进入 P5，更不得进入 P11/P12。

## 2. 服务器环境

执行位置：`/root/RR_GID_CN/`

连接方式：

```text
ssh -p 40882 root@connect.weste.seetacloud.com
```

已知环境：

- Python：3.12.3
- PyTorch：2.13.0+cu130
- CUDA：13.0
- 完整测试最近一次：`32 passed`
- 正式运行必须设置：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
- 推荐解释器：`/root/rrgid_env/bin/python3`

进入服务器后统一使用：

```bash
cd /root/RR_GID_CN
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## 3. PDF 冻结的关键规格

### Synthetic S1/P4

- nonlinear four-component latent GMM + sinh warp，`d=16`
- `alpha=1`
- `r=12` bounded feature map
- 120 个 coordinate-pair panels
- `beta*` 使用冻结方向和约 `ESS/N=0.5` 的尺度
- policies：Uniform SQD、A-OSQD、oracle RR-GID
- budgets：`{2000, 4000, 8000, 16000, 32000}`
- 每档至少 200 个独立 replication
- `b_B=ceil(10 B^(1/3))`
- `J=2`
- 同一 replication 的三种 policy 必须共享 target draws
- 主 S1 final observed score 使用 exact conditional oracle，以隔离 allocation 贡献
- primary metric：`B * KL(Q_beta* || Q_beta_hat)`
- 同时报 design ratio、MC SE、95% CI、J ablation 和 oracle horizontal line
- `J` ablation：固定 `B=8000`，`J in {0,1,2}`

### 其他阶段冻结规格

- P5：加入 mask-conditioned MLP 的 Discriminative Score OED 和冻结 VAEAC 的 learned RR-GID，四策略使用相同 final RR estimator
- P6：Synthetic VAEAC，必须有 canonical checkpoint、arbitrary-mask conditional interface、重建和 conditional quality 验收
- P7：`alpha={0,0.5,1.0,1.5}`、`B=8000`；reuse `T={1,5,20,50}`；每个 campaign 重抽 feature dictionary、beta 和 panel 子集；RR-GID 不重训 generator，Discriminative 每 campaign 重训
- P8：真实 UCI Gas Sensor Array Drift，13,910 条、128 raw features、16 sensors；reference batches 1-6，targets batch 7、batches 8-9、batch 10
- P9：四档 `{400,800,1600,3200}`，四 policy，每档 200 reps；独立冻结 Gas VAEAC；`b_B=min(0.2B,ceil(10B^(1/3)))`，`J=2`
- P10：3 campaigns × 4 budgets × 4 policies × 50 paired reps = 2400 rows；campaign pool 只用于 acquisition，full-test 只用于 evaluation；Eq.17 Bregman projection、32 held-out functions、5-fold C2ST、实测 acceptance/ESS/FW gap/PSD

## 4. 阶段状态表

| 阶段 | 状态 | 证据与备注 | 是否可进入下一阶段 |
|---|---|---|---|
| P0 | 已完成 | Git、目录、配置入口和环境结构已建立；Python 约束已统一到 3.12 | 是 |
| P1 | 已完成 | Synthetic GMM+sinh warp、exact conditional sampler、16 维/4 成分/seed 2026、120 panels、12 维 feature map | 是 |
| P2 | 已完成 | Uniform SQD、A-OSQD、oracle RR-GID、budget rounding、Frank-Wolfe | 是 |
| P3 | 已完成 | balanced pilot、HT moment、cross-completion、PSD projection、Fisher update；单元测试覆盖 | 是 |
| P4 | 进行中，未通过整体门禁 | B=2000、B=4000 已有完整独立结果；B=8000 正在运行；B=16000/32000 和 J ablation 尚未完成 | 否 |
| P5 | 未开始正式验收 | 代码接口和部分修补存在，但必须等待 P4 理论/统计门禁 | 否 |
| P6 | 代码和 checkpoint 有，但正式质量门禁需重验 | 旧 acceptance 曾记录质量失败，不能直接接受 | 否 |
| P7 | 结果文件存在，但依赖 P6 | 必须 P6 通过后重跑并更新 acceptance | 否 |
| P8 | 原则上完成 | 真实 Gas preprocessing 和 data card 已有；需最终回归和 hash 核对 | 暂不推进 |
| P9 | 结果数量已有，正式验收未通过 | 必须确认 final estimator 使用冻结 Gas VAEAC，而非 empirical kernel | 否 |
| P10 | 旧结果/代码修补存在，正式 R2 未重验 | 必须重做 2400 条记录并检查指标非负、policy-dependent、FW/PSD/预算 | 否 |
| P11 | 不允许进入 | figures/tables 必须只读取重新验收后的 summaries | 否 |
| P12 | 不允许进入 | manifest、commit、tag 尚未最终冻结 | 否 |

## 5. P4 已完成的代码修复

最近相关提交：

- `6db69d0`：更新阶段只对 active panels 重估 information，并增加覆盖测试
- `6067712`：pilot beta 投影到 compact Theta，记录 pilot residual 和 beta error
- `9726ded`：runner 暴露 `--lu`、`--h-tilted`、`--h-cond`、`--kl-samples`，便于可追溯的 MC 探针
- `72e1df1` / `7cd3d50`：KL 使用固定 large reference moments，清理重复配置
- `716ff03`：保存固定 seed profile：`experiments/p4_profile_diagnostics_fix.json`
- `ab0bab8`：清理重复 KL 配置
- `fe5b84a`：缓存 exact conditional Gaussian Cholesky，减少重复分解开销

这些提交修复了实现偏差和性能问题，但不等于正式实验门禁已通过。

## 6. P4 当前 artifact

### 已完成的独立结果

`results/p4_formal_fix4_2000.jsonl`

- 600 行 = 200 replications × 3 policies
- `scripts/p4_validate.py`：通过，0 个 pairing/budget/KL 结构失败
- 独立 summary：`results/p4_formal_fix4_summary.json`
- 统计结果：
  - oracle RR-GID：mean `44.11`，MC SE `1.6675`，95% CI `[40.845, 47.382]`
  - Uniform SQD：mean `82.90`，MC SE `2.8953`，95% CI `[77.226, 88.575]`
  - A-OSQD：mean `128.33`，MC SE `4.1719`，95% CI `[120.149, 136.503]`

`results/p4_formal_fix5_4000.jsonl`

- 600 行 = 200 replications × 3 policies
- 使用独立 `fix5` 前缀和显式 MC overrides：`LU=64`、`h_tilted=128`、`h_cond=4`、`kl_samples=2000`
- 结构验收通过
- 独立 summary：`results/p4_formal_fix5_summary_4000.json`
- 统计结果：
  - oracle RR-GID：mean `58.35`，MC SE `2.6212`，95% CI `[53.209, 63.485]`
  - Uniform SQD：mean `96.49`，MC SE `3.2225`，95% CI `[90.178, 102.810]`
  - A-OSQD：mean `180.91`，MC SE `6.6121`，95% CI `[167.949, 193.868]`

### 正在运行的结果

`results/p4_formal_fix5_8000.jsonl`

- 当前交接时：318 行，约 106/200 replications
- 12 个 CPU shard，replication 区间不重叠
- 启动参数：

```bash
python3 -u -m scripts.p4_formal_run \
  --budget 8000 --max-replications 200 --rep-range START END \
  --prepared experiments/p4_prepared_oracle.pkl \
  --out-prefix p4_formal_fix5 \
  --lu 64 --h-tilted 128 --h-cond 4 --kl-samples 2000
```

- 服务器后台进程由 12 个不重叠 range 组成；不要重复启动同一个 range
- 完成标准：600 行，replication 0-199，各 replication 恰好 3 policies

### 尚未完成的 P4 artifact

- `p4_formal_fix5_16000.jsonl`：尚未完成
- `p4_formal_fix5_32000.jsonl`：尚未完成
- `J=0/1/2` 的 B=8000 独立输出：尚未完成
- P4 五档合并 summary：尚未生成
- 新的 P4 acceptance record：尚未通过

## 7. P4 继续执行流程

### 7.1 监控 B=8000

```bash
cd /root/RR_GID_CN
wc -l results/p4_formal_fix5_8000.jsonl
pgrep -af 'scripts.p4_formal_run.*p4_formal_fix5'
```

完成 600 行后：

```bash
/root/rrgid_env/bin/python3 scripts/p4_validate.py \
  results/p4_formal_fix5_8000.jsonl --required-replications 200

/root/rrgid_env/bin/python3 scripts/p4_summary.py \
  results/p4_formal_fix5_8000.jsonl \
  --output results/p4_formal_fix5_summary_8000.json
```

### 7.2 顺序启动 B=16000、B=32000

仅在前一档完成结构验收后启动下一档。推荐 12 个不重叠 range：

```bash
for b in 16000 32000; do
  for r in $(seq 0 11); do
    s=$((r*17)); e=$(((r+1)*17));
    if [ "$r" -eq 11 ]; then e=200; fi
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH=src /root/rrgid_env/bin/python3 -u -m scripts.p4_formal_run \
      --budget "$b" --max-replications 200 --rep-range "$s" "$e" \
      --prepared experiments/p4_prepared_oracle.pkl \
      --out-prefix p4_formal_fix5 \
      --lu 64 --h-tilted 128 --h-cond 4 --kl-samples 2000 \
      > "/tmp/p4_fix5_${b}_${r}.log" 2>&1 &
  done
  # 等该 budget 完成并运行 p4_validate 后再启动下一个 budget。
done
```

不要同时运行两个高预算档，避免 CPU 争用和不可解释的 MC 性能变化。

### 7.3 J ablation

固定 `B=8000`，每个 `J` 使用独立输出后缀，不覆盖主结果：

```bash
for j in 0 1 2; do
  # 使用与主实验相同的 prepared artifact、seed scheme 和 MC overrides
  python3 -u -m scripts.p4_formal_run \
    --budget 8000 --max-replications 200 --rep-range START END \
    --prepared experiments/p4_prepared_oracle.pkl \
    --out-prefix p4_formal_fix5 \
    --scoring-steps "$j" \
    --lu 64 --h-tilted 128 --h-cond 4 --kl-samples 2000
done
```

实际运行时必须对每个 `J` 使用 12 个不重叠 range；runner 会生成 `_J0`、`_J1`、`_J2` 后缀文件。

### 7.4 P4 汇总与门禁

只有五档和 J ablation 都完成后，才执行：

1. 每档 `p4_validate.py`：600 行、200 reps、三 policy、预算严格相等、target draw seed 配对、KL 非负。
2. `p4_summary.py`：mean、MC SE、95% CI、design ratio。
3. 检查 oracle RR-GID 排序和 `B*KL` 随预算趋近理论平台；理论参考线约为 oracle `31.9`、Uniform `64.8`、A-OSQD `95.8`。
4. 检查 `J=0` 为 pilot-only，`J=1` 改善，`J=2` 达到预期的渐近修正方向。
5. 生成新 acceptance 文档，记录 commit、config hash、prepared artifact hash、seed scheme、运行日志、异常分类和结果 hash。
6. P4 仍不通过时，保留全部结果并继续定位；不能进入 P5。

## 8. P5-P12 后续任务

### P5：四策略 Synthetic S1 闭环

前置：P4 五档理论/统计门禁通过。

- 实现并正式验收 mask-conditioned MLP Discriminative Score OED
- 使用冻结 Synthetic VAEAC 的 generator-aware information 实现 learned RR-GID
- 四策略共享 target draws 和 final RR estimator
- exact conditional oracle 仅作为 Synthetic S1 observed-score evaluation，不得把 learned RR-GID 写成 oracle
- 生成四策略 paired manifest、B·KL、design ratio、CI/SE、Fig.1 输入 summary

### P6：Synthetic VAEAC 质量门禁

前置：P5 中需要 generator 的部分前，且正式结果依赖 P6。

- 使用 canonical VAEAC checkpoint loader
- checkpoint 必须包含 `proposal`、`conditional_prior`、`decoder`、`data_mean`、`data_std`
- 禁止接受旧 empirical generator / `z_std` / checkpoint-free generator
- 重训 Synthetic VAEAC，验证 unconditional moments、conditional reconstruction、acceptance、ESS/N、PSD information
- 质量失败时修模型/训练实现，不得放宽阈值冒充通过

### P7：Synthetic S2

- P6 质量通过后重跑 alpha sweep 和 generator reuse
- 每 campaign 重抽 feature dictionary、beta、panel 子集
- RR-GID 不重训 generator，Discriminative 每 campaign 重训
- 输出 design ratio、operator information error、compute、regret、CI

### P8/P9：Gas R1

- P8 最终核对真实 UCI 数据 hash、data card、reference/target 隔离
- P9 使用独立 Gas VAEAC checkpoint
- final estimator 必须走冻结 Gas VAEAC conditional interface，禁止 empirical kernel 绕过 generator
- 四档预算、四 policy、每档 200 reps；记录 ESS/N、acceptance、ranking、information error、full-law KL

### P10：Gas R2

- 重新生成 2400 条 paired records
- projection loss 必须是 PDF Eq.17 的非负 Bregman divergence
- 每个 policy 必须得到自己的 `beta_hat`，样本来自各自 `Q_beta_hat`
- 固定 32 held-out functions；generated/target 等量；C2ST 固定 5-fold
- acquisition 只能读取 campaign pool；full-test 只能 evaluation；`beta_dagger` 只能来自 full-test evaluation
- 记录并分类 acceptance collapse、低 ESS、FW gap、PSD 失败和预算异常，不删除异常结果

### P11：图表和表格

- 只读取 P4/P5/P7/P9/P10 重新验收后的 summaries
- Fig.1：四策略、B·KL、J ablation、oracle horizontal line
- Fig.2：真实 P7 summary
- Fig.3：真实 R1/R2
- Fig.4：information fidelity/compute
- Table 1：三 campaign 的 projection loss、held-out moment RMSE、C2ST
- 自动核对图表数字与 JSONL，保存图表输入 manifest

### P12：版本冻结

- 干净环境重新运行 pytest 和关键 smoke
- 抽查至少一个 S1、一个 R1、三个 R2 campaign
- manifest 写入正式结果、配置 hash、数据 hash、checkpoint hash、figures、tables、commit
- manifest commit、HEAD、最终 tag 必须一致
- 旧 tag `v0.1-iclr-stage1` 不得冒充最终结果；创建新的明确最终版本 tag
- 最终报告列出全部未解决问题，不得把未通过阶段写成完成

## 9. 验收记录模板

每个阶段必须使用以下字段，不能只写“已完成”：

```text
阶段与 commit：
实际修改文件：
输入与配置 hash：
输出 artifact：
单元测试：
数值/理论检查：
统计检查：
配对与预算检查：
失败/异常及定位：
待确认问题：
是否允许进入下一阶段：
```

## 10. 当前待解决问题

1. P4 的 B=8000、16000、32000 正式 replication 尚未完成。
2. P4 J=0/1/2 ablation 尚未完成。
3. P4 五档合并理论/统计门禁尚未通过；B=2000 和 B=4000 的 B·KL 高于理论线，需观察高预算平台趋势。
4. P4 fix5 使用的中等 MC 配置必须在 acceptance 中明确记录，不能与旧正式配置混合。
5. P5-P10 正式实验尚未按当前修复重新验收。
6. P11/P12 不得执行。
7. 未提交的 `gas_data_card.json` 是用户修改，必须保留并在最终 freeze 前由用户变更纳入明确 commit。

## 11. 交接原则

- 不删除旧结果、不覆盖旧 acceptance、不把 smoke/schema 当正式实验。
- 不使用绘图脚本或理论水平线修正实验结果。
- 不为了通过门禁而放宽阈值、减少 replication、改变预算、改变 seed、改变 target/full-test 隔离或替换模型。
- 任何 PDF 未冻结的工程参数必须在 acceptance 中记录；若参数改变，使用新输出前缀并重新运行对应验收。
- 只有阶段所有验收通过、artifact 可追溯、异常已分类后，才允许进入依赖阶段。
