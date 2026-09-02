# RR-GID-CN

Reference-Relative Generative Information Design 的工作快照仓库。账号 `15628925702` 下的新项目 **RR**。

当前停在 **P4 正式实验前的诊断阶梯 G3**。G4 plugin 与正式 3 政策 × 5 预算 × 50 次尚未作为完成结果写入。

## 权威文档

只认这两份，不认旧诊断目录：

- `核心文档/RR_GID_CN.pdf`
- `核心文档/RR_GID_CN_P4代码审查与修正指南.md`

`legacy_diagnostic_20260826/` 不进本仓库，也不作验收。

## 代码

| 路径 | 内容 |
| --- | --- |
| `src/rr_gid_cn/` | 核心实现（oracle、integrity、S1 gate） |
| `scripts/` | P4 Phase 0–3 入口与正式 runner（正式格子未跑） |
| `tests/` | Phase 0–3 与 integrity 单测 |
| `configs/` | G0–G4 与 Phase 0–2 yaml |
| `experiments/` | target / ladder seed manifest |

正式 replication 一律 **50**，禁止 200。种子公式 `202600000 + budget*1000 + replication`。

## 中间结果（`results/p4/`）

| 目录 | 阶梯 |
| --- | --- |
| `phase0/`、`phase0_*` | 冻结与 profiler |
| `phase2_gold_*` | gold φ、centering |
| `g0_b2000_20260831` | G0 oracle-start |
| `g1_20260831` / `g1_stepcap_20260831` | G1 无帽 / 带帽 |
| `g2_20260831` | G2 估计 H |
| `g3_qmc_20260831` | G3 QMC |
| `g3_rejection_20260831` | G3 rejection（LU=415） |

证书看各目录的 `diagnostics.json` 与 `rows.jsonl`。运行日志 `*.log` 按 gitignore 不入库。

## 汇报

`汇报/` 里是 2026-09-02 进度汇报、停在 G3 的实现说明、墙时瓶颈说明（md / html / pdf）。

## 本仓库刻意不包含

- 本地 Python 环境（`.venv`、`.phase0-env`）
- `.apodex/` 内部轨迹
- Gas 传感器原始数据（`data/gas/`，P8 以后才用）
- G4 未完成跑次、正式 750 格
