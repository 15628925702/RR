# RR-GID-CN

执行依据：`核心文档/RR_GID_实验重规划与代码改造执行指南_20260902.md`。  
G0–G4 诊断阶梯已归档，不再进入论文实验。

论文只回答三个问题：固定预算下 RR-GID 是否降低完整 target 估计误差；优势是否来自 nonlinear conditional information；frozen generator 能否跨 campaign 复用。

## 复现入口（论文）

```text
python scripts/build_oracle_artifact.py --config configs/paper/oracle_calibration.yaml
python scripts/calibrate_fast_backend.py --config configs/paper/oracle_calibration.yaml
python scripts/run_synthetic_main.py --config configs/paper/synthetic_main.yaml --resume
```

默认 `pytest` 只跑快速测试，不触发 G-ladder / rejection / adaptive order 16。

## 目录

| 路径 | 用途 |
| --- | --- |
| `configs/paper/` | 论文实验配置 |
| `scripts/` | 论文 runner（建设中） |
| `scripts/diagnostics_legacy/` | 已冻结的 G0–G4 / Phase 0–3 脚本 |
| `results/diagnostics_legacy/` | 已冻结的 G0–G3 证书 |
| `汇报/` | 停在 G3 之前的进度与墙时说明 |

## 权威文档

- `核心文档/RR_GID_CN.pdf`
- `核心文档/RR_GID_CN_P4代码审查与修正指南.md`
- `核心文档/RR_GID_实验重规划与代码改造执行指南_20260902.md`
