# RR-GID_CN

Reference-Relative Generative Information Design 项目工程实现。研究设计规范以项目根目录的 **`RR_GID_CN.pdf`** 为唯一依据，工程按 `docs/PROJECT_PLAN.md` 拆成 P0–P12 阶段。

## 目录结构

```
RR_GID_CN/
├── RR_GID_CN.pdf              # 唯一研究规范（冻结）
├── src/rr_gid_cn/             # 核心源码（oracle / policies / synthetic oracle / vaeac / s1_gate）
├── scripts/                   # 各阶段运行脚本（p0–p11 的 smoke / formal / diagnostic）
├── configs/                   # 各阶段 YAML 配置（p1–p11_formal）
├── tests/                     # 单元测试（当前 32 passed）
├── results/                   # 实验输出
│   ├── p4/                    # P4 逐 replication 原始结果（按子实验分目录）
│   │   ├── diag/{h1024,h2048,oracleH,q12,q13,qmcinfo}/   # 信息估计层诊断
│   │   ├── gate_probe/  gate_j2/  gate_true/              # 最优性门禁探测/消融
│   │   ├── repair/                                       # pilot 修复实验
│   │   └── formal/                                       # 正式批次 jsonl
│   ├── p{1,2,3,5,6,7,8,9,10}_*_summary.json              # 各阶段汇总
│   └── p{4,5,9,10}_*_*.jsonl                              # 既有正式批次行数据
├── experiments/              # 实验资产（acceptance 记录、diagnostic 记录、checkpoint 等）
├── paper_tables/             # 论文格式数据表
├── figures/                  # 图表脚本产物（P11）
├── data/gas/                 # UCI Gas 数据卡片与预处理产物
└── docs/                     # 规划、验收、交接与汇报文档
    ├── PROJECT_PLAN.md / EXECUTION_PROMPT.md
    ├── HANDOVER_STATUS*.md / PERSISTENT_CONTEXT_*.md   # 交接
    └── 汇报给师兄_*.md/.pdf                            # 阶段性汇报
```

## 当前状态（2026-08-26）

- **已完成并验证**：P0–P3 工程与四策略算法、Synthetic 精确 oracle（16 维 GMM + sinh warp、120 panels、12 维 feature map）、RR-GID 完整算法（balanced pilot → HT moment → cross-completion 信息估计 → Frank-Wolfe design → J=2 Fisher scoring），单元测试 `32 passed`。
- **P4（Synthetic S1 最优性门禁）未通过**：oracle RR-GID 的 `B·KL` 未随预算收敛到理论常数 `½·Φ(p*)`。卡点在 PDF 冻结的 pilot 预算公式 `b_B=ceil(10B^(1/3))` 与理论相合性假设的有限样本张力（pilot 系统偏差 O(1)，非理论要求的 O(B^{-1/6})）。详见 `docs/汇报给师兄_项目进展与P4阻塞分析.md` 与 `experiments/p4_*_record.md`。
- **P5–P12 依赖 P4 门禁，暂未进入正式验收**；P6 VAEAC 生成器质量为第二个阻塞项。

## 运行

```bash
# 环境（conda）
conda env create -f environment.yml
pip install -e .

# 测试
python -m pytest tests/ -q

# 单阶段 smoke / formal
python scripts/p{1..11}_smoke.py --config configs/pX_formal.yaml
```
