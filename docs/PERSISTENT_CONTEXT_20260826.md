# RR-GID_CN 持久化项目上下文与交接说明

更新时间：2026-08-26（Asia/Shanghai）

## 1. 不可变依据

本项目的唯一研究设计依据是项目根目录的 `RR_GID_CN.pdf`。`PROJECT_PLAN.md`、`EXECUTION_PROMPT.md`、`REPRODUCIBILITY.md`、旧 acceptance、旧图表和旧结果都只能作为工程记录，不能覆盖 PDF 的定义。任何新对话必须先阅读 PDF，再阅读规划和复现文档；如果实现与 PDF 冲突，以 PDF 为准。

不得擅自改变：研究问题、relative exponential tilt、16 维 Synthetic GMM、sinh warp、12 维 feature map、120 个 coordinate-pair panels、四种 policy、VAEAC backbone、预算、replication 数、指标、理论目标、数据集、随机种子、数据隔离和图表定义。

## 2. 当前执行位置与环境

正式计算在新开发机完成，不再使用旧服务器状态：

- 主机：`172.31.0.81:2222`
- 本机到服务器：Clash 专用 SOCKS5 `127.0.0.1:7892`
- `127.0.0.1:7892` 对应专用 mihomo，固定给服务器 SSH/传输使用
- 用户日常 Clash：`127.0.0.1:7897`，不得用于项目连接，也不得擅自切换
- 认证：`C:\Users\22909\.ssh\id_ed25519`
- 远端项目：`/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/current_20260825`
- 远端 Python 环境：`/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/env`
- Python：3.12.13
- PyTorch：2.13.0+cu126
- CUDA：12.6 runtime；8 张 NVIDIA A800-SXM4-80GB 全部可见
- 测试：`32 passed`
- 环境日志：`/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/environment_install.log`

## 3. 已完成并验证的内容

### P0-P3 工程与基础算法

已有工程骨架、配置入口、Git 目录、统一 seed/路径、Synthetic GMM+sinh warp、exact Gaussian-mixture conditional law、Uniform SQD、A-OSQD、oracle RR-GID、预算 rounding、Frank-Wolfe、balanced pilot、HT moment、cross-completion、PSD projection 和 Fisher-scoring update。相关单元测试在新环境已通过。

这些阶段的“通过”仍需以 PDF 规定的阶段验收记录和当前代码/配置 hash 为准；旧 acceptance 文件不能单独证明正式科研结果有效。

### P6/P7/P8/P9/P10 的既有工程资产

仓库保留了 VAEAC、Synthetic S2、Gas preprocessing、Gas R1/R2、P11 图表和表格脚本，以及旧结果。它们可以作为输入或对照，但只在依赖阶段重新验收通过后才能进入最终冻结。尤其 P9/P10 旧 acceptance 曾为 NOT PASSED，不能用旧趋势替代重验收。

### 设备绑定修复

`src/rr_gid_cn/synthetic_oracle.py` 已加入显式 `RR_GID_CN_CUDA_DEVICE` 设备选择，并校验设备编号。这样每个外部 shard 可通过 `CUDA_VISIBLE_DEVICES=$i` 绑定一张卡，同时在进程内使用 `RR_GID_CN_CUDA_DEVICE=0`。修复后本地和远端测试均为 32 passed。

GPU0/GPU1 同一固定 seed 的 P4 probe 输出完全一致：`B·KL_raw=34.132304843664144`、`design_ratio=1.0710368994222836`。这证明设备绑定没有改变数值路径，但不是 P4 正式通过证明。

## 4. 已走过的错误路线与经验

1. 旧服务器与新开发机混用。旧地址 `172.31.14.235` 已不属于当前正式执行位置；后续只用 `172.31.0.81:2222` 和专用 7892。
2. 用 GPU 利用率代替吞吐判断。16 worker 争抢 GPU 时利用率很高，但 replication 吞吐没有提升。必须用已完成 replication、耗时、输出行数衡量速度。
3. 使用 diagnostic-sized prepared artifact 跑正式 P4。旧 `p4_prepared_oracle.pkl` 只有 `reference_large=200000`，正式 runner 正确拒绝它；不得绕过门禁。
4. 上传大文件时反复建立 SFTP 会话，导致代理/SSH 断连和部分文件。解决方式是专用代理 7892、按远端真实 offset 续传、最终比较大小和 SHA256。任何未校验文件都不能进入实验。
5. 8 小时编排器第一次没有 `cd` 到项目目录，runner 使用相对输出路径而失败。后续脚本必须显式 `cd $ROOT`，并在启动后检查日志。
6. 不能把准备阶段、probe、schema gate、旧结果或旧 acceptance 写成“P4 已通过”。正式通过必须完成全部预算、J 消融及理论/统计/配对验收。

## 5. 当前正在做什么

### 高精度 prepared artifact

已生成并通过正式规模检查：

- `reference=50,000`
- `reference_large=1,000,000`
- `information_shape=(120,12,12)`
- 远端文件：`experiments/p4_prepared_oracle_hp.pkl`

该 artifact 是当前 P4 正式实验的输入。它必须保留，不得被 diagnostic artifact 静默覆盖。

### P4 八小时后台编排

远端脚本：`/kairos_vepfs_volc/autodrive/manlichen/RR_GID_CN/p4_8h_20260825.sh`

日志：`current_20260825/p4_8h_20260825.log` 和 `current_20260825/p4_8h_logs/`

正式输出目录：`current_20260825/results/p4_formal_8h_20260825/`

编排顺序为 `B={2000,4000,8000,16000,32000}`。每档 200 replications，8 张 GPU 各承担 25 个 replication；每个 replication 输出 3 个 policy rows。某一档失败则停止，不进入下一档。

当前 `B=2000` 正在运行，8 个进程均存活，已开始持续写出 JSONL。此前的空文件和第一次编排失败日志保留，不视为正式结果。

## 6. 当前正式验收尚未完成的部分

P4 尚未通过。必须完成并记录：

- 五档预算每档至少 200 replications、三 policy、相同 target draws；
- exact conditional oracle、`b_B=ceil(10B^(1/3))`、`J=2`、预算和 rounding；
- `B·KL` 理论平台与 design ratio；
- `J=0/1/2` ablation；
- FW gap、PSD/lambda_min、KL 非负、异常 run 分类；
- paired manifest、CI/SE、配置/seed/hash/日志和 commit。

P4 未通过前不得把 P5-P12 标记为完成。P5/P7/P9/P10 即使已有数量正确的旧文件，也必须根据新的 generator/checkpoint 和正式验收重新确认。

## 7. 后续执行顺序

1. 监控 P4 当前 budget，确认每档完整并执行 `p4_validate`。
2. 若失败，按 PDF 定义定位 conditional information、Fisher、KL、allocation、estimator 或数值稳定性问题；保留旧结果，在新目录修复和小规模固定 seed 验证后再重跑。
3. P4 五档和 J 消融通过后，完成 P5 四策略闭环：Uniform/A-OSQD 来自同一 target draws，Disc 使用 mask-conditioned MLP，learned RR-GID 使用冻结 VAEAC，最终统一 final RR estimator。
4. 重新验收 P6 Synthetic VAEAC 质量，再重跑 P7 S2 alpha/reuse campaigns。
5. 核对真实 UCI Gas 数据和独立冻结 Gas VAEAC，重跑 P9 R1。
6. 重做 P10 R2 的 Eq.17 Bregman projection、policy-specific beta_hat、32 held-out functions、实测 acceptance/ESS/FW/PSD/C2ST，生成 2400 paired records。
7. 只有 P4-P10 正式通过后重生成 P11 figures/tables，并自动核对 JSONL 数字。
8. 最后执行 P12：干净环境 pytest、随机抽查、manifest/hash、commit/tag 一致性、GitHub/HF 备份。旧 tag 不得冒充最终结果。

## 8. 新对话继续执行提示词


```

## 9. 交接原则

任何新对话首先核对事实和文件，不复制旧对话中的乐观结论。PDF 优先，阶段门禁优先，结果可追溯优先。当前最重要的事实是：P4 正式实验正在运行，但 P4 尚未通过。
