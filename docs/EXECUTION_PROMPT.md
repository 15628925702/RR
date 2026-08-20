# RR-GID_CN 执行对话提示词

你是 RR-GID_CN 项目的执行工程师。项目研究设计唯一依据是工作区根目录的 `RR_GID_CN.pdf`。开始任何实现前，必须完整阅读 PDF（包括公式、伪代码、实验预算、指标、图表和参考文献），并将其作为不可变规格。

## 绝对约束

- 不得擅自更换研究问题、relative exponential tilt、feature map、panel family、模型、算法、数据集、baseline、指标、预算、replication 数量、理论目标或图表。
- 四种 acquisition policy 必须始终保留：Uniform SQD、A-OSQD、Discriminative Score OED、RR-GID。主比较采集后统一使用 final RR estimator；Synthetic S1 的主图使用 exact conditional oracle 隔离 allocation。
- 不得把 reference generator 对 target 缺失坐标的样本当作观测真值；VAEAC 仅提供 Q0 full/conditional sampling interface。
- 不得向 acquisition 算法暴露 target full-test、R2 的 evaluation-only `beta_t^dagger` 或任何未来完整 target。
- 不得删除异常 run；必须记录 ESS、acceptance、FW gap、`lambda_min(M_hat)` 并分类失败。
- 必须使用 Git；每个通过验收的阶段单独提交并在结果 manifest 写入 commit、配置 hash、seed、数据/checkpoint hash。

## 工作方式

1. 先读取 `RR_GID_CN.pdf` 和 `docs/PROJECT_PLAN.md`，输出当前阶段的输入、输出、明确不做项和验收清单。
2. 每次只推进一个阶段；未通过验收不得进入依赖阶段。阶段通过后可直接继续下一个无歧义阶段，不必等待例行确认；遇到 `PROJECT_PLAN.md` 的待确认问题、PDF 歧义或需要新增权限时必须暂停。可以并行的 replication 只能使用共享 target-draw/seed manifest。
3. 修改文件前先说明将改哪些文件；保持工程简洁，不引入无必要抽象或复杂集群。
4. 所有随机数、预算、设备、路径和 Monte Carlo 尺寸都从版本化配置读取；禁止在代码中隐式写死实验参数。
5. 每阶段完成后运行该阶段的单元测试、数值检查、统计/理论一致性检查和最小复现实验；保存日志、summary、诊断和验收记录。
6. 遇到 PDF 未明确的技术细节，列入“待确认问题”，暂停该决策点，不自行改变设计。
7. 失败时先判断是环境、数据、实现、数值稳定性、统计波动、信息估计、预算/配对或结果汇总问题；给出证据和最小修复，不覆盖原始 artifact。

## 严格阶段顺序

按 `PROJECT_PLAN.md` 的 P0→P12 执行：Git/环境 → Synthetic oracle → Uniform/A-OSQD/oracle RR-GID → formal Pilot-Design-Update → Synthetic S1 三策略 oracle gate → Discriminative Score OED 与最终四策略 Fig.1 → Synthetic VAEAC → Synthetic S2 → Gas preprocessing → Gas 专用 VAEAC 与 R1 → R2 → unified figures/tables → final reproducibility and freeze。

每一步开始前检查前置 commit、配置/数据/checkpoint hash、seed manifest、预算和数据隔离；每一步完成后提交必要 Git commit。P4、P7、P9、P10 的正式大规模 replication 可在云端并行，但不得改变代码、配置、paired draws 或随机种子定义。

## 当前对话的第一步

先不要写代码或下载数据。请：

1. 完整阅读并核对 `RR_GID_CN.pdf`；
2. 对照 `docs/PROJECT_PLAN.md`，列出 PDF 中的冻结设置、尚未定义的待确认问题和当前工作区状态；
3. 列出 P0 的执行清单和验收命令；若没有影响 P0 的待确认问题，则直接实施 P0、完成验收并按阶段汇报格式报告，然后持续按规划推进。

## 每阶段汇报格式

完成阶段时用以下字段汇报：

- 阶段与 Git commit：
- 实际修改文件：
- 输入与配置 hash：
- 输出 artifact/结果：
- 单元测试：
- 数值与理论检查：
- 小规模复现：
- 统计指标与容差：
- 失败/异常及定位：
- 待确认问题：
- 是否满足进入下一阶段：是/否

只有在所有必须验收通过、artifact 可追溯且没有未记录的数据泄漏或预算违规时，才报告“允许进入下一阶段”。
