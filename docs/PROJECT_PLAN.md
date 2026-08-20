# RR-GID_CN 项目规划与阶段验收方案

## 0. 依据、范围与冻结约束

本规划唯一依据是项目根目录的 `RR_GID_CN.pdf`（2026 年 8 月研究规格草案）。本轮只完成工程规划，不执行代码、数据下载、训练或实验。后续实现不得改变 PDF 已冻结的研究问题、relative exponential tilt、feature map、panel family、四种 acquisition policy、VAEAC backbone、预算、指标、理论目标和图表。

PDF 中必须保持的关键常数与设置：

- Synthetic：`d=16`；四成分 latent GMM；`seed=2026` 只生成并冻结 mixture 参数；可逆 `sinh` warp；`r=12`；120 个等成本坐标 pair panels；主强度 `alpha=1`。
- Relative family：`dQ_beta/dQ0=exp(beta^T phi-A(beta))`；`phi` 为 bounded transform；以完整 target KL 为风险；设计目标 `Phi_beta(p)=tr(F(beta) M_beta(p)^-1)`。
- RR-GID：balanced pilot、一次 generator-aware Frank-Wolfe design、主采集、`J` 次 Fisher-scoring update；默认 `gamma=1/3`、`J=2`；使用 cross-completion information estimator 与 PSD projection。
- 四种 policy：Uniform SQD、A-OSQD、Discriminative Score OED、RR-GID。主比较四者采集后统一使用 final RR estimator，并在 Synthetic S1 使用 exact conditional oracle 隔离 allocation 贡献。
- Synthetic S1：`B in {2000,4000,8000,16000,32000}`，`b_B=ceil(10 B^(1/3))`，每个 B、policy 至少 200 个独立 target replications。
- Synthetic S2：`B=8000`；`alpha in {0,0.5,1.0,1.5}`；reuse `T in {1,5,20,50}`。
- Gas Sensor：UCI Gas Sensor Array Drift；13,910 measurements、16 sensors、每 sensor 8 features、128 维；reference batches 1--6，target 为 batch 7、batches 8--9、batch 10；120 个任意两 sensor panels；每 sensor block PC1，`r=16`。
- Gas R1：reference validation empirical base 的 well-specified 半合成 tilt；`B in {400,800,1600,3200}`，`b_B=min(0.2B,ceil(10B^(1/3)))`，`J=2`。
- Gas R2：自然 drift robustness；至少 50 个 paired replications/budget/campaign；family projection loss、held-out moment RMSE、C2ST AUC、overlap diagnostics。
- 主文结果：Fig.1 S1 `B*KL` 与 J ablation；Fig.2 nonlinearity/reuse；Fig.3 Gas R1/R2 budget curves；Fig.4 information fidelity/cumulative compute；Table 1 三个 natural campaigns 的 projection loss、held-out moment RMSE、C2ST。

## 1. 阶段规划总览

每个阶段都必须保存配置、随机种子、代码 commit、日志、摘要结果和验收记录。阶段未通过时停止，不进入依赖它的阶段。

### P0. Git 仓库与项目环境初始化

**目标**：建立最小可迁移 Python 工程、Git 追踪和统一配置入口。

**前置依赖**：仅需本 PDF 和空工作区。

**实现内容**：初始化 Git；建立 `pyproject.toml`、`environment.yml`、`configs/`、`src/`、`scripts/`、`tests/`、`experiments/`、`results/`、`figures/`、`docs/`；统一 seed、设备、dtype、路径和 YAML/JSON 配置读取；固定 Python 版本（建议 3.11，若依赖冲突则在环境文件中明确唯一替代版本）；设置 `.gitignore`。

**输入**：PDF、机器资源说明。

**输出**：可创建环境并运行 `--help` 的最小包；环境锁定文件；初始 Git commit。

**明确不做**：不下载 UCI 数据、不实现模型、不提交任何数据或 checkpoint。

**验收与进入条件**：

1. 单元测试：配置 schema、seed 初始化、CPU/GPU device 选择、路径解析各至少 1 个测试。
2. 数值/复现：同一 seed 的 toy RNG 输出逐字节一致；不同 seed 确实改变样本。
3. 小实验：空 pipeline 可在本机 1 分钟内启动并写出 run manifest。
4. 失败定位：环境安装失败归为依赖问题；seed 不一致归为随机源/worker 问题；路径失败归为配置问题。
5. 进入下一阶段条件：`pytest` 通过、`python -m ... --help` 成功、Git 工作区干净。

### P1. Synthetic oracle pipeline

**目标**：实现冻结的 nonlinear GMM+warp reference、精确 conditional sampler、tilt oracle 和统一评估原语。

**前置依赖**：P0。

**实现内容**：按 PDF 用 seed 2026 生成并冻结 GMM 参数；实现 `T_alpha`/逆变换、full sampler、任意 panel 的 Gaussian-mixture conditional；实现 `phi`、`A(beta)`、`mu_beta`、`F(beta)`、full-target KL；实现 exact tilted full/conditional oracle、ESS/acceptance 记录。

**输入**：Synthetic 配置、冻结 mixture 参数、beta 与 alpha。

**输出**：`synthetic_oracle` 模块；冻结参数 artifact/hash；conditional sampling diagnostics；可计算 exact `I_S(beta)`、`F(beta)`、`p*` 的小规模脚本。

**明确不做**：不使用 VAEAC；不近似 conditional law；不改变 mixture、warp、feature 定义。

**验收与进入条件**：

1. 单元测试：warp 与 inverse 在随机点误差 `<1e-10`；panel index/feature support 正确。
2. 数值正确性：conditional sampler 的样本均值/协方差与解析 Gaussian-mixture conditional 的标准误差范围一致；增大样本量时误差按 `O(n^-1/2)` 下降。
3. 理论一致性：`F` 与 Monte Carlo covariance 的相对 Frobenius 误差在预设容差内（小规模 `<=5%`）；每个 `I_S` 对称，`lambda_min >= -1e-10`；全观测信息与 `F` 一致。
4. 小复现：固定 beta、alpha、seed 连续两次导出完全相同的 oracle summary；`J=2`、小 B 能完成端到端 toy run。
5. 失败定位：conditional 偏差随 panel 增大不降是变换/条件公式问题；PSD 失败是中心化或数值问题；KL 不非负是 log-partition/积分实现问题。
6. 进入条件：以上通过，并生成 machine-readable oracle summary。

### P2. Uniform SQD、A-OSQD、oracle RR-GID

**目标**：在 exact oracle 上建立三个 allocation policy 和共同 cost-aware solver。

**前置依赖**：P1。

**实现内容**：Uniform SQD 在 120 panels 均匀随机；A-OSQD 按 Jang et al. 的完整 reference covariance、multivariate-normal mean A-optimal split-questionnaire 目标 `tr(M_A^-1)`；oracle RR-GID 用 exact `I_S`、`F`、等成本 panels 求 `p*`。实现 cost-share 到整数 counts 的 rounding、balanced safe allocation、budget accounting。

**输入**：exact `F`、`I_S`、panel costs、B、`p_safe`。

**输出**：三种 policy 的 allocation/counts；oracle `p*`；`Phi`、design ratio、预算日志。

**明确不做**：不加入 VAEAC、不加入 discriminative network、不做 Fisher update。

**验收与进入条件**：

1. 单元测试：Uniform 概率和为 1；rounding 后 `sum n_S c(S)<=B`；panel counts 非负整数。
2. 数值正确性：FW 每轮 `Phi` 不上升（允许 `1e-10` 数值噪声）；`g_FW=max gamma_S-Phi`；终止时 `g_FW<=tau_FW`；`M` PSD/正定（safe allocation 下 `lambda_min>0`）。
3. 理论一致性：oracle solver 与独立高精度 constrained optimizer 的 `Phi` 差异在 `tau_FW` 内；`J=2` 的理论条件记录为 `2J gamma=4/3>1`。
4. 小复现：d=4、4 panels、B=100 的 toy allocation 可手工核对。
5. 失败定位：预算超限是 rounding；FW gap 不降是 sensitivity/line-search；allocation 不稳定是 information conditioning。
6. 进入条件：三种 policy 均能输出可审计 allocation 和 certificate。

### P3. Formal RR-GID Pilot-Design-Update algorithm

**目标**：严格实现 PDF Algorithm 2 的 balanced pilot、HT moment、generator-aware design 和 J-step Fisher scoring。

**前置依赖**：P1、P2。

**实现内容**：reference pool `R_A` 与 `A_hat_R`；pilot `n0_S=floor(b p_bal,S/c(S))`；coverage `rho_a` 与 HT estimator；`beta_tilde` constrained solve；cross-completion `I_hat_S`；PSD projection；FW 一次设计；main campaign；每步 `mu_hat_j`、`H_j`、`U_hat_j`、投影到 `Theta` 的 scoring update；输出 `beta_hat_B` 与 target generator density ratio。

**输入**：G0 oracle、feature supports、Theta、panels/costs、B、pilot、`R_A, N_g, L_I, L_U, J, tau_FW`。

**输出**：pilot/main datasets（仅在实验 artifact 中）、allocation/counts、每步 beta、FW gap、`Q_beta_hat`、overlap diagnostics。

**明确不做**：不把 cost share 当 unit-level probability；不使用未来 full target；不在 acquisition 阶段使用 full-test。

**验收与进入条件**：

1. 单元测试：HT 只访问 `A_a subseteq S` 的观测；`rho_a>0` 才允许估计；Theta projection 保持边界；cross-completion 两批独立。
2. 数值正确性：每个 run `sum n c<=B`；`M` 正定或明确记录 safe fallback；FW gap 满足阈值；每次 update 的 log-likelihood/estimating-equation residual 不出现异常爆炸。
3. 统计一致性：重复 toy runs 中 HT bias 接近 0；`B*KL` 随 B 增大接近 `1/2 min_p Phi_beta*(p)`；oracle allocation 与 estimated allocation 的 L1/`Phi` 差异随 pilot/MC 增大下降。
4. 小复现：固定 seed 的 B=200 toy run 两次 allocation、counts、beta trajectory 一致。
5. 失败定位：HT bias 非零是 coverage/support；MC upward bias 是未使用 cross-completion；beta 发散是 tilt overlap/Theta；FW gap 大是 information estimate 或 solver。
6. 进入条件：P2 三种 policy 与 oracle RR-GID 在 toy 上可同一评估接口运行。

### P4. Synthetic S1 statistical optimality oracle gate

**目标**：按 PDF 的执行顺序，先用 Uniform SQD、A-OSQD、oracle RR-GID 验证 `B*KL` 向理论常数收敛，并完成 J ablation；本阶段是加入 discriminative baseline 前的理论门禁，不是最终四策略 Fig.1。

**前置依赖**：P3；先通过 P2/P3 toy 验收。

**实现内容**：alpha=1；reference train/validation 50,000/10,000；B 五档；`b_B=ceil(10B^(1/3))`、J=2；Uniform SQD、A-OSQD、oracle RR-GID 每个 policy/B 至少 200 replications；三者使用相同 target draws；final observed score 使用 exact conditional oracle；B=8000 做 J in {0,1,2} ablation。Formal RR-GID 只做接口和估计 allocation 对 oracle allocation 的诊断，最终四策略主比较留到 P5 闭环。

**输入**：P1--P3 modules、冻结 beta direction、独立 Q0 pool、replication seed table。

**输出**：S1 oracle-gate 数据与诊断图（非最终 Fig.1）；`B*KL`、oracle horizontal line、design ratio、J ablation summary；每 replication manifest。

**明确不做**：不引入 VAEAC；不改变 target beta scale（ESS/N 约 0.5）；不增加未定义 baseline。

**验收与进入条件**：

1. 单元/数值：同一 replication 的三种 gate policy 使用相同 target draws；KL 计算非负；预算约束逐 run 通过。
2. 统计：报告均值、Monte Carlo SE、95% CI；`B*KL` 在大 B 接近理论常数，且 oracle RR-GID design ratio 接近 1；`J=2` 相比 J=0/1 符合 `gamma=1/3` 的理论预期。
3. 可复现：抽取至少 5 个固定 seed，重跑摘要完全一致；200 replications 的 seed manifest 无重复。
4. 失败定位：三种 gate policy 的 target draws 不同是实验设计错误；KL 常数偏离但 allocation 正确是 estimator/评估错误；仅大 B 失败是预算/MC scaling；所有 B 失败是 oracle 或 beta 设定。
5. 进入条件：oracle-gate 数据冻结并经人工检查、`B*KL` 收敛门禁通过，才允许加入 discriminative baseline；不得在本阶段把三策略图冒充最终 Fig.1。

### P5. Discriminative Score OED and final S1 closure

**目标**：实现强非生成式 ablation，与 true `I_S` 比较 panel ranking/design regret，并在 P4 门禁通过后完成 PDF 要求的最终四策略 S1/Fig.1。

**前置依赖**：P4 通过；P1 的完整 reference data 接口。

**实现内容**：按当前 `beta_tilde` 在完整 reference samples 构造 `s_beta(X)` labels；mask-conditioned MLP 输入 masked X+mask、输出 r 维 score；mask 从候选 panels 均匀抽取；tilt weights `w proportional exp(beta^T phi)`；独立 validation set 估计 `I_hat_S=Cov_w(g(X_S))`；每个 campaign 重新训练并重新设计。随后按 P4 的冻结 seed/target-draw manifest 补跑 Discriminative Score OED，并用 Uniform SQD、A-OSQD、Discriminative Score OED、正式 RR-GID 四种 acquisition policy、相同 target draws、相同 final RR estimator 生成最终 Fig.1；exact conditional oracle 继续用于 S1 final observed score。

**输入**：完整 reference train/validation、当前 beta、panel library。

**输出**：网络 checkpoint/配置（不入 Git）、训练日志、panel information、allocation、ranking correlation、design regret；最终四策略 Fig.1 数据/图、oracle line、J ablation 和 paired manifest。

**明确不做**：不让网络替代 RR final estimator；不使用 target full-test；不宣称生成器必然更优。

**验收与进入条件**：

1. 单元测试：mask 编码、未观测坐标不泄漏、权重归一化、validation/train 分离。
2. 数值：预测 score 的 validation loss 与 seed stability；`I_hat_S` PSD；与 true `I_S` 的 operator/Frobenius 误差可量化。
3. 统计：panel ranking Spearman/Kendall、`Phi` design ratio、regret；MLP 增大训练量时误差应不系统恶化。
4. 小复现：d=4 小网络可在 CPU 完成，固定 seed checkpoint 输出一致；抽取至少 5 个 S1 seeds 重跑四策略摘要一致。
5. 失败定位：信息矩阵偏差但预测 loss 正常是 covariance/weighting；ranking 乱序是 mask 或 panel mapping；训练不稳是优化/归一化。
6. 进入条件：四 policy 统一接口可运行，P4 的 exact-score isolation 仍保持，最终 Fig.1 每个 policy/B 至少 200 replications、四策略 target draws 完全配对且图表数值通过核对。

### P6. VAEAC generator

**目标**：一次训练 VAEAC，验证 arbitrary-conditioning、tilt accept-reject 和 conditional information 接口。

**前置依赖**：P5；P0 环境；Synthetic reference train/validation。

**实现内容**：VAEAC 完整 16 维输入、任意 mask conditioning；训练一次并冻结；验证 arbitrary-pair reconstruction；实现 Q0 full/conditional sample、tilted accept-reject、unconditional/conditional acceptance rate、importance ESS；learned-generator `A_hat_R` 与 cross-completion。

**输入**：Synthetic reference train/validation；固定 mask sampling/config。

**输出**：VAEAC checkpoint、训练 manifest、重建/conditional quality、acceptance/ESS diagnostics、generator version hash。

**明确不做**：不更换 backbone；不把缺失坐标当 target 真值；不将 checkpoint 提交 Git。

**验收与进入条件**：

1. 单元测试：mask 全观测/空观测/任意 pair；输出维度与 observed coordinates 保持约束；采样 shape 正确。
2. 数值/质量：unconditional moments 与 reference 一致；pair conditional reconstruction error、coverage、acceptance rate、ESS/N 全部记录；低 overlap run 自动标记。
3. 理论/接口：learned `I_hat_S` 对称 PSD；增大 conditional samples 时估计稳定；VAEAC full-sample pool 可复现。
4. 小复现：小数据/少 epoch smoke test 可在本机 GPU 完成，固定 seed 结果一致。
5. 失败定位：重建误差高是模型/训练；acceptance 低是 tilt overlap；conditional 偏差是 mask/conditioning；ESS 异常是权重溢出。
6. 进入条件：Synthetic pair conditional 达到预设质量阈值，且 learned generator diagnostics 完整。

### P7. Synthetic S2 nonlinearity 与 generator reuse

**目标**：验证 nonlinear conditional information 和单次 generator 跨 campaign reuse。

**前置依赖**：P6；P4 的评估脚本。

**实现内容**：固定 B=8000，alpha 四档；报告四 policy design ratio 与 `max_S ||I_hat_S-I_S||_op`；generator reuse 使用 T={1,5,20,50} 个 campaign，每次重抽 bounded unary/pairwise feature dictionary 中 12 个 features、beta 与部分 pair panels；RR-GID 不重训 G0，Discriminative Score OED 每 campaign 重训；绘制累计训练+推断 compute 与平均 design regret。

**输入**：冻结 VAEAC、Synthetic oracle、reuse seed table。

**输出**：Fig.2 数据/图、alpha sweep、reuse frontier、误差与 compute logs。

**明确不做**：不预设 generator 必然优于 discriminative；不增加第二 backbone/real dataset。

**验收与进入条件**：

1. 数值：四 alpha、四 T、四 policy 全部有完整 paired records；op-norm、FW gap、ESS/acceptance 无缺失。
2. 统计：alpha 增大时 A-OSQD 与 oracle 的偏离可解释；RR-GID/Discriminative 的 nonlinear information 误差不被线性基线掩盖；reuse 报告均值与 CI。
3. 可复现：同一 generator hash 在所有 T 中一致；重复摘要一致。
4. 失败定位：alpha=0 也偏离是 baseline 实现；reuse compute 异常是计时边界；T 增大性能漂移是 feature/beta 重抽样问题。
5. 进入条件：Fig.2 完成且 generator reuse 没有重训证据。

### P8. Gas Sensor 数据预处理

**目标**：严格构造 reference/target、128 维标准化特征、PC1 feature map 和 120 个 sensor-pair panels。

**前置依赖**：P7；只在此阶段下载并登记 UCI 数据。

**实现内容**：下载并校验 UCI 数据版本/hash；按 batches 1--6 reference、batch 7、8--9、10 target；reference 80/20 train/validation；所有 raw features 用 reference train mean/std；每 sensor block PCA 取 PC1；定义 `phi_j=tanh(z_j), phi_{8+j}=tanh(z_j z_{j+8})`；balanced pilot 用 8 个 `(j,j+8)` pairs。本阶段只准备 Gas VAEAC 的 128 维 train/validation 输入，不训练模型。

**输入**：UCI archive、PDF 预处理定义。

**输出**：原始数据不入 Git；校验 manifest；处理后 cache、split indices、scaler/PCA artifacts、panel map、数据卡。

**明确不做**：不使用 target full-test 参与 fit；不改变 batch 划分、PCA、feature map、panel size。

**验收与进入条件**：

1. 单元测试：128 维、16 sensor blocks、每 block 8 features；120 panels；PC1 只用 reference train 拟合；split 无泄漏。
2. 数值：标准化 reference train 均值约 0、方差约 1；PC1 重复拟合符号按固定规则对齐；`phi` 在 [-1,1]。
3. 统计：reference train/validation 与 PDF 样本量、batch 计数一致；同一 hash 重跑结果一致。
4. 小复现：抽取固定子集完成全 preprocessing smoke test。
5. 失败定位：维度错误是 parser；分布漂移异常是 split/scaler；PC1 不稳定是随机 solver/符号。
6. 进入条件：数据卡、hash、split、feature/panel manifest 齐全。

### P9. Gas Sensor R1 半合成实验

**目标**：先在 Gas reference train 上训练并冻结独立的 128 维 VAEAC，再在真实 reference joint structure 上验证 learned generator 的 panel ranking 与 synthetic 现象；该 checkpoint 供 R1、R2 共用。

**前置依赖**：P8、P6 的 VAEAC pipeline。

**实现内容**：复用 P6 已验收的 VAEAC 代码和接口，但不复用 Synthetic checkpoint；以 Gas reference train 的完整 128 维标准化 sensor features 训练一次 Gas VAEAC，完成 arbitrary-pair conditional 验收后冻结。随后从 reference validation empirical base 按 `w_i(beta*) proportional exp(beta*^T phi(x_i))` 重采样 target；调 beta scale 使 ESS/N 约 0.5；B={400,800,1600,3200}，`b_B=min(0.2B,ceil(10B^(1/3)))`，J=2；四 policy 使用相同 target draws 和统一 final RR estimator；Discriminative Score OED 按 campaign 重训 score network。

**输入**：reference train/validation records、P6 的 VAEAC 实现与验收接口、R1 seeds/config。

**输出**：独立 Gas VAEAC checkpoint/hash 与条件生成验收记录；R1 budget curves、panel ranking diagnostics、projection/full-law KL（相对 empirical base）与 overlap logs。

**明确不做**：不把 R1 当自然 drift；不使用 full-test；不改预算或 pilot 公式。

**验收与进入条件**：

1. 单元/数值：Gas VAEAC 输入/输出保持 128 维且任意 sensor-pair mask 无泄漏；重采样权重正确归一化；每 replication 预算与 target draw 对齐；四 policy target draws 完全相同。
2. 统计：R1 能稳定复现 Synthetic 的 panel-ranking 方向；报告均值、SE、CI；learned generator 与 empirical oracle 的 information 误差达预设阈值。
3. 可复现：Gas VAEAC 固定 seed/config 的 checkpoint hash 与验收摘要可重建；至少 5 个固定 R1 seeds 重跑一致；R1 full-law KL 高精度可由 empirical base 直接核对。
4. 失败定位：Gas conditional quality 失败是 128 维 mask/模型训练；R1 失败但 preprocessing 正确是 generator/information；ESS 低是 beta scale；ranking 正确但 loss 失败是 final estimator。
5. 进入条件：Gas VAEAC 冻结且任意 pair conditional 验收通过，R1 稳定后才允许 R2。

### P10. Gas Sensor R2 natural drift

**目标**：评估三次真实后续 campaign 的 robustness，不将其宣称为 T3/T4 exact verification。

**前置依赖**：P9 通过；P8 target split。

**实现内容**：每 campaign 一半 campaign pool（只可被 panel masking 观察）、一半 full-test pool（仅 evaluation）；固定大规模 Gas VAEAC full-sample pool 估计 `A_hat`；用 full-test 仅求 evaluation `beta_t^dagger`；沿用 R1 的 `B={400,800,1600,3200}` sweep；至少 50 个 paired replications/budget/campaign；四 policy 使用相同 target draws；RR-GID 复用冻结的 Gas VAEAC，Discriminative Score OED 每个 campaign 重训 score network。

**输入**：batch 7、8--9、10 的 split、冻结 generator、R2 seed table。

**输出**：Fig.3/4 数据；Table 1；projection loss、held-out moment RMSE、C2ST AUC、ESS、acceptance、FW gap、`lambda_min(M_hat)`。

**明确不做**：不向 acquisition 暴露 full-test 或 `beta_t^dagger`；不把 drift 当 exponential tilt 真值；不增加新数据集。

**验收与进入条件**：

1. 单元/数值：campaign pool/full-test 严格隔离；`D_A_hat` 与 Bregman 公式实现核对；C2ST 使用等量生成样本、固定 5-fold classifier；held-out 32 functions 固定。
2. 统计：每 campaign/budget >=50 paired reps；报告 projection loss、moment RMSE、C2ST AUC（越接近 0.5 越好）及 CI；四 policy paired comparison。
3. 诊断：所有 run 报 overlap ESS、conditional acceptance、FW gap、`lambda_min(M_hat)`；异常 run 单独标记，不静默删除。
4. 可复现：相同 target draw manifest 和 seed 可重建配对结果。
5. 失败定位：projection 好但 C2ST 差是 family misspecification/secondary metric；AUC 异常是 classifier split；某 campaign 独有失败是 drift overlap/数据问题。
6. 进入条件：三 campaign 的主指标、secondary metrics、diagnostics 完整并可生成最终图表。

### P11. 统一评估、绘图、表格与论文结果整理

**目标**：将全部实验转为可追溯、可复核的最终结果包。

**前置依赖**：P5、P7、P9、P10。

**实现内容**：统一结果 schema、paired seed join、CI/SE 计算、图表脚本、表格脚本、异常标记；生成四张主图和 Table 1；保留数据版本、配置、代码 commit、generator hash、环境摘要。

**输入**：各阶段 machine-readable summaries。

**输出**：`figures/fig1--fig4.*`、`results/table1.*`、论文可直接引用的 CSV/JSON、结果 manifest、limitations/待确认问题记录。

**明确不做**：不在绘图阶段改变实验数据或删除不利 run；不补充 PDF 未指定的 baseline/metric 作为主结果。

**验收与进入条件**：

1. 单元：每张图/表的输入列、分组、单位、CI、legend 自动检查；缺失 policy/budget 直接失败。
2. 数值：图表数值与原始 summary 抽样逐项一致；Fig.1 oracle line 与理论常数一致；Table 1 三 campaign 行完整。
3. 科研一致性：四 policy 同 target draws；所有主指标定义与 PDF 一致；R2 明确标注 robustness。
4. 可复现：清空派生 figures/results 后，从冻结 summaries 一条命令重建完全相同文件 hash。
5. 失败定位：图表不一致是 join/聚合；CI 错误是 replication count；理论 line 错是 oracle summary。
6. 进入条件：主图表人工目检通过，结果 manifest 可审计。

### P12. 最终复现实验与版本冻结

**目标**：在干净环境中重跑关键 smoke/full verification，冻结可交付版本。

**前置依赖**：P11；本机与云端 artifacts 已同步。

**实现内容**：从 release candidate commit 在干净环境重建；本机运行 smoke subset；云端运行正式 replication；校验所有 seed/config/checkpoint/hash；生成 final manifest、环境锁定、Git tag。

**输入**：release candidate、配置、数据/checkpoint artifact、结果 manifest。

**输出**：最终 figures/tables、复现实验日志、`REPRODUCIBILITY.md`、release tag（建议 `v0.1-iclr-stage1`）。

**明确不做**：不在冻结后修改算法、数据划分、指标或主图；新增探索必须另开 tag/分支并标为非主结果。

**验收与进入条件**：

1. 单元/数值：全测试通过；关键 summary hash 与冻结 manifest 匹配。
2. 统计：正式 replications 数量、CI、预算、paired draws、诊断字段完整。
3. 复现：干净环境至少重复一个 Synthetic S1 B 档、一个 R1 档、三个 R2 campaign smoke；固定 seed 输出一致。
4. 失败定位：环境失败不改代码先修锁定文件；结果 hash 失败查 artifact/seed；统计失败回到对应阶段，禁止直接覆盖。
5. 进入条件：全部签收后才可交付论文结果。

## 2. 本机 WSL2 方案

本机完成 P0--P3、P5 的小规模开发验收、P6 的 VAEAC smoke/小数据训练、P8 preprocessing、P9/R2 smoke、P11 绘图重建和 P12 复现抽样。Windows 文件放在 WSL 可访问的工作区（建议 `~/work/RR_GID_CN`，与 Windows 工作区通过 Git 同步；不把大数据放 `/mnt/c` 作为训练盘）。

- Python：3.11；`venv`/conda 二选一，以 `environment.yml` 和 `pyproject.toml` 为唯一依赖声明；CUDA/PyTorch 版本记录在环境 manifest。
- 资源策略：16 CPU 线程；并行 worker 默认 4--8；内存上限留出 2--3 GB；GPU batch/latent size 以 6 GB 显存为上限；所有 smoke 配置显式写入 `configs/local_*.yaml`。
- 本机小规模：Synthetic reference 2,000/500，S1 每格 3--10 reps，B={200,400,800}；VAEAC 少 epoch；Gas 只做固定子集、每格 3--5 paired reps；仅验收逻辑和数值，不宣称统计结论。
- 迁移不改代码：设备、路径、batch、replication、artifact URI 全由配置覆盖；代码只读相对路径；seed manifest 与 commit 随 run 保存。
- 本机通过标准：所有 smoke 的 unit/numerical/预算/seed/图表 schema 检查通过；GPU 与 CPU 结果在容差内一致（随机算法明确记录允许差异）。

## 3. 云端方案

云端只承载 P4、P7 正式规模、P9/R2 正式 replications 和最终 P12 full verification。无需集群：一台单 GPU/多核实例即可，replication 以进程或 job array 并行。

- 推荐：NVIDIA GPU 16--24 GB 显存（VAEAC/MLP 共用即可）、8--16 vCPU、32--64 GB RAM、至少 200 GB SSD；若只跑 oracle/评估，可用 16 vCPU CPU 实例。
- 并行：每个 replication 独立 seed、独立输出目录；同一 policy/budget/campaign 的 paired draws 由共享 manifest 分发，禁止各 job 自行抽样。
- 保存：每次 run 保存 `config_resolved.yaml`、Git commit、环境 lock/hash、seed、generator hash、日志、checkpoint、summary；checkpoint 和原始数据放对象存储/挂载盘，不入 Git。
- 迁移：本机提交代码与配置，云端 clone 同一 commit；下载登记过 hash 的数据/checkpoint；先运行 smoke，再提交正式 job；结果回传后只通过 manifest 合并。
- 可追溯：run id 由 commit+config hash+seed 组成；禁止手工改结果 CSV；所有异常 run 保留并分类。
- 云端验收：目标 replication 数量达到 PDF 要求；四 policy 配对 draws；预算/diagnostics 无缺失；随机抽取 runs 可从 checkpoint/日志重建；最终图表 hash 可重建。

## 4. 简洁项目目录

```text
RR_GID_CN/
├── RR_GID_CN.pdf
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── configs/        # 版本化实验配置、local/cloud 覆盖
├── data/           # 原始/处理数据；除 manifest 和小索引外不提交
├── src/            # oracle、RR-GID、baselines、VAEAC、评估库
├── scripts/        # 训练、实验、绘图、表格入口
├── tests/          # 单元、数值、统计 smoke tests
├── experiments/    # run manifest、seed table、非大体积元数据
├── results/        # machine-readable summaries；大文件不提交
├── figures/        # 生成图；可提交最终小图，原始中间图不提交
└── docs/           # 本规划、执行提示词、复现说明
```

不提交 Git：原始 UCI 数据、处理后大 cache、VAEAC/MLP checkpoint、完整 replication logs、临时 tensor、云端缓存。通过 `.gitignore` 忽略，并在 `data/`/`results/` 中仅提交 manifest、hash、schema 和必要的小型示例。

## 5. Git 版本管理

只保留 `main` 与短期 `work/<stage>` 分支。初始化提交为 `P0-init`；每个阶段通过验收后提交一次（例如 `P1-synthetic-oracle`、`P4-s1-verified`、`P10-r2-verified`）；实验结果必须在 summary manifest 中记录生成它的 commit。阶段一最终通过后打 tag `v0.1-iclr-stage1`。探索性改动不得覆盖已冻结 tag。

本机和云端只同步 Git 提交、配置和 manifest；使用 `git pull --ff-only`/显式 commit，不做复杂 Git Flow。大数据、checkpoint、完整结果使用对象存储或挂载盘，文件名包含 run id/hash；Git 只保存下载地址、版本和 SHA256。每次发布前检查 `git status` 干净、commit/tag 与结果 manifest 一致。

## 6. 最终验收总表

| 阶段 | 主要产出 | 必须通过的验收 | 执行位置 | 允许进入下一阶段 | Git commit/tag |
|---|---|---|---|---|---|
| P0 | 环境、目录、配置入口 | 环境/seed/path/pytest smoke | 本机 | 是 | `P0-init` |
| P1 | Synthetic exact oracle | conditional、Fisher、PSD、KL 一致 | 本机 | 是 | `P1-synthetic-oracle` |
| P2 | Uniform/A-OSQD/oracle RR-GID | FW gap、预算、oracle optimizer 对照 | 本机 | 是 | `P2-oracle-policies` |
| P3 | Formal Pilot-Design-Update | HT 无偏、cross-completion、scoring 可复现 | 本机 | 是 | `P3-rrgid-formal` |
| P4 | Synthetic S1 oracle gate | 三种 gate policy 的 `B*KL` 理论常数、J=2 预期、>=200 reps | 云端（本机 smoke） | 是 | `P4-s1-oracle-gate` |
| P5 | Discriminative Score OED + final Fig.1 | information/ranking/regret；最终四策略配对、每格 >=200 reps | 本机开发+云端正式 | 是 | `P5-s1-four-policy` |
| P6 | 冻结 VAEAC | conditional quality、acceptance、ESS | 本机 GPU + 云端正式 | 是 | `P6-vaeac-frozen` |
| P7 | Synthetic S2/Fig.2 | alpha sweep、reuse frontier、generator 不重训 | 云端 | 是 | `P7-s2-verified` |
| P8 | Gas preprocessing | 128 维、split、PC1、120 panels、hash | 本机 | 是 | `P8-gas-preprocess` |
| P9 | Gas VAEAC + R1 | 128 维 arbitrary-pair conditional；ranking、paired draws、KL/overlap | 本机 smoke+云端正式 | 是 | `P9-gas-vaeac-r1` |
| P10 | Gas R2/Fig.3/4/Table 1 | 3 campaigns、>=50 reps、三类指标与诊断 | 云端 | 是 | `P10-r2-verified` |
| P11 | 统一图表/结果包 | schema、数值逐项一致、可重建 | 本机/云端 | 是 | `P11-results-locked` |
| P12 | 最终复现与冻结 | clean env、hash、关键 smoke/full、tag | 云端+本机 | 交付 | `v0.1-iclr-stage1` |

## 7. 严格执行顺序

1. 先检查 PDF hash、工作区状态和 Python/CUDA 版本；保存检查结果与 P0 manifest。
2. 线性执行 P0→P1→P2→P3；每一步结束提交阶段 commit。任何 unit、数值、预算或复现失败，停止并只修复该阶段。
3. P3 通过后，P4 必须先以 Uniform SQD、A-OSQD、oracle RR-GID 完成 S1 理论门禁；P5 只能在该门禁通过后实现，并负责补齐最终四策略 Fig.1。P5 与 P6 的开发 smoke 可并行，但正式结果汇合前都必须有各自 commit。
4. P6 冻结后执行 P7；P8 可与 P7 的纯 Synthetic 正式 replications 并行，但 Gas 实验必须等待 P8 验收。
5. P9 必须等待 P8+P6，先训练并冻结 Gas 专用 VAEAC，再执行 R1；P10 必须等待 P9 稳定通过，并复用同一 Gas checkpoint。P4/P5/P7/P9/P10 的 replication 可按独立 seed 并行，但每个 job 开始前检查共享 target-draw manifest。
6. P11 只读取已冻结 summaries；不在此阶段改算法或补主指标。P12 从 release candidate 的干净环境重建并打 tag。
7. 每一步开始前检查：依赖 commit、配置 hash、数据/checkpoint hash、seed manifest、预算和访问权限。每一步完成后保存：日志、summary、诊断、环境信息、commit、artifact hash 和验收记录。
8. 必须停止的情况：预算超限、target/full-test 泄漏、四 policy 未配对 draws、`M` 非正定且无记录、FW gap 未达标、`B*KL`/理论常数系统偏离、ESS/acceptance 低于配置阈值未标记、结果不可由固定 seed 重建。

## 8. 待确认问题（不改变设计）

以下事项 PDF 未给出唯一工程数值，执行时必须记录决定并获得确认，不能自行改变研究设计：

1. `Theta` 的具体边界、`tau_FW`、PSD projection cutoff、各阶段 `R_A/N_g/L_I/L_U` 与 accept-reject 数值阈值。
2. VAEAC 的层数、latent size、optimizer、epoch/early stopping、mask 采样比例和 conditional quality 的具体通过线。
3. Synthetic `beta*` 的随机方向生成方式、ESS/N=0.5 的二分容差和每个 alpha 是否重新调 scale。
4. A-OSQD 的 covariance 正则化、不可观测 panel 的处理和与等成本 120 panels 的数值实现细节。
5. 真实数据的 UCI 下载镜像、文件 hash、缺失值/异常记录处理；PC1 符号对齐规则。
6. R2 的 32 个 held-out functions、C2ST classifier 架构/超参、投影 loss 数值容差。
7. 云端具体实例型号、对象存储位置、wall-time 配额，以及正式 run 的并发上限。
8. 图表配色、字体、文件格式和论文版面尺寸；不得借此改变指标或结果。
9. P4/P5 中“接近理论常数”、oracle/estimated allocation 差异、Gas VAEAC conditional quality、R1 panel-ranking 接近 oracle 的定量验收阈值；阈值必须在看正式结果前冻结。
