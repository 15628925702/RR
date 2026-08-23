# 论文数据表（P11/P12 生成）

所有表格从冻结的正式实验结果（`results/`）生成，数值为 mean ± SE（标准差/√n）。
CSV 供脚本/表格软件使用，LaTeX（`*.tex`）供论文直接粘贴，JSON 为机器可读版。

## 表格清单

### Table 1 (S1)：Synthetic S1 四策略 B·KL
- 文件：`Table_1_S1.csv` / `Table_1_S1.tex`
- 内容：B ∈ {2000,4000,8000,16000,32000}，四策略（Uniform SQD、A-OSQD、
  Discriminative Score OED、learned RR-GID）的 `B·KL(Q_{β*}\|Q_β̂)`，n=200 reps。
- 关键结论：learned RR-GID 与 Discriminative 远优于线性基线（B8000: 180/188 vs
  Uniform 1234 / A-OSQD 4463）；learned RR-GID 普遍略优于 Discriminative。
- 注意：四策略 B·KL 差 10-50 倍，论文配图用 log 轴。

### Table 2 (S2 alpha)：S2 nonlinearity sweep
- 文件：`Table_2_S2_alpha.csv` / `Table_2_S2_alpha.tex`
- 内容：α ∈ {0, 0.5, 1.0, 1.5}（B=8000），四策略 design ratio + learned op error。
- 关键结论：A-OSQD 随 α 增大偏离（2.02→5.60），learned RR-GID 更稳定。

### Table 3 (S2 reuse)：generator reuse frontier
- 文件：`Table_3_S2_reuse.csv` / `Table_3_S2_reuse.tex`
- 内容：T ∈ {1,5,20,50} 的 mean design regret 与 cumulative compute。
- 关键结论：RR-GID regret 更低（1.69-1.83 vs 2.16-2.21）且 compute 便宜 15-35 倍。

### Table 4 (R1)：Gas semi-synthetic budget curves
- 文件：`Table_4_R1.csv` / `Table_4_R1.tex`
- 内容：B ∈ {400,800,1600,3200}，四策略 B·KL（relative to empirical base），n=200 reps。

### Table 5 (R2)：natural-drift per-campaign metrics（论文 Table 1）
- 文件：`Table_5_R2.csv` / `Table_5_R2.tex`
- 内容：三 campaign（batch7、batches89、batch10）× 四策略的 projection loss、
  held-out moment RMSE、C2ST AUC、importance ESS。
- 关键结论：RR-GID 在 batch7 (0.85) / batches89 (0.96) 上 ESS 最高，A-OSQD 最差
  （0.31-0.77）；projection loss / RMSE / C2ST 对四策略相同（β_dag 为 evaluation-only，
  差异主要体现在 ESS）。

## 生成方式

```bash
python scripts/make_paper_tables.py
```
