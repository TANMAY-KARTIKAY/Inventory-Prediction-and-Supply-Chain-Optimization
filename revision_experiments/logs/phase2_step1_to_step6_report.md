# Report: Revision Experiments Steps 1 to 6
**Status:** Completed and Verified  
**Date:** 2026-10-12  
**Working Directory:** `C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments`  
**Generated Files:**
- `outputs/tables/step1_feasibility_check.csv`
- `outputs/tables/step2_initialization_sensitivity.csv`
- `outputs/tables/step3_causality_audit.csv`
- `outputs/tables/step3_static_vs_rolling_comparison.csv`
- `outputs/tables/step4_e2_grid_search.csv`
- `outputs/tables/step4_e1_tuning_seeds.csv`
- `outputs/tables/step4_e1_summary.csv`
- `outputs/tables/step4_e3_factorial_ablation.csv`
- `outputs/tables/step5_pareto_matched_fill_rate.csv`
- `outputs/figures/step5_pareto_frontiers.png`

---

## 1. Step 1: Feasibility Check (Deliberately Generous Policy)

Prior to running optimization, a deliberately generous policy ($\alpha = 1.5, \beta = 5.0$) was simulated under the primary initialization (pipeline pre-filled with $\bar{D}_{\text{pre}}$ units/day for all 7 lead-time slots; on-hand inventory $= 2 \times \bar{D}_{\text{pre}}$) on both validation and test splits across all three forecasters.

*File: `revision_experiments/outputs/tables/step1_feasibility_check.csv`*

| Forecaster | $\alpha$ | $\beta$ | Val Fill Rate (%) | Val Cost (\$) | Test Fill Rate (%) | Test Cost (\$) | Meets 97% Feasibility Threshold? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT + Conformal** | 1.50 | 5.00 | **99.07** | 177,534.57 | **96.17** | 205,212.55 | **True** |
| **Huber TFT + Gaussian** | 1.50 | 5.00 | **98.50** | 127,594.46 | **94.51** | 164,117.59 | **True** |
| **Mean-Demand Gaussian** | 1.50 | 5.00 | **98.76** | 154,903.54 | **95.54** | 156,035.07 | **True** |

- **Validation Maximum Achievable Fill Rate:** **99.07%** (Quantile), **98.50%** (Huber), **98.76%** (Mean-Demand).
- **Feasibility Assertion:** Passed (`max_val_sl >= 97.0%`).
- **Test Fill Rate at Generous Bound:** **96.17%** (Quantile), **94.51%** (Huber), **95.54%** (Mean-Demand).

---

## 2. Step 2: Redefined Initialization & Sensitivity Analysis

The simulator was evaluated under the standardized benchmark setting ($\alpha = 1.0, \beta = 4.0$) across four policy-independent initialization variants, alongside the legacy test-tuned oracle baseline.

*File: `revision_experiments/outputs/tables/step2_initialization_sensitivity.csv`*

| Forecaster | Initialization Mode | On-Hand Multiplier | Warmup Excluded? | $\alpha$ | $\beta$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT** | **legacy (I0 = 0.7*S; includes generous starting stock)** | $0.70 \times S$ | False | 0.1000 | 3.4879 | — | — | **156,920.63** | 109,275.92 | 37,052.71 | 10,592.00 | **93.61** |
| Quantile TFT + Conformal | Primary (2x Dbar_pre, prefilled pipe) | $2.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 144,235.78 | 98.94 | **166,593.14** | 134,117.51 | 22,491.62 | 9,984.00 | **96.12** |
| Huber TFT + Gaussian | Primary (2x Dbar_pre, prefilled pipe) | $2.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 114,319.10 | 97.65 | **144,187.58** | 101,245.36 | 33,430.22 | 9,512.00 | **94.23** |
| Mean-Demand Gaussian | Primary (2x Dbar_pre, prefilled pipe) | $2.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 134,168.93 | 98.28 | **137,696.99** | 96,927.80 | 31,033.18 | 9,736.00 | **94.65** |
| Quantile TFT + Conformal | Sensitivity 1x (1x Dbar_pre, prefilled pipe) | $1.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 138,028.66 | 97.87 | **171,006.03** | 125,565.98 | 35,592.05 | 9,848.00 | **93.86** |
| Huber TFT + Gaussian | Sensitivity 1x (1x Dbar_pre, prefilled pipe) | $1.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 106,957.95 | 97.05 | **148,815.33** | 92,926.10 | 46,345.23 | 9,544.00 | **92.01** |
| Mean-Demand Gaussian | Sensitivity 1x (1x Dbar_pre, prefilled pipe) | $1.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 126,949.12 | 97.42 | **139,667.27** | 88,033.13 | 41,978.14 | 9,656.00 | **92.76** |
| Quantile TFT + Conformal | Sensitivity 3x (3x Dbar_pre, prefilled pipe) | $3.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 154,960.24 | 99.21 | **171,730.60** | 145,889.00 | 15,865.60 | 9,976.00 | **97.26** |
| Huber TFT + Gaussian | Sensitivity 3x (3x Dbar_pre, prefilled pipe) | $3.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 125,012.72 | 97.86 | **148,784.81** | 112,387.15 | 27,197.66 | 9,200.00 | **95.31** |
| Mean-Demand Gaussian | Sensitivity 3x (3x Dbar_pre, prefilled pipe) | $3.0 \times \bar{D}_{\text{pre}}$ | False | 1.0000 | 4.0000 | 143,974.90 | 98.42 | **141,993.39** | 107,829.34 | 24,740.04 | 9,424.00 | **95.73** |
| Quantile TFT + Conformal | Secondary (Warmup Excluded, days 8-15) | $2.0 \times \bar{D}_{\text{pre}}$ | True | 1.0000 | 4.0000 | 99,850.81 | 99.71 | **115,211.76** | 110,480.38 | 371.38 | 4,360.00 | **99.86** |
| Huber TFT + Gaussian | Secondary (Warmup Excluded, days 8-15) | $2.0 \times \bar{D}_{\text{pre}}$ | True | 1.0000 | 4.0000 | 70,694.12 | 97.04 | **92,990.21** | 77,608.23 | 11,309.98 | 4,072.00 | **95.66** |
| Mean-Demand Gaussian | Secondary (Warmup Excluded, days 8-15) | $2.0 \times \bar{D}_{\text{pre}}$ | True | 1.0000 | 4.0000 | 89,687.96 | 98.34 | **86,299.61** | 73,290.67 | 8,912.94 | 4,096.00 | **96.58** |

---

## 3. Step 3: Causality Audit & Static vs Rolling Policy Analysis

### Causality Audit Table
*File: `revision_experiments/outputs/tables/step3_causality_audit.csv`*

| Vector Name | Source File | Row Filter / Date Range | Rows Feeding | Causality Status | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `val_dbar_vec` | `Temporal Split(6)/train_split.csv` | `date >= '2017-06-19' and date <= '2017-07-16'` (28 trailing days) | All active pair sales in trailing 28 days | Strictly Causal (0 test-window rows) | Mean actual sales per pair over 28 days preceding validation start (`2017-07-17`). |
| `test_dbar_vec` | `train_split.csv` (Jul 04-16) + `validation_split.csv` (Jul 17-31) | `date >= '2017-07-04' and date <= '2017-07-31'` (28 trailing days) | All active pair sales in trailing 28 days | Strictly Causal (0 test-window rows) | Mean actual sales per pair over 28 days preceding test start (`2017-08-01`). Zero rows from test window. |
| `val_sigma_vec` | `Temporal Split(6)/train_split.csv` | `date >= '2017-06-19' and date <= '2017-07-16'` (28 trailing days) | All active pair sales in trailing 28 days | Strictly Causal (0 test-window rows) | Standard deviation of sales over 28 trailing days $\times \sqrt{7} + 1.0$. |
| `test_sigma_vec`| `train_split.csv` (Jul 04-16) + `validation_split.csv` (Jul 17-31) | `date >= '2017-07-04' and date <= '2017-07-31'` (28 trailing days) | All active pair sales in trailing 28 days | Strictly Causal (0 test-window rows) | Standard deviation of sales over 28 trailing days $\times \sqrt{7} + 1.0$. |
| `val_sig_res_vec`| `validation_split.csv` vs `validation_forecasts_huber.csv` | `date >= '2017-07-17' and date <= '2017-07-31'` (15 validation days) | 363 pairs $\times$ 15 days = 5,445 residuals | Strictly Causal (frozen on validation, used unchanged on test) | Residual standard deviation on validation split $(y - \hat{y}) \times \sqrt{7} + 1.0$. Zero test rows. |

### Static vs Rolling Origin Policy Comparison
In the legacy implementation, thresholds $s$ and $S$ were computed **once per pair on Day 0** and held static across the entire 15-day simulation. Using the available 9 rolling forecast origins ($t=0 \dots 8$), we constructed a rolling origin variant that updates $s_t$ and $S_t$ daily from the 7-day multi-horizon forecast issued at origin $t$.

*File: `revision_experiments/outputs/tables/step3_static_vs_rolling_comparison.csv`*

| Policy Variant | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Fill Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Static Policy ($s, S$ fixed on Day 0)** | 144,235.78 | 98.94 | **166,593.14** | 96.12 |
| **Rolling Policy ($s_t, S_t$ updated daily from origin $t$)** | 144,153.10 | 98.81 | **162,262.23** | 96.01 |

- Updating thresholds daily reduces Test Cost from **\$166,593.14 to \$162,262.23 (-\$4,330.91 / -2.6%)** at comparable service level (96.01% vs 96.12%).

---

## 4. Step 4: Rerun E2 (Grid Search), E1 (Multi-Seed Tuning), and E3 (Factorial Ablation)

### A. Rerun E1: Validation Policy Tuning (10 Seeds Each) & Test Evaluation
Tuned strictly on validation ($\alpha \in [0.0, 1.5], \beta \in [0.5, 5.0]$, 25 agents, 20 iterations = 500 evaluations per seed, 10 seeds each) under the primary initialization.

*Files: `revision_experiments/outputs/tables/step4_e1_summary.csv` and `revision_experiments/outputs/tables/step4_e1_tuning_seeds.csv`*

| Policy | Optimizer | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) | Misses 95% on Test? | Bound Hit? | Evaluation Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Quantile TFT + Conformal PSO** | **PSO (Legacy Oracle)** | **0.1000** | **3.4879** | — | — | **136,378.10** | 103,576.04 | 21,690.06 | 11,112.00 | **96.41** | **False** | $\alpha=0.10$ (at legacy lower bound) | legacy (I0 = 0.7*S; includes generous starting stock) |
| **Quantile TFT + Conformal** | PSO | 0.1499 | 1.7514 | 104,475.59 | 96.25 | **118,981.11** | 68,915.96 | 33,081.14 | 16,984.00 | **94.29** | **True** | None | fair_val_tuned (frozen on validation) |
| **Quantile TFT + Conformal** | GWO | 0.1540 | 1.7574 | 104,491.51 | 96.27 | **119,031.02** | 69,139.69 | 32,963.33 | 16,928.00 | **94.32** | **True** | None | fair_val_tuned (frozen on validation) |
| **Huber TFT + Gaussian** | PSO | 0.6221 | 1.4666 | 103,025.26 | 96.28 | **127,793.12** | 71,805.21 | 36,995.91 | 18,992.00 | **93.62** | **True** | None | fair_val_tuned (frozen on validation) |
| **Huber TFT + Gaussian** | GWO | 0.6106 | 1.4669 | 102,987.81 | 96.25 | **127,623.77** | 71,768.64 | 36,903.13 | 18,952.00 | **93.61** | **True** | None | fair_val_tuned (frozen on validation) |
| **Mean-Demand Gaussian** | PSO | 0.0000 | 1.6756 | 106,921.70 | 95.59 | **122,457.92** | 54,679.52 | 51,706.40 | 16,072.00 | **91.09** | **True** | $\alpha=0.0$ (at lower bound) | fair_val_tuned (frozen on validation) |
| **Mean-Demand Gaussian** | GWO | 0.0016 | 1.6671 | 106,868.22 | 95.60 | **122,521.08** | 54,589.65 | 51,807.43 | 16,124.00 | **91.08** | **True** | $\alpha=0.0$ (at lower bound) | fair_val_tuned (frozen on validation) |

#### E1 Findings:
1. **Quantile TFT Test Cost vs Baselines:**
   - Quantile TFT Test Cost: **\$118,981.11** (Test Fill Rate: **94.29%**).
   - Huber TFT + Gaussian Test Cost: **\$127,793.12** (Test Fill Rate: **93.62%**).
   - Mean-Demand Gaussian Test Cost: **\$122,457.92** (Test Fill Rate: **91.09%**).
   - Quantile TFT achieves **\$8,812.01 lower test cost (-6.9%)** and **0.67% higher fill rate** than Huber TFT + Gaussian.
   - Quantile TFT achieves **\$3,476.81 lower test cost (-2.8%)** and **3.20% higher fill rate** than Mean-Demand Gaussian.
2. **Service Level Target on Test:**
   - All validation-tuned policies achieve $\ge 95.5\%$ on validation, but fall slightly below 95.0% on test: Quantile reaches 94.29% (misses by 0.71%), Huber reaches 93.62% (misses by 1.38%), and Mean reaches 91.09% (misses by 3.91%).
3. **Bound Hits:**
   - Quantile TFT: $\alpha^* = 0.1499, \beta^* = 1.7514$ (Interior solution; no bound hits).
   - Huber TFT: $\alpha^* = 0.6221, \beta^* = 1.4666$ (Interior solution; no bound hits).
   - Mean-Demand: $\alpha^* = 0.0000$ (Hits lower bound $\alpha = 0.0$).

---

### B. Rerun E3: Factorial Ablation (5 Separate Buffers & Forecasters)
Evaluating 5 distinct safety stock buffers (None, Gaussian, Empirical Residual Quantile, Raw Quantile, Conformal Quantile) across forecasters.

*File: `revision_experiments/outputs/tables/step4_e3_factorial_ablation.csv`*

| Ablation Component | Forecaster | Safety Buffer | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) | Misses 95% on Test? | Bound Hit? | Evaluation Note |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Legacy Quantile PSO** | Quantile TFT | Conformal Quantile (Legacy Test Tuned) | 0.1000 | 3.4879 | — | — | **136,378.10** | 103,576.04 | 21,690.06 | 11,112.00 | **96.41** | False | $\alpha=0.10$ (at legacy lower bound) | legacy (I0 = 0.7*S) |
| **No Safety Stock ($\alpha=0$)** | Quantile TFT | none | **0.0000** | 2.1803 | 106,858.88 | 94.95 | **118,887.27** | 65,293.95 | 39,857.32 | 13,736.00 | 93.12 | True | $\alpha$ fixed to 0.0 | fair_val_tuned |
| **No Safety Stock (Huber, $\alpha=0$)** | Huber TFT | none | **0.0000** | 4.3955 | 115,987.66 | 94.24 | **131,901.08** | 83,946.08 | 39,891.01 | 8,064.00 | 93.12 | True | $\alpha$ fixed to 0.0 | fair_val_tuned |
| **Gaussian Buffer (Validation Residuals)**| Quantile TFT | gaussian_residual | 0.7991 | 1.7592 | 101,743.73 | 97.41 | **124,169.18** | 73,171.51 | 33,749.67 | 17,248.00 | 94.18 | True | None | fair_val_tuned |
| **Gaussian Buffer (Huber Residuals)** | Huber TFT | gaussian_residual | 0.6221 | 1.4666 | 103,025.26 | 96.28 | **127,793.12** | 71,805.21 | 36,995.91 | 18,992.00 | 93.62 | True | None | fair_val_tuned |
| **Empirical Residual Quantile Buffer** | Quantile TFT | empirical_residual_quantile | 0.7112 | 2.2439 | 98,887.14 | 97.57 | **123,663.21** | 74,281.93 | 35,285.28 | 14,096.00 | 93.91 | True | None | fair_val_tuned |
| **Empirical Residual Quantile (Huber)** | Huber TFT | empirical_residual_quantile | 1.0773 | 2.1796 | 97,462.09 | 98.02 | **131,456.38** | 81,502.28 | 35,338.10 | 14,616.00 | 93.90 | True | None | fair_val_tuned |
| **Raw Quantile Buffer ($P_{90}^{\text{raw}} - P_{50}$)** | Quantile TFT | raw_quantile | 0.1139 | 1.6747 | 104,506.75 | 95.95 | **118,105.59** | 66,388.83 | 34,492.77 | 17,224.00 | 94.05 | True | None | fair_val_tuned |
| **Conformal Quantile Buffer (Calibrated)** | Quantile TFT | conformal_quantile | 0.1499 | 1.7514 | 104,475.59 | 96.25 | **118,981.11** | 68,915.96 | 33,081.14 | 16,984.00 | 94.29 | True | None | fair_val_tuned |
| **Rolling Origin Conformal Buffer** | Quantile TFT | conformal_quantile (daily) | 0.1289 | 1.7642 | 107,274.39 | 95.45 | **117,224.82** | 61,017.62 | 40,743.20 | 15,464.00 | 92.97 | True | None | fair_val_tuned |

#### E3 Findings:
1. **Raw Quantile vs Conformal Quantile:**
   - Raw Quantile: Test Cost = **\$118,105.59**, Test Fill Rate = **94.05%** ($\alpha^* = 0.1139, \beta^* = 1.6747$).
   - Conformal Quantile: Test Cost = **\$118,981.11**, Test Fill Rate = **94.29%** ($\alpha^* = 0.1499, \beta^* = 1.7514$).
   - Difference: +\$875.52 cost (+0.7%) for +0.24% fill rate.
2. **Quantile Forecaster vs Huber Forecaster under Identical Buffers:**
   - Under Gaussian Residual Buffer: Quantile TFT achieves **\$124,169.18** (94.18% SL) vs Huber TFT **\$127,793.12** (93.62% SL).
   - Under Empirical Residual Quantile Buffer: Quantile TFT achieves **\$123,663.21** (93.91% SL) vs Huber TFT **\$131,456.38** (93.90% SL).
   - Under No Safety Stock ($\alpha = 0$): Quantile TFT achieves **\$118,887.27** (93.12% SL) vs Huber TFT **\$131,901.08** (93.12% SL).

---

## 5. Step 5: Pareto Frontier & Matched Fill-Rate Comparison

Over the exhaustive $(\alpha, \beta)$ grid (5,704 evaluations in `step4_e2_grid_search.csv`), the non-dominated Pareto frontier was extracted for each policy.

*Plot: `revision_experiments/outputs/figures/step5_pareto_frontiers.png`*  
*File: `revision_experiments/outputs/tables/step5_pareto_matched_fill_rate.csv`*

### Matched Service Level Comparison Table (Cheapest Configuration Reaching $\ge \text{Target SL}$ on Validation):
| Target Fill Rate (%) | Policy | Cheapest $\alpha$ | Cheapest $\beta$ | Achieved Val SL (%) | Cheapest Val Cost (\$) | Realized Test SL (%) | Realized Test Cost (\$) | Realized Test Holding (\$) | Realized Test Stockout (\$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **95.0%** | **Quantile TFT (Conformal Buffer)** | 0.10 | 2.20 | 95.99 | 104,683.58 | **94.10** | **118,217.76** | 69,938.50 | 34,223.27 |
| 95.0% | Quantile TFT (Raw Buffer) | 0.10 | 2.20 | 95.94 | 104,732.59 | 94.09 | 118,087.21 | 69,767.34 | 34,279.87 |
| 95.0% | Huber TFT + Gaussian | 0.60 | 1.50 | 96.14 | 103,514.05 | 93.61 | **127,427.78** | 71,695.84 | 37,043.93 |
| 95.0% | Mean-Demand Gaussian | 0.00 | 1.70 | 95.58 | 107,093.85 | 91.10 | **122,374.25** | 54,876.40 | 51,617.85 |
| **96.0%** | **Quantile TFT (Conformal Buffer)** | 0.15 | 2.20 | 96.37 | 104,684.21 | **94.35** | **119,292.88** | 72,256.52 | 32,772.36 |
| 96.0% | Quantile TFT (Raw Buffer) | 0.15 | 2.20 | 96.29 | 104,899.22 | 94.33 | 119,091.83 | 71,994.65 | 32,849.17 |
| 96.0% | Huber TFT + Gaussian | 0.60 | 1.50 | 96.14 | 103,514.05 | 93.61 | **127,427.78** | 71,695.84 | 37,043.93 |
| 96.0% | Mean-Demand Gaussian | 0.15 | 2.20 | 96.14 | 108,768.68 | 92.09 | **120,796.73** | 61,451.69 | 45,873.04 |
| **97.0%** | **Quantile TFT (Conformal Buffer)** | 0.25 | 2.50 | 97.05 | 106,293.58 | **94.89** | **123,153.80** | 80,102.77 | 29,651.03 |
| 97.0% | Quantile TFT (Raw Buffer) | 0.25 | 2.50 | 97.01 | 106,064.22 | 94.88 | 122,736.08 | 79,661.23 | 29,698.85 |
| 97.0% | Huber TFT + Gaussian | 0.85 | 2.20 | 97.05 | 104,373.64 | 93.88 | **132,453.34** | 82,382.92 | 35,486.43 |
| 97.0% | Mean-Demand Gaussian | 0.45 | 2.30 | 97.04 | 112,865.69 | 93.24 | **121,752.27** | 68,773.34 | 39,202.92 |
| **98.0%** | **Quantile TFT (Conformal Buffer)** | 0.50 | 2.10 | 98.00 | 111,316.98 | **95.59** | **131,147.75** | 89,827.80 | 25,583.96 |
| 98.0% | Quantile TFT (Raw Buffer) | 0.45 | 2.70 | 98.01 | 110,751.73 | 95.58 | 130,343.57 | 91,650.79 | 25,628.79 |
| 98.0% | Huber TFT + Gaussian | 1.40 | 2.20 | 98.04 | 109,705.68 | 94.07 | **142,981.29** | 93,671.85 | 34,357.44 |
| 98.0% | Mean-Demand Gaussian | 1.00 | 2.20 | 98.01 | 122,881.48 | 94.54 | **127,423.04** | 80,575.45 | 31,655.59 |

---

## 6. Step 6: Direct Reporting of Required Checkpoints

1. **Feasibility Result:**
   - The generous policy ($\alpha = 1.5, \beta = 5.0$) achieved **99.07%** validation fill rate (Quantile), **98.50%** (Huber), and **98.76%** (Mean-Demand).
   - The feasibility threshold condition ($\text{Fill Rate} \ge 97.0\%$) was met across all forecasters.
   - On the test split, the generous policy achieved **96.17%** (Quantile), **94.51%** (Huber), and **95.54%** (Mean-Demand).

2. **Bound Hit Audit:**
   - **Quantile TFT + Conformal:** Optimized parameters were $\alpha^* = 0.1499, \beta^* = 1.7514$ (PSO) and $\alpha^* = 0.1540, \beta^* = 1.7574$ (GWO). Both sit within the interior of the search space ($\alpha \in [0.0, 1.5], \beta \in [0.5, 5.0]$). Neither $\alpha$ nor $\beta$ hit a bound.
   - **Huber TFT + Gaussian:** Optimized parameters were $\alpha^* = 0.6221, \beta^* = 1.4666$ (PSO) and $\alpha^* = 0.6106, \beta^* = 1.4669$ (GWO). Both sit within the interior. Neither $\alpha$ nor $\beta$ hit a bound.
   - **Mean-Demand Gaussian:** Optimized parameters were $\alpha^* = 0.0000, \beta^* = 1.6756$ (PSO) and $\alpha^* = 0.0016, \beta^* = 1.6671$ (GWO). $\alpha^*$ hit the lower bound ($\alpha = 0.0$).
   - **Legacy Mode Oracle:** $\alpha = 0.1000$ sat at the lower bound defined in the legacy notebook.

3. **Validation-Tuned Policy Test Service Level ($\ge 95\%$):**
   - On the validation split, all validation-tuned policies met the 95% service level: Quantile TFT achieved **96.25%**, Huber TFT achieved **96.28%**, and Mean-Demand achieved **95.59%**.
   - On the test split, the one-shot evaluation of these validation-tuned policies yielded:
     - Quantile TFT + Conformal: **94.29%** (misses 95.0% by 0.71%).
     - Huber TFT + Gaussian: **93.62%** (misses 95.0% by 1.38%).
     - Mean-Demand Gaussian: **91.09%** (misses 95.0% by 3.91%).
   - When the validation tuning target was increased to 98.0% on the grid, Quantile TFT achieved **95.59%** on test (exceeding 95.0%), Huber TFT achieved **94.07%**, and Mean-Demand achieved **94.54%**.
