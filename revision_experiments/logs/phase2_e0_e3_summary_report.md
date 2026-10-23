# Phase 2 Summary Report: Experiments E0, E1, E2, E3
**Status:** Completed and Verified  
**Date:** 2026-10-12  
**Working Directory:** `C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments`  
**Artifacts Generated:**
- `outputs/tables/simulation_inputs.npz` (Precomputed causal matrices)
- `outputs/tables/e0_fair_mode_benchmark.csv` (Causal baseline and unit-tested simulator)
- `outputs/tables/e2_grid_search_summary.csv` (1,426 grid evaluations on validation split)
- `outputs/figures/e2_fitness_surface_validation.png` (2D contour fitness surface)
- `outputs/tables/e1_validation_tuned_seeds_detail.csv` (60 optimization runs across 10 seeds)
- `outputs/tables/e1_validation_tuned_evaluation.csv` (Validation-tuned vs test evaluation summary)
- `outputs/tables/e3_factorial_ablation.csv` (Component ablation from No-SS to Conformal Quantile)

---

## 1. Executive Summary: What the Fair Experiments Actually Show

Under the strict execution of Phase 2 with causal separation, frozen validation tuning, and 10 seeds each, we evaluated whether the proposed Quantile TFT + conformal calibration pipeline outperforms tuned classical baselines. 

### Key Empirical Findings:
1. **The Proposed Policy Does NOT Beat Tuned Baselines Under Fair Validation Tuning:**
   - In the standardized fair benchmark (15-day un-warmed simulation with causal $I_0 = 0.70 \times 7 \times \bar{D}$):
     - **Quantile TFT (PSO Tuned):** Test Cost = **\$305,965.05**, Test Fill Rate = **77.79%**
     - **Huber TFT + Gaussian (PSO Tuned):** Test Cost = **\$278,670.61**, Test Fill Rate = **76.14%**
     - **Mean-Demand Gaussian (PSO Tuned):** Test Cost = **\$270,488.58**, Test Fill Rate = **77.08%**
   - Quantile TFT incurs **$27,294.44 higher cost (+9.8%)** than Huber TFT + Gaussian, and **$35,476.47 higher cost (+13.1%)** than the simple Mean-Demand Gaussian baseline.
   
2. **Reviewer Critique Confirmed: Heuristic Agreement is NOT a Global Optimality Proof:**
   - The 2D exhaustive grid search (1,426 evaluations over $\alpha \in [0.0, 1.5], \beta \in [0.5, 5.0]$) reveals the exact topology of the validation fitness landscape.
   - PSO and GWO both converge to the same neighborhood ($\alpha^* \approx 1.46 - 1.50, \beta^* \approx 2.14$), but this agreement occurs because both optimizers are trapped by a structural constraint penalty, not because of mathematical optimality.

3. **Structural Root Cause of the Service Level Drop & Boundary Solution:**
   - In a 15-day simulation with a 7-day supplier lead time and an empty initial pipeline, **no replenishment order placed on Day 0 can arrive until Day 7**.
   - With initial inventory standardized at $I_0 = 4.9 \bar{D}$ and 7-day demand averaging $7.0 \bar{D}$, retail stores unavoidably experience stockouts on Days 5–7 regardless of ordering decisions.
   - Consequently, **the maximum mathematically achievable 15-day fill rate is bounded at ~85.6%**.
   - Because the 95% service level threshold cannot be reached over 15 un-warmed days, the quadratic penalty $50,000 \times (95 - \text{SL})^2$ remained permanently active across the entire parameter space. This drove the optimizers to push $\alpha$ to its upper bound ($\alpha^* = 1.50$) to accumulate excessive safety stock, drastically inflating holding costs to \$163,703 on test without preventing the Day 5–7 stockouts.

4. **Factorial Ablation (E3) Disproves the Conformal Quantile Headline Claim:**
   - **No Safety Stock ($\alpha = 0$):** Achieves the lowest total cost among all policies (**\$247,407.28**).
   - Adding quantile safety stock increases cost from \$247,407 to \$304,377.
   - Adding split-conformal calibration on top of raw quantiles increases cost further from \$304,377.85 to \$305,965.05 while producing the exact same fill rate (77.79%).
   - In steady state (warmup excluded), the optimal safety stock multiplier collapses to **$\alpha^* = 0.0$**, explicitly verifying the reviewer's critique that the quantile safety stock buffer is redundant or switched off.

---

## 2. Unit Testing & Causal Verification (Experiment E0)

Prior to running tuning, the modular simulator (`revision_experiments/code/e0_fair_simulator.py`) was subjected to 4 strict automated unit tests:

| Test Name | Target Condition | Result | Evidence / Metric |
| :--- | :--- | :--- | :--- |
| **Unit Test 1** | Non-Negative Inventory ($I_t \ge 0$) | **PASSED** | Minimum on-hand inventory across all 363 pairs and 15 days = `0.000000` |
| **Unit Test 2** | Cost Accounting Balance | **PASSED** | $\lvert C_{\text{tot}} - (C_{\text{hold}} + C_{\text{so}} + C_{\text{ord}}) \rvert = 0.00000000\times 10^0$ |
| **Unit Test 3** | Units-Based Fill Rate Definition | **PASSED** | $\lvert \text{SL}_{\text{sim}} - (\sum \text{Sales} / \sum \text{Demand} \times 100) \rvert = 0.00000000\times 10^0$ |
| **Unit Test 4** | Temporal Causality (Tamper Test) | **PASSED** | Injected random noise into Day 8–14 demand; decisions on Days 0–7 showed **0.00000 variance** |

### Benchmark Results from Experiment E0:
*File: `revision_experiments/outputs/tables/e0_fair_mode_benchmark.csv`*

| Policy | $\alpha$ | $\beta$ | Evaluation Mode | $\bar{D}$ Source | Total Cost (\$) | Holding (\$) | Stockout (\$) | Ordering (\$) | Fill Rate (%) |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT + Conformal PSO** | 0.1000 | 3.4879 | **oracle (uses test information)** | Realized Test (15d) | **136,378.10** | 103,576.04 | 21,690.06 | 11,112.00 | 96.41 |
| Mean-Demand Gaussian | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Trailing 28d Sales | 246,961.29 | 110,136.94 | 127,920.35 | 8,904.00 | 78.82 |
| Huber TFT + Gaussian | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Trailing 28d Sales | 248,872.74 | 112,460.01 | 127,508.73 | 8,904.00 | 78.89 |
| Quantile Default | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Trailing 28d Sales | 283,509.87 | 148,828.24 | 125,713.63 | 8,968.00 | 79.19 |
| Quantile (Legacy PSO params) | 0.1000 | 3.4879 | Fair Causal (Un-warmed) | Trailing 28d Sales | 238,498.30 | 101,004.40 | 127,845.90 | 9,648.00 | 78.84 |
| Mean-Demand Gaussian ($P_{50}$ variant) | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Forecast $P_{50}$ | 254,942.31 | 119,747.60 | 126,530.71 | 8,664.00 | 79.05 |
| Huber TFT ($P_{50}$ variant) | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Forecast $P_{50}$ | 248,913.83 | 112,988.90 | 127,292.92 | 8,632.00 | 78.93 |
| Quantile Default ($P_{50}$ variant) | 1.0000 | 4.0000 | Fair Causal (Un-warmed) | Forecast $P_{50}$ | 283,947.69 | 149,378.85 | 125,848.84 | 8,720.00 | 79.17 |
| Quantile (Legacy params, $P_{50}$) | 0.1000 | 3.4879 | Fair Causal (Un-warmed) | Forecast $P_{50}$ | 237,112.07 | 99,744.65 | 128,079.42 | 9,288.00 | 78.80 |
| Mean-Demand Gaussian (Warmup Excl.) | 1.0000 | 4.0000 | Fair Steady-State | Trailing 28d Sales | 100,501.97 | 91,471.24 | 4,094.74 | 4,936.00 | 98.59 |
| Huber TFT + Gaussian (Warmup Excl.) | 1.0000 | 4.0000 | Fair Steady-State | Trailing 28d Sales | 102,437.42 | 93,794.31 | 3,683.11 | 4,960.00 | 98.73 |
| Quantile Default (Warmup Excl.) | 1.0000 | 4.0000 | Fair Steady-State | Trailing 28d Sales | 137,050.55 | 130,162.54 | 1,888.01 | 5,000.00 | 99.35 |
| Quantile (Legacy params, Warmup Excl.)| 0.1000 | 3.4879 | Fair Steady-State | Trailing 28d Sales | **91,014.99** | 82,338.70 | 4,020.29 | 4,656.00 | 98.61 |

---

## 3. Exhaustive 2D Grid Search & Fitness Surface (Experiment E2)

To evaluate the reviewer's concern regarding the "global optimality proof," we evaluated the full 2D grid over $\alpha \in [0.0, 1.5]$ (step 0.05, 31 points) and $\beta \in [0.5, 5.0]$ (step 0.10, 46 points), totaling 1,426 discrete simulations of all 363 store-item pairs.

*Files: `revision_experiments/outputs/tables/e2_grid_search_summary.csv` and `revision_experiments/outputs/figures/e2_fitness_surface_validation.png`*

- **Validation Grid True Global Minimum:**
  - $\alpha^* = 1.5000$, $\beta^* = 2.1000$
  - Validation Cost = **\$231,617.62**, Validation Fill Rate = **85.64%**
  - Validation Fitness = \$231,617.62 + $50,000 \times (95 - 85.64)^2$ = **\$4,611,817.62**
- **One-Shot Evaluation of True Grid Minimum on Test Window:**
  - Test Cost = **\$307,537.73**
  - Test Fill Rate = **77.79%**
  - Breakdown: Holding = \$165,198.80, Stockout = \$128,738.93, Ordering = \$13,600.00

### Key Insights from the Fitness Landscape:
1. The landscape is dominated by the steep cliff of the service-level constraint penalty ($50,000 \times (95 - \text{SL})^2$).
2. Across the entire parameter space $\alpha \in [0.0, 1.5]$ and $\beta \in [0.5, 5.0]$, the validation fill rate is mathematically constrained between $82.93\%$ and $85.64\%$.
3. Because the fill rate cannot reach $95.0\%$ over 15 un-warmed days, the gradient points directly toward the upper bound of $\alpha$ ($\alpha \to 1.50$), proving why heuristic optimizers drive $\alpha$ to the boundary.

---

## 4. Multi-Seed Validation Policy Tuning & Test Benchmark (Experiment E1)

All policies were tuned strictly on the validation window (2017-07-17 to 2017-07-31) across 10 random seeds (`[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`), identical agent budget (25 agents, 20 iterations = 500 evaluations per seed), and identical parameter bounds ($\alpha \in [0.0, 1.5], \beta \in [0.5, 5.0]$).

*Files: `revision_experiments/outputs/tables/e1_validation_tuned_evaluation.csv` and `revision_experiments/outputs/tables/e1_validation_tuned_seeds_detail.csv`*

### Benchmark Summary Table:
| Policy | Optimizer | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) | Misses 95% on Test? | Evaluation Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Quantile TFT + Conformal PSO** | **PSO (Legacy Oracle)** | **0.1000** | **3.4879** | — | — | **136,378.10** | 103,576.04 | 21,690.06 | 11,112.00 | **96.41** | **False** | oracle (uses test information) |
| Quantile TFT | PSO | 1.4633 | 2.1400 | 230,179.24 | 85.64 | 305,965.05 | 163,703.27 | 128,773.78 | 13,488.00 | 77.79 | **True** | fair_val_tuned (frozen on validation) |
| Quantile TFT | GWO | 1.4594 | 2.1486 | 230,040.97 | 85.64 | 305,833.53 | 163,618.98 | 128,774.55 | 13,440.00 | 77.79 | **True** | fair_val_tuned (frozen on validation) |
| Huber TFT + Gaussian | PSO | 1.5000 | 3.2723 | 196,696.03 | 85.43 | 278,670.61 | 130,653.16 | 138,345.45 | 9,672.00 | 76.14 | **True** | fair_val_tuned (frozen on validation) |
| Huber TFT + Gaussian | GWO | 1.5000 | 3.2722 | 196,695.28 | 85.43 | 278,669.89 | 130,652.38 | 138,345.51 | 9,672.00 | 76.14 | **True** | fair_val_tuned (frozen on validation) |
| Mean-demand Gaussian | PSO | 1.5000 | 3.2260 | 219,906.56 | 85.48 | 270,488.58 | 127,829.66 | 132,842.92 | 9,816.00 | 77.08 | **True** | fair_val_tuned (frozen on validation) |
| Mean-demand Gaussian | GWO | 1.5000 | 3.2258 | 219,902.22 | 85.48 | 270,484.87 | 127,825.39 | 132,843.48 | 9,816.00 | 77.08 | **True** | fair_val_tuned (frozen on validation) |

### Key Observations:
1. **PSO vs GWO Discrepancy is Negligible:** Across all 3 policies, PSO and GWO found practically identical parameters and costs (difference $< 0.05\%$).
2. **Quantile TFT Fails to Beat Huber TFT or Mean Baseline:**
   - Test Cost: Mean Gaussian (\$270,488) $<$ Huber TFT (\$278,670) $<$ Quantile TFT (\$305,965).
   - Quantile TFT incurs substantial inventory holding penalties because the quantile spread $(P_{90} - P_{50})$ scaled by $\alpha = 1.46$ produces much higher order quantities than the Gaussian residual buffer.
3. **Every Validation-Tuned Policy Misses 95% Service Level on Test:**
   - Because the 15-day un-warmed evaluation includes 7 days of supply starvation, the achieved test fill rates are all between $76.1\%$ and $77.8\%$.

---

## 5. Factorial Ablation Study (Experiment E3)

To isolate the value of each component (Forecaster vs Safety Stock Mechanism vs Calibration), we executed the full factorial ablation.

*File: `revision_experiments/outputs/tables/e3_factorial_ablation.csv`*

### Factorial Ablation Table:
| Ablation Component | Forecaster | Safety Stock Buffer | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) | Misses 95% on Test? | Evaluation Note |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Legacy Quantile PSO** | Quantile TFT | Conformal Quantile (Test Tuned) | 0.1000 | 3.4879 | — | — | **136,378.10** | 103,576.04 | 21,690.06 | 11,112.00 | 96.41 | False | oracle (uses test information) |
| **No SS ($\alpha=0$)** | Quantile TFT | No SS ($\alpha=0$) | **0.0000** | 3.2723 | 175,077.23 | 84.79 | **247,407.28** | 99,466.17 | 138,437.12 | 9,504.00 | 76.12 | True | fair_val_tuned |
| **Gaussian SS (Mean)** | Mean-Demand Baseline | Gaussian SS (Mean) | 1.5000 | 3.2260 | 219,906.56 | 85.48 | **270,488.58** | 127,829.66 | 132,842.92 | 9,816.00 | 77.08 | True | fair_val_tuned |
| **Gaussian SS (Huber)** | Huber TFT | Gaussian SS (Huber Residual) | 1.5000 | 3.2723 | 196,696.03 | 85.43 | **278,670.61** | 130,653.16 | 138,345.45 | 9,672.00 | 76.14 | True | fair_val_tuned |
| **Quantile Empirical** | Quantile TFT | Raw Quantile Buffer (Uncalibrated) | 1.5000 | 2.0564 | 228,474.11 | 85.63 | **304,377.85** | 161,848.15 | 128,761.70 | 13,768.00 | 77.79 | True | fair_val_tuned |
| **Quantile Conformal** | Quantile TFT | Calibrated Quantile Buffer | 1.4633 | 2.1400 | 230,179.24 | 85.64 | **305,965.05** | 163,703.27 | 128,773.78 | 13,488.00 | 77.79 | True | fair_val_tuned |

### Ablation Findings:
1. **The Safety Stock Buffer Adds No Financial Advantage:**
   - Setting $\alpha = 0$ (No Safety Stock) reduces test cost to **\$247,407.28**, saving **\$58,557.77** compared to the calibrated quantile policy (\$305,965.05), with only a 1.67% difference in fill rate ($76.12\%$ vs $77.79\%$).
2. **Conformal Offsets Add No Detectable Benefit Over Raw Quantiles:**
   - Raw Quantile: Test Cost = \$304,377.85, Fill Rate = 77.79%
   - Conformal Quantile: Test Cost = \$305,965.05, Fill Rate = 77.79%
   - Calibration simply inflates holding costs by \$1,855 without improving the realized service level.

---

## 6. Steady-State Evaluation (Warmup Excluded Analysis)

To understand how the policies behave when retail operations are not artificially starved by an empty pipeline, we analyzed the steady-state regime where the 7-day lead-time pipeline is allowed to warm up:

| Policy | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Fill Rate (%) | Met 95% on Test? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT** | **0.0000** | 1.2000 | 59,929.72 | 95.99 | 82,231.32 | 94.95 | Borderline ($\approx 95\%$) |
| **Huber TFT + Gaussian** | **0.1000** | 1.2000 | 58,685.58 | 95.02 | 86,915.34 | 94.15 | Borderline |
| **Mean-Demand Gaussian** | **0.0000** | 1.2000 | 66,100.86 | 96.44 | 79,946.62 | 92.51 | Missed |

### Steady-State Insights:
- In steady state, **the optimal safety stock multiplier $\alpha^*$ collapses to 0.0 for Quantile TFT and Mean-Demand, and 0.1 for Huber TFT**.
- This directly confirms the reviewer's suspicion: **in an inventory policy where the order-up-to level $S = s + \beta \bar{D}$ already incorporates cycle stock $\beta \bar{D}$, the dedicated safety stock buffer is redundant and gets deactivated by optimization**.
- The savings claimed in the legacy paper (\$136k vs \$248k) were largely artifacts of initial inventory formulation differences ($0.70 \times S$ in legacy vs fixed baselines), not superior multi-horizon quantile forecasting.

---

## 7. Direct Answers to the Reviewer Critiques

| Reviewer Critique | Empirical Finding from Experiments E0–E3 | Implication for Journal Revision |
| :--- | :--- | :--- |
| **1. Test-set tuning invalidates reported cost.** | Confirmed. When tuned strictly on validation, test cost increases from \$136,378 to \$305,965 (un-warmed) or \$82,231 (warmed). The legacy \$136,378 was an oracle result. | Must report validation-tuned numbers as the primary benchmark and label legacy results as test-tuned oracle. |
| **2. Two heuristics agreeing is not a global proof.** | Confirmed. The E2 grid search maps the actual 2D surface. PSO and GWO agreed because the boundary penalty dominated the landscape. | Replace the "optimality proof" claim with an empirical fitness-surface analysis and explain why metaheuristics were tested. |
| **3. $\alpha$ sits on its lower bound in legacy.** | Confirmed. In steady-state validation tuning, $\alpha^*$ hits 0.0000 (lower bound). In un-warmed validation tuning, $\alpha^*$ hits 1.5000 (upper bound). | Acknowledge that the optimal policy minimizes or turns off the safety buffer in favor of cycle coverage. |
| **4. Quantile safety buffer adds no value.** | Confirmed by E3 factorial ablation. No-SS ($\alpha=0$) yields the lowest cost (\$247k), outperforming calibrated quantiles (\$306k). Conformal calibration adds \$1.8k in cost with zero gain in fill rate. | Refocus the paper contribution from "novel conformal safety stock" to a rigorous benchmarking of multi-horizon forecasting in retail inventory control. |
| **5. Savings trade service for cost.** | Confirmed. In un-warmed fair mode, fill rates fall from 96.4% to ~77% due to pipeline starvation, and holding costs trade directly off against stockouts. | Present Pareto trade-offs (E5) and stockout penalty sensitivities (E6). |

---

## 8. Checkpoint & Next Steps

In accordance with Hard Rule 5 and Condition 5 of the Phase 2 approval:
- **E0, E1, E2, and E3 are complete, logged, and audited.**
- **We pause here for your review and explicit approval before running Experiments E4 through E9.**
