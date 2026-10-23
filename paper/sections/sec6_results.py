# -*- coding: utf-8 -*-
"""
Section 6: Empirical Results and Comparative Evaluations
"""

SECTION_6_TEXT = """
# 6. Empirical Results and Comparative Evaluations

In this section, we present the consolidated experimental results, hypothesis tests, and ablation analyses evaluated across all 363 active store-item pairs on the unseen test window (August 1 to August 15, 2017). All policies were tuned strictly on the validation window (July 17–31, 2017) and evaluated one-shot forward.

## 6.1 Multi-Horizon Point Forecasting Performance
Table 6.1 (Table 1 in Manuscript) evaluates multi-horizon point forecast accuracy across all 3,237 sequence evaluation windows ($3,237 \\times 7 = 22,659$ row observations).

**Table 6.1 (Table 1 in Manuscript). Test-Window Point Forecast Accuracy Benchmark**  
*(Evaluated Across All 7 Multi-Horizon Days; 22,659 Sequence Forecast Observations)*

| Forecasting Model | Architectural Paradigm | MAE (units) | RMSE (units) | **WAPE (%) [Primary]** | **Strict sMAPE (%) [0/0=0]** | Pinball $P_{50}$ Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT ($P_{50}$)** | Multi-Quantile Transformer Head | **12.4326** | 35.2266 | **28.09%** | **37.15%** | 6.2163 |
| **Huber TFT** | Single-Point Conditional Mean Head | **12.3874** | **29.5425** | **27.99%** | 38.13% | **6.1937** |
| **PatchTST Benchmark** | Channel-Independent Patch Transformer | 13.2641 | 31.8012 | 29.95% | 39.42% | N/A |
| **Trailing 28-Day Mean** | Classical Non-Parametric Moving Average | 16.6099 | 42.7669 | 37.52% | 45.06% | 8.3050 |
| **Seasonal Naive (Lag 7)** | Deterministic Day-of-Week Heuristic | 18.4068 | 43.1502 | 41.58% | 50.09% | 9.2034 |

*Note: sMAPE enforces the strict division rule where $0/0 = 0.0$. PatchTST represents an architectural benchmark trained with MSE loss across 2 channels (sales + promotion) without Optuna tuning.*

### Key Statistical Findings on Point Forecasting:
1. **Dominance Over Classical Baselines:** Both TFT models achieve substantial, statistically decisive accuracy gains over traditional supply chain methods, reducing MAE by approximately 25.1% relative to the trailing 28-day mean and 32.7% relative to Seasonal Naive. Non-parametric pair-level Wilcoxon signed-rank tests ($N = 363$) confirm statistical superiority at $p < 0.0001$ against both baselines (pair win rates: 69.4% vs. Mean, 91.7% vs. Seasonal Naive).
2. **Statistical Point Equivalence of TFTs:** Comparing Quantile TFT ($P_{50}$) against Huber TFT, the models exhibit virtually identical point accuracy: WAPE is 28.09% vs. 27.99% (a 0.10% difference), and MAE is 12.43 vs. 12.39 units. The pair-level Wilcoxon signed-rank test yields $W = 31,136.0$ ($p = 0.3430$), and paired bootstrap resamples on per-pair MAE ($B = 2,000$) yield a mean difference of $-0.2466$ units with a 95% confidence interval of $[-1.1493, +0.5831]$, which firmly spans zero. The pair win rate is 50.69% to 49.31%. **Statistically, the two TFT models possess identical point forecasting capability.**
3. **The RMSE/MAE Divergence Explained:** While MAE is identical, Huber TFT achieves a significantly lower RMSE (29.54 vs. 35.23 units). This divergence directly reflects the mathematics of loss functions: Huber loss penalizes large deviations, pulling the conditional prediction toward extreme outlier spikes, whereas the $P_{50}$ median minimizes absolute deviations, maintaining operational baseline stability while allowing the $P_{90}$ head to absorb spike risk.
4. **TFT vs. PatchTST:** PatchTST achieves a respectable 29.95% WAPE, but trails the TFT models by 1.86% to 1.96% absolute WAPE. Because PatchTST treats feature channels independently, it cannot leverage cross-series entity embeddings (store clusters, perishability) or future holiday schedules as effectively as the TFT's Variable Selection Networks.

## 6.2 Empirical Conformal Coverage Calibration
Table 6.2 (Table 2 in Manuscript) evaluates empirical coverage of the upper-tail $P_{90}$ bound before and after Mondrian split-conformal calibration, verified across all 23,093 validation and 22,659 test sequence observations with series-bootstrap 95% confidence intervals ($B = 2,000$).

**Table 6.2 (Table 2 in Manuscript). Empirical Coverage of the One-Sided $P_{90}$ Bound**  
*(Proportion of Sequence Observations Satisfying $Y \\le \\hat{y}^{P_{90}}$; 95% Series-Bootstrap Confidence Intervals)*

| Partition | Demand Stratum | Observation Count ($N$) | Raw Uncalibrated Coverage (%) | Calibrated Conformal Coverage (%) | Offset Applied | Nominal Target (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation** | **Overall Cohort** | 23,093 | 89.26% [88.76, 89.74] | **90.01% [89.53, 90.47]** | Tier-Specific | 90.00% |
| Validation | Standard Tier ($<100$) | 19,880 | 89.57% [89.06, 90.08] | **90.01% [89.51, 90.51]** | +0.1785 units | 90.00% |
| Validation | High-Volume Tier ($\ge 100$) | 3,213 | 87.36% [85.87, 88.76] | **90.04% [88.67, 91.38]** | +4.5766 units | 90.00% |
| **Test** | **Overall Cohort** | 22,659 | 90.49% [90.05, 90.93] | **91.01% [90.57, 91.43]** | Tier-Specific | 90.00% |
| Test | Standard Tier ($<100$) | 19,523 | 90.61% [90.12, 91.07] | **90.98% [90.52, 91.44]** | +0.1785 units | 90.00% |
| Test | High-Volume Tier ($\ge 100$) | 3,136 | 89.80% [88.55, 90.98] | **91.20% [90.02, 92.32]** | +4.5766 units | 90.00% |

### Key Conformal Calibration Findings:
1. **Restoration of Nominal SLA:** On validation, raw Quantile TFT under-covered volatile high-volume SKUs (87.36% coverage, CI $[85.87, 88.76]$), which would cause frequent stockouts during promotional surges. Adding the conformal offset $+4.5766$ restores validation coverage to exactly 90.04% (CI $[88.67, 91.38]$).
2. **Generalization to Test Data:** Transferred to unseen test data, calibrated coverage achieves **91.20%** on high-volume SKUs (2,860 of 3,136 observations covered) and **91.01%** overall. The one-sided bound is slightly conservative by design, fully satisfying enterprise service-level coverage requirements.
3. **Bounded Interval Sharpness:** Conformal calibration achieved nominal coverage without interval explosion: the calibrated two-sided $P_{10}$–$P_{90}$ interval covered **80.03%** of test observations (against an 80.0% nominal width) with an average interval width of $35.81$ physical units.

## 6.3 Downstream Replenishment Performance: Primary Selection Regime
Table 6.3 (Table 4 in Manuscript) summarizes downstream replenishment outcomes for the primary policy variants selected strictly by minimum validation fitness.

**Table 6.3 (Table 4 in Manuscript). Validation-Selected Replenishment Policy Performance**  
*(Evaluated on 363 Active Test Pairs; Primary Window Days 7–14, Arriving-Order Fee Rule)*

| Forecaster Model | Selected Buffer Architecture | Execution Form | Optimal $\\alpha^*$ | Optimal $\\beta^*$ | Val Fitness ($) | Val Fill (7-14) (%) | Test Cost (Prim 7-14) ($) | Test Cost (Full 0-14) ($) | Test Fill (Prim 7-14) (%) | Test Fill (Full 0-14) (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantile TFT** | `empirical_residual` | Static | 0.9286 | 1.2163 | **92,154.21** | 96.57% | **$72,759.10** | **$118,516.48** | **94.66%** | 93.78% |
| **Huber TFT** | `empirical_residual` | Static | 1.0805 | 0.8435 | 92,968.11 | 96.58% | $81,010.06 | $126,767.43 | 92.97% | 93.02% |
| **Seasonal Naive** | `empirical_residual` | Static | 0.8272 | 1.2164 | 97,208.26 | 94.54% | $69,755.20 | $115,512.57 | 92.69% | 92.90% |
| **Trailing Mean** | `empirical_residual` | Rolling | 0.7725 | 0.9256 | 94,785.63 | 96.91% | $70,806.09 | $116,563.46 | 91.87% | 92.53% |

### Key Policy Selection Findings:
1. **The Cost Divergence of Equally Accurate Models:** Despite possessing equal point accuracy, the Quantile TFT policy achieves a primary test cost of **$\\$72,759.10** compared to **$\\$81,010.06** for Huber TFT—delivering a net operating cost saving of **$\\$8,250.96** (a 10.2% cost reduction).
2. **Service Level Superiority:** Quantile TFT achieves a test fill rate of **94.66%** over days 7 to 14, outperforming Huber TFT (92.97%), Seasonal Naive (92.69%), and Trailing Mean (91.87%).
3. **Validation-to-Test Degradation:** Applying parameter bounds tuned to a 95% service constraint over days 0 to 14 experiences natural degradation on unseen test data: fill rates degraded by 1.91 percentage points for Quantile TFT, 3.61 points for Huber TFT, 1.85 points for Seasonal Naive, and 5.04 points for Trailing Mean. Quantile TFT demonstrated the highest operational resilience to unseen demand shifts.

## 6.4 Paired Bootstrap Statistical Tests and Cost Breakdown
To verify that the $\\$8,251$ cost saving achieved by Quantile TFT is statistically robust, we execute paired non-parametric bootstrap resampling ($B = 2,000$ iterations) across store-item pairs.

**Table 6.4 (Table 5 in Manuscript). Paired Bootstrap Statistical Significance Benchmark**  
*(B = 2,000 Resamples Across Store-Item Pairs; Quantile Policy minus Comparator)*

| Comparator Model | Evaluated Store-Item Cohort | Mean Cost $\\Delta$ ($) | 95% Bootstrap Confidence Interval ($) | $p$-value (Cost) | Mean Fill $\\Delta$ (%) | 95% Bootstrap Confidence Interval (%) | Pair Win Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Huber TFT** | **All 363 Active Pairs** | **-$8,250.95** | **[-$19,104.83, -$2,126.05]** | **0.0010** | **+1.69%** | [-0.70%, +5.57%] | **66.39%** |
| Huber TFT | Excl. Top 20 High-Volume ($N=343$) | **-$2,813.11** | **[-$3,862.10, -$1,798.18]** | **0.0005** | +0.01% | [-0.41%, +0.43%] | **66.18%** |
| **Seasonal Naive** | All 363 Active Pairs | +$3,003.91 | [-$2,545.25, +$10,040.93] | 0.3610 | +1.97% | [-1.51%, +4.63%] | 38.02% |
| Seasonal Naive | Excl. Top 20 High-Volume ($N=343$) | +$1,110.20 | [-$2,012.47, +$3,941.53] | 0.4370 | **+2.95%** | **[+2.07%, +3.96%]** | 37.03% |
| **Trailing Mean** | All 363 Active Pairs | +$1,953.02 | [-$2,133.14, +$5,946.68] | 0.3210 | **+2.79%** | **[+1.25%, +4.16%]** | 46.56% |
| Trailing Mean | Excl. Top 20 High-Volume ($N=343$) | +$218.08 | [-$2,447.79, +$3,066.21] | 0.8800 | **+3.24%** | **[+2.00%, +4.41%]** | 47.52% |

### Economic Cost Decomposition:
Decomposing the $\\$8,250.95$ saving achieved by Quantile TFT over Huber TFT reveals that cost reductions stem from balanced physical inventory control, not fee artifacts:
* **Holding Cost Reduction:** **-$4,400.12** (53.3% of total savings) — Quantile TFT drastically reduces unnecessary buffer stocks on non-surge days.
* **Stockout Penalty Reduction:** **-$3,248.83** (39.4% of total savings) — Quantile $P_{90}$ head absorbs surge spikes, preventing lost sales.
* **Order Fee Reduction:** **-$602.00** (7.3% of total savings) — More stable base-stock trajectories reduce reorder chatter.

Crucially, **both 95% bootstrap intervals for the Huber comparison exclude zero**, confirming statistical significance at $p = 0.0010$ across the full cohort and $p = 0.0005$ when excluding the 20 largest SKUs. Against naive heuristics, cost differences are within bootstrap noise, but Quantile TFT provides statistically significant fill rate gains (+2.79% over Trailing Mean, and +2.95% over Seasonal Naive on standard SKUs).

## 6.5 Secondary Regime: Factorial Buffer Ablation Without Fitted Residuals
Table 6.5 (Table 6 in Manuscript) evaluates the secondary operational regime, restricting safety buffers to model-derived spreads without fitting 363 pair-specific validation residual statistics.

**Table 6.5 (Table 6 in Manuscript). Secondary Regime: Replenishment Without Fitted Residuals**  
*(Evaluated on 363 Active Test Pairs; Days 7–14 Primary Evaluation Horizon)*

| Replenishment Policy | Forecaster Model | Buffer Type | Execution Form | Optimal $\\alpha^*$ | Optimal $\\beta^*$ | Test Cost ($) | Test Fill Rate (%) | Methodological Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Quantile Conformal** | Quantile TFT | `conformal_quantile` | Static | 0.1648 | 1.2172 | **$66,918.16** | **94.74%** | **Primary Proposed Policy** |
| **Quantile Raw Spread** | Quantile TFT | `raw_quantile` | Static | 0.1892 | 1.2172 | $67,157.98 | 94.99% | Uncalibrated Model Head |
| **Trailing Mean Gaussian**| Trailing Mean | `gaussian_demand` | Rolling | 0.0147 | 1.4765 | $65,345.74 | 90.59% | Classical Parametric Baseline |
| **Seasonal Naive None** | Seasonal Naive | `none` | Rolling | 0.0000 | 2.4629 | $75,417.49 | 92.77% | Heuristic Base-Stock Baseline |
| **Huber None (Bound $\\le 5$)**| Huber TFT | `none` | Rolling | 0.0000 | 4.9979 | $88,078.45 | 90.46% | Bound-Ceiling Failure |
| **Huber None (Bound $\\le 8$)**| Huber TFT | `none` | Rolling | 0.0000 | 8.0000 | $107,955.34 | 91.77% | Inflated Holding Penalty |

### Key Buffer Ablation Insights:
1. **The Bound-Hitting Failure of Huber Without Buffers:** Without a dedicated safety buffer ($\\alpha = 0$), Huber TFT cannot bridge 7-day replenishment delays. Under a 95% service penalty, the optimizer drives $\\beta^*$ to the search boundary ($\beta^* = 4.9979$). Widening the search space to $\beta \\le 8.0$ inflates holding costs to $\\$107,955.34$ on test without reaching feasibility (achieving only 91.77% fill). This proves that conditional point forecasters cannot protect against lead-time shocks through cycle stock alone.
2. **Quantile Conformal Outperforms Unbuffered Models:** Quantile Conformal achieves a test cost of **$\\$66,918.16** and **94.74% fill**, saving $\\$21,160.29$ over Huber None (95% CI $[-\\$32,245, -\\$13,114]$, $p = 0.0005$) and $\\$8,499.34$ over Seasonal Naive None ($p = 0.0050$).
3. **Conformal vs. Raw Spread:** Conformal calibration provides finite-sample coverage guarantees while operating at identical cost to the raw quantile spread ($\\$66,918$ vs. $\\$67,158$, cost delta within bootstrap noise: $-\\$239.83$, CI $[-\\$625, +\\$320]$).

## 6.6 The Non-Dominated Matched-Fill Pareto Frontier
To evaluate efficiency without relying on penalty function weights, we construct the **Non-Dominated Matched-Fill Pareto Frontier** (Figure 6.1 / `r4_matched_fill_frontier.png`). The frontier is defined as the monotone lower envelope over all grid configurations achieving a test customer fill rate at or above target $T$:
$$C^*(T) = \\min \\{ J(c) : \\text{Fill}(c) \\ge T \\}$$

**Table 6.6 (Table 7 in Manuscript). Matched-Fill Lower Envelope Cost Frontier ($C^*(T)$)**  
*(Minimum Operating Cost ($) Required to Guarantee Test Customer Fill Rate $\\ge T$)*

| Target Fill Rate ($T$) | Quantile Conformal ($) | Trailing Mean ($) | Seasonal Naive ($) | Huber Gaussian ($) | Operational Frontier Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **90.0%** | **$64,183.77** | **$63,786.58** | $70,697.54 | $69,147.68 | Mean and Quantile match within 0.6% |
| **92.0%** | **$64,183.77** | **$63,786.58** | $70,697.54 | $70,311.05 | Mean and Quantile match within 0.6% |
| **94.0%** | **$65,155.90** | $65,169.56 | $71,786.12 | $93,345.65 | Huber cost inflates (+43.3%); Quantile matches Mean |
| **95.0%** | **$69,253.95** | $70,731.65 | $82,141.57 | **Unreachable** | **Huber Infeasible; Quantile beats Naive by $12,888 (15.7%)** |
| **96.0%** | **$97,967.62** | **Unreachable** | $112,046.54 | **Unreachable** | **Mean Infeasible; Quantile beats Naive by $14,079 (12.6%)** |
| **97.0%** | **Unreachable** | **Unreachable** | **Unreachable** | **Unreachable** | System physical capacity bound |

Table 6.6 demonstrates the definitive commercial value of conformalized quantile transformers:
* At moderate service levels ($90\\%$ to $92\\%$), trailing mean baselines are marginally cheaper ($\\$63,786$ vs. $\\$64,183$, a 0.6% gap) because lower target fill rates permit frequent stockouts, favoring policies that carry minimal safety stock.
* At the enterprise standard **95.0% Fill Rate**, Huber Gaussian cannot reach feasibility, while Quantile Conformal achieves **$\\$69,253.95**, outperforming Mean Gaussian by $\\$1,477.70$ (2.1%) and Seasonal Naive by **$\\$12,887.62$ (15.7% net savings)**.
* At **96.0% Fill Rate**, Trailing Mean becomes completely infeasible. Quantile Conformal achieves **$\\$97,967.62**, beating Seasonal Naive by **$\\$14,078.92$ (12.6% net savings)**. Conformal quantile buffering expands the achievable operational frontier.

## 6.7 Metaheuristic Convergence Benchmark: PSO vs. GWO vs. 2D Grid
Table 6.7 (Table 8 in Manuscript) evaluates the multi-seed optimization benchmark across 10 random seeds on the primary fair specification (Quantile TFT + Conformal Buffer, rolling mode).

**Table 6.7 (Table 8 in Manuscript). Multi-Seed Metaheuristic Benchmark: PSO vs. GWO vs. 2D Grid**  
*(Rolling Conformal Quantile Policy; 10 Random Seeds; Mean $\\pm$ Standard Deviation)*

| Optimization Algorithm | Optimal Multiplier $\\alpha^*$ | Optimal Multiplier $\\beta^*$ | Val Fitness ($) | Test Operating Cost ($) | Test Fill Rate (%) | Iteration Convergence Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Particle Swarm Optimization (PSO)** | $0.2075 \\pm 0.0312$ | $1.1607 \\pm 0.0903$ | $\\$100,412.26 \\pm 85.83$ | **$\\$64,447.86 \\pm 345.14$** | $93.57 \\pm 0.35\\%$ | $21.4 \\pm 1.2$ s |
| **Grey Wolf Optimizer (GWO)** | $0.2274 \\pm 0.0216$ | $1.0858 \\pm 0.0868$ | $\\$100,369.60 \\pm 75.94$ | $\\$64,546.64 \\pm 265.59$ | $93.82 \\pm 0.23\\%$ | $24.8 \\pm 1.5$ s |
| **Exhaustive 2D Grid ($31 \\times 46 = 1,426$ pts)** | $0.2500$ | $0.9000$ | $\\$100,658.29$ | $\\$64,950.35$ | $94.04\\%$ | $284.2$ s |

Figure 6.2 (`pso_vs_gwo_convergence_quantile.png`) displays the iteration convergence trajectories. Both algorithms exhibit rapid cost decay, descending from uncoordinated initial exploration states to the optimal parameter valley within 5 iterations. Across all 10 seeds, PSO and GWO differ by only 0.04% in validation fitness and 0.15% in test cost. Benchmarked against the exhaustive 2D grid, both metaheuristics locate the true minimum basin within 0.8% of grid cost in less than 10% of the computation time.

## 6.8 Economic Stress Testing: 14-Scenario Cost Sensitivity Grid
To verify operational robustness across changing economic environments, we execute full re-optimization across 14 economic scenarios spanning stockout penalty variations ($c_p \\in [1.0, 10.0]$), warehouse holding multipliers ($h_{\\text{mult}} \\in [0.5, 1.5]$), and order fee multipliers ($o_{\\text{mult}} \\in [0.5, 1.5]$).

### Key Sensitivity Findings:
1. **Rank 1 Dominance in 13 of 14 Scenarios:** Quantile Conformal achieves Rank 1 (lowest operating cost) in 13 out of 14 scenarios ($c_p \\ge 2.5$, across all holding and order fee multipliers).
2. **Mechanistic Penalty Flips:** When the stockout penalty is low ($c_p = \\$1.00$), stockouts carry negligible financial impact. Mean-Demand achieves Rank 1 ($\\$50,889.41$) vs. Quantile Conformal ($\\$52,047.70$, +2.28% gap), but Mean-Demand achieves a lower customer fill rate (89.79% vs. 91.82%).
3. **Severe Cost Divergence as $c_p$ Increases:** As penalty costs rise ($c_p = \\$5.20$ to $\\$10.00$), carrying insufficient tail safety stock triggers massive financial penalties. At $c_p = \\$10.00$, Quantile Conformal achieves $\\$75,491.09$, outperforming Mean Gaussian ($\\$82,061.11$) and Huber Residual ($\\$120,791.63$). In contrast, Huber None is strictly Rank 6 in all 14 scenarios, incurring cost penalties up to +84.48% above the champion policy.
"""
