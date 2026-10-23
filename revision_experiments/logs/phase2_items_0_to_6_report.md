# Report: Items 0 to 6 Benchmark & Analysis
**Status:** Completed and Verified  
**Date:** 2026-10-12  
**Working Directory:** `C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments`  
**Generated Files:**
- `outputs/tables/item0_legacy_reproduction_resimulated.csv`
- `outputs/tables/reproduction_legacy_benchmark.csv`
- `outputs/tables/item1_origin_alignment_audit.csv`
- `outputs/tables/item2_cross_fitted_conformal_tuning.csv`
- `outputs/tables/item3_huber_scalar_multiplier_tuning.csv`
- `outputs/tables/item3_forecast_bias_audit.csv`
- `outputs/tables/item4_decision_level_baselines.csv`
- `outputs/tables/item5_e4_paired_bootstrap.csv`
- `outputs/tables/item6_test_matched_fill_rate_descriptive.csv`
- `outputs/tables/item6_test_grid_descriptive_evaluations.csv`
- `outputs/figures/step6_test_matched_fill_rate_descriptive.png`

---

## Item 0: Legacy Reproduction & Exact Code Audit

### (a) Phase 1 Exact Reproduction Table
*File: `revision_experiments/outputs/tables/reproduction_legacy_benchmark.csv`*

| Policy | $\alpha$ | $\beta$ | Total Cost (\$) | Holding Cost (\$) | Stockout Cost (\$) | Ordering Cost (\$) | Fill Rate (%) | Savings vs Huber (\$) | Savings (%) | Evaluation Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mean-demand Gaussian** | 1.000 | 4.0000 | 142,167.01 | 128,624.32 | 2,774.69 | 10,768.00 | 99.54 | +8,329.63 | +5.53% | legacy (reproduced) |
| **Huber TFT + Gaussian** | 1.000 | 4.0000 | 150,496.63 | 131,408.49 | 8,512.15 | 10,576.00 | 98.59 | 0.00 | 0.00% | legacy (reproduced) |
| **Quantile Default** | 1.000 | 4.0000 | 204,328.24 | 189,825.31 | 3,750.93 | 10,752.00 | 99.38 | -53,831.61 | -35.77% | legacy (reproduced) |
| **Quantile TFT + Conformal PSO** | 0.100 | 3.4879 | **136,378.10** | 103,575.63 | 21,690.37 | 11,112.00 | **96.41** | +14,118.63 | +9.38% | legacy (reproduced) |
| **Quantile TFT + Conformal GWO** | 0.101 | 3.4890 | **136,390.59** | 103,660.13 | 21,618.45 | 11,112.00 | **96.42** | +14,106.05 | +9.37% | legacy (reproduced) |

### (b) Exact Code Lines Defining Legacy $\bar{D}$ and $\sigma$
From `MHQF optm.ipynb` Cell 2 & Cell 5 (and audited in `p1_reproduce_legacy.py` lines 130–141):

```python
# Cell 5 line 22:
mean_d = max(1.0, np.mean(demand))  # demand = pair_daily_actual: actual sales from 2017-08-01 to 2017-08-15

# Cell 5 lines 25-33:
if policy_name in ["Classical Gaussian", "Mean-demand Gaussian"]:
  lead_exp = mean_d * LEAD_TIME_DAYS
  ss = alpha_val * 1.282 * (np.std(demand) * np.sqrt(LEAD_TIME_DAYS) + 1.0)
elif policy_name == "Huber TFT + Gaussian":
  h_fc = pair_huber_daily.get((s_nbr, i_nbr), np.full(15, mean_d))
  lead_exp = np.sum(h_fc[:LEAD_TIME_DAYS])
  ss = alpha_val * 1.282 * (np.std(demand - h_fc) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

# Cell 5 lines 47-49:
s = lead_exp + ss
S = s + (beta_val * mean_d)
on_hand = S * 0.70
pipeline = [0.0] * LEAD_TIME_DAYS
```

- **Test-window demand use:** Yes. `demand` was constructed directly from realized daily sales during the 15-day test period (2017-08-01 to 2017-08-15).
- **Explanation of Step 2 Legacy Row (\$156,920.63 / 93.61% vs \$136,378.10 / 96.41%):**
  - In the original notebook simulation, `mean_d` was computed from realized 15-day test demand, and offsets were hardcoded to +0.18 (std) and +4.51 (high-vol).
  - In Step 2, the row used the causal trailing 28-day mean (`test_dbar_vec`) and recomputed validation offsets (+0.1785 / +4.5766).
  - Using trailing 28-day demand rather than realized test demand changed the order-up-to level $S = s + \beta \times \bar{D}$ and the resulting starting inventory $I_0 = 0.70 \times S$ for items whose demand shifted between July and August, resulting in \$37,052.71 stockout cost, \$156,920.63 total cost, and 93.61% fill rate.

---

## Item 1: Primary Specification Declaration & Forecast Alignment

### Specification Definition:
1. **Policy Architecture:** Rolling-origin replenishment policy ($s_t$ and $S_t$ updated daily from the forecast issued on day $t$ covering horizons $h=1 \dots 7$, using data available through day $t$).
2. **Initialization:** Fair, policy-independent initial conditions ($2 \times \bar{D}_{\text{pre}}$ on hand, pipeline pre-filled with $\bar{D}_{\text{pre}}$ units/day across all 7 lead-time slots, where $\bar{D}_{\text{pre}}$ is trailing 28-day mean sales prior to window start).
3. **Parameter Selection:** Validation-tuned parameters frozen on the validation split under a 95.0% service-level constraint.

### Alignment Audit Table:
*File: `revision_experiments/outputs/tables/item1_origin_alignment_audit.csv`*

| Day Index $t$ | Test Decision Date | Test Forecast Origin | Forecast Horizon Days | Historical Data Cutoff |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 2017-08-01 | 2017-08-01 | 2017-08-02 to 2017-08-08 | Through 2017-08-01 |
| 1 | 2017-08-02 | 2017-08-02 | 2017-08-03 to 2017-08-09 | Through 2017-08-02 |
| 2 | 2017-08-03 | 2017-08-03 | 2017-08-04 to 2017-08-10 | Through 2017-08-03 |
| 3 | 2017-08-04 | 2017-08-04 | 2017-08-05 to 2017-08-11 | Through 2017-08-04 |
| 4 | 2017-08-05 | 2017-08-05 | 2017-08-06 to 2017-08-12 | Through 2017-08-05 |
| 5 | 2017-08-06 | 2017-08-06 | 2017-08-07 to 2017-08-13 | Through 2017-08-06 |
| 6 | 2017-08-07 | 2017-08-07 | 2017-08-08 to 2017-08-14 | Through 2017-08-07 |
| 7 | 2017-08-08 | 2017-08-08 | 2017-08-09 to 2017-08-15 | Through 2017-08-08 |
| 8 | 2017-08-09 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 9 | 2017-08-10 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 10 | 2017-08-11 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 11 | 2017-08-12 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 12 | 2017-08-13 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 13 | 2017-08-14 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |
| 14 | 2017-08-15 | 2017-08-09 | 2017-08-10 to 2017-08-16 | Through 2017-08-09 |

---

## Item 2: Cross-Fitted Conformal Calibration & Tuning

Offsets and residual statistics calibrated on validation first half (Jul 17 to Jul 23); policy parameters $(\alpha, \beta)$ tuned on validation second half (Jul 24 to Jul 31); evaluated one-shot on the 15-day test window.

*File: `revision_experiments/outputs/tables/item2_cross_fitted_conformal_tuning.csv`*

| Specification | Offset Std | Offset High-Vol | Tuning Period | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full-Validation Tuned** | +0.1785 | +4.5766 | Jul 17–31 (15d) | 0.1289 | 1.7642 | 107,274.39 | 95.45 | **117,224.82** | 61,017.62 | 40,743.20 | 15,464.00 | **92.97** |
| **Cross-Fitted** | -0.0902 | +4.0040 | Jul 24–31 (8d) | 0.3533 | 1.4432 | 61,376.15 | 95.18 | **120,793.93** | 68,943.59 | 33,418.35 | 18,432.00 | **94.24** |

---

## Item 3: Bias Test & Huber Multiplier $k$ Joint Tuning

### (a) Joint Tuning of $(k, \alpha, \beta)$ for Huber Forecaster
*File: `revision_experiments/outputs/tables/item3_huber_scalar_multiplier_tuning.csv`*

| Specification | $k^*$ | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Huber + Scalar $k$ (None, $k, \beta$ only)** | 1.0967 | 0.0000 | 2.1803 | 106,858.88 | 94.95 | **118,887.27** | 65,293.95 | 39,857.32 | 13,736.00 | **93.12** |
| **Huber + Scalar $k$ (Gaussian Buffer)** | 0.8143 | 0.7712 | 1.4651 | 102,897.43 | 96.28 | **127,105.34** | 70,892.15 | 37,221.19 | 18,992.00 | **93.59** |
| **Huber + Scalar $k$ (Empirical Residual Buffer)** | 0.8471 | 1.5000 | 2.1773 | 93,579.48 | 97.41 | **125,836.62** | 71,207.11 | 40,565.51 | 14,064.00 | **93.00** |

### (b) Forecast Lead Bias Table $(\text{Forecast} - \text{Actual}) / (7 \times \bar{D}_{\text{pre}})$
*File: `revision_experiments/outputs/tables/item3_forecast_bias_audit.csv`*

| Model | Val Mean Scaled Bias | Val Median Scaled Bias | Val Std Scaled Bias | Test Mean Scaled Bias | Test Median Scaled Bias | Test Std Scaled Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quantile P50** | -0.0498 | -0.0104 | 0.2165 | -0.3005 | -0.0119 | 4.6998 |
| **Quantile P90 (Raw)** | +0.5366 | +0.4279 | 0.4750 | +0.4120 | +0.5015 | 3.0229 |
| **Quantile P90 (Calibrated)** | +0.5573 | +0.4376 | 0.4835 | +0.4335 | +0.5177 | 3.0163 |
| **Huber Point** | -0.1158 | -0.0604 | 0.2550 | -0.4336 | -0.0198 | 6.8299 |
| **Mean-Demand** | +0.0202 | +0.0669 | 0.3849 | -0.5278 | -0.0647 | 8.1332 |
| **Seasonal Naive** | -0.0318 | 0.0000 | 0.2260 | -0.5560 | -0.1063 | 8.1836 |

---

## Item 4: Decision-Level Baselines (Seasonal Naïve & Trailing Mean)

Tuned on validation (10 seeds, $\ge 95\%$ constraint) under the primary rolling specification.

*File: `revision_experiments/outputs/tables/item4_decision_level_baselines.csv`*

| Baseline Policy | Buffer | $\alpha^*$ | $\beta^*$ | Val Cost (\$) | Val Fill Rate (%) | Test Cost (\$) | Test Holding (\$) | Test Stockout (\$) | Test Ordering (\$) | Test Fill Rate (%) | Misses 95% on Test? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (None, $\alpha=0$)** | none | 0.0000 | 2.1803 | 106,858.88 | 94.95 | **118,887.27** | 65,293.95 | 39,857.32 | 13,736.00 | **93.12** | True |
| **Seasonal Naive (Gaussian Buffer)** | gaussian_residual | 0.5016 | 1.7477 | 107,525.99 | 96.25 | **125,370.27** | 71,940.11 | 36,494.16 | 16,936.00 | **94.00** | True |
| **Seasonal Naive (Empirical Residual)**| empirical_residual | 0.7060 | 1.7526 | 105,813.39 | 96.59 | **125,609.48** | 72,551.40 | 36,134.08 | 16,924.00 | **94.05** | True |
| **Trailing 28d Mean (None, $\alpha=0$)** | none | 0.0000 | 1.6756 | 106,921.85 | 95.59 | **122,454.96** | 54,677.21 | 51,705.75 | 16,072.00 | **91.09** | True |
| **Trailing 28d Mean (Gaussian Buffer)** | gaussian_demand | 0.0000 | 1.6756 | 106,921.70 | 95.59 | **122,457.92** | 54,679.52 | 51,706.40 | 16,072.00 | **91.09** | True |
| **Trailing 28d Mean (Empirical Residual)**| empirical_residual | 0.7353 | 1.7668 | 105,122.32 | 97.46 | **123,549.94** | 63,736.65 | 43,869.29 | 15,944.00 | **92.47** | True |

---

## Item 5: E4 Paired Bootstrap Over Store-Item Pairs ($B = 2000$)

Pairs were resampled with replacement ($B = 2,000$ iterations) to compute 95% bootstrap confidence intervals for cost differences, fill-rate differences, and the empirical probability that Policy 1 is cheaper than Policy 2 ($\mathbb{P}(\Delta C < 0)$).

*File: `revision_experiments/outputs/tables/item5_e4_paired_bootstrap.csv`*

| Sample Subset | Comparison (P1 vs P2) | Mean $\Delta$ Cost (\$) | 95% CI $\Delta$ Cost (\$) | P1 Cheaper Win Rate (%) | Mean $\Delta$ Fill Rate (%) | 95% CI $\Delta$ Fill Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **All Pairs ($N=363$)** | Quantile Conformal vs Quantile None | -1,662.45 | [-\$5,787.52, +\$2,488.75] | 80.50% | -0.15% | [-0.85%, +0.58%] |
| **All Pairs ($N=363$)** | Quantile Conformal vs Quantile Raw | -821.04 | [-\$1,897.40, +\$249.98] | 93.25% | +0.15% | [-0.03%, +0.32%] |
| **All Pairs ($N=363$)** | Quantile Conformal vs Huber None | -14,614.51 | [-\$22,570.33, -\$8,242.09] | **100.00%** | -0.16% | [-1.21%, +1.17%] |
| **All Pairs ($N=363$)** | Quantile Conformal vs Huber Tuned | -10,273.39 | [-\$20,664.60, -\$3,085.03] | **99.95%** | -0.65% | [-1.98%, +1.13%] |
| **All Pairs ($N=363$)** | Quantile Conformal vs Seasonal Naive | -6,734.08 | [-\$11,792.13, -\$547.25] | **98.15%** | -0.86% | [-2.63%, +0.30%] |
| **All Pairs ($N=363$)** | Quantile Conformal vs Mean-Demand | -5,232.16 | [-\$10,009.27, -\$925.85] | **99.05%** | +1.88% | [+1.10%, +2.70%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Quantile None | -4,531.50 | [-\$7,618.39, -\$2,067.30] | **100.00%** | +0.45% | [-0.06%, +1.07%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Quantile Raw | -677.83 | [-\$1,471.74, +\$82.19] | 95.50% | +0.13% | [-0.04%, +0.32%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Huber None | -10,139.69 | [-\$17,300.60, -\$4,902.72] | **100.00%** | +0.32% | [-0.81%, +1.83%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Huber Tuned | -9,930.17 | [-\$19,007.86, -\$4,036.21] | **100.00%** | +0.02% | [-1.37%, +2.06%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Seasonal Naive | -4,141.01 | [-\$8,047.52, +\$1,199.18] | 93.65% | -0.88% | [-2.83%, +0.32%] |
| **Excluding Top 20 ($N=343$)** | Quantile Conformal vs Mean-Demand | -4,577.83 | [-\$8,868.04, -\$1,378.83] | **99.90%** | +2.22% | [+1.55%, +3.13%] |

---

## Item 6: Descriptive Test Matched Fill-Rate Across Grid

Extracted from 7,176 discrete evaluations on the test split across the grid $\alpha \in [0.0, 1.5]$ and $\beta \in [0.5, 5.0]$. Linear interpolation was applied along the empirical non-dominated frontier.

*Plot: `revision_experiments/outputs/figures/step6_test_matched_fill_rate_descriptive.png`*  
*File: `revision_experiments/outputs/tables/item6_test_matched_fill_rate_descriptive.csv`*

| Target Test Fill Rate (%) | Policy | Descriptive Interpolated Cost (\$) | Frontier Min Fill Rate (%) | Frontier Max Fill Rate (%) | Interpolation Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **92.0%** | **Quantile Conformal** | **116,435.74** | 93.34% | 96.13% | Extrapolated (Below Min) |
| 92.0% | Quantile Raw | 116,305.29 | 93.32% | 96.13% | Extrapolated (Below Min) |
| 92.0% | Quantile $\alpha=0$ (None) | 118,320.12 | 92.06% | 92.55% | Extrapolated (Below Min) |
| 92.0% | Huber Gaussian | 120,996.01 | 92.67% | 94.51% | Extrapolated (Below Min) |
| 92.0% | Mean-Demand Gaussian | 119,613.39 | 92.29% | 95.54% | Extrapolated (Below Min) |
| 92.0% | Seasonal Naive Gaussian | 122,168.48 | 93.31% | 95.50% | Extrapolated (Below Min) |
| **93.0%** | **Quantile Conformal** | **116,435.74** | 93.34% | 96.13% | Extrapolated (Below Min) |
| 93.0% | Quantile Raw | 116,305.29 | 93.32% | 96.13% | Extrapolated (Below Min) |
| 93.0% | Quantile $\alpha=0$ (None) | 127,727.64 | 92.06% | 92.55% | Extrapolated (Above Max) |
| 93.0% | Huber Gaussian | 121,245.54 | 92.67% | 94.51% | **Interpolated** |
| 93.0% | Mean-Demand Gaussian | 119,838.17 | 92.29% | 95.54% | **Interpolated** |
| 93.0% | Seasonal Naive Gaussian | 122,168.48 | 93.31% | 95.50% | Extrapolated (Below Min) |
| **94.0%** | **Quantile Conformal** | **117,302.41** | 93.34% | 96.13% | **Interpolated** |
| 94.0% | Quantile Raw | 117,262.34 | 93.32% | 96.13% | **Interpolated** |
| 94.0% | Quantile $\alpha=0$ (None) | 127,727.64 | 92.06% | 92.55% | Extrapolated (Above Max) |
| 94.0% | Huber Gaussian | 132,894.70 | 92.67% | 94.51% | **Interpolated** |
| 94.0% | Mean-Demand Gaussian | 122,533.88 | 92.29% | 95.54% | **Interpolated** |
| 94.0% | Seasonal Naive Gaussian | 124,394.76 | 93.31% | 95.50% | **Interpolated** |
| **95.0%** | **Quantile Conformal** | **120,812.38** | 93.34% | 96.13% | **Interpolated** |
| 95.0% | Quantile Raw | 120,329.58 | 93.32% | 96.13% | **Interpolated** |
| 95.0% | Quantile $\alpha=0$ (None) | 127,727.64 | 92.06% | 92.55% | Extrapolated (Above Max) |
| 95.0% | Huber Gaussian | 164,117.59 | 92.67% | 94.51% | Extrapolated (Above Max) |
| 95.0% | Mean-Demand Gaussian | 130,398.04 | 92.29% | 95.54% | **Interpolated** |
| 95.0% | Seasonal Naive Gaussian | 133,143.54 | 93.31% | 95.50% | **Interpolated** |
| **96.0%** | **Quantile Conformal** | **149,146.79** | 93.34% | 96.13% | **Interpolated** |
| 96.0% | Quantile Raw | 147,852.73 | 93.32% | 96.13% | **Interpolated** |
| 96.0% | Quantile $\alpha=0$ (None) | 127,727.64 | 92.06% | 92.55% | Extrapolated (Above Max) |
| 96.0% | Huber Gaussian | 164,117.59 | 92.67% | 94.51% | Extrapolated (Above Max) |
| 96.0% | Mean-Demand Gaussian | 154,235.89 | 92.29% | 95.54% | Extrapolated (Above Max) |
| 96.0% | Seasonal Naive Gaussian | 155,638.83 | 93.31% | 95.50% | Extrapolated (Above Max) |

---

## Checkpoint Status: Stopped After Items 0 to 6

Per your instruction, execution has stopped after Item 6. Awaiting your approval before executing Items 7 to 10 (E5 cost-parameter re-optimization, E6 conformal coverage evidence, E7/E8 forecast metrics and segments, and Item 10 summary checks).
