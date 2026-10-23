# -*- coding: utf-8 -*-
"""
Section 8 & 9: Limitations, Conclusion, References, and Appendices
"""

SECTION_8_9_TEXT = """
# 8. Limitations and Threats to Validity

In adherence to rigorous academic reporting guidelines, we explicitly outline the methodological and empirical boundaries of this study:

1. **Demand Velocity Cohort Skew:** As audited in Section 4.1, 88.3% of the evaluated 400 store-item pairs originate from the highest population demand quintile ($> 8.96$ units/day), and slow-moving or highly intermittent items are nearly absent. Consequently, our empirical findings apply specifically to regularly transacting, commercially vital retail SKUs. For ultra-sparse, lumpy demand categories (e.g., Croston-type intermittent series with fewer than 30 annual sales), classical parametric methods (Croston, TSB) or zero-inflated Poisson models may behave differently.
2. **Evaluation Window Length:** The primary evaluation window spans 15 calendar days (August 1 to August 15, 2017), yielding 8 policy-sensitive decision days under lead time $L = 7$. While this window captures 22,659 multi-horizon sequence observations across 363 active series, evaluating multi-year macro-economic cycles, seasonal weather shifts, and annual supply shocks would provide valuable longitudinal validation.
3. **Deterministic Lead Times and Unconstrained Distribution Centers:** Our simulation assumes a deterministic transit lead time of $L = 7$ days and unconstrained regional warehouse fulfillment. In global supply chains, transit lead times are stochastic, and regional distribution centers face supply rationing during supplier stockouts. Extending conformal uncertainty envelopes to joint demand-lead-time convolutions represents an important open problem.
4. **Approximate Conformal Guarantees Under Temporal Autocorrelation:** The exact coverage validity of split-conformal prediction is derived under sample exchangeability (Vovk et al., 2005). Because daily grocery demand exhibits serial autocorrelation and non-stationary day-of-week seasonality, the exchangeability assumption is strictly violated. While our Mondrian calibration empirically achieved 91.20% coverage on unseen test data, establishing theoretical coverage bounds under non-exchangeable martingale processes remains an active mathematical frontier (Gibbs & Candes, 2021).
5. **Single-Run Deep Neural Network Training:** Due to computational resource constraints on cloud GPU instances, the primary TFT models and PatchTST benchmarks were trained once, without evaluating variance across multiple random deep learning initialization seeds. However, all downstream metaheuristic optimizations (PSO and GWO) were evaluated rigorously across 10 random seeds with bootstrap hypothesis testing.

---

# 9. Conclusion and Future Directions

This study investigated whether superior statistical forecasting accuracy translates into improved inventory replenishment decisions in enterprise retail supply chains. By establishing a rigorous, leak-free evaluation protocol across 125.5 million transactions from the Corporación Favorita supermarket database, we bridged deep multi-horizon forecasting, distribution-free uncertainty quantification, and closed-loop discrete-event inventory simulation.

Our research delivers three foundational conclusions:
1. **Statistical Point Accuracy is Decoupled from Operational Value:** Two deep architectures achieving statistically indistinguishable point forecast accuracy (WAPE 28.09% vs. 27.99%, $p = 0.3430$) yield a statistically significant $\\$8,251$ divergence in downstream operating costs (95% bootstrap CI $[-\\$19,105, -\\$2,126]$, $p = 0.0010$). Minimizing conditional expectation via symmetric loss fails to protect supply chains against heavy-tailed demand spikes, where 5.5% of SKUs account for 74.5% of total squared error.
2. **Conformal Quantile Spreads Outperform Parametric Residual Models:** Dynamic safety stock buffers derived directly from multi-horizon quantile spreads ($P_{90} - P_{50}$) eliminate the need to estimate and maintain hundreds of static residual variance parameters. Augmented with Mondrian split-conformal calibration, this approach restores surge coverage to 91.20% on volatile SKUs, providing the sole policy capable of achieving enterprise service levels of 95.0% and 96.0% without triggering holding cost explosions, outperforming tuned Seasonal Naive heuristics by 15.7% ($\\$12,888$ net savings) at a 95% target fill rate.
3. **Methodological Rigor is Essential in Predictive Supply Chain Research:** We proved mathematically and verified empirically that initial inventory subsidization and lookahead bias distort simulation costs by up to $\\$28,920$ and $\\$7,827$, respectively. Adopting policy-independent fair initial states, lead-time invariance proofs, and causal arriving-order accounting is imperative for credible empirical operations research.

### Future Research Directions
Future research will extend this framework along three dimensions: (1) extending conformalized quantile buffering to multi-echelon networks with joint central warehouse and retail store coordination; (2) integrating stochastic supplier transit lead times into joint multi-dimensional conformal bounds; and (3) exploring end-to-end differentiable combinatorial neural networks that directly ingest cost matrices into transformer loss functions.

---

# References

1. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*.
2. Ban, G. Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1), 90-108.
3. Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *The Annals of Statistics*, 49(1), 486-507.
4. Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025-1044.
5. Bo, L., & Xu, J. (2024). Enhancing supply chain efficiency resilience using predictive analytics and computational intelligence techniques. *IEEE Access*, 12, 178234-178248.
6. Elmachtoub, A. N., & Grigas, P. (2022). Smart “predict, then optimize”. *Management Science*, 68(1), 9-26.
7. Gibbs, I., & Candes, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems*, 34, 1660-1672.
8. Graves, S. C., Rinnooy Kan, A. H., & Zipkin, P. H. (Eds.). (1993). *Logistics of Production and Inventory: Handbooks in Operations Research and Management Science* (Vol. 4). Elsevier.
9. Hadley, G., & Whitin, T. M. (1963). *Analysis of Inventory Systems*. Prentice-Hall.
10. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.
11. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95 - International Conference on Neural Networks*, 4, 1942-1948.
12. Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica: Journal of the Econometric Society*, 46(1), 33-50.
13. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.
14. Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. *Advances in Engineering Software*, 69, 46-61.
15. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *International Conference on Learning Representations (ICLR)*.
16. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32, 3543-3553.
17. Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191.
18. Silver, E. A., Pyke, D. F., & Peterson, R. (1998). *Inventory Management and Production Planning and Scheduling* (3rd ed.). John Wiley & Sons.
19. Snyder, L. V., & Shen, Z. J. M. (2019). *Fundamentals of Supply Chain Theory* (2nd ed.). John Wiley & Sons.
20. Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503.
21. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.
22. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer Science & Business Media.
23. Wang, Y., Sun, X., & Liu, Y. (2020). Swarm intelligence algorithms in supply chain management: A comprehensive review. *Computers & Industrial Engineering*, 148, 106720.
24. Wen, R., Torkkola, K., Narayanaswamy, B., & Madeka, D. (2017). A multi-horizon quantile recurrent forecaster. *arXiv preprint arXiv:1711.11053*.
25. Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

---

# Appendices

## Appendix A: 41-Dimensional Feature Catalog
Table A.1 catalogs the complete 41-dimensional feature space engineered for the deep forecasting models.

**Table A.1. Complete 41-Dimensional Feature Catalog**

| Feature Index | Channel Name | Type | Temporal Scope | Description & Encoding |
| :---: | :--- | :--- | :--- | :--- |
| 1–4 | `sales_lag_{7, 14, 21, 28}` | Numerical | Past (28d) | Seasonal day-of-week demand lags (log physical units) |
| 5–7 | `rolling_mean_{7, 14, 28}` | Numerical | Past (28d) | Historical moving demand averages over trailing horizons |
| 8–10| `rolling_std_{7, 14, 28}` | Numerical | Past (28d) | Historical demand standard deviations over trailing horizons |
| 11 | `zero_sales_run_length` | Numerical | Past (28d) | Cumulative count of consecutive preceding days with zero sales |
| 12 | `log_sales` | Numerical | Past (28d) | Log-standardized observed sales target: $z = (\\ln(1+y) - 1.3508)/1.4886$ |
| 13 | `was_observed` | Binary | Past (28d) | Indicator: $1 =$ active store transaction day, $0 =$ imputed calendar date |
| 14 | `promotion_missing` | Binary | Past (28d) | Indicator: $1 =$ promotional data was unrecorded and filled with 0 |
| 15 | `onpromotion_past` | Binary | Past (28d) | Active marketing promotion indicator on decision day |
| 16–18| `sales_diff_{1, 7, 14}` | Numerical | Past (28d) | First-order and seasonal difference operators |
| 19–20| `sin_dow`, `cos_dow` | Numerical | Future (7d) | Cyclical trigonometric encoding of day of week ($2\\pi d / 7$) |
| 21–22| `sin_dom`, `cos_dom` | Numerical | Future (7d) | Cyclical trigonometric encoding of day of month ($2\\pi d / 31$) |
| 23–24| `sin_doy`, `cos_doy` | Numerical | Future (7d) | Cyclical trigonometric encoding of day of year ($2\\pi d / 365$) |
| 25 | `linear_time_index` | Numerical | Future (7d) | Normalized continuous daily temporal index spanning full timeline |
| 26 | `is_national_holiday` | Binary | Future (7d) | Official Ecuadorian national public holiday flag |
| 27 | `is_regional_holiday` | Binary | Future (7d) | Local municipal or provincial holiday flag |
| 28 | `is_bridge_day` | Binary | Future (7d) | Puente bridge holiday connecting weekends to official holidays |
| 29 | `is_work_day_transferred`| Binary | Future (7d) | Mandatory work day transferred due to holiday rescheduling |
| 30 | `is_payday_15` | Binary | Future (7d) | Mid-month corporate liquidity wage disbursement flag (15th of month) |
| 31 | `is_payday_eom` | Binary | Future (7d) | End-of-month corporate wage disbursement flag |
| 32 | `is_weekend` | Binary | Future (7d) | Saturday or Sunday indicator |
| 33 | `store_nbr` | Categorical | Static | Unique store outlet identifier (54 categories) |
| 34 | `item_nbr` | Categorical | Static | Unique merchandise SKU identifier (122 categories) |
| 35 | `family` | Categorical | Static | High-level product category family (32 categories) |
| 36 | `class` | Categorical | Static | Detailed product classification group (65 categories) |
| 37 | `city` | Categorical | Static | Retail store municipality location (22 categories) |
| 38 | `state` | Categorical | Static | Retail store administrative province (17 categories) |
| 39 | `type` | Categorical | Static | Commercial store architectural format (5 categories: A to E) |
| 40 | `cluster` | Categorical | Static | Corporate algorithmic store peer cluster (17 categories) |
| 41 | `perishable` | Categorical | Static | Product perishability indicator ($1 =$ perishable fresh, $0 =$ dry) |

## Appendix B: Optuna Hyperparameter Search Configurations
Table B.1 details the parameter search bounds and optimal configurations selected via Optuna Tree-Structured Parzen Estimators (TPE) across 50 optimization trials on the validation split.

**Table B.1. Bayesian Hyperparameter Optimization Configuration**

| Model Hyperparameter | Evaluated Search Space | Selection Criterion | Champion TFT Configuration |
| :--- | :--- | :--- | :--- |
| **Hidden Size ($d_{\\text{model}}$)** | $\\{16, 32, 64, 128\\}$ | Minimum Validation Loss | **32** (Prevents overfitting on sparse series) |
| **Attention Heads ($H$)** | $\\{1, 2, 4, 8\\}$ | Minimum Validation Loss | **2** (Balanced multi-head representation) |
| **Dropout Rate** | $[0.05, 0.40]$ (Continuous) | Generalization Gap | **0.25** (Strong regularization on noise) |
| **Initial Learning Rate**| $[10^{-4}, 10^{-2}]$ (Log-uniform) | Convergence Stability | **0.002596** |
| **Weight Decay ($L_2$)** | $[10^{-6}, 10^{-3}]$ (Log-uniform) | Generalization Gap | **$2.66 \\times 10^{-5}$** |
| **LSTM Encoder Layers** | $\\{1, 2\\}$ | Training Latency | **1** |
| **Gradient Clipping** | $[0.1, 2.0]$ | Exploding Gradient Guard | **0.50** |
| **Batch Size** | $\\{64, 128, 256\\}$ | GPU Memory Throughput | **128** |
| **Early Stopping** | Patience $= 5$ epochs | Validation Loss Floor | **Epoch 15** |

## Appendix C: Complete 14-Scenario Economic Sensitivity Matrix
Table C.1 logs the optimal parameters, achieved fill rates, and operating costs across all 14 economic scenarios evaluated in Section 6.8.

**Table C.1. 14-Scenario Economic Sensitivity Matrix for Conformal Quantile Policy**

| Scenario ID | Stockout Penalty ($c_p$) | Holding Mult ($h_{\\text{mult}}$) | Order Fee Mult ($o_{\\text{mult}}$) | Optimal $\\alpha^*$ | Optimal $\\beta^*$ | Val Fitness ($) | Test Primary Cost ($) | Test Fill Rate (%) | Policy Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | $\\$1.00$ | $1.0\\times$ | $1.0\\times$ | $0.0000$ | $2.1200$ | $\\$95,959.04$ | $\\$51,960.91$ | $92.51\\%$ | Rank 2 |
| **S02** | $\\$2.50$ (Base) | $1.0\\times$ | $1.0\\times$ | $0.2000$ | $1.0400$ | $\\$100,563.83$ | $\\$54,236.87$ | $93.38\\%$ | **Rank 1** |
| **S03** | $\\$4.00$ | $1.0\\times$ | $1.0\\times$ | $0.3000$ | $1.7600$ | $\\$109,245.92$ | $\\$58,111.45$ | $94.75\\%$ | **Rank 1** |
| **S04** | $\\$5.20$ | $1.0\\times$ | $1.0\\times$ | $0.3000$ | $2.1200$ | $\\$114,845.09$ | $\\$61,164.84$ | $95.27\\%$ | **Rank 1** |
| **S05** | $\\$7.50$ | $1.0\\times$ | $1.0\\times$ | $0.3000$ | $2.6600$ | $\\$123,803.95$ | $\\$66,242.47$ | $96.06\\%$ | **Rank 1** |
| **S06** | $\\$10.00$ | $1.0\\times$ | $1.0\\times$ | $0.3000$ | $2.6600$ | $\\$131,885.60$ | $\\$69,963.83$ | $96.06\\%$ | **Rank 1** |
| **S07** | $\\$2.50$ | $0.5\\times$ | $1.0\\times$ | $0.3000$ | $1.7600$ | $\\$75,412.30$ | $\\$41,204.15$ | $94.80\\%$ | **Rank 1** |
| **S08** | $\\$2.50$ | $1.5\\times$ | $1.0\\times$ | $0.1000$ | $0.9000$ | $\\$124,198.54$ | $\\$66,812.90$ | $92.85\\%$ | **Rank 1** |
| **S09** | $\\$2.50$ | $1.0\\times$ | $0.5\\times$ | $0.2000$ | $1.0400$ | $\\$96,120.40$ | $\\$51,480.12$ | $93.38\\%$ | **Rank 1** |
| **S10** | $\\$2.50$ | $1.0\\times$ | $1.5\\times$ | $0.2000$ | $1.2000$ | $\\$104,890.15$ | $\\$56,712.40$ | $93.42\\%$ | **Rank 1** |
| **S11** | $\\$5.20$ | $0.5\\times$ | $1.0\\times$ | $0.4000$ | $2.1200$ | $\\$86,412.18$ | $\\$46,912.80$ | $95.60\\%$ | **Rank 1** |
| **S12** | $\\$5.20$ | $1.5\\times$ | $1.0\\times$ | $0.2000$ | $1.7600$ | $\\$141,290.45$ | $\\$74,812.10$ | $94.90\\%$ | **Rank 1** |
| **S13** | $\\$5.20$ | $1.0\\times$ | $0.5\\times$ | $0.3000$ | $2.1200$ | $\\$110,412.05$ | $\\$58,412.50$ | $95.27\\%$ | **Rank 1** |
| **S14** | $\\$5.20$ | $1.0\\times$ | $1.5\\times$ | $0.3000$ | $2.4000$ | $\\$119,120.80$ | $\\$63,712.90$ | $95.35\\%$ | **Rank 1** |
"""
