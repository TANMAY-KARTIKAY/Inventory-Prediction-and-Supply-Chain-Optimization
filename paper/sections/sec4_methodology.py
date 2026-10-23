# -*- coding: utf-8 -*-
"""
Section 4: Methodology: The Conformalized Forecast-then-Optimize Framework
"""

SECTION_4_TEXT = """
# 4. Methodology: The Conformalized Forecast-then-Optimize Framework

In this section, we present the end-to-end predictive and optimization architecture. Figure 4.1 (`sysarchi.png`) illustrates the modular data flow, connecting raw transactional data ingestion, multi-modal feature engineering, deep multi-horizon forecasting, split-conformal uncertainty calibration, discrete replenishment simulation, and swarm intelligence policy optimization.

```
+----------------------------------------------------------------------------------------------------+
|                                  SYSTEM ARCHITECTURE OVERVIEW                                      |
|                                                                                                    |
|  [Corporacion Favorita Data]                                                                       |
|  125,497,040 Transactions (54 Stores, 33 Families)                                                 |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Sampling & Preprocessing]                                                                        |
|  PriorityScore Selection → 400 Cohort (363 Active Test Pairs)                                      |
|  Continuous Daily Reindexing (was_observed, promotion_missing flags)                               |
|  Log-Standardized Target: z = (ln(1 + y) - 1.3508) / 1.4886                                        |
|         │                                                                                          |
|         ▼                                                                                          |
|  [41-Dimensional Feature Space & Memory-Mapped Tensors]                                            |
|  18 Past Covariates (28d) | 14 Known Future Covariates (7d) | 9 Static Embeddings                  |
|  Train: 624,451 sequences | Validation: 3,299 sequences | Test: 3,237 sequences                    |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Deep Time-Series Forecasting Stage]                                                              |
|  ├─► Multi-Horizon Quantile TFT (230k params, Pinball Loss: P10, P50, P90) ──► WAPE: 28.09%         |
|  ├─► Huber TFT (228k params, Huber delta=1.0)                             ──► WAPE: 27.99%         |
|  └─► Channel-Independent PatchTST Benchmark (Patch 7, Stride 7, MSE)       ──► WAPE: 29.95%         |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Uncertainty Quantification: Split-Conformal Calibration]                                         |
|  Mondrian Tiered Calibration on Validation Nonconformity Scores:                                   |
|  Standard Tier (<100 units/day): +0.1785 units | High-Volume Tier (≥100 units/day): +4.5766 units  |
|  Test High-Volume Coverage restored from 89.80% to 91.20% (Finite-Sample SLA Guarantee)            |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Dynamic Replenishment & Safety Stock Formulation]                                                |
|  Reorder Point: s_t = sum(P50) + SS_t(alpha) | Base Stock: S_t = s_t + beta * D_pre                |
|  Dynamic Conformal Buffer: SS_t = alpha * sum( max(0, P90_calibrated - P50) )                      |
|         │                                                                                          |
|         ▼                                                                                          |
|  [Closed-Loop Policy Optimization (PSO vs. GWO vs. 2D Grid)]                                       |
|  Tuned on Validation (Jul 17-31) under 95% Fill Rate Penalty Barrier:                              |
|  Fitness = Cost_val + 50000 * max(0, 95.0 - SL_val)^2                                              |
|  Evaluated One-Shot Forward on Test Window (Aug 1-15, 2017)                                        |
+----------------------------------------------------------------------------------------------------+
```

## 4.1 Data Pipeline, Sampling Strategy, and Representativeness Audit
The empirical foundation of this study is the open-access *Corporación Favorita Grocery Sales* dataset, comprising 125,497,040 daily transaction records from Ecuador's leading national grocery supermarket chain between January 1, 2013, and August 15, 2017. The raw database captures transactions across 54 commercial stores, 4,100 unique item SKUs, and 33 product families, supplemented by geographic store attributes, daily crude oil prices (West Texas Intermediate), and national/regional holiday calendars.

### A. Candidate Pair Filtering
Training deep sequence transformers on the complete cartesian product of stores and items ($54 \\times 4,100 = 221,400$ series) is both computationally prohibitive and operationally redundant, as a large majority of pairs represent discontinued items, sparse seasonal items, or newly opened retail outlets. To ensure operational relevance for multi-horizon inventory optimization, candidate store-item pairs were required to possess at least 180 non-zero sales transactions prior to the July 16, 2017 training cutoff date, yielding an eligible universe of $N_{\\text{pop}} = 138,951$ active pairs.

### B. Subsampling Pipeline and Selection Scoring
To extract a robust, computationally tractable sample, candidate pairs were ranked using a multi-criteria optimization priority score:
$$\\text{PriorityScore}_i = 0.40 \\cdot \\text{Coverage}_i + 0.35 \\cdot \\text{HistoryLength}_i + 0.25 \\cdot \\ln(1 + \\text{TotalSales}_i) + \\text{ScarcityBonus}_i$$
where $\\text{Coverage}_i$ denotes the proportion of active transaction days, $\\text{HistoryLength}_i$ measures uninterrupted calendar tenure, and $\\text{ScarcityBonus}_i$ penalizes over-represented store-family clusters, enforcing caps of approximately 14 pairs per store and 25 pairs per product family.

The resulting primary cohort comprises 400 store-item pairs spanning 53 retail outlets, 122 unique item SKUs, 32 product families, and 65 merchandise classes. During the August 1–15, 2017 test evaluation window, exactly 363 pairs remained active; the remaining 37 pairs experienced permanent commercial discontinuation or zero customer demand, and were logged in our audit ledger.

### C. Honest Population Representativeness Disclosure
In the interest of scholarly integrity, we disclose that this 400-pair cohort is not a uniform random sample of the general supermarket population. Scanning all 125.5 million rows across the 138,951 eligible population pairs reveals the following distributional skews:
1. **Demand Velocity Concentration:** Quintile partitioning of the full population based on mean daily sales yields quintile cutoffs at $[2.24, 3.30, 5.00, 8.96]$ units/day. In our 400-pair sample, 353 pairs (88.25%) originate from Tier 5 (Very High Demand, $> 8.96$ units/day), with only 2 pairs in Tier 1, 6 in Tier 2, 17 in Tier 3, and 22 in Tier 4. Within the active 363 test cohort, 322 pairs (88.71%) reside in Tier 5. This concentration was intentional: dynamic $(s, S)$ inventory policies and multi-day lead-time buffers require consistent sales velocity to justify automated continuous review.
2. **Perishability Oversampling:** Across the full population, perishable goods account for 23.23% (32,282 pairs). In our 400-pair sample, perishable goods were deliberately oversampled to 51.00% (204 perishable vs. 196 dry pairs; 175 perishable vs. 188 dry in the active test set). This deliberate oversampling ensures sufficient statistical power to measure how asymmetric holding rates ($h_{\\text{perish}} = \\$0.15$ vs. $h_{\\text{dry}} = \\$0.10$) penalize over-buffering.

## 4.2 Data Preprocessing and 41-Dimensional Feature Space
Raw transactional logs contain missing dates, inventory stockouts recorded as zero sales, and unrecorded promotional attributes. We implement rigorous calendar standardization:

1. **Continuous Daily Calendar Re-Indexing:** For each pair, a continuous timeline was synthesized spanning exactly 1,688 days (January 1, 2013 to August 15, 2017). Unobserved calendar dates were imputed with zero sales and flagged via a binary indicator `was_observed = 0`.
2. **Return Clipping and Promotion Imputation:** Negative sales entries (customer returns) were clipped to zero. Missing promotional flags were filled with zero and tracked via a binary indicator `promotion_missing = 1`. To maintain strict causality, **promotions were treated strictly as past observed covariates and were excluded from future known inputs**, reflecting enterprise environments where supplier promotions are subject to cancellation.
3. **Log-Transform Target Normalization:** Given extreme right-skewness, customer sales $y$ were transformed using standard variance-stabilizing log-scaling:
   $$z_{i,t} = \\frac{\\ln(1 + y_{i,t}) - \\mu_{\\text{train}}}{\\sigma_{\\text{train}}}$$
   where parameters were fit strictly on training-set observations: $\\mu_{\\text{train}} = 1.3508$ and $\\sigma_{\\text{train}} = 1.4886$. Inverse transformations back to physical units were applied prior to inventory cost calculations.

### The 41-Dimensional Feature Architecture
The resulting input tensor comprises 41 heterogeneous feature channels partitioned into three architectural categories:
* **18 Past Observed Covariates (28-Day Lookback):** Log-sales, `was_observed`, `promotion_missing`, past promotion intensity, historical rolling demand means (7, 14, 28 days), rolling demand standard deviations (7, 14, 28 days), day-of-week demand lags (7, 14, 21, 28 days), and historical zero-sales run-lengths.
* **14 Known Future Covariates (7-Day Forecast Horizon):** Cyclical temporal encodings (sine and cosine transformations of day-of-week, day-of-month, and day-of-year), a normalized linear temporal counter, binary national holiday flags, regional holiday flags, bridge day indicators, official work-day transfers, and bi-monthly payday indicators (15th and end of month).
* **9 Static Categorical Entity Embeddings:** Dense vector representations of store ID (54 categories), item ID (122 categories), product family (32 categories), merchandise class (65 categories), city (22 categories), state (17 categories), store type (5 categories), geographic cluster (17 categories), and binary perishability flag.

## 4.3 Zero-Copy Memory-Mapped Sequence Generation
Sliding sequences were generated using a fixed lookback window of 28 consecutive days ($[T-27, T]$) and a forward multi-horizon forecast window of 7 consecutive days ($[T+1, T+7]$), yielding a total sequence length of 35 days with a daily sliding stride of 1.

Chronological splits were enforced strictly to eliminate data leakage:
* **Training Partition:** January 1, 2013 to July 16, 2017 ($624,451$ sequences across 400 pairs; 1,658 training days).
* **Validation Partition:** July 17, 2017 to July 31, 2017 ($3,299$ sequences across 367 pairs).
* **Test Partition:** August 1, 2017 to August 15, 2017 ($3,237$ sequences across 363 active pairs; evaluating $3,237 \\times 7 = 22,659$ row observations).

To eliminate RAM bottlenecks during GPU training, sequence tensors were serialized into NumPy memory-mapped containers (`np.memmap` mode). Virtual memory paging allowed multi-worker PyTorch DataLoaders to ingest batches directly from disk page caches, achieving a throughput exceeding 26,000 sequences/second on an NVIDIA Tesla T4 GPU and reducing training time to under 38 seconds per epoch.

## 4.4 Deep Time-Series Architectures: TFT vs. PatchTST
We formulate and evaluate three deep learning architectures:

### A. The Temporal Fusion Transformer (TFT) Architecture
Both TFT variants share an identical structural configuration selected via Bayesian optimization (Optuna TPE) on the Huber loss model:
* Model dimension: $d_{\\text{model}} = 32$
* Multi-head self-attention: 2 attention heads
* Sequence processing: Single-layer LSTM encoder-decoder
* Regularization: Dropout rate $0.25$, weight decay $2.66 \\times 10^{-5}$
* Optimization: Adam optimizer with initial learning rate $\\eta = 0.002596$, cosine learning rate scheduling, and early stopping.

Static entity embeddings are processed through Variable Selection Networks (VSNs) to produce context vectors that condition both the historical LSTM encoder and the multi-horizon decoder. Gated Residual Networks (GRNs) provide adaptive non-linear feature gating, allowing the model to bypass inactive feature channels.

### B. Loss Function Formulations: Huber vs. Multi-Quantile Pinball Loss
1. **Huber TFT (Single-Point Champion):** Trained with 228,503 trainable parameters to predict the conditional mean $\\mathbb{E}[Y \\mid X]$ via Huber loss ($\\delta = 1.0$):
   $$\\mathcal{L}_{\\text{Huber}}(y, \\hat{y}) = \\begin{cases} \\frac{1}{2}(y - \\hat{y})^2, & \\text{if } |y - \\hat{y}| \\le \\delta \\\\ \\delta |y - \\hat{y}| - \\frac{1}{2}\\delta^2, & \\text{otherwise} \\end{cases}$$
2. **Quantile TFT (Multi-Quantile Champion):** Trained with 230,135 trainable parameters to predict the conditional quantiles $\\mathbf{\\hat{y}}_t = [\\hat{y}_{t, 0.10}, \\hat{y}_{t, 0.50}, \\hat{y}_{t, 0.90}]^T$ simultaneously across all 7 horizons using the multi-quantile Pinball loss:
   $$\\mathcal{L}_{\\text{Pinball}}(y, \\mathbf{\\hat{y}}) = \\frac{1}{3} \\sum_{q \\in \\{0.1, 0.5, 0.9\\}} \\frac{1}{7} \\sum_{h=1}^7 \\max\\left(q(y_{t+h} - \\hat{y}_{t+h, q}), (1-q)(\\hat{y}_{t+h, q} - y_{t+h})\\right)$$

### C. Architectural Baseline: Channel-Independent PatchTST
To benchmark against modern patching architectures, we train a channel-independent PatchTST (Nie et al., 2023). Following standard configurations, PatchTST divides continuous sequences into patches of length 7 with a stride of 7, deploying a 3-layer transformer with MSE loss over sales and promotional channels. Because PatchTST treats feature channels independently, it cannot natively fuse high-cardinality static embeddings or future calendar schedules, serving as an architectural reference.

## 4.5 The Heavy-Tail Error Concentration Diagnostic
The motivation for pivoting from single-point Huber estimation to multi-quantile forecasting arose from an empirical diagnostic. Across initial validation evaluations, the Champion Huber TFT model exhibited an RMSE of 29.542 units against an MAE of 12.387 units—yielding an RMSE-to-MAE ratio of approximately **2.39x**. Under well-behaved Gaussian error distributions, this ratio is analytically bounded near $\\sqrt{\\pi / 2} \\approx 1.25$.

To diagnose this discrepancy, we conducted an 8-point error concentration audit on validation residuals:
1. **Error Sorting:** Sorting sequence observations by descending squared error revealed that **the top 20 store-item pairs accounted for 74.5% of total squared forecast error**, despite comprising only 5.5% of active series.
2. **Spike Asymmetry:** Examining error residuals across promotional and payday cycles revealed severe right-skewed spikes: on quiet baseline days, point errors were small (MAE $\\approx 4.2$), but during payday promotional surges, realized sales spiked up to 10x baseline volume.
3. **The Compromise Dilemma:** When trained under $L_2$ or Huber loss, single-point models attempt to minimize squared deviations by over-predicting baseline demand. This inflates working capital holding costs on 95% of normal days without providing sufficient protection during extreme surges.

Quantile regression resolves this dilemma through functional specialization:
* **$P_{50}$ (Median Head):** Optimizing Pinball loss at $q=0.50$ is mathematically equivalent to minimizing Mean Absolute Error ($L_1$ loss). The median forecast is theoretically immune to extreme outliers, allowing $P_{50}$ to trace baseline demand with record operational precision (WAPE 28.09%).
* **$P_{90}$ (Surge Ceiling Head):** Penalizing under-predictions 9x more heavily than over-predictions ($0.9$ vs. $0.1$) forces the $P_{90}$ trajectory to trace peak promotional surge demand, isolating the safety stock buffer directly from the model's output spread ($P_{90} - P_{50}$).
* **$P_{10}$ (Demand Floor Head):** Traces baseline absorption, preventing over-ordering on slow-moving perishable items.

## 4.6 Mondrian Split-Conformal Calibration
While Quantile TFT models tail uncertainty effectively, empirical evaluation reveals that uncalibrated neural networks suffer from localized under-coverage. On the validation partition, the raw $P_{90}$ trajectory achieved an empirical coverage of only **87.36%** on high-volume SKUs ($\ge 100$ units/day), violating the nominal 90.0% coverage requirement.

To establish exact finite-sample coverage guarantees, we apply **Mondrian Split-Conformal Calibration** (Angelopoulos & Bates, 2021). The calibration space is partitioned into two distinct volume strata: Standard Volume ($< 100$ units/day) and High-Volume ($\ge 100$ units/day).

For each validation sequence observation $j \\in \\{1, \\dots, n_k\\}$ in stratum $k$, we evaluate the signed nonconformity score:
$$s_{j,k} = y_{j,k} - \\hat{y}_{j,k}^{P_{90}}$$

The conformal calibration offset $\\hat{c}_k$ is defined as the empirical quantile of nonconformity scores at the finite-sample adjusted rank:
$$p_k = \\frac{\\lceil (n_k + 1) \\times 0.90 \\rceil}{n_k}$$
$$\\hat{c}_k = \\text{Quantile}\\left(\\{s_{j,k}\\}_{j=1}^{n_k}, p_k\\right)$$

Evaluating across $n_{\\text{std}} = 19,880$ and $n_{\\text{hvol}} = 3,213$ validation sequence observations yields:
* **Standard Tier Offset:** $\\hat{c}_{\\text{std}} = +0.1785$ physical units
* **High-Volume Tier Offset:** $\\hat{c}_{\\text{hvol}} = +4.5766$ physical units

The calibrated upper-tail bound deployed in the downstream inventory policy is:
$$\\hat{y}_{i, t+h}^{P_{90}, \\text{cal}} = \\hat{y}_{i, t+h}^{P_{90}} + \\hat{c}_{\\text{tier}(i)}$$

Under exchangeability, this calibration provides the finite-sample marginal guarantee $\\mathbb{P}(Y \\le \\hat{y}^{P_{90}, \\text{cal}}) \\ge 0.90$. While temporal autocorrelation introduces mild serial dependence, Section 6 confirms that this formulation achieves 91.20% empirical test coverage.

## 4.7 Factorial Safety Stock Buffer Architectures
To isolate the operational mechanism responsible for inventory cost reductions, we formulate and factorially ablate six distinct safety stock architectures $\\text{SS}_{i,t}(\\alpha)$:

1. **`none` (Zero Safety Buffer):**
   $$\\text{SS}_{i,t} = 0$$
   Represents a pure base-stock policy relying entirely on order-up-to multiplier $\\beta$.
2. **`gaussian_demand` (Historical Demand Variability):**
   $$\\text{SS}_{i,t} = \\alpha \\cdot 1.282 \\cdot \\hat{\\sigma}_{i, \\text{demand}} \\sqrt{L}$$
   where $\\hat{\\sigma}_{i, \\text{demand}}$ is the standard deviation of customer demand over the preceding 28 days, and $1.282$ represents the standard normal 90th percentile $z_{0.90}$.
3. **`raw_quantile` (Uncalibrated Multi-Horizon Quantile Spread):**
   $$\\text{SS}_{i,t} = \\alpha \\sum_{h=1}^L \\max\\left(0, \\hat{y}_{i, t+h}^{P_{90}} - \\hat{y}_{i, t+h}^{P_{50}}\\right)$$
   Derives dynamic safety stock directly from the raw neural network output spread without conformal offsets or fitted residual statistics.
4. **`conformal_quantile` (Calibrated Multi-Horizon Quantile Spread):**
   $$\\text{SS}_{i,t} = \\alpha \\sum_{h=1}^L \\max\\left(0, \\hat{y}_{i, t+h}^{P_{90}, \\text{cal}} - \\hat{y}_{i, t+h}^{P_{50}}\\right)$$
   Combines multi-horizon quantile spreads with validation-anchored conformal offsets $\\hat{c}_{\\text{tier}}$.
5. **`gaussian_residual` (Fitted Forecast Residual Variance):**
   $$\\text{SS}_{i,t} = \\alpha \\cdot 1.282 \\cdot \\hat{\\sigma}_{i, \\text{res}} \\sqrt{L}$$
   where $\\hat{\\sigma}_{i, \\text{res}}$ represents the pair-specific standard deviation of validation forecast residuals ($363$ fitted parameters assuming normality).
6. **`empirical_residual` (Fitted Validation Residual Quantiles):**
   $$\\text{SS}_{i,t} = \\alpha \\cdot \\hat{q}_{i, 0.90}(\\text{res}) \\sqrt{L}$$
   where $\\hat{q}_{i, 0.90}(\\text{res})$ represents the empirical 90th percentile of validation forecast residuals ($363$ fitted parameters without distributional assumptions).

Each policy is evaluated in two operational execution modes: **Static** (policy parameters utilize multi-horizon forecasts issued at epoch $t=0$) and **Rolling** (forecasts and buffers are recomputed daily from newly arrived inputs).

## 4.8 Metaheuristic Parameter Tuning: PSO vs. GWO vs. 2D Exhaustive Grid
Downstream control parameters $\\boldsymbol{\\theta} = [\\alpha, \\beta]^T$ are calibrated strictly on the validation partition (July 17–31, 2017) by minimizing the penalized fitness function:
$$\\min_{\\alpha \\in [0.0, 1.5], \\beta \\in [0.5, 5.0]} \\text{Fitness}(\\alpha, \\beta) = J_{\\text{val}}(\\alpha, \\beta) + 50,000 \\cdot \\left[ \\max\\left(0, 95.0 - \\text{SL}_{\\text{val}}(\\alpha, \\beta)\\right) \\right]^2$$

To ensure algorithmic robustness, we deploy and benchmark two distinct nature-inspired metaheuristics across 10 independent random seeds (`42, 101, 202, 303, 404, 505, 606, 707, 808, 909`):

### A. Particle Swarm Optimization (PSO)
A swarm of $P = 25$ particles navigates the 2D parameter space over $K = 20$ iterations. Each particle $p$ maintains position $\\mathbf{x}_p^{(k)} = [\\alpha_p, \\beta_p]^T$ and velocity $\\mathbf{v}_p^{(k)}$, updated according to:
$$\\mathbf{v}_p^{(k+1)} = w \\mathbf{v}_p^{(k)} + c_1 r_1 (\\mathbf{p}_{\\text{best}, p} - \\mathbf{x}_p^{(k)}) + c_2 r_2 (\\mathbf{g}_{\\text{best}} - \\mathbf{x}_p^{(k)})$$
$$\\mathbf{x}_p^{(k+1)} = \\text{clip}\\left(\\mathbf{x}_p^{(k)} + \\mathbf{v}_p^{(k+1)}, \\mathbf{x}_{\\min}, \\mathbf{x}_{\\max}\\right)$$
with inertia weight $w = 0.72$, cognitive coefficient $c_1 = 1.49$, social coefficient $c_2 = 1.49$, and $r_1, r_2 \\sim \\text{Uniform}(0, 1)$.

### B. Grey Wolf Optimizer (GWO)
A pack of $W = 25$ wolves navigates the search space over $K = 20$ iterations, guided by the three best solutions logged as alpha ($\\boldsymbol{\\alpha}$), beta ($\\boldsymbol{\\beta}$), and delta ($\\boldsymbol{\\delta}$) leaders:
$$\\mathbf{D}_\\alpha = |\\mathbf{C}_1 \\cdot \\mathbf{X}_\\alpha - \\mathbf{X}|, \\quad \\mathbf{D}_\\beta = |\\mathbf{C}_2 \\cdot \\mathbf{X}_\\beta - \\mathbf{X}|, \\quad \\mathbf{D}_\\delta = |\\mathbf{C}_3 \\cdot \\mathbf{X}_\\delta - \\mathbf{X}|$$
$$\\mathbf{X}_1 = \\mathbf{X}_\\alpha - \\mathbf{A}_1 \\cdot \\mathbf{D}_\\alpha, \\quad \\mathbf{X}_2 = \\mathbf{X}_\\beta - \\mathbf{A}_2 \\cdot \\mathbf{D}_\\beta, \\quad \\mathbf{X}_3 = \\mathbf{X}_\\delta - \\mathbf{A}_3 \\cdot \\mathbf{D}_\\delta$$
$$\\mathbf{X}^{(k+1)} = \\text{clip}\\left(\\frac{\\mathbf{X}_1 + \\mathbf{X}_2 + \\mathbf{X}_3}{3}, \\mathbf{x}_{\\min}, \\mathbf{x}_{\\max}\\right)$$
where coefficients $\\mathbf{A} = 2a \\cdot r_1 - a$ and $\\mathbf{C} = 2r_2$, with exploration parameter $a$ linearly decaying from $2.0$ to $0.0$ over iterations.

### C. Exhaustive 2D Grid Search Ground Truth
To avoid false claims of mathematical optimality, we construct a brute-force ground-truth grid over $\\alpha \\in [0.0, 1.5]$ (31 uniformly spaced points, $\\Delta \\alpha = 0.05$) and $\\beta \\in [0.5, 5.0]$ (46 uniformly spaced points, $\\Delta \\beta = 0.10$), evaluating all $31 \\times 46 = 1,426$ configurations. Figure 4.2 (`e2_fitness_surface_validation.png`) illustrates the resulting non-linear fitness landscape.
"""
