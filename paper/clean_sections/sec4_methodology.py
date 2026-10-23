# -*- coding: utf-8 -*-
"""
Clean Section 4: Methodology: The Conformalized Forecast-then-Optimize Framework
Free of raw LaTeX syntax; contains prominent Figure placeholder callout boxes.
"""

SECTION_4_PARAGRAPHS = [
    ("4. METHODOLOGY: THE CONFORMALIZED FORECAST-THEN-OPTIMIZE FRAMEWORK", "h1"),
    ("In this section, we present the end-to-end predictive and optimization architecture. The framework connects raw transactional data ingestion, multi-modal feature engineering, deep multi-horizon forecasting, split-conformal uncertainty calibration, discrete replenishment simulation, and swarm intelligence policy optimization.", "p"),
    
    ("4.1 Data Pipeline, Demand Categorization & Representativeness Audit", "h2"),
    ("The empirical foundation of this study is the open-access Corporación Favorita Grocery Sales dataset, comprising 125,497,040 daily transaction records from Ecuador's leading national grocery supermarket chain between January 1, 2013, and August 15, 2017. The raw database captures transactions across 54 commercial stores, 4,100 unique item SKUs, and 33 product families, supplemented by geographic store attributes and national/regional holiday calendars. Daily crude oil prices (West Texas Intermediate) were examined during preliminary exploratory data analysis but excluded from final model inputs to eliminate external macroeconomic dependencies.", "p"),
    ("To ensure operational relevance for multi-horizon inventory optimization, candidate store-item pairs were required to possess at least 180 non-zero sales transactions prior to the July 16, 2017 training cutoff date, yielding an eligible universe of N_pop = 138,951 active pairs. From this universe, candidate pairs were ranked using a multi-criteria optimization priority score:", "p"),
    ("PriorityScore_i = 0.40 * Coverage_i + 0.35 * HistoryLength_i + 0.25 * ln(1 + TotalSales_i) + ScarcityBonus_i", "eq", "Eq. 15"),
    ("where Coverage denotes the proportion of active transaction days, HistoryLength measures uninterrupted calendar tenure, and ScarcityBonus penalizes over-represented store-family clusters, enforcing caps of approximately 14 pairs per store and 25 pairs per product family.", "p"),
    ("The resulting primary cohort comprises 400 store-item pairs spanning 53 retail outlets (out of 54 in the national dataset), 122 unique item SKUs, 32 product families, and 65 merchandise classes across 16 Ecuadorian provinces. During the August 1–15, 2017 test evaluation window, exactly 363 pairs remained active; the remaining 37 pairs experienced permanent commercial discontinuation or zero customer demand, and were logged in our audit ledger.", "p"),
    ("In the interest of scholarly integrity, we disclose that this 400-pair cohort is not a uniform random sample of the general supermarket population:", "p"),
    ("1. Demand Velocity Concentration: Quintile partitioning of the full population based on mean daily sales yields quintile cutoffs at [2.24, 3.30, 5.00, 8.96] units/day. In our 400-pair sample, 353 pairs (88.25%) originate from Tier 5 (Very High Demand, > 8.96 units/day), with only 2 pairs in Tier 1, 6 in Tier 2, 17 in Tier 3, and 22 in Tier 4. Within the active 363 test cohort, 322 pairs (88.71%) reside in Tier 5. This concentration was intentional: dynamic (s, S) inventory policies and multi-day lead-time buffers require consistent sales velocity to justify automated continuous review.", "bullet"),
    ("2. Perishability Oversampling: Across the full population, perishable goods account for 23.23% (32,282 pairs). In our 400-pair sample, perishable goods were deliberately oversampled to 51.00% (204 perishable vs. 196 dry pairs; 175 perishable vs. 188 dry in the active test set). This deliberate oversampling ensures sufficient statistical power to measure how asymmetric holding rates ($0.15 vs. $0.10) penalize over-buffering.", "bullet"),

    ("4.2 Deep Multi-Horizon Forecasting (TFT vs. PatchTST & Baselines)", "h2"),
    ("For each pair, a continuous timeline was synthesized spanning exactly 1,688 days (January 1, 2013 to August 15, 2017). Unobserved calendar dates were imputed with zero sales and flagged via was_observed = 0. Negative sales (customer returns) were clipped to zero. Missing promotional flags were filled with zero and tracked via promotion_missing = 1. To maintain strict causality, promotions were treated strictly as past observed covariates and were excluded from future known inputs, reflecting enterprise environments where supplier promotions are subject to cancellation.", "p"),
    ("Customer sales were normalized via variance-stabilizing log-scaling: z = (ln(1 + y) - 3.2521) / 1.2694, fit strictly on training data.", "p"),
    ("The resulting input tensor comprises 41 heterogeneous feature channels partitioned into three categories:", "p"),
    ("• 18 Past Observed Covariates (28-Day Lookback): Log-sales, was_observed, promotion_missing, past promotion intensity, historical rolling demand means (7, 14, 28 days), rolling demand standard deviations (7, 14, 28 days), day-of-week demand lags (7, 14, 21, 28 days), and historical zero-sales run-lengths.", "bullet"),
    ("• 14 Known Future Covariates (7-Day Forecast Horizon): Cyclical temporal encodings (sine and cosine of day-of-week, day-of-month, day-of-year), a normalized linear temporal counter, national holiday flags, regional holiday flags, bridge day indicators, work-day transfers, and bi-monthly payday indicators (15th and end of month).", "bullet"),
    ("• 9 Static Categorical Entity Embeddings: Dense vector representations of store ID (54 categories, 53 active), item ID (122 categories), product family (32 categories), merchandise class (65 categories), city (22 categories), state (17 categories, 16 active), store type (5 categories), geographic cluster (17 categories), and binary perishability flag.", "bullet"),
    ("Sliding sequence tensors (28 lookback, 7 horizon) were partitioned chronologically into Training (624,451 sequences, Jan 1, 2013 to Jul 16, 2017), Validation (3,299 sequences, Jul 17 to Jul 31, 2017), and Test (3,237 sequences across 363 pairs, Aug 1 to Aug 15, 2017; evaluating 22,659 row observations). Sequences were serialized into zero-copy memory-mapped containers (np.memmap), enabling ingestion of over 26,000 sequences/second on an NVIDIA Tesla T4 GPU.", "p"),
    ("We evaluate three deep architectures:", "p"),
    ("1. Huber TFT (Single-Point Champion): 228,503 trainable parameters, hidden capacity d_model = 32, 2 attention heads, 1 LSTM layer, dropout 0.25, initial learning rate 0.002596, weight decay 2.66e-5, trained with Huber loss (delta = 1.0) to estimate the conditional robust location.", "bullet"),
    ("2. Quantile TFT (Multi-Quantile Champion): 230,135 trainable parameters, identical capacity and hyperparameters, trained with multi-horizon Pinball loss across quantiles q in {0.10, 0.50, 0.90} to predict conditional quantiles simultaneously across all 7 horizons.", "bullet"),
    ("3. Channel-Independent PatchTST Benchmark: 167,751 trainable parameters, patch length 7, stride 7, 3 layers, dropout 0.10, initial learning rate 0.001, MSE loss over sales and promotional channels, serving as an architectural benchmark.", "bullet"),
    ("The pivot to multi-quantile forecasting was driven by an empirical diagnostic: the Huber TFT model exhibited an RMSE of 29.54 units against an MAE of 12.38 units (a 2.39x ratio, far exceeding the Gaussian bound of 1.25). Sorting validation residuals revealed that the top 20 store-item pairs accounted for 74.5% of total squared error. Pinball loss resolves this: P50 converges to the median, ignoring extreme spikes to achieve record operational precision (WAPE 28.09%), while P90 traces upper promotional surge risk.", "p"),

    ("4.3 Split-Conformal Calibration & Finite-Sample Coverage", "h2"),
    ("While Quantile TFT models tail uncertainty effectively, empirical evaluation reveals that uncalibrated neural networks suffer from localized under-coverage. On the validation partition, the raw P90 trajectory achieved an empirical coverage of only 87.36% on high-volume SKUs (>= 100 units/day), violating the nominal 90.0% coverage requirement.", "p"),
    ("To improve empirical tail coverage on held-out horizons, we apply Mondrian Split-Conformal Calibration across two volume tiers: Standard Volume (< 100 units/day) and High-Volume (>= 100 units/day). For validation nonconformity scores s_i = y_i - P90_i, the calibration threshold is evaluated at the finite-sample order statistic rank k = ceil((n+1)*0.90) (corresponding to empirical quantile fraction ceil((n+1)*0.90)/n), yielding calibration offsets of +0.1785 physical units for standard SKUs and +4.5766 physical units for high-volume SKUs. The calibrated upper bound deployed in inventory control is P90_calibrated = P90_raw + offset_tier.", "p"),

    ("4.4 Dynamic (s_t, S_t) Replenishment Policy & Safety Stock Architectures", "h2"),
    ("We formulate and factorially ablate six safety stock buffer architectures SS_{i,t}(alpha):", "p"),
    ("1. none: SS = 0 (pure base-stock policy relying on beta).", "bullet"),
    ("2. gaussian_demand: SS = alpha * 1.282 * sigma_demand * sqrt(L), where sigma_demand is trailing 28-day demand standard deviation.", "bullet"),
    ("3. raw_quantile: SS = alpha * sum_{h=1..L} max(0, P90_raw - P50), deriving safety stock directly from the uncalibrated neural network output spread.", "bullet"),
    ("4. conformal_quantile: SS = alpha * sum_{h=1..L} max(0, P90_calibrated - P50), combining multi-horizon quantile spreads with validation-derived conformal offsets.", "bullet"),
    ("5. gaussian_residual: SS = alpha * 1.282 * sigma_res * sqrt(L), where sigma_res is the pair-specific validation residual standard deviation (363 fitted parameters).", "bullet"),
    ("6. empirical_residual: SS = alpha * q90(residuals) * sqrt(L), where q90 is the empirical 90th percentile of validation forecast residuals (363 fitted parameters).", "bullet"),
    ("Policies are evaluated in both Static mode (issued at window start) and Rolling mode (recomputed daily from newly arrived inputs).", "p"),

    ("4.5 Metaheuristic Parameter Tuning (PSO, GWO) vs. 2D Exhaustive Grid", "h2"),
    ("Policy control parameters [alpha, beta] are calibrated strictly on validation (July 17–31, 2017) by minimizing the penalized fitness function with exterior barrier lambda = 50,000.0. We deploy two nature-inspired metaheuristics across 10 independent random seeds:", "p"),
    ("• Particle Swarm Optimization (PSO): 25 particles, 20 iterations, inertia w = 0.72, cognitive c1 = 1.49, social c2 = 1.49.", "bullet"),
    ("• Grey Wolf Optimizer (GWO): 25 wolves, 20 iterations, guided by alpha, beta, delta leaders with parameter a decaying from 2.0 to 0.0.", "bullet"),
    ("• Exhaustive 2D Grid Ground Truth: 31 alpha points in [0.0, 1.5] by 46 beta points in [0.5, 5.0], evaluating all 1,426 configurations.", "bullet")
]

FIG_1_SPEC = {
    "box_title": "FIGURE 1 PLACEHOLDER & SPECIFICATION: End-to-End System Architecture",
    "items": [
        ("Figure Number", "Figure 1"),
        ("Caption", "Figure 1. End-to-End Closed-Loop Framework Architecture: From raw transactional ingestion, multi-modal feature engineering, and memory-mapped sequence generation, to Quantile TFT forecasting, Mondrian conformal calibration, and metaheuristic (s, S) replenishment simulation."),
        ("Target Image Filename", "sysarchi.png"),
        ("Image Location", "Root Directory (sysarchi.png)"),
        ("Key Components Shown", "125M Data Ingestion -> Feature Tensor -> Quantile TFT vs PatchTST -> Conformal Head -> Simulator -> PSO/GWO")
    ]
}

FIG_2_SPEC = {
    "box_title": "FIGURE 2 PLACEHOLDER & SPECIFICATION: Dynamic Quantile Buffer vs. Static Gaussian",
    "items": [
        ("Figure Number", "Figure 2"),
        ("Caption", "Figure 2. Empirical Dynamic Safety Stock Sizing: Comparison between the adaptive multi-horizon quantile spread (P90 - P50) and traditional rigid Gaussian safety stock (z * sigma * sqrt(L)) across promotional surges and baseline sales regimes."),
        ("Target Image Filename", "quantile_safety_stock_distribution.png"),
        ("Image Location", r"Multi-Horizon Quantile Forecasting(process)\quantile_safety_stock_distribution.png"),
        ("Key Finding Shown", "Quantile buffers dynamically expand during marketing promotions and payday spikes, and contract during quiet baseline periods.")
    ]
}

FIG_3_SPEC = {
    "box_title": "FIGURE 3 PLACEHOLDER & SPECIFICATION: Validation Fitness Landscape & Optimal Basin",
    "items": [
        ("Figure Number", "Figure 3"),
        ("Caption", "Figure 3. Validation Fitness Landscape and Algorithmic Convergence: 2D contour surface of total operating cost plus service penalty across alpha in [0.0, 1.5] and beta in [0.5, 5.0], showing the steep 95.0% SLA barrier and convergence points for PSO, GWO, and the 2D grid optimum."),
        ("Target Image Filename", "e2_fitness_surface_validation.png"),
        ("Image Location", r"revision_experiments\outputs\figures\e2_fitness_surface_validation.png"),
        ("Key Finding Shown", "All metaheuristics independently navigate the penalty barrier and land inside the true global minimum basin verified by exhaustive grid search.")
    ]
}
