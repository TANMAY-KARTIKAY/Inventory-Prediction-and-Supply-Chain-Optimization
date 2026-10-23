# -*- coding: utf-8 -*-
"""
Clean Section 8, Section 9, References, and Appendices
Contains:
- Section 8: Limitations & Threats to Validity (8.1 to 8.5)
- Section 9: Conclusion & Future Directions (9.1 to 9.3)
- References: 25 authoritative academic citations
- Appendices:
  - Appendix A: Feature Engineering & 41-Dimensional Covariate Architecture (Table A.1)
  - Appendix B: Optuna Hyperparameter Optimization & Model Architectures (Table B.1)
  - Appendix C: Complete 14-Scenario Economic Stress-Testing Ledger (Table C.1)
  - Appendix D: Supplementary Supply Chain Optimization Dashboard (Figure D.1)
"""

SECTION_8_9_PARAGRAPHS = [
    ("8. LIMITATIONS & THREATS TO VALIDITY", "h1"),
    ("To maintain scientific rigor and preserve the integrity of our empirical findings, we explicitly delineate the methodological boundaries, modeling assumptions, and threats to internal and external validity governing this study.", "p"),
    
    ("8.1 Finite Test Horizon and Boundary Effects", "h2"),
    ("A primary constraint of our empirical evaluation is the 15-day test horizon (August 1 to August 15, 2017). While evaluating 363 active store-item pairs over 15 daily rolling decision steps yields 5,445 pair-day replenishment decisions and 22,659 multi-horizon forecast instances, a 15-day window remains vulnerable to local demand shocks, short-term promotional campaigns, and end-of-horizon truncation artifacts.", "p"),
    ("In particular, replenishment orders dispatched after Day 8 cannot arrive at the store before Day 15 under a deterministic lead time of L = 7 days. We resolved this boundary distortion via Theorem 1 and our causal arriving-order accounting rule (W1), proving exact cost reconciliation. Nonetheless, multi-month or multi-season evaluations will be necessary to capture long-cycle seasonality, quarterly demand drift, and annual promotional peaks.", "p"),

    ("8.2 Cohort Representativeness and Survival Bias", "h2"),
    ("As established in our representativeness audit (Section 4.1), our 400 store-item cohort was sampled using a priority scoring mechanism that favored regularly selling, higher-volume items. Consequently, 88.3% of the evaluated SKUs fall within the top demand quintile of the broader Corporacion Favorita population (mean sales > 8.96 units/day), and 51.0% represent perishable categories (compared to 23.2% in the population). Slow-moving, lumpy, and ultra-sparse demand series (which represent approximately 60% of retail SKUs in grocery retail) are deliberately under-represented.", "p"),
    ("Furthermore, 37 of the 400 pairs exhibited zero sales throughout the August test window due to store closures or product phase-outs. Defining the active cohort (N = 363) using ex-post realization introduces mild survivorship bias. While our findings apply directly to high-velocity, high-holding-cost retail goods that dominate working capital commitments, caution must be exercised before extrapolating these results to slow-moving spare parts or long-tail items.", "p"),

    ("8.3 Conformal Exchangeability under Temporal and Spatial Autocorrelation", "h2"),
    ("Split-conformal prediction guarantees marginal finite-sample coverage under the formal assumption of data exchangeability. In retail time series, temporal autocorrelation, day-of-week seasonality, and spatial clustering across stores within the same metropolitan area violate strict exchangeability.", "p"),
    ("We mitigated this dependency through Mondrian stratification by sales volume tiers and by establishing that the 7-day multi-horizon forecasting architecture internalizes cyclical and calendar patterns. Our empirical test coverage (91.01% overall and 91.20% on volatile high-volume SKUs) closely mirrored nominal 90.0% targets. However, from a theoretical standpoint, our conformal coverage guarantees must be classified as approximate rather than exact distribution-free bounds.", "p"),

    ("8.4 Single-Echelon vs. Multi-Echelon Network Dynamics", "h2"),
    ("Our simulation environment models a single-echelon store inventory node served by an uncapacitated regional distribution center. In enterprise retail supply chains, stores compete for finite warehouse capacity, and upstream stockouts can cascade into downstream stockouts (the bullwhip effect).", "p"),
    ("While our replenishment model captures store-level ordering, pipeline delays, and lost sales penalties, it abstracts away multi-echelon inventory allocation, cross-docking bottlenecks, and joint replenishment batching constraints across multiple store orders sharing a single logistics carrier.", "p"),

    ("8.5 Deterministic Lead Time and Censored Demand Dynamics", "h2"),
    ("We evaluated inventory policies under a fixed, deterministic transit lead time of L = 7 days. In actual retail logistics, supplier lead times exhibit stochastic variance driven by port congestion, highway disruptions, and warehouse handling delays.", "p"),
    ("Additionally, customer demand was assumed to be fully observed through recorded sales transactions, with stockouts resulting in unfulfilled lost sales. In physical store retail, stockouts censor true demand: when an item is out of stock, customer purchase intent is unrecorded. Incorporating stochastic lead times and censored demand estimation remains an important frontier.", "p"),

    ("9. CONCLUSION & FUTURE DIRECTIONS", "h1"),
    
    ("9.1 Synthesis of Theoretical and Empirical Findings", "h2"),
    ("This study investigated the critical interface between deep sequence forecasting and downstream replenishment optimization in enterprise retail supply chains. By developing a leakage-audited evaluation framework, we resolved four pervasive methodological pitfalls that distort reported artificial intelligence savings in operations management literature.", "p"),
    ("Our primary findings deliver three fundamental insights:", "p"),
    ("1. Decoupling of Statistical Point Accuracy and Operational Cost: Two Temporal Fusion Transformers exhibiting statistically indistinguishable point accuracy (WAPE 28.09% vs. 27.99%; Wilcoxon p = 0.3430) produced an $8,251 cost disparity in downstream replenishment. Single-point conditional mean forecasting systematically inflates holding costs ($4,400 excess capital) or causes service failures during promotional surges.", "bullet"),
    ("2. Conformal Safety Buffers Eliminate Residual Tuning: Model-derived quantile spreads (P90 - P50) calibrated via split-conformal prediction restored high-volume service coverage to 91.20% without requiring 363 pair-specific residual variance estimations. At enterprise service levels (>= 94.0%), Conformal Quantile TFT expanded the achievable Pareto frontier, saving $12,888 (15.7%) over Seasonal Naive and achieving operational feasibility where Huber point forecasting failed completely.", "bullet"),
    ("3. Parity with Classical Baselines under Lean Regimes: At moderate fill rate targets (90% to 92%), classical trailing mean heuristics matched deep learning costs within 0.6% ($63,786 vs. $64,183). Deep sequence models provide decisive operational advantages specifically in asymmetric penalty regimes, high-service service level agreements, and heavy-tailed promotional environments.", "bullet"),

    ("9.2 Practical Impact for Enterprise Operations", "h2"),
    ("For enterprise retailers, our findings provide a concrete blueprint for implementing machine learning in inventory planning. Rather than deploying complex neural networks to predict single-point mean demand and attempting to buffer lead-time variance through static cycle stock, retailers should leverage multi-quantile architectures to separate baseline volume tracking from tail risk buffering.", "p"),
    ("Conformal calibration provides an actionable, distribution-free safeguard that guarantees target customer fill rates across volatile promotional spikes without exposing supply chain teams to parametric distributional errors.", "p"),

    ("9.3 Future Research Trajectories", "h2"),
    ("Three key avenues emerge for future investigation:", "p"),
    ("1. End-to-End Differentiable Decision Losses: Integrating inventory cost functions directly into neural network training via differentiable convex programming or policy gradient algorithms could align representation learning with asymmetric downstream costs.", "bullet"),
    ("2. Multi-Echelon Network Coordination: Extending conformal quantile buffering to multi-echelon networks with stochastic lead times and upstream warehouse capacity constraints.", "bullet"),
    ("3. Online Adaptive Conformal Prediction: Developing rolling, non-stationary conformal calibration algorithms that dynamically update nonconformity offsets in response to real-time macroeconomic shifts and abrupt supply disruptions.", "bullet")
]

REFERENCES_DATA = [
    ("Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2623–2631).", "ref"),
    ("Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511.", "ref"),
    ("Babai, M. Z., Ali, M. M., & Nikolopoulos, K. (2013). Impact of forecasting methods on stockout performance. International Journal of Production Economics, 143(1), 159–169.", "ref"),
    ("Ban, G. Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. Operations Research, 67(1), 90–108.", "ref"),
    ("Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. The Annals of Statistics, 49(1), 486–507.", "ref"),
    ("Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. Management Science, 66(6), 2535–2548.", "ref"),
    ("Croston, J. D. (1972). Forecasting and stock control for intermittent demands. Operational Research Quarterly, 23(3), 289–303.", "ref"),
    ("Elmachtoub, A. N., & Grigas, P. (2022). Smart 'predict, then optimize'. Management Science, 68(1), 9–26.", "ref"),
    ("Fildes, R., Ma, S., & Kolassa, S. (2022). Retail forecasting: Research and practice. International Journal of Forecasting, 38(4), 1283–1318.", "ref"),
    ("Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477), 359–378.", "ref"),
    ("Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.", "ref"),
    ("Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In Proceedings of IEEE International Conference on Neural Networks (Vol. 4, pp. 1942–1948).", "ref"),
    ("Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748–1764.", "ref"),
    ("Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in Engineering Software, 69, 46–61.", "ref"),
    ("Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR).", "ref"),
    ("Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations (ICLR).", "ref"),
    ("Prestwich, S. D., Rossi, R., Armagan Tarim, S., & Kingsman, B. G. (2014). Mean-variance approximations for inventory models with non-stationary demand. European Journal of Operational Research, 234(2), 430–440.", "ref"),
    ("Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. Advances in Neural Information Processing Systems (NeurIPS), 32, 3543–3553.", "ref"),
    ("Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181–1191.", "ref"),
    ("Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). Inventory and Production Management in Supply Chains (4th ed.). CRC Press.", "ref"),
    ("Snyder, L. V., & Shen, Z. J. M. (2019). Fundamentals of Supply Chain Theory (2nd ed.). John Wiley & Sons.", "ref"),
    ("Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. International Journal of Forecasting, 21(2), 303–314.", "ref"),
    ("Tibshirani, R. J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019). Conformal prediction under covariate shift. Advances in Neural Information Processing Systems (NeurIPS), 32, 2530–2540.", "ref"),
    ("Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 5998–6008.", "ref"),
    ("Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer Science & Business Media.", "ref")
]

APPENDICES_PARAGRAPHS = [
    ("APPENDICES & SUPPLEMENTARY MATERIAL", "h1"),
    
    ("Appendix A: Feature Engineering and 41-Dimensional Covariate Architecture", "h2"),
    ("To enable accurate multi-horizon demand forecasting across diverse retail categories, our data pipeline constructs a 41-dimensional feature tensor spanning three distinct covariate types: past time-varying variables (28-day lookback), known-future variables (7-day direct horizon), and static entity embeddings. Table A.1 provides the complete technical schema.", "p"),
    
    ("Appendix B: Optuna Hyperparameter Optimization and Model Architectures", "h2"),
    ("Model hyperparameters were tuned using Optuna over 30 Bayesian optimization trials evaluated on the validation window (July 17–31, 2017). The selected architecture was shared across both the Huber TFT and Quantile TFT models to ensure strict comparability. Table B.1 details the parameter search space and optimal configurations.", "p"),

    ("Appendix C: Complete 14-Scenario Economic Stress-Testing Ledger", "h2"),
    ("To assess policy robustness against severe macroeconomic and operational shifts, all replenishment policies were subjected to full re-optimization across 14 distinct economic environments. Table C.1 details the primary operating costs, fill rates, and policy rankings across variations in stockout penalty (c_p), holding cost multipliers (h_mult), and fixed ordering fees (o_mult).", "p"),

    ("Appendix D: Supplementary Supply Chain Optimization Dashboard", "h2"),
    ("Figure D.1 illustrates the end-to-end interactive dashboard developed for retail category managers, integrating multi-horizon forecast fan charts, conformal coverage intervals, simulated inventory trajectories, and cost-penalty breakdowns.", "p")
]

TABLE_A_1_DATA = {
    "box_title": "TABLE A.1 PLACEHOLDER & SPECIFICATION: 41-Dimensional Input Feature Architecture",
    "items": [
        ("Table Number", "Table A.1"),
        ("Table Title", "Complete 41-Dimensional Input Feature Schema for Temporal Fusion Transformers"),
        ("Key Categories", "Past Time-Varying (18 channels), Known Future (14 channels), Static Entity Embeddings (9 channels)"),
        ("Technical Purpose", "Defines exact channel inputs, temporal lookback windows, encodings, and scaling protocols.")
    ],
    "headers": ["Channel Category", "Channel Count", "Feature Names", "Temporal Window", "Encoding & Normalization Protocol", "Operational Role"],
    "rows": [
        ["Past Time-Varying Covariates", "18 Channels", "Daily Unit Sales, log(1+y), was_observed, promotion_missing, Day-of-Week (one-hot, 7 dims), Rolling Means (7, 14, 28 days), Rolling Stds (7, 14, 28 days), Lag Sales (1, 7, 14, 28 days)", "t - 27 to t (28 days)", "Standardized via training-set μ = 3.2521, σ = 1.2694; binary flags for missingness and observation status", "Captures historical sales velocity, volatility trends, and recent transaction spikes."],
        ["Known Future Covariates", "14 Channels", "Cyclical Calendar Encodings (sin/cos of Day-of-Week, Day-of-Month, Month-of-Year), Linear Time Index, Holiday Flag, Event Flag, Bridge Flag, Work-Day Flag, Transfer Flag", "t + 1 to t + 7 (7 days direct)", "Trigonometric sin/cos transformations normalized to [-1, +1]; binary indicators for official Ecuadorian calendar events", "Provides forward visibility into calendar rhythms, holiday demand surges, and payday shopping spikes."],
        ["Static Entity Embeddings", "9 Channels", "Store Number (54 categories, 53 active), Item Number (122 categories), Family (32 categories), Class (65 categories), Perishability Flag (binary), City (22 categories), State (17 categories, 16 active), Store Type (5 categories), Cluster (17 categories)", "Static (Invariant)", "Learned dense categorical embedding layers (embedding dimension = 8 to 16) mapped to continuous representation space", "Encodes permanent store hierarchy, geographic characteristics, and SKU perishability dynamics."]
    ]
}

TABLE_B_1_DATA = {
    "box_title": "TABLE B.1 PLACEHOLDER & SPECIFICATION: Optuna Hyperparameter Search Space & Model Architecture",
    "items": [
        ("Table Number", "Table B.1"),
        ("Table Title", "Hyperparameter Search Space and Optimal Configuration for Deep Forecasting Models"),
        ("Key Architectures", "Temporal Fusion Transformer (TFT Quantile & Huber) and PatchTST Benchmark"),
        ("Technical Purpose", "Documents complete search ranges, optimal parameters found, and hardware training configuration.")
    ],
    "headers": ["Hyperparameter", "Search Range / Options", "Optimal Value (TFT)", "PatchTST Benchmark Value", "Architectural Role"],
    "rows": [
        ["Hidden State Dimension", "[16, 32, 64, 128]", "32", "64", "Determines latent state capacity across temporal processing layers."],
        ["Self-Attention Heads", "[1, 2, 4, 8]", "2", "4", "Controls multi-head self-attention representation subspaces."],
        ["Recurrent Layer Depth", "[1, 2]", "1 (LSTM Layer)", "3 (Transformer Layers)", "Captures sequential temporal dynamics across lookback horizon."],
        ["Dropout Rate", "[0.10, 0.40]", "0.25", "0.10", "Prevents co-adaptation and regularizes neural representations."],
        ["Learning Rate", "[1e-4, 1e-2] (log scale)", "0.002596", "0.001000", "Adam optimizer initial step size with cosine learning rate schedule."],
        ["Weight Decay", "[1e-6, 1e-3] (log scale)", "2.66e-5", "1.00e-4", "L2 weight regularization penalty."],
        ["Total Trainable Parameters", "Calculated Architecture", "230,135 (Quantile) / 228,503 (Huber)", "167,751", "Model capacity footprint optimized for single NVIDIA T4 GPU."],
        ["Hardware & Batching", "Single NVIDIA T4 GPU", "Batch Size = 128, Early Stopping = 10 epochs", "Batch Size = 64, Early Stopping = 10 epochs", "Training completed in 3.4 hours across 624,451 sequence windows."]
    ]
}

TABLE_C_1_DATA = {
    "box_title": "TABLE C.1 PLACEHOLDER & SPECIFICATION: Full 14-Scenario Economic Stress-Testing Ledger",
    "items": [
        ("Table Number", "Table C.1"),
        ("Table Title", "Comprehensive 14-Scenario Economic Stress-Testing Ledger Across Asymmetric Cost Regimes"),
        ("Key Sensitivity Dimensions", "Stockout Penalty Variation (c_p), Holding Multiplier (h_mult), Ordering Fee Multiplier (o_mult)"),
        ("Key Finding Shown", "Quantile Conformal achieves Rank 1 in 13 of 14 scenarios; Huber None exhibits catastrophic cost explosion (+84.5%).")
    ],
    "headers": ["Sensitivity Regime", "Scenario Value", "Quantile Conformal ($)", "Quantile Rank", "Mean Gaussian ($)", "Huber Residual ($)", "Huber None ($)", "Huber None Penalty (%)"],
    "rows": [
        ["Stockout Penalty", "c_p = $1.00", "$52,047.70", "Rank 2", "$50,889.41", "$56,217.23", "$73,177.22", "+40.60%"],
        ["Stockout Penalty", "c_p = $2.50 (Baseline)", "$65,276.40", "Rank 1", "$65,588.48", "$73,485.26", "$88,090.13", "+34.95%"],
        ["Stockout Penalty", "c_p = $4.00", "$72,361.35", "Rank 1", "$74,556.87", "$89,691.85", "$103,003.04", "+42.34%"],
        ["Stockout Penalty", "c_p = $5.20", "$77,178.78", "Rank 1", "$78,976.57", "$98,317.20", "$114,933.37", "+48.92%"],
        ["Stockout Penalty", "c_p = $7.50", "$84,219.50", "Rank 1", "$84,531.61", "$114,524.08", "$137,799.83", "+63.62%"],
        ["Stockout Penalty", "c_p = $10.00", "$88,171.09", "Rank 1", "$93,957.11", "$129,239.63", "$162,654.68", "+84.48%"],
        ["Holding Multiplier", "h_mult = 0.50x", "$41,991.70", "Rank 1", "$45,428.91", "$52,582.27", "$58,552.49", "+39.44%"],
        ["Holding Multiplier", "h_mult = 0.75x", "$55,280.49", "Rank 1", "$57,696.05", "$62,794.12", "$73,321.31", "+32.64%"],
        ["Holding Multiplier", "h_mult = 1.25x", "$72,609.16", "Rank 1", "$75,953.29", "$81,565.08", "$102,858.95", "+41.66%"],
        ["Holding Multiplier", "h_mult = 1.50x", "$80,288.95", "Rank 1", "$82,099.36", "$88,230.85", "$117,627.77", "+46.51%"],
        ["Ordering Fee Multiplier", "o_mult = 0.50x", "$56,745.36", "Rank 1", "$61,719.21", "$65,504.50", "$86,010.13", "+51.57%"],
        ["Ordering Fee Multiplier", "o_mult = 0.75x", "$60,837.36", "Rank 1", "$65,763.21", "$70,121.90", "$87,050.13", "+43.09%"],
        ["Ordering Fee Multiplier", "o_mult = 1.25x", "$67,554.40", "Rank 1", "$67,898.48", "$75,969.26", "$89,130.13", "+31.94%"],
        ["Ordering Fee Multiplier", "o_mult = 1.50x", "$69,832.40", "Rank 1", "$70,208.48", "$77,477.10", "$90,170.13", "+29.12%"]
    ]
}

FIG_D_1_SPEC = {
    "box_title": "FIGURE D.1 PLACEHOLDER & SPECIFICATION: Supply Chain Optimization Dashboard",
    "items": [
        ("Figure Number", "Figure D.1 (Supplementary)"),
        ("Caption", "Figure D.1. Enterprise Retail Replenishment Decision Dashboard: Interactive visual analytics interface integrating multi-horizon forecast fan charts, conformal uncertainty intervals, simulated stock trajectories, and cost-breakdown waterfalls."),
        ("Target Image Filename", "supply_chain_optimization_dashboard.png"),
        ("Image Location", r"Phase_4_Inventory_Optimization\supply_chain_optimization_dashboard.png"),
        ("Key Finding Shown", "Demonstrates practitioner deployment workflow linking conformalized deep forecasts directly to real-time order generation.")
    ]
}
