# -*- coding: utf-8 -*-
"""
Part 1 of Comprehensive Academic Research Monograph
Covers: Front Matter, Abstract, Chapters 1, 2, 3, and 4
"""

PART1_CONTENT = {
    "title": "An End-to-End Deep Probabilistic Forecasting and Metaheuristic Inventory Optimization Framework for Retail Supply Chains",
    "subtitle": "Integrating Quantile Temporal Fusion Transformers, Split-Conformal Calibration, and Swarm Intelligence under Asymmetric Cost Structures",
    "author": "Tanmay",
    "affiliation": "Department of Data Science & Operations Research",
    "target_venue": "IEEE Transactions on Engineering Management / International Journal of Production Economics",
    "date": "October 2026",
    "dataset_info": "Corporación Favorita Grocery Dataset (Ecuador National Supermarket Chain)",
    
    "abstract": """
Multi-horizon demand forecasting in large-scale grocery retail is severely challenged by intermittent sales, localized temporal non-stationarity, promotional volatility, and heavy-tailed consumption spikes. Classical operational paradigms rely on point forecasting models optimized via symmetric loss functions—such as Mean Squared Error (MSE) or Huber Loss—which fundamentally estimate conditional expectations. When paired with traditional Gaussian safety stock assumptions, these models systematically misallocate working capital: they overestimate baseline buffer stocks for slow-moving stock-keeping units (SKUs) while catastrophically underestimating surge demand during promotional events and payday cycles. This structural mismatch precipitates excessive inventory holding and perishable spoilage on one extreme, and severe stockout penalties on the other.

To resolve this critical disconnect, this research establishes an exhaustive, leak-free, 17-step end-to-end operational framework spanning from raw transactional data ingestion to closed-loop metaheuristic policy optimization. Utilizing 125,497,040 transaction logs from Corporación Favorita, we formulate a stratified subsampling methodology that extracts a mathematically representative cohort of 400 store-item pairs spanning 53 retail outlets and 122 product categories. A continuous calendar timeline is synthesized to rectify intermittent sparsity through explicit observation tracking and missing promotion attribution. We engineer a 41-dimensional multivariate feature space comprising 18 historical covariates, 14 future known temporal drivers, and 9 categorical static entity embeddings.

At the modeling core, we design and train a Multi-Horizon Quantile Temporal Fusion Transformer (Quantile-TFT) that replaces point estimation with simultaneous prediction of the 10th, 50th, and 90th conditional demand percentiles (P10, P50, P90) across a 7-day replenishment horizon via multi-quantile Pinball Loss. We address and mathematically resolve the academic conflict regarding the elevation of the Root Mean Squared Error (RMSE) to Mean Absolute Error (MAE) ratio (2.833 vs. 2.385 in the point model): via an 8-point tail residual diagnostic and the Fundamental Theorem of Loss Functions, we prove that P50's convergence to the median inherently ignores extreme outliers to achieve an operational best-in-class Weighted Absolute Percentage Error (WAPE) of 28.09%, while the P90 trajectory specifically isolates surge dynamics.

To eliminate localized under-coverage on highly volatile SKUs, we introduce a validation-anchored Split-Conformal Calibration layer that yields finite-sample marginal coverage guarantees, elevating empirical surge protection on test data to 91.01% overall (91.17% on volatile SKUs). Finally, we embed these calibrated probabilistic trajectories into a 15-day continuous discrete-event supermarket replenishment simulator operating under an empirical dynamic (s, S) policy with asymmetric perishability costs ($0.15/unit/day perishable vs. $0.10/unit/day dry goods, $2.50 stockout penalty, and $8.00 order dispatch). Parameterized via Particle Swarm Optimization (PSO) and independently validated via Grey Wolf Optimization (GWO) under a 95.0% minimum service level constraint, our proposed policy converges in 23.42 seconds to alpha = 0.1000 and beta = 3.4879. Benchmarked across 363 active test store-item pairs, our framework slashes holding and spoilage costs by 21.18% (-$27,832.45) and achieves a 9.38% net reduction in total supply chain operating costs (+$14,118.53 savings) compared to the state-of-the-art Huber TFT baseline, while securing a 96.41% customer fill rate.
""",

    "keywords": [
        "Multi-Horizon Time-Series Forecasting",
        "Temporal Fusion Transformer (TFT)",
        "Quantile Regression",
        "Pinball Loss",
        "Split-Conformal Prediction",
        "Dynamic (s, S) Inventory Control",
        "Particle Swarm Optimization (PSO)",
        "Grey Wolf Optimizer (GWO)",
        "Perishable Spoilage Reduction",
        "Supply Chain Resilience"
    ],

    "abbreviations": [
        ("TFT", "Temporal Fusion Transformer"),
        ("GRN", "Gated Residual Network"),
        ("VSN", "Variable Selection Network"),
        ("GLU", "Gated Linear Unit"),
        ("LSTM", "Long Short-Term Memory"),
        ("MHA", "Multi-Head Attention"),
        ("PSO", "Particle Swarm Optimization"),
        ("GWO", "Grey Wolf Optimizer"),
        ("MAE", "Mean Absolute Error"),
        ("RMSE", "Root Mean Squared Error"),
        ("WAPE", "Weighted Absolute Percentage Error"),
        ("SKU", "Stock Keeping Unit"),
        ("EDA", "Exploratory Data Analysis"),
        ("SL", "Service Level / Customer Fill Rate"),
        ("PO", "Purchase Order"),
        ("EOQ", "Economic Order Quantity"),
        ("PERT", "Program Evaluation and Review Technique"),
        ("WTI", "West Texas Intermediate (Crude Oil)")
    ],

    "chapter1": {
        "title": "Chapter 1: Introduction & Problem Formulation",
        "sections": [
            {
                "heading": "1.1 The Complex Realities of Modern Grocery Retail Supply Chains",
                "body": """
Modern grocery retail operates under extreme operational pressures characterized by razor-thin profit margins (typically 1% to 3%), highly non-stationary consumer buying habits, and localized competitive pressures. Unlike consumer packaged goods (CPG) manufacturing or durable goods fulfillment, grocery retail logistics must constantly navigate the physical decay of perishable commodities alongside strict consumer expectations of instantaneous product availability. In large retail supermarket chains—such as Corporación Favorita in Ecuador, the empirical foundation of this study—inventory managers oversee tens of thousands of stock-keeping units (SKUs) distributed across geographically diverse urban and rural store topologies.

In this environment, consumer demand is governed by a multitude of interacting temporal, macroeconomic, and promotional drivers. Payday schedules (notably the bi-weekly 'quincena' occurring on the 15th and penultimate day of each month) induce massive demand surges, while regional public holidays, calendar shifts, and local carnival festivities disrupt baseline shopping behavior. Simultaneously, external macroeconomic shocks—such as fluctuations in the spot price of West Texas Intermediate (WTI) crude oil, which directly influences national economic liquidity in dollarized economies—inflict broad shifts in aggregate consumer spending power.

Compounding these exogenous drivers is the operational reality of commercial promotions. Temporary price reductions, end-cap displays, and cross-merchandising campaigns cause demand spikes that can exceed 500% to 800% of typical baseline consumption. When retail supply chains rely on disjointed, heuristic-based replenishment rules, they suffer from the classical Bullwhip Effect, where minor fluctuations in consumer demand amplify upstream into massive order oscillations, driving excessive buffer inventory and catastrophic operational waste.
"""
            },
            {
                "heading": "1.2 Physical and Financial Asymmetries: Holding, Spoilage, Stockouts, and Dispatch",
                "body": """
A central deficiency in conventional inventory optimization literature is the treatment of operational costs as symmetric or uniform across catalog categories. In real-world supermarket operations, cost parameters exhibit profound structural asymmetries that dictate inventory viability:

1. Perishable vs. Non-Perishable Holding Asymmetry: Dry, shelf-stable groceries (e.g., canned foods, dry pasta, household goods) incur primarily capital financing and ambient warehousing costs, modeled conservatively at $0.10 per unit per day. In sharp contrast, fresh, chilled, and perishable categories (e.g., dairy, poultry, meats, produce) carry strict cold-chain refrigeration requirements, accelerated expiration schedules, and imminent spoilage risks. Excess inventory of fresh poultry held past its shelf-life is not merely carried over at capital cost; it must be completely condemned and discarded. To capture this physical reality, perishable holding costs are parameterized at $0.15 per unit per day—a 50% premium over dry inventory.

2. Stockout Penalty Asymmetry: When a customer encounters an empty shelf for a grocery staple, the retailer faces far more than the immediate loss of retail margin. Stockouts cause severe reputational friction, drive customers to competing retail chains, and disrupt weekly basket sizes. In our modeling, the stockout penalty is established at $2.50 per unit of unmet consumer demand—exceeding daily holding costs by an order of magnitude (16.7x to 25.0x).

3. Fixed Order Dispatch Costs: Replenishing physical supermarket shelves is not a friction-free, continuous process. Placing a supplier purchase order triggers administrative review, dock scheduling, labor allocation for unloading, and dedicated freight transportation. Every dispatched purchase order carries a fixed transaction cost of $8.00 per order, regardless of batch volume. Consequently, an inventory policy that attempts to maintain razor-thin buffers through incessant, micro-scale purchase orders will rapidly collapse under unsustainable logistics dispatch expenses.
"""
            },
            {
                "heading": "1.3 The Failure of Classical Single-Point Forecasting and Gaussian Assumptions",
                "body": """
Historically, retail inventory replenishment has been divided into a two-stage sequential workflow: first, generate a single-point forecast representing expected demand over the supplier lead time; second, compute a buffer margin (safety stock) using classical Gaussian statistical theory. 

The standard textbook safety stock formula dictates:
SS_gaussian = z * sigma_L = z * (sigma_d * sqrt(L))

where z is the standard normal critical value corresponding to a desired service level (e.g., z = 1.282 for a 90% cycle service level, or z = 1.645 for 95%), sigma_d is the standard deviation of historical daily demand, and L is the supplier delivery lead time in days.

This classical paradigm collapses under empirical retail conditions due to three fatal theoretical flaws:
* The Normal Distribution Fallacy: Retail sales data violate Gaussian symmetry. Daily sales distributions are zero-bounded, highly right-skewed, and exhibit extreme leptokurtosis (fat tails). Demand errors do not follow a Gaussian bell curve; rather, sales remain low or zero on standard weekdays and erupt during promotional weekends or paydays.
* The Conditional Mean Trap: Point forecasting models trained on Mean Squared Error (MSE) or Huber Loss are mathematically guaranteed to converge to the conditional mean: E[Y | X]. In a right-skewed distribution, the conditional mean lies substantially above the median (the 50th percentile). Consequently, an MSE-trained model constantly over-predicts everyday baseline sales, forcing warehouses to hold excessive buffer stock on 85% of days, which triggers catastrophic spoilage in perishable categories.
* Static Error Assumptions: Classical safety stock treats volatility (sigma_d) as a static scalar property of the SKU. In reality, demand uncertainty is highly dynamic and context-dependent. Demand uncertainty during an active promotion or leading up to Christmas is orders of magnitude greater than uncertainty on a quiet Tuesday in February. Applying a static Gaussian buffer guarantees stockouts during promotional surges and overstocking during seasonal troughs.
"""
            },
            {
                "heading": "1.4 Research Objectives and Guiding Inquiries",
                "body": """
To overcome the limitations of classical forecasting and disconnected inventory planning, this study formulates and executes a rigorous end-to-end framework. We articulate four primary research questions (RQs):

* RQ1 (Probabilistic Architecture): Can a multi-horizon deep learning architecture utilizing temporal self-attention and Pinball loss simultaneously forecast operational baseline demand and extreme surge boundaries across heterogeneous retail catalog hierarchies without suffering from temporal leakage?
* RQ2 (The Evaluation Paradox): How can academic researchers and practitioners reconcile the apparent divergence between Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) when transitioning from conditional mean models to quantile-based median models, and what mathematical theorems govern this relationship?
* RQ3 (Finite-Sample Calibration): Can distribution-free Split-Conformal Prediction provide mathematically provable coverage guarantees that shield retail supply chains against catastrophic stockouts during extreme demand surges without inflating inventory holding costs?
* RQ4 (Closed-Loop Metaheuristic Optimization): Does optimizing dynamic (s, S) order-up-to inventory thresholds directly against a continuous, multi-echelon discrete-event simulator via metaheuristic swarm intelligence (PSO and GWO) achieve superior financial savings compared to isolated statistical safety stock policies?
"""
            },
            {
                "heading": "1.5 Summary of Major Scientific and Industrial Contributions",
                "body": """
This monograph documents five distinct contributions to the fields of operational forecasting, machine learning, and supply chain management:

1. End-to-End Leak-Free Pipeline Architecture: We design, implement, and rigorously validate a 17-step operational pipeline that bridges raw, unstructured retail transaction logs (125 million rows) with downstream replenishment dispatch. The pipeline enforces strict chronological split boundaries and train-only parameter fitting, eliminating lookahead bias.

2. Multi-Horizon Quantile Temporal Fusion Transformer (Quantile-TFT): We construct a deep sequence-to-sequence neural network utilizing Gated Residual Networks (GRNs), Variable Selection Networks (VSNs), LSTM recurrent layers, and Interpretable Multi-Head Temporal Self-Attention. Trained under a multi-quantile Pinball objective across P10, P50, and P90, the architecture decouples everyday operational replenishment (P50) from surge protection (P90) and spoilage prevention (P10).

3. Definitive Resolution of the RMSE/MAE Evaluation Paradox: We present an 8-point tail residual diagnostic and formal statistical proof establishing that evaluating the P50 median forecast with L2-based RMSE is mathematically invalid. We demonstrate that 78.1% of squared forecast errors are driven by extreme, non-Gaussian sales surges in a tiny subset of perishable meat/poultry SKUs, and show how full distribution integration via Simpson's rule recovers the conditional expectation.

4. Stratified Split-Conformal Calibration Layer: We implement an empirical non-conformity scoring framework stratified across baseline sales volume tiers. This layer yields finite-sample marginal coverage guarantees, elevating out-of-sample empirical protection against stockouts to 91.01% overall and 91.17% on high-volatility SKUs.

5. Metaheuristic-Driven Continuous Supermarket Simulator: We develop a modular, 15-day continuous discrete-event simulation engine modeling physical warehouse arrivals, consumer sales fulfillment, asymmetric perishable holding costs, stockout penalties, and fixed dispatch expenses across 363 active store-item pairs. Optimizing dynamic (s, S) thresholds via Particle Swarm Optimization (PSO) and Grey Wolf Optimization (GWO) cuts perishable holding costs by 21.18% and reduces total supply chain costs by 9.38% (+$14,118.53 net savings) while maintaining a 96.41% fill rate.
"""
            },
            {
                "heading": "1.6 Organization of the Monograph",
                "body": """
The remainder of this research monograph is organized as follows:
* Chapter 2 reviews relevant theoretical literature in classical time series, deep neural architectures, conformal prediction, and inventory control.
* Chapter 3 outlines the master 17-step architecture and operational data flow.
* Chapter 4 details Phase 1: Data Engineering, Stratified Sampling, Feature Formulation, and Normalization (Steps 1–7).
* Chapter 5 covers Phase 2: Sliding Sequence Generation, Memory-Mapped DataLoaders, and Quantile-TFT Training Dynamics (Steps 8–10).
* Chapter 6 presents the mathematical rebuttal, loss function theorems, and the 8-point diagnostic resolving the RMSE vs. MAE paradox.
* Chapter 7 details Phase 3: Split-Conformal Calibration and Finite-Sample Coverage Guarantees (Step 11).
* Chapter 8 presents Phase 4: Dynamic Safety Stock Formulation and the 15-Day Continuous Discrete-Event Simulator (Steps 12–13).
* Chapter 9 details Phase 5: Metaheuristic Policy Search via Particle Swarm Optimization and Grey Wolf Optimizer (Step 14).
* Chapter 10 presents Steps 15 and 16: Comparative Four-Way Baseline Benchmarking, Ablation Studies, and Financial Ledger Analysis.
* Chapter 11 details Step 17: Executive Dashboard Architecture, Production Deployment, and Real-Time Monitoring.
* Chapter 12 discusses threats to validity, empirical boundaries, and methodology limitations.
* Chapter 13 concludes the study and outlines future research trajectories.
* Extensive Appendices provide full feature matrices, hyperparameter logs, and iteration ledgers.
"""
            }
        ]
    },

    "chapter2": {
        "title": "Chapter 2: Literature Review & Theoretical Foundations",
        "sections": [
            {
                "heading": "2.1 Classical Time Series Forecasting in Retail Operations",
                "body": """
For over half a century, retail demand forecasting has been anchored in classical parametric statistical methodologies. Autoregressive Integrated Moving Average (ARIMA) models and their seasonal variants (SARIMAX), formalized by Box and Jenkins (1970), model univariate time series through linear combinations of past autoregressive lags and moving average forecast errors. Holt-Winters Exponential Smoothing (Hyndman et al., 2008) provides additive and multiplicative frameworks for tracking evolving baseline levels, linear trends, and repeating seasonal cycles.

While theoretically elegant, these classical methods fail in modern grocery retail settings due to three core structural limitations:
1. Univariate Isolation: Classical ARIMA and ETS models are inherently designed for individual univariate trajectories. In a supermarket chain with 50,000 store-item combinations, fitting 50,000 isolated models ignores shared cross-series dynamics, such as store-wide promotional events, regional holiday impacts, and brand-level substitution effects.
2. Inability to Ingest High-Dimensional Exogenous Features: While SARIMAX permits exogenous covariates, it assumes strict linear relationships and requires future covariate trajectories to be independently estimated. It cannot effectively ingest high-dimensional interactions between calendar cycles, oil price shocks, and promotional flags.
3. Intermittent Demand Breakdown: When applied to slow-moving or intermittent SKUs characterized by frequent zero-demand days interspersed with sudden lump purchases, classical models produce fractional continuous forecasts that distort physical reorder triggers. While Croston's method (Croston, 1972) and the Syntetos-Boylan Approximation (Syntetos & Boylan, 2005) decouple demand size from inter-arrival intervals, they assume stationary Poisson demand arrivals and cannot capture complex seasonal or promotional multipliers.
"""
            },
            {
                "heading": "2.2 Deep Learning for Multi-Horizon Sequence Modeling",
                "body": """
The advent of deep recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) architectures (Hochreiter & Schmidhuber, 1997) revolutionized time-series analysis by enabling non-linear state transitions over extended historical sequences. Salinas et al. (2020) introduced DeepAR, an autoregressive recurrent network that estimates parameters of a parametric distribution (e.g., Gaussian or Negative Binomial) across thousands of interrelated time series. Deep State Space Models (Rangapuram et al., 2018) combined linear state space formulations with deep recurrent emission networks to maintain statistical interpretability.

However, standard autoregressive deep architectures exhibit notable operational vulnerabilities when deployed for multi-horizon operational replenishment:
* Error Accumulation via Recursive Unrolling: In autoregressive models, predicting a 7-day lead-time horizon requires feeding the model's own Day 1 forecast back into its input to generate Day 2, repeating recursively through Day 7. Any prediction error or outlier shock on Day 1 is propagated and compounded across subsequent horizons, severely degrading multi-step accuracy.
* Black-Box Opacity: Standard LSTMs and gated recurrent units (GRUs) provide zero visibility into which input features (e.g., promotions, holidays, day-of-week) drove a particular forecast spike, making operational validation impossible for retail category managers.
"""
            },
            {
                "heading": "2.3 The Temporal Fusion Transformer (TFT) Architecture",
                "body": """
To overcome error accumulation and black-box opacity, Lim et al. (2021) developed the Temporal Fusion Transformer (TFT). The TFT is an attention-based deep sequence-to-sequence architecture purpose-built for multi-horizon time-series forecasting. The network uniquely segregates input variables into three distinct structural classes:
1. Static Metadata: Time-invariant entity descriptors (e.g., store geographic cluster, store architectural format, item commodity family, item perishable flag).
2. Observed Past Covariates: Dynamic features known only historically (e.g., historical sales volumes, past promotion statuses, past observation indicators).
3. Known Future Covariates: Dynamic temporal signals known with deterministic certainty across the entire forecast horizon (e.g., calendar day-of-week, future promotion schedules, national and regional public holidays).

The architectural power of the TFT stems from its specialized internal building blocks:
* Variable Selection Networks (VSNs): Operating across static, past, and future feature inputs, VSNs utilize Gated Residual Networks (GRNs) and softmax gating to dynamically assign attention weights to individual features at each time step. Uninformative or noisy features are mathematically suppressed, preventing overfitting in high-dimensional feature spaces.
* Gated Residual Networks (GRNs): GRNs provide non-linear feature transformations equipped with Gated Linear Units (GLUs) (Dauphin et al., 2017) and skip connections, allowing the network to dynamically bypass non-linear processing when simple linear transformations suffice.
* Interpretable Multi-Head Temporal Self-Attention: A modified multi-head self-attention layer learns long-range temporal dependencies across the historical 28-day window and 7-day forecast horizon, producing interpretable attention weight matrices that explicitly reveal which historical days most heavily influence future replenishment needs.
"""
            },
            {
                "heading": "2.4 Quantile Regression & Pinball Loss Optimization",
                "body": """
Rather than predicting a single scalar point estimate representing conditional mean demand, operational decision-making requires estimating conditional quantiles of the underlying demand distribution. Quantile regression, pioneered by Koenker and Bassett (1978), replaces symmetric quadratic loss with the asymmetric Pinball Loss (Quantile Loss).

For a target quantile q in (0, 1), actual demand y, and predicted quantile y_hat_q, the Pinball Loss is formally defined as:
L_q(y, y_hat_q) = max( q * (y - y_hat_q), (1 - q) * (y_hat_q - y) )

The mathematical elegance of Pinball Loss lies in its asymmetric gradient structure:
* Under-prediction (y > y_hat_q): The error (y - y_hat_q) is positive, and the loss scales linearly by the factor q.
* Over-prediction (y < y_hat_q): The error (y_hat_q - y) is positive, and the loss scales linearly by the factor (1 - q).

When optimizing for the upper surge boundary at q = 0.90, under-predicting demand is penalized with weight 0.90, whereas over-predicting demand is penalized with weight 0.10. Consequently, under-forecasting is penalized 9 times more heavily than over-forecasting, mathematically compelling the neural network's P90 output head to envelope 90% of observed historical demand spikes. Conversely, at q = 0.10, over-forecasting is penalized 9 times more heavily, compelling the P10 output head to hug the minimum baseline demand floor. At q = 0.50, the loss reduces to symmetric L1 absolute deviation:
L_0.50(y, y_hat_0.50) = 0.5 * |y - y_hat_0.50|
which is mathematically guaranteed to estimate the conditional Median.
"""
            },
            {
                "heading": "2.5 Distribution-Free Conformal Prediction in Time Series",
                "body": """
While quantile neural networks asymptotically estimate nominal quantiles under ideal training conditions, empirical deep learning models frequently suffer from miscalibration when evaluated on out-of-sample data. Deep networks are prone to localized over-confidence or under-coverage, particularly in sparse, heavy-tailed data regions such as high-volume retail promotions. In high-stakes supply chains, an uncalibrated 90th percentile forecast that empirically captures only 82% of demand surges will trigger severe stockouts.

To provide rigorous mathematical guarantees without imposing restrictive parametric distribution assumptions, we turn to Conformal Prediction (Vovk, Gammerman, & Shafer, 2005; Angelopoulos & Bates, 2021). Conformal prediction constructs finite-sample valid prediction regions using an independent calibration dataset. 

In Split-Conformal Prediction, an independent validation split is utilized to compute non-conformity scores:
R_i = s(y_i, y_hat_i)
which measure the degree to which true validation observations deviate from model predictions. By computing the empirical (1 - alpha)(1 + 1/n)-th quantile of these calibration non-conformity scores, conformal prediction applies a calibrated additive offset delta to test forecasts. Under the mild assumption of exchangeability, the calibrated prediction intervals satisfy the exact finite-sample coverage guarantee:
P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha
regardless of the underlying sample size n or probability distribution family.
"""
            },
            {
                "heading": "2.6 Inventory Control Theory: Periodic Review (R, s, S) Policies",
                "body": """
In inventory control literature (Silver, Pyke, & Peterson, 1998; Zipkin, 2000), replenishment systems are broadly classified into continuous review (s, Q) policies and periodic review (R, s, S) policies. In modern supermarket operations, daily replenishment follows a periodic review cadence (review period R = 1 day) subject to a multi-day supplier delivery lead time L.

Under a periodic (s, S) policy:
* Every evening at review epoch t, the inventory manager calculates the SKU's total Inventory Position (IP_t):
  IP_t = I_t + Pipeline_t - Backorders_t
  where I_t is physical on-hand inventory, and Pipeline_t represents in-transit orders dispatched during previous review epochs that have not yet arrived at the retail store.
* If the inventory position falls to or below the reorder threshold s (IP_t <= s), a replenishment purchase order is placed of sufficient quantity Q_t to elevate the inventory position back up to the target order-up-to ceiling S:
  Q_t = S - IP_t
* If the inventory position remains strictly above s (IP_t > s), no order is placed (Q_t = 0), avoiding unnecessary logistics dispatch expenses.

The fundamental operational challenge lies in dynamically setting s and S. In our framework, s and S are dynamically modulated by our deep quantile trajectories:
s_t = sum_{k=1}^L P50_{t+k} + alpha * SS_dynamic
S_t = s_t + beta * mu_daily
where SS_dynamic represents the empirical quantile safety buffer derived directly from the calibrated (P90 - P50) trajectory.
"""
            },
            {
                "heading": "2.7 Metaheuristic Optimization in Supply Chain Engineering",
                "body": """
Optimizing inventory control policies within complex, stochastic supermarket simulation environments is an NP-hard, non-linear, non-convex problem. The presence of discrete order dispatch costs, non-linear service-level threshold penalties, and asymmetric holding costs creates a rugged, multi-modal objective function landscape where classical gradient descent and linear programming cannot operate.

To discover global cost-optimal policy parameters, operations researchers utilize population-based metaheuristic algorithms. Two prominent nature-inspired algorithms stand out:
1. Particle Swarm Optimization (PSO): Introduced by Kennedy and Eberhart (1995), PSO simulates the collective social foraging behavior of bird flocks or fish schools. A swarm of candidate particles navigates the multi-dimensional parameter space, with each particle adjusting its velocity and position based on a weighted combination of its personal historical best position (cognitive memory) and the entire swarm's global best position (social collaboration).
2. Grey Wolf Optimizer (GWO): Developed by Mirjalili et al. (2014), GWO models the strict social leadership hierarchy and cooperative hunting mechanisms of grey wolf packs in nature. The population is categorized into Alpha (leader/best solution), Beta (deputy advisor/second best), Delta (sentinels/third best), and Omega (subordinate search agents). Candidate solutions encircle and track prey by mathematically averaging the guiding directional vectors provided by the Alpha, Beta, and Delta leaders.

In this research, we harness PSO as our primary optimization engine and deploy GWO as an independent benchmark to prove convergence toward the global Pareto-optimal frontier.
"""
            }
        ]
    },

    "chapter3": {
        "title": "Chapter 3: End-to-End System Architecture (The 17-Step Sequential Engineering Pipeline)",
        "sections": [
            {
                "heading": "3.1 Macro Architecture and Information Flow",
                "body": """
The proposed research framework is structured as an integrated, closed-loop 17-step operational architecture. To prevent subtle data leakage and ensure reproducibility, the system enforces strict unidirectional information flow across three overarching operational phases:

Phase 1: Data Engineering & Preprocessing Pipeline (Steps 1 to 7)
Phase 2: Deep Probabilistic Modeling & Calibration (Steps 8 to 11)
Phase 3: Operational Inventory Simulation & Metaheuristic Optimization (Steps 12 to 17)

Every parameter transformation, normalization scaler, categorical encoder, and conformal offset is computed strictly on historical training observations and stored as an immutable serialized artifact before being projected forward onto out-of-sample validation and testing splits.
"""
            },
            {
                "heading": "3.2 The 17-Step Master Architecture Ledger",
                "body": """
The table below delineates the exhaustive 17-step execution workflow, detailing the inputs, operational transformations, mathematical validation criteria, and persistent artifacts produced at each stage:
"""
            }
        ]
    },

    "chapter4": {
        "title": "Chapter 4: Phase 1 — Data Engineering & Preprocessing Pipeline (Steps 1–7)",
        "sections": [
            {
                "heading": "4.1 Step 1: Raw Data Ingestion & Corporación Favorita Overview",
                "body": """
The empirical foundation of this study is the open-access commercial dataset provided by Corporación Favorita, Ecuador's largest grocery retail supermarket chain. The raw transactional corpus comprises five relational tables:
1. `train.csv`: 125,497,040 individual daily transaction logs recorded between January 1, 2013 and August 15, 2017 across 54 physical supermarket stores and 4,100 unique commercial products. Attributes include `date`, `store_nbr`, `item_nbr`, `unit_sales` (numerical sales volume), and `onpromotion` (boolean promotion status).
2. `items.csv`: SKU-level metadata cataloging each product's commercial `family` (33 major categories such as GROCERY I, BEVERAGES, MEATS, POULTRY, DAIRY, PRODUCE), granular commodity `class` (337 hierarchical subclasses), and a binary `perishable` flag (where 1 indicates short shelf-life chilled goods, and 0 indicates shelf-stable dry items).
3. `stores.csv`: Physical retail footprint containing store `city` (22 distinct municipalities across Ecuador), `state` (16 administrative provinces), store format `type` (A through E, reflecting store square footage and customer demographics), and spatial store `cluster` (17 operational clusters).
4. `oil.csv`: Daily spot price of West Texas Intermediate (WTI) crude oil ($/barrel). Ecuador is a petroleum-exporting, dollarized economy; crude oil prices serve as the primary macroeconomic barometer governing national household disposable income. Missing weekend and holiday oil prices are imputed using forward-fill followed by backward-fill.
5. `holidays_events.csv`: Official calendar documenting 350 national, regional, and municipal public holidays, transferred work days, bridge days, and emergency earthquake relief operations (following the devastating 7.8 magnitude earthquake in Manabí on April 16, 2016).
"""
            },
            {
                "heading": "4.2 Step 2: Exploratory Data Analysis & Empirical Distribution Anomalies",
                "body": """
Rigorous exploratory data analysis on the raw 125-million row corpus revealed five structural properties that invalidate classical linear forecasting:

1. Extreme Right-Skewed Leptokurtosis: Across the catalog, daily sales exhibit extreme right-skewness. While the median daily sales for slow-moving items is 0 or 1 unit, promotional weekends frequently record spikes of 150 to 500 units. The sample kurtosis across active SKUs exceeds 45.2, indicating massive fat-tailed behavior that violates Gaussian error models.
2. Intermittent Demand Sinks: Over 32.4% of catalog SKU-store combinations exhibit intermittent or lumpy demand patterns. On standard non-promotional weekdays, sales drop to zero. In classical relational databases, zero-sales days are omitted entirely, creating artificial temporal discontinuities if unaddressed.
3. Payday Surge Synchronicity ('Quincena'): In Ecuador, public and private sector employees receive salary disbursements bi-weekly on the 15th and penultimate day of each month. Exploratory time-series decomposition reveals massive, statistically significant demand spikes on these exact calendar days (+85% to +140% above baseline sales), requiring explicit cyclical calendar feature engineering.
4. Perishability Cost Exposure: Approximately 24.8% of active catalog items are tagged with `perishable = 1`. In fresh meat, dairy, and poultry departments, unsold stock spoils rapidly, imposing substantial economic loss and refrigeration waste.
5. Promotional Non-Linearity: Active promotions (`onpromotion = True`) induce asymmetric demand elasticity. A promotion on a fast-moving beverage SKU in an urban Type A store produces a 400% sales surge, whereas the same promotion in a rural Type D store produces only a 45% uplift.
"""
            },
            {
                "heading": "4.3 Step 3: Stratified Subsampling Framework (125M to 400 Core Pairs)",
                "body": """
Processing 125 million raw transaction rows across multi-year temporal sliding windows generates astronomical computational overhead, requiring hundreds of gigabytes of RAM. To establish a rigorous yet computationally tractable research benchmark without sacrificing statistical diversity, we engineer a Stratified Subsampling Selection Algorithm.

Rather than taking a naive random sample—which would disproportionately select slow-moving, uninformative items—we formulate a multi-objective SKU scoring function:
PriorityScore_i = w1 * HistoryScore_i + w2 * CoverageScore_i + w3 * ActivityScore_i + ScarcityBonus_i

where:
* HistoryScore_i: Ratio of active SKU trading lifespan to the total dataset duration (1,672 days).
* CoverageScore_i: Percentage of non-missing observations during the active lifespan.
* ActivityScore_i: Average daily transaction sales volume, ensuring robust representations of high-volume anchor SKUs.
* ScarcityBonus_i: An inverse-frequency category weighting that guarantees proportional representation for under-represented product families (e.g., SEAFOOD, MEATS, DAIRY) and rural store clusters.

Weights were calibrated at w1 = 0.30, w2 = 0.30, w3 = 0.40. Executing this stratified algorithm extracted exactly 400 representative store-item pairs spanning 53 distinct retail stores, 122 unique items, 32 product families, and 65 commodity classes. This 400-pair cohort mirrors the macro distributional properties of the 125-million-row population while permitting deep sequence-to-sequence neural training.
"""
            },
            {
                "heading": "4.4 Step 4: Temporal Rectification, Continuous Grid Expansion & Zero-Demand Imputation",
                "body": """
In enterprise transactional databases, store registers log transactions only when an item is sold. Days on which zero units are sold are omitted from `train.csv`. Ingesting this raw table directly into a temporal neural network causes catastrophic temporal leakage and distortion, as the network perceives adjacent rows as consecutive days when they may be separated by weeks.

To establish mathematical temporal integrity:
1. Continuous Calendar Grid: For each of the 400 selected store-item pairs, we generate a complete Cartesian product with a continuous daily calendar spanning January 1, 2013 to August 15, 2017 (exactly 1,688 consecutive days per pair, yielding 675,200 continuous daily observations).
2. Zero Demand Imputation: Omitted dates are explicitly imputed with `unit_sales = 0.0`.
3. Observation Flagging: We engineer a binary indicator feature:
   was_observed = 1 if transaction record existed in raw train.csv, else 0
   This enables downstream neural attention layers to explicitly differentiate between a genuine zero-demand trading day and days where the store was closed (e.g., New Year's Day) or stock was entirely absent.
4. Promotion Imputation: Missing promotional values are imputed with 0, accompanied by an explicit boolean indicator `promotion_missing = 1`.
"""
            },
            {
                "heading": "4.5 Step 5: Multivariate Feature Engineering & 41-Feature Taxonomy",
                "body": """
To provide the Temporal Fusion Transformer with rich temporal, seasonal, and categorical context, we construct a 41-dimensional feature matrix partitioned strictly according to the TFT input specification:

A. Observed Past Covariates (18 Features, Window t in [T-27, T]):
1. `unit_sales_scaled`: Log-transformed, standardized historical sales volume.
2. `onpromotion`: Binary indicator of active promotional merchandising.
3. `promotion_missing`: Indicator tracking imputed promotional status.
4. `was_observed`: Historical transaction logging indicator.
5-6. `day_of_week_sin`, `day_of_week_cos`: Cyclical sine/cosine encodings with period 7.
7-8. `month_sin`, `month_cos`: Cyclical sine/cosine encodings with period 12.
9-10. `day_of_month_sin`, `day_of_month_cos`: Cyclical sine/cosine encodings with period 31 (captures quincena payday surges).
11-12. `week_of_year_sin`, `week_of_year_cos`: Cyclical sine/cosine encodings with period 53 (smoothly handles ISO week 53).
13. `time_idx`: Normalized continuous linear trend index.
14. `is_holiday`: Binary indicator for official national/regional holiday.
15. `is_event`: Binary indicator for special calendar events (e.g., World Cup, Black Friday).
16. `is_additional`: Binary indicator for extended holiday bridge days.
17. `is_work_day`: Binary indicator for transferred work shifts.
18. `is_transfer`: Binary indicator for official holiday date reallocations.

B. Known Future Covariates (14 Features, Horizon t in [T+1, T+7]):
Known future covariates exclude past sales and unobservable telemetry, retaining only deterministic calendar signals:
1-2. `day_of_week_sin`, `day_of_week_cos`
3-4. `month_sin`, `month_cos`
5-6. `day_of_month_sin`, `day_of_month_cos`
7-8. `week_of_year_sin`, `week_of_year_cos`
9. `time_idx` (linear trend projected forward across the 7-day horizon)
10-14. `is_holiday`, `is_event`, `is_additional`, `is_work_day`, `is_transfer`

C. Static Categorical Entity Embeddings (9 Metadata Identifiers):
1. `store_nbr`: Physical store entity ID (Cardinality: 53).
2. `item_nbr`: SKU entity ID (Cardinality: 122).
3. `family`: Commodity department category (Cardinality: 32).
4. `class`: Granular commodity subclass (Cardinality: 65).
5. `perishable`: Binary shelf-life classification (Cardinality: 2).
6. `city`: Municipality footprint (Cardinality: 22).
7. `state`: Administrative province footprint (Cardinality: 16).
8. `type`: Store architectural format A-E (Cardinality: 5).
9. `cluster`: Operational store cluster (Cardinality: 17).
"""
            },
            {
                "heading": "4.6 Step 6: Strict Chronological Splitting (Zero Lookahead Leakage)",
                "body": """
To ensure strict methodological validity, the dataset is partitioned chronologically into three non-overlapping temporal windows:
* Training Split: January 1, 2013 to July 16, 2017 (1,657 continuous days). Used exclusively for fitting normalization scalers, categorical encoders, and neural network weights. Exactly 624,451 valid 35-day sliding sequences are generated across all 400 store-item pairs.
* Validation Split: July 17, 2017 to July 31, 2017 (15 continuous days). Used for Optuna hyperparameter tuning, early stopping checkpoint selection, and non-conformity score calculation in Split-Conformal Calibration. Comprises 3,299 sequences across 367 active pairs.
* Out-of-Sample Test Split: August 1, 2017 to August 15, 2017 (15 continuous days). Preserved under strict isolation. Used exclusively for final out-of-sample forecast evaluation, conformal coverage verification, and discrete-event supermarket inventory simulation. Comprises 3,237 sequences across 363 active pairs.

Preserved SHA-256 Hash Signatures:
* Training Split Hash: `40ecd28df406067208842c3ad4d4f26447d049dec63c0df472bd2bf588d24f23`
* Validation Split Hash: `b15c162c148a24339ec85b8c7299ce682c74b6491b3010594a0236642f0e5261`
* Test Split Hash: `5afd571764b1d325b518c72a61da0442cb58b0c32bf00b47ea0dc27a108c5edc`
"""
            },
            {
                "heading": "4.7 Step 7: Leak-Free Scaling and Target Transformation",
                "body": """
To handle right-skewed sales volumes and stabilize deep gradient descent, target sales volumes are normalized using a two-stage transformation fit strictly on the training partition:

Stage 1: Logarithmic Compression:
y_log = log(y + 1) = log1p(unit_sales)
This compresses extreme sales spikes (e.g., 500 units becomes 6.21) while mapping zero demand to log1p(0) = 0.

Stage 2: Standard Gaussian Normalization:
y_scaled = (y_log - mu_train) / sigma_train
where mu_train and sigma_train are computed exclusively from the 624,451 training sequences and serialized into `target_sales_scaler.joblib`. During inference and simulation, model quantile outputs are transformed back onto physical retail units via exact inverse transformation:
y_physical = exp(y_scaled * sigma_train + mu_train) - 1.0

Linear Trend Index Normalization:
The continuous linear trend index `time_idx` is normalized using fixed training boundaries:
time_idx = (date - Timestamp('2013-01-01')).days / 1657.0
On the training split, time_idx spans exactly [0.0, 1.0]. On validation and test splits, time_idx linearly continues into [1.0006, 1.0181], enabling the network's recurrent and feed-forward layers to capture long-term linear macroeconomic trend continuation without distributional distortion.
"""
            }
        ]
    }
}
