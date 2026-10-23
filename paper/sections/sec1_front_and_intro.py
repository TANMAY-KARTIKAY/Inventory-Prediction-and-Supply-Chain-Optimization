# -*- coding: utf-8 -*-
"""
Section 1: Front Matter and Introduction
"""

TITLE = "Bridging Prediction and Inventory Optimization in Retail Supply Chains: Conformalized Temporal Fusion Transformers with Guaranteed Coverage"

AUTHORS = [
    {"name": "Tanmay Kartikay", "affiliation": "1", "email": "tanmay.kartikay2023@vitstudent.ac.in"},
    {"name": "Aditya Singh", "affiliation": "1", "email": "aditya.singh2023@vitstudent.ac.in"},
    {"name": "Arti Tiwari", "affiliation": "2", "email": "arti.tiwari@vit.ac.in"}
]

AFFILIATIONS = [
    "Department of Computer Science and Engineering (Artificial Intelligence and Machine Learning), School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, Tamil Nadu, 632014, India",
    "Assistant Professor Senior Grade 1, School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology, Vellore, Tamil Nadu, 632014, India"
]

ABSTRACT = """
Multi-horizon demand forecasting in enterprise grocery retail is severely challenged by intermittent transactions, high promotional elasticity, non-stationary seasonality, and heavy-tailed consumption spikes. Classical operational paradigms rely on point forecasting models trained via symmetric objective functions—such as Mean Squared Error (MSE) or Huber loss—which fundamentally estimate conditional expectations. When paired with standard parametric Gaussian safety stock assumptions, these models systematically misallocate working capital: they overstock slow-moving stock-keeping units (SKUs) while catastrophically under-buffering surge demand during promotional events and payday cycles. This structural mismatch precipitates excessive inventory holding and perishable spoilage on one extreme, and punitive stockout penalties on the other.

To resolve this disconnect, this study develops a leakage-free, distribution-free 'Forecast-then-Optimize' replenishment framework integrating Multi-Horizon Quantile Temporal Fusion Transformers (TFT), split-conformal calibration, and nature-inspired metaheuristic policy optimization. Benchmarking against a channel-independent PatchTST baseline, classical Seasonal Naive, and trailing moving averages across 125,497,040 transactions from the Corporación Favorita supermarket dataset (audited across 400 representative store-item combinations and 363 active test pairs), we establish several critical empirical findings. First, while Quantile TFT (median P50) and Huber TFT achieve statistically indistinguishable point accuracy (WAPE 28.09% vs. 27.99%; paired Wilcoxon p = 0.343), their downstream operational values diverge sharply: the quantile-calibrated policy reduces total replenishment operating costs by $8,251 (95% bootstrap CI [-$19,105, -$2,126], p = 0.0010) under identical parameter tuning, driven by $4,400 lower holding costs and $3,200 lower stockout penalties. Second, an empirical error concentration audit reveals that the top 20 high-volume SKUs account for 74.5% of total squared forecast error, explaining why single-point conditional expectations fail to insulate supply chains against tail risk. Third, Mondrian split-conformal calibration eliminates localized under-coverage on volatile SKUs, elevating test-period coverage from 89.80% to 91.20% while maintaining tight interval sharpness without fitting pair-specific residual variances. Finally, by constructing a non-dominated monotone matched-fill Pareto frontier, we demonstrate that conformalized quantile buffering is the only policy capable of achieving enterprise service levels of 95.0% and 96.0% without triggering holding cost explosions, outperforming tuned Seasonal Naive heuristics by $12,888 (15.7% cost reduction) at a 95% target fill rate. Methodological audits further prove that standard simulation practices—specifically lookahead bias in cycle stock sizing and unequal initial inventory subsidization—artificially distort reported savings by up to $28,921, underscoring the necessity of invariant evaluation protocols in data-driven operations research.
"""

KEYWORDS = [
    "Retail Supply Chain Management",
    "Multi-Horizon Demand Forecasting",
    "Temporal Fusion Transformer (TFT)",
    "Conformal Prediction",
    "Dynamic (s, S) Replenishment",
    "Particle Swarm Optimization (PSO)",
    "Grey Wolf Optimizer (GWO)",
    "Perishable Inventory Control",
    "Forecast-then-Optimize"
]

SECTION_1_TEXT = """
# 1. Introduction

## 1.1 Operational Context and the Multi-Echelon Retail Dilemma
Modern retail supply chains operate under severe economic tension between customer service levels and working capital efficiency. In large-scale supermarket networks, category managers oversee tens of thousands of stock-keeping units (SKUs) spanning dry groceries, household goods, and highly perishable fresh produce. The operational reality of these networks is defined by three pervasive structural characteristics:

1. **Non-Zero Replenishment Lead Times:** Retail distribution centers operate under positive supplier lead times ($L = 7$ days in our empirical setting), requiring store replenishment orders placed today to bridge customer demand across an extended forward horizon. Consequently, point forecasts for the immediate next day ($t+1$) are insufficient; store managers require coherent multi-horizon trajectories extending across the full pipeline horizon $[t+1, t+L]$.
2. **Heavy-Tailed, Intermittent Consumption Spikes:** Supermarket sales exhibit severe intermittency punctuated by localized demand shocks driven by marketing promotions, calendar holidays, and bi-monthly payday liquidity cycles (specifically the 15th and final days of each calendar month). Because demand distributions are strongly right-skewed, conditional expectations fail to capture the probability mass residing in extreme tail events.
3. **Severe Economic Cost Asymmetries:** The financial consequences of misestimating demand are fundamentally asymmetric. Carrying excess dry goods incurs warehouse holding charges ($h_{\\text{dry}} = \\$0.10$/unit-day), while holding excess fresh foods incurs both physical capital holding and accelerated biological obsolescence ($h_{\\text{perish}} = \\$0.15$/unit-day). Conversely, under-ordering triggers immediate stockouts, resulting in unrecoverable lost sales, brand erosion, and contractual service-level agreement (SLA) penalties ($c_p = \\$2.50$/unit). Under high penalty ratios ($c_p / h \\ge 16.7$), symmetric error minimization leads directly to systematic under-stocking during critical revenue surges.

## 1.2 The "Forecast-then-Optimize" Disconnect: Decoupling Accuracy and Operational Value
The dominant paradigm in data-driven supply chain management is the sequential two-stage "Forecast-then-Optimize" architecture. In the first stage, machine learning practitioners train statistical or deep neural networks to minimize standard loss functions—most notably Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or Huber loss. In the second stage, operations researchers ingest these conditional demand predictions into inventory replenishment models, typically applying classical $(s, S)$ or base-stock policies supplemented by parametric Gaussian safety stocks ($z \\cdot \\sigma \\sqrt{L}$).

This traditional decoupling rests on an implicit, widely accepted premise: *improving statistical forecasting accuracy automatically translates to superior downstream inventory performance.*

In this paper, we demonstrate empirically that this foundational premise is flawed. When demand distributions are non-Gaussian and cost structures are asymmetric, statistical point accuracy and inventory decision value become decoupled. Two deep neural network architectures achieving statistically identical point forecast errors (WAPE 28.09% vs. 27.99%) can yield dramatically divergent supply chain costs. A conditional expectation forecaster (Huber TFT) trained to balance positive and negative deviations produces point estimates that systematically underestimate the variance required to absorb multi-day replenishment shocks. To prevent stockouts, downstream optimizers are forced to inflate order-up-to base stocks, artificially bloating warehouse working capital. Conversely, a multi-quantile architecture explicitly trained on asymmetric pinball loss allows downstream policies to decouple median cycle demand from upper-tail protection, yielding lower total costs at higher service levels without requiring post-hoc parametric residual fitting.

## 1.3 Methodological Traps in Existing Machine Learning Inventory Literature
A secondary, equally critical motivation for this study arises from widespread methodological vulnerabilities in empirical machine learning supply chain literature. In reviewing recent predictive inventory studies, three recurrent simulation flaws distort reported performance gains:

1. **Test-Window Parameter Tuning Leakage:** Numerous published studies tune metaheuristic hyperparameters (such as safety stock multipliers and order-up-to targets) directly on the evaluation test horizon, or allow downstream policies to ingest realized test-period statistics (such as test mean sales). This introduces severe lookahead bias, producing optimistic cost reductions that collapse in real-world deployment.
2. **Initial Inventory Subsidization:** In discrete-event simulation studies spanning finite evaluation windows (e.g., 14 to 30 days), on-hand starting inventory is frequently initialized as a function of the policy's own order-up-to target ($I_0 = \\gamma \\cdot S$). Because base-stock targets vary across policies, this convention awards policies with larger target inventories substantially more initial stock for free, artificially inflating their fill rates over short simulation horizons.
3. **Horizon Truncation Cost Artifacts:** When simulating supply chains with positive lead times ($L = 7$) over finite windows (e.g., 15 days), purchase orders placed toward the end of the window ($t \\in [8, 14]$) incur purchase order dispatch fees but cannot physically arrive before the simulation terminates ($t + 7 \\ge 15$). Charging order fees for trailing orders whose delivery benefits are never credited introduces severe boundary distortion.

Without establishing rigorous, leak-free, and mathematically invariant simulation protocols, empirical comparisons between machine learning models and classical heuristics remain fundamentally unreliable.

## 1.4 Core Contributions of This Work
To resolve these foundational challenges, this study presents an end-to-end empirical investigation spanning data engineering, deep time-series modeling, distribution-free uncertainty quantification, and discrete-event inventory simulation. Our primary contributions are fourfold:

1. **Multi-Horizon Quantile Modeling vs. Patch Transformers:** We formulate and train a Multi-Horizon Quantile Temporal Fusion Transformer (Quantile-TFT) predicting the 10th, 50th, and 90th percentiles ($P_{10}, P_{50}, P_{90}$) over a 7-day replenishment pipeline across 400 store-item pairs from Corporación Favorita (125.5 million transactions). We conduct an 8-point error diagnostic explaining why single-point models exhibit RMSE/MAE divergence (2.39x), proving that 74.5% of squared error is driven by 20 high-volume SKUs. We benchmark TFT against a channel-independent PatchTST transformer and classical heuristics, proving that TFT's multi-modal embedding layers are vital for multi-echelon retail.
2. **Mondrian Split-Conformal Coverage Guarantees:** We demonstrate that uncalibrated deep quantile heads suffer localized under-coverage on volatile high-volume SKUs (87.36% coverage against a 90% target). We integrate a validation-anchored Mondrian split-conformal calibration mechanism that provides distribution-free finite-sample guarantees, restoring test-window high-volume coverage to 91.20% while preserving sharp interval bounds.
3. **The Locked Invariant Simulation Protocol:** We formulate and mathematically prove the Lead-Time Invariance Theorem: under lead time $L=7$ and fair steady-state initialization ($I_0 = 2\\bar{D}_{\\text{pre}}$, prefilled pipeline), inventory trajectories across Days 0 to 6 are mathematically invariant across all policies ($0.00 difference), locking historical holding and stockout costs to an exact constant ($45,757.37). We establish causal arriving-order accounting (W1 rule) and provide a comprehensive accounting waterfall ledger (Table 3) quantifying how historical lookahead bias artificially distorted costs by +$7,827 and baseline subsidization by -$28,921.
4. **Comprehensive Empirical and Metaheuristic Benchmarking:** We evaluate downstream dynamic $(s, S)$ policies using Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO) across 10 random seeds, validated against an exhaustive 1,426-point 2D grid search. We conduct paired bootstrap hypothesis testing ($B=2,000$ resamples) proving that Quantile TFT saves $8,251 over Huber TFT ($p=0.0010$). Finally, we construct a non-dominated monotone matched-fill Pareto frontier ($C^*(T)$), proving that conformal quantile buffering dominates all classical heuristics at commercial service levels ($\ge 94.0\%$).

## 1.5 Organization of the Manuscript
The remainder of this article is structured as follows: Section 2 reviews related literature across deep time-series forecasting, conformal prediction, and inventory optimization. Section 3 details the formal problem formulation, supply chain environment, and asymmetric cost model. Section 4 presents our methodology, covering data engineering, TFT and PatchTST architectures, conformal calibration, dynamic safety stocks, and metaheuristic tuning. Section 5 details the locked invariant simulation protocol and mathematical proofs. Section 6 presents comprehensive empirical results, bootstrap significance tests, matched-fill Pareto frontiers, and economic stress testing. Section 7 discusses managerial implications, Section 8 outlines research limitations, and Section 9 concludes with future directions.
"""
