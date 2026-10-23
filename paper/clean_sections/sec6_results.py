# -*- coding: utf-8 -*-
"""
Clean Section 6: Experimental Results and Audit Findings
Contains clean narrative text, Table callout boxes, data tables, and Figure callouts with embedded graphics.
"""

SECTION_6_PARAGRAPHS = [
    ("6. EXPERIMENTAL RESULTS & AUDIT FINDINGS", "h1"),
    ("In this section, we present the consolidated experimental results, hypothesis tests, and ablation analyses evaluated across all 363 active store-item pairs on the unseen test window (August 1 to August 15, 2017). All policies were tuned strictly on the validation window (July 17–31, 2017) and evaluated one-shot forward.", "p"),
    
    ("6.1 Multi-Horizon Point and Quantile Forecast Performance", "h2"),
    ("Table 1 evaluates multi-horizon point forecast accuracy across all 3,237 sequence evaluation windows (22,659 row observations). Both TFT models achieve substantial, statistically decisive accuracy gains over traditional supply chain methods, reducing MAE by 25.1% relative to the trailing 28-day mean and 32.7% relative to Seasonal Naive. Non-parametric pair-level Wilcoxon signed-rank tests (N = 363) confirm statistical superiority at p < 0.0001 against both baselines (pair win rates: 69.4% vs. Mean, 91.7% vs. Seasonal Naive).", "p"),
    ("Crucially, comparing Quantile TFT (median P50) against Huber TFT, the models exhibit virtually identical point accuracy: WAPE is 28.09% vs. 27.99% (a 0.10% difference), and MAE is 12.43 vs. 12.39 units. The pair-level Wilcoxon signed-rank test yields W = 31,136.0 (p = 0.3430), and paired bootstrap resamples on per-pair MAE (B = 2,000) yield a mean difference of -0.2466 units with a 95% confidence interval of [-1.1493, +0.5831], firmly spanning zero. Statistically, the two TFT models possess identical point forecasting capability.", "p"),
    ("While MAE is identical, Huber TFT achieves a lower RMSE (29.54 vs. 35.23 units) because Huber loss pulls predictions toward outlier spikes, whereas the P50 median minimizes absolute deviations, maintaining baseline operational stability. PatchTST achieves a 29.95% WAPE, trailing TFT by 1.86% to 1.96% absolute WAPE due to its channel-independent design.", "p"),

    ("6.2 Conformal Coverage Calibration & Sharpness Guarantees", "h2"),
    ("Table 2 evaluates empirical coverage of the upper-tail P90 bound before and after Mondrian split-conformal calibration across all 23,093 validation and 22,659 test sequence observations with series-bootstrap 95% confidence intervals (B = 2,000).", "p"),
    ("On validation, raw Quantile TFT under-covered volatile high-volume SKUs (87.36% coverage, CI [85.87, 88.76]), violating the nominal 90.0% coverage requirement. Adding the validation conformal offset +4.5766 restores validation coverage to exactly 90.04% (CI [88.67, 91.38]). Transferred to unseen test data, calibrated coverage achieves 91.20% on high-volume SKUs (2,860 of 3,136 observations covered) and 91.01% overall, fully satisfying enterprise service-level requirements without interval explosion (the calibrated P10–P90 interval covered 80.03% of test observations with an average width of 35.81 units).", "p"),

    ("6.3 Downstream Replenishment Performance & Cost Breakdown", "h2"),
    ("Table 4 summarizes downstream replenishment outcomes for the primary policy variants selected strictly by minimum validation fitness. Despite possessing equal point accuracy, the Quantile TFT policy achieves a primary test cost of $72,759.10 compared to $81,010.06 for Huber TFT—delivering a net operating cost saving of $8,250.96 (a 10.2% cost reduction). Quantile TFT achieves a test fill rate of 94.66% over days 7 to 14, outperforming Huber TFT (92.97%), Seasonal Naive (92.69%), and Trailing Mean (91.87%).", "p"),

    ("6.4 Factorial Safety Stock Ablation (6 Buffer Architectures)", "h2"),
    ("Table 6 evaluates the secondary operational regime, restricting safety buffers to model-derived spreads without fitting 363 pair-specific validation residual statistics. Without a dedicated safety buffer (alpha = 0), Huber TFT cannot bridge 7-day replenishment delays. Under a 95% service penalty, the optimizer drives beta to the ceiling (beta* = 4.9979). Widening the search space to beta <= 8.0 inflates holding costs to $107,955.34 on test without reaching feasibility (achieving only 91.77% fill). This proves that conditional point forecasters cannot protect against lead-time shocks through cycle stock alone. In contrast, Quantile Conformal achieves $66,918.16 and 94.74% fill, saving $21,160.29 over Huber None (p = 0.0005) and $8,499.34 over Seasonal Naive None (p = 0.0050).", "p"),

    ("6.5 Paired Bootstrap Significance (B = 2,000 Resamples)", "h2"),
    ("To verify that the $8,251 cost saving achieved by Quantile TFT is statistically robust, Table 5 executes paired non-parametric bootstrap resampling (B = 2,000 iterations) across store-item pairs. Decomposing the $8,250.95 saving achieved by Quantile TFT over Huber TFT reveals that cost reductions stem from balanced physical inventory control: Holding Cost Reduction = -$4,400.12 (53.3% of savings); Stockout Penalty Reduction = -$3,248.83 (39.4% of savings); Order Fee Reduction = -$602.00 (7.3% of savings). Both 95% bootstrap intervals for the Huber comparison exclude zero (p = 0.0010 across all pairs, and p = 0.0005 excluding the 20 largest SKUs). Against naive heuristics, cost differences are within bootstrap noise, but Quantile TFT provides statistically significant fill rate gains (+2.79% over Trailing Mean, and +2.95% over Seasonal Naive on standard SKUs).", "p"),

    ("6.6 Non-Dominated Monotone Matched-Fill Pareto Frontier", "h2"),
    ("To evaluate policy capacity limits independently of exterior penalty weights, Table 7 and Figure 4 present the Non-Dominated Matched-Fill Pareto Lower Envelope, defined as the empirical lower bound over all grid configurations achieving a test customer fill rate at or above target T: C*(T) = min { Cost(c) : Fill(c) >= T }. We emphasize that while our primary deployed replenishment policies (Table 4) were tuned strictly out-of-sample on validation data with zero test leakage, this matched-fill frontier functions as a retrospective capacity benchmark under parameter hindsight—establishing the theoretical frontier of what each forecasting paradigm can achieve when tuned to hit exact test service levels. At moderate service levels (90% to 92%), trailing mean baselines are marginally cheaper ($63,786 vs. $64,183, a 0.6% gap) because lower fill rates permit frequent stockouts, favoring policies that carry minimal safety stock. However, at the enterprise standard 95.0% Fill Rate, Huber Gaussian cannot reach feasibility, while Quantile Conformal achieves $69,253.95, outperforming Mean Gaussian by $1,477.70 (2.1%) and Seasonal Naive by $12,887.62 (15.7% net savings). At 96.0% Fill Rate, Trailing Mean becomes completely infeasible, and Quantile Conformal achieves $97,967.62, beating Seasonal Naive by $14,078.92 (12.6% net savings). Conformal quantile buffering expands the achievable operational frontier.", "p"),

    ("6.7 Economic Stress Testing (c_p, Holding, and Fee Sensitivity)", "h2"),
    ("Table 8 and Figure 5 evaluate the multi-seed metaheuristic optimization benchmark across 10 random seeds on the primary fair specification (Quantile TFT + Conformal Buffer, rolling mode). We report both the Primary Arrival-Aligned Diagnostic Cost (Days 7–14, W1 rule: $64,447.86 +- $345.14) and the Full-Window Operational Cost (Days 0–14: $110,205.23 +- $345.14), reconciling simulation costs across accounting definitions. Both PSO and GWO exhibit rapid convergence within 5 iterations, differing by only 0.04% in validation fitness and 0.15% in test cost across all 10 seeds. Benchmarked against the exhaustive 2D grid, both metaheuristics locate the true minimum basin within 0.8% of grid cost in less than 10% of the computation time.", "p"),
    ("Executing full re-optimization across 14 economic scenarios spanning stockout penalties (c_p in [$1.0, $10.0]), holding multipliers (0.5x to 1.5x), and order fee multipliers (0.5x to 1.5x) confirms that Quantile Conformal achieves Rank 1 (lowest operating cost) in 13 out of 14 scenarios (c_p >= $2.50). When penalty costs rise to c_p = $10.00, Quantile Conformal achieves $75,491.09, outperforming Mean Gaussian ($82,061.11) and Huber Residual ($120,791.63), while Huber None incurs cost penalties up to +84.48% above the champion policy.", "p")
]

TABLE_1_DATA = {
    "box_title": "TABLE 1 PLACEHOLDER & SPECIFICATION: Test-Window Point Forecast Accuracy Benchmark",
    "items": [
        ("Table Number", "Table 1 (Table 6.1 in Text)"),
        ("Table Title", "Test-Window Point Forecast Accuracy Benchmark (22,659 Sequence Forecast Observations)"),
        ("Key Evaluation Metrics", "MAE, RMSE, WAPE (Primary), Strict sMAPE (0/0=0), Pinball P50 Loss"),
        ("Key Finding Shown", "Quantile TFT (P50) and Huber TFT achieve statistically equal point accuracy (WAPE 28.09% vs 27.99%; Wilcoxon p = 0.343), while dominating classical baselines (p < 0.0001).")
    ],
    "headers": ["Forecasting Model", "Architectural Paradigm", "MAE (units)", "RMSE (units)", "WAPE (%) [Primary]", "Strict sMAPE (%) [0/0=0]", "Pinball P50 Loss"],
    "rows": [
        ["Quantile TFT (P50)", "Multi-Quantile Transformer Head", "12.4326", "35.2266", "28.09%", "37.15%", "6.2163"],
        ["Huber TFT", "Single-Point Conditional Mean Head", "12.3874", "29.5425", "27.99%", "38.13%", "6.1937"],
        ["PatchTST Benchmark", "Channel-Independent Patch Transformer", "13.2641", "31.8012", "29.95%", "39.42%", "N/A"],
        ["Trailing 28-Day Mean", "Classical Non-Parametric Moving Avg", "16.6099", "42.7669", "37.52%", "45.06%", "8.3050"],
        ["Seasonal Naive (Lag 7)", "Deterministic Day-of-Week Heuristic", "18.4068", "43.1502", "41.58%", "50.09%", "9.2034"]
    ]
}

TABLE_2_DATA = {
    "box_title": "TABLE 2 PLACEHOLDER & SPECIFICATION: Empirical Coverage of One-Sided P90 Bound",
    "items": [
        ("Table Number", "Table 2 (Table 6.2 in Text)"),
        ("Table Title", "Empirical Coverage of the One-Sided P90 Bound Across Validation and Test Splits"),
        ("Key Evaluation Metrics", "Observation Count (N), Raw Uncalibrated Coverage (%), Calibrated Coverage (%), 95% Bootstrap CIs"),
        ("Key Finding Shown", "Split-conformal offsets (+0.18 standard, +4.58 high-volume) restore high-volume test coverage from 89.80% to 91.20%, guaranteeing nominal 90% SLA.")
    ],
    "headers": ["Partition", "Demand Stratum", "Observation Count (N)", "Raw Uncalibrated Coverage (%)", "Calibrated Conformal Coverage (%)", "Offset Applied", "Nominal Target (%)"],
    "rows": [
        ["Validation", "Overall Cohort", "23,093", "89.26% [88.76, 89.74]", "90.01% [89.53, 90.47]", "Tier-Specific", "90.00%"],
        ["Validation", "Standard Tier (<100)", "19,880", "89.57% [89.06, 90.08]", "90.01% [89.51, 90.51]", "+0.1785 units", "90.00%"],
        ["Validation", "High-Volume Tier (>=100)", "3,213", "87.36% [85.87, 88.76]", "90.04% [88.67, 91.38]", "+4.5766 units", "90.00%"],
        ["Test", "Overall Cohort", "22,659", "90.49% [90.05, 90.93]", "91.01% [90.57, 91.43]", "Tier-Specific", "90.00%"],
        ["Test", "Standard Tier (<100)", "19,523", "90.61% [90.12, 91.07]", "90.98% [90.52, 91.44]", "+0.1785 units", "90.00%"],
        ["Test", "High-Volume Tier (>=100)", "3,136", "89.80% [88.55, 90.98]", "91.20% [90.02, 92.32]", "+4.5766 units", "90.00%"]
    ]
}

TABLE_4_DATA = {
    "box_title": "TABLE 4 PLACEHOLDER & SPECIFICATION: Validation-Selected Replenishment Policy Performance",
    "items": [
        ("Table Number", "Table 4 (Table 6.3 in Text)"),
        ("Table Title", "Downstream Replenishment Performance on Primary Selection Regime (363 Active Test Pairs)"),
        ("Key Evaluation Metrics", "Optimal Parameters (alpha*, beta*), Validation Fitness, Primary Window Cost ($), Test Fill Rate (%)"),
        ("Key Finding Shown", "Quantile TFT achieves $72,759.10 test cost vs $81,010.06 for Huber TFT ($8,251 savings / 10.2% cost reduction) and higher fill rate (94.66% vs 92.97%).")
    ],
    "headers": ["Forecaster Model", "Selected Buffer Architecture", "Execution Form", "Optimal alpha*", "Optimal beta*", "Val Fitness ($)", "Val Fill (7-14) (%)", "Test Cost (Prim 7-14) ($)", "Test Fill (Prim 7-14) (%)"],
    "rows": [
        ["Quantile TFT", "empirical_residual", "Static", "0.9286", "1.2163", "92,154.21", "96.57%", "$72,759.10", "94.66%"],
        ["Huber TFT", "empirical_residual", "Static", "1.0805", "0.8435", "92,968.11", "96.58%", "$81,010.06", "92.97%"],
        ["Seasonal Naive", "empirical_residual", "Static", "0.8272", "1.2164", "97,208.26", "94.54%", "$69,755.20", "92.69%"],
        ["Trailing Mean", "empirical_residual", "Rolling", "0.7725", "0.9256", "94,785.63", "96.91%", "$70,806.09", "91.87%"]
    ]
}

TABLE_5_DATA = {
    "box_title": "TABLE 5 PLACEHOLDER & SPECIFICATION: Paired Bootstrap Statistical Significance Benchmark",
    "items": [
        ("Table Number", "Table 5 (Table 6.4 in Text)"),
        ("Table Title", "Paired Bootstrap Significance Benchmark (B = 2,000 Resamples Across Pairs)"),
        ("Key Evaluation Metrics", "Mean Cost Delta ($), 95% Bootstrap CI ($), Cost p-value, Fill Delta (%), Win Rate (%)"),
        ("Key Finding Shown", "Quantile TFT statistically outperforms Huber TFT by -$8,251 (p = 0.0010; holding -$4,400, stockout -$3,200), with both CIs strictly excluding zero.")
    ],
    "headers": ["Comparator Model", "Evaluated Store-Item Cohort", "Mean Cost Delta ($)", "95% Bootstrap CI ($)", "p-value (Cost)", "Mean Fill Delta (%)", "95% Bootstrap CI (%)", "Pair Win Rate (%)"],
    "rows": [
        ["Huber TFT", "All 363 Active Pairs", "-$8,250.95", "[-$19,104.83, -$2,126.05]", "0.0010", "+1.69%", "[-0.70%, +5.57%]", "66.39%"],
        ["Huber TFT", "Excl. Top 20 High-Volume (N=343)", "-$2,813.11", "[-$3,862.10, -$1,798.18]", "0.0005", "+0.01%", "[-0.41%, +0.43%]", "66.18%"],
        ["Seasonal Naive", "All 363 Active Pairs", "+$3,003.91", "[-$2,545.25, +$10,040.93]", "0.3610", "+1.97%", "[-1.51%, +4.63%]", "38.02%"],
        ["Seasonal Naive", "Excl. Top 20 High-Volume (N=343)", "+$1,110.20", "[-$2,012.47, +$3,941.53]", "0.4370", "+2.95%", "[+2.07%, +3.96%]", "37.03%"],
        ["Trailing Mean", "All 363 Active Pairs", "+$1,953.02", "[-$2,133.14, +$5,946.68]", "0.3210", "+2.79%", "[+1.25%, +4.16%]", "46.56%"],
        ["Trailing Mean", "Excl. Top 20 High-Volume (N=343)", "+$218.08", "[-$2,447.79, +$3,066.21]", "0.8800", "+3.24%", "[+2.00%, +4.41%]", "47.52%"]
    ]
}

TABLE_6_DATA = {
    "box_title": "TABLE 6 PLACEHOLDER & SPECIFICATION: Secondary Regime: Replenishment Without Fitted Residuals",
    "items": [
        ("Table Number", "Table 6 (Table 6.5 in Text)"),
        ("Table Title", "Secondary Regime: Factorial Replenishment Evaluation Without Fitted Residuals"),
        ("Key Evaluation Metrics", "Optimal Parameters, Primary Test Cost ($), Test Customer Fill Rate (%), Methodological Role"),
        ("Key Finding Shown", "Huber None experiences bound-hitting ceiling failure (beta -> 5.0, 8.0) and $107k cost; Quantile Conformal achieves $66,918 and 94.74% fill rate.")
    ],
    "headers": ["Replenishment Policy", "Forecaster Model", "Buffer Type", "Execution Form", "Optimal alpha*", "Optimal beta*", "Test Cost ($)", "Test Fill Rate (%)", "Methodological Role"],
    "rows": [
        ["Quantile Conformal", "Quantile TFT", "conformal_quantile", "Static", "0.1648", "1.2172", "$66,918.16", "94.74%", "Primary Proposed Policy"],
        ["Quantile Raw Spread", "Quantile TFT", "raw_quantile", "Static", "0.1892", "1.2172", "$67,157.98", "94.99%", "Uncalibrated Model Head"],
        ["Trailing Mean Gaussian", "Trailing Mean", "gaussian_demand", "Rolling", "0.0147", "1.4765", "$65,345.74", "90.59%", "Classical Parametric Baseline"],
        ["Seasonal Naive None", "Seasonal Naive", "none", "Rolling", "0.0000", "2.4629", "$75,417.49", "92.77%", "Heuristic Base-Stock Baseline"],
        ["Huber None (Bound <= 5)", "Huber TFT", "none", "Rolling", "0.0000", "4.9979", "$88,078.45", "90.46%", "Bound-Ceiling Failure"],
        ["Huber None (Bound <= 8)", "Huber TFT", "none", "Rolling", "0.0000", "8.0000", "$107,955.34", "91.77%", "Inflated Holding Penalty"]
    ]
}

TABLE_7_DATA = {
    "box_title": "TABLE 7 PLACEHOLDER & SPECIFICATION: Matched-Fill Lower Envelope Cost Frontier (C*(T))",
    "items": [
        ("Table Number", "Table 7 (Table 6.6 in Text)"),
        ("Table Title", "Matched-Fill Lower Envelope Cost Frontier: Minimum Cost to Reach Target Fill Rate"),
        ("Key Evaluation Metrics", "Target Fill Rate (90% to 96%), Operating Cost Across Models, Operational Frontier Status"),
        ("Key Finding Shown", "Quantile Conformal dominates all policies at fill rates >= 94%: at 95% fill, Huber is unreachable and Quantile beats Naive by $12,888 (15.7% savings); at 96% fill, Mean is unreachable and Quantile beats Naive by $14,079 (12.6% savings).")
    ],
    "headers": ["Target Fill Rate (T)", "Quantile Conformal ($)", "Trailing Mean ($)", "Seasonal Naive ($)", "Huber Gaussian ($)", "Operational Frontier Status"],
    "rows": [
        ["90.0%", "$64,183.77", "$63,786.58", "$70,697.54", "$69,147.68", "Mean and Quantile match within 0.6%"],
        ["92.0%", "$64,183.77", "$63,786.58", "$70,697.54", "$70,311.05", "Mean and Quantile match within 0.6%"],
        ["94.0%", "$65,155.90", "$65,169.56", "$71,786.12", "$93,345.65", "Huber cost inflates (+43.3%); Quantile matches Mean"],
        ["95.0%", "$69,253.95", "$70,731.65", "$82,141.57", "Unreachable", "Huber Infeasible; Quantile beats Naive by $12,888 (15.7%)"],
        ["96.0%", "$97,967.62", "Unreachable", "$112,046.54", "Unreachable", "Mean Infeasible; Quantile beats Naive by $14,079 (12.6%)"],
        ["97.0%", "Unreachable", "Unreachable", "Unreachable", "Unreachable", "System physical capacity bound"]
    ]
}

TABLE_8_DATA = {
    "box_title": "TABLE 8 PLACEHOLDER & SPECIFICATION: Multi-Seed Metaheuristic Benchmark (PSO vs. GWO vs. 2D Grid)",
    "items": [
        ("Table Number", "Table 8 (Table 6.7 in Text)"),
        ("Table Title", "Multi-Seed Metaheuristic Benchmark Across 10 Random Seeds (Mean +- Std)"),
        ("Key Evaluation Metrics", "Optimal alpha*, Optimal beta*, Validation Fitness ($), Test Operating Cost ($), Test Fill Rate (%), Runtime"),
        ("Key Finding Shown", "PSO and GWO independently converge to virtually identical parameter regimes (within 0.04% in fitness, 0.15% in test cost) matching the 2D grid ground truth in under 10% of computation time.")
    ],
    "headers": ["Optimization Algorithm", "Optimal alpha*", "Optimal beta*", "Val Fitness ($)", "Test Operating Cost ($)", "Test Fill Rate (%)", "Convergence Time (s)"],
    "rows": [
        ["Particle Swarm Optimization (PSO)", "0.2075 +- 0.0312", "1.1607 +- 0.0903", "$100,412.26 +- 85.83", "$64,447.86 +- 345.14", "93.57 +- 0.35%", "21.4 +- 1.2 s"],
        ["Grey Wolf Optimizer (GWO)", "0.2274 +- 0.0216", "1.0858 +- 0.0868", "$100,369.60 +- 75.94", "$64,546.64 +- 265.59", "93.82 +- 0.23%", "24.8 +- 1.5 s"],
        ["Exhaustive 2D Grid (31x46 = 1,426 pts)", "0.2500", "0.9000", "$100,658.29", "$64,950.35", "94.04%", "284.2 s"]
    ]
}

FIG_4_SPEC = {
    "box_title": "FIGURE 4 PLACEHOLDER & SPECIFICATION: Non-Dominated Matched-Fill Pareto Frontier",
    "items": [
        ("Figure Number", "Figure 4"),
        ("Caption", "Figure 4. Non-Dominated Matched-Fill Pareto Frontier C*(T): Minimum supply chain operating cost required to achieve customer service levels from 90.0% to 96.0%. Illustrates how Quantile Conformal expands the achievable operational frontier at enterprise service levels (>= 94.0%) where single-point models become infeasible."),
        ("Target Image Filename", "r4_matched_fill_frontier.png"),
        ("Image Location", r"revision_experiments\outputs\figures\r4_matched_fill_frontier.png"),
        ("Key Finding Shown", "Quantile Conformal achieves $69,254 at 95% fill where Huber is unreachable, delivering 15.7% net savings over Seasonal Naive.")
    ]
}

FIG_5_SPEC = {
    "box_title": "FIGURE 5 PLACEHOLDER & SPECIFICATION: Swarm Convergence Dynamics (PSO vs. GWO)",
    "items": [
        ("Figure Number", "Figure 5"),
        ("Caption", "Figure 5. Nature-Inspired Swarm Convergence Dynamics: 20-iteration convergence curves for Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO) across 10 random seeds on the primary fair specification, demonstrating rapid descent to the optimal cost plateau within 5 iterations."),
        ("Target Image Filename", "pso_vs_gwo_convergence_quantile.png"),
        ("Image Location", r"Multi-Horizon Quantile Forecasting(process)\pso_vs_gwo_convergence_quantile.png"),
        ("Key Finding Shown", "Both stochastic metaheuristics independently replicate the optimal basin without premature entrapment.")
    ]
}
