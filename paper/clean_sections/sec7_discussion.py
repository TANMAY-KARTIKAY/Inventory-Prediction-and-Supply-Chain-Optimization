# -*- coding: utf-8 -*-
"""
Clean Section 7: Discussion and Managerial Implications
"""

SECTION_7_PARAGRAPHS = [
    ("7. DISCUSSION & MANAGERIAL IMPLICATIONS", "h1"),
    ("The empirical findings presented in Section 6 provide critical insights for both operations research theory and enterprise supply chain engineering. In this section, we synthesize the broader implications of our results, examine why statistical accuracy decouples from decision value, detail the methodological lessons uncovered during our audits, and outline actionable heuristics for retail practitioners.", "p"),
    
    ("7.1 Why Point Accuracy Does Not Equal Operational Efficiency", "h2"),
    ("For decades, supply chain management literature has operated under the working assumption that improving demand forecasting accuracy directly translates into inventory cost reductions. Our findings decisively challenge this premise in asymmetric retail environments.", "p"),
    ("As established in Table 1 and confirmed via non-parametric Wilcoxon hypothesis testing (p = 0.3430), Quantile TFT (P50) and Huber TFT are statistically indistinguishable in point forecasting accuracy: WAPE is 28.09% vs. 27.99%, and MAE is 12.43 vs. 12.39 units. Under traditional forecasting benchmarks, an operations manager would judge these two models as functionally equivalent, or marginally favor the Huber model due to its superior RMSE (29.54 vs. 35.23 units).", "p"),
    ("Yet, when deployed into an identical discrete-event replenishment simulator under identical parameter tuning, the Quantile TFT policy achieves an operational cost reduction of $8,251 (95% bootstrap CI [-$19,105, -$2,126], p = 0.0010). Conversely, a Seasonal Naive heuristic with a 48.0% worse MAE (18.41 vs. 12.43 units) achieves a total cost within bootstrap noise of the deep learning model at low service targets.", "p"),
    ("Why does point accuracy decouple from inventory performance? The answer lies in the interaction between loss functions and asymmetric replenishment dynamics:", "p"),
    ("1. The Flaw of Conditional Expectation in Supply Chains: When trained via symmetric loss, single-point models estimate the conditional expectation. Because supermarket demand is punctuated by intermittent, heavy-tailed spikes (where 20 SKUs generate 74.5% of squared error), the conditional mean is pulled upward to balance large positive errors. On normal transaction days, the point forecast over-estimates baseline demand, leading to chronic over-ordering.", "bullet"),
    ("2. The Inability to Absorb Multi-Day Lead Times: Despite pulling upward, the conditional mean does not carry explicit variance information over the 7-day transit pipeline. When a true promotional surge arrives, the point forecast is insufficient to cover the demand shock. Downstream optimizers are forced to inflate the order-up-to multiplier (beta -> 5.0), locking up working capital in static cycle stock.", "bullet"),
    ("3. The Advantage of Functional Quantile Specialization: By utilizing Pinball loss, Quantile TFT separates baseline tracking from risk buffering. The median head (P50) ignores outlier spikes, tracking baseline demand with minimal inventory holding (-$4,400 in holding savings). Concurrently, the upper quantile head (P90) isolates surge volatility, dynamically scaling safety stock (P90 - P50) only when promotional flags or calendar indicators signal impending spikes (-$3,200 in stockout savings).", "bullet"),

    ("7.2 Methodological Pitfalls in ML-Driven Supply Chain Literature", "h2"),
    ("The accounting waterfall audit documented in Table 3 provides cautionary lessons for the machine learning and operations research communities:", "p"),
    ("1. Beware of Initial Inventory Subsidization: Simulating inventory policies with initial stock proportional to target levels (I_0 = gamma * S) shifted absolute simulation costs by $28,920—completely confounding policy comparisons. Researchers must adopt policy-independent starting states (I_0 = 2 * D̄_pre, prefilled pipeline) and prove lead-time invariance to ensure fair evaluation.", "bullet"),
    ("2. Beware of Test Lookahead in Replenishment Baselines: Allowing cycle stock targets to scale with realized test mean demand masked $7,827 in true operating costs. In production, future sales are unknown; policy targets must scale strictly with causal trailing statistics.", "bullet"),
    ("3. Beware of Truncated Horizon Boundary Distortion: Charging fixed order fees for orders placed toward the end of finite simulations without crediting arriving stock penalizes proactive ordering. Causal arriving-order fee accounting (the W1 rule) resolves this distortion and proves exact cent-level cost reconciliation.", "bullet"),

    ("7.3 Practical Implementation Guidelines for Retailers", "h2"),
    ("Based on our multi-horizon experiments, Pareto frontiers, and economic sensitivity grid, Table 7.1 synthesizes a concrete decision matrix for retail category managers across demand velocity and penalty regimes.", "p")
]

TABLE_7_1_DATA = {
    "box_title": "TABLE 7.1 PLACEHOLDER & SPECIFICATION: Enterprise Retail Replenishment Decision Matrix",
    "items": [
        ("Table Number", "Table 7.1"),
        ("Table Title", "Enterprise Retail Replenishment Decision Matrix across Operational Regimes"),
        ("Key Dimensions", "Product Segment, Cost Regime, Recommended Policy, Expected Operational Benefit"),
        ("Managerial Guidance", "Direct decision rules linking SKU velocity, perishability, and penalty asymmetry to optimal policy choice.")
    ],
    "headers": ["Product Segment & Cost Regime", "Demand & Operational Characteristics", "Recommended Replenishment Strategy", "Expected Operational Benefit"],
    "rows": [
        ["Low-Margin, Non-Perishable (c_p <= $2.00)", "Low stockout penalties; high holding sensitivity; steady baseline sales.", "Trailing Mean Base-Stock Policy (Lean Cycle Stock, beta ~ 1.2, Zero Buffer)", "Matches deep model costs within 0.6%; zero neural network inference overhead."],
        ["Moderate Velocity Dry Goods (c_p in [$2.5, $4.0])", "Standard commercial contracts; 90–92% target fill rate; moderate promo elasticity.", "Quantile TFT with Raw Spread (P50 base orders + raw [P90 - P50] buffer)", "Delivers 92–94% fill rates with minimal holding capital; eliminates residual fitting."],
        ["High-Velocity & Perishable (c_p >= $4.00, SLA >= 95%)", "Severe stockout penalties; high spoilage cost ($0.15); heavy-tailed promo spikes.", "Conformalized Quantile TFT (P50 orders + Mondrian Calibrated P90 buffer)", "Sole policy reaching >= 95% fill without holding explosion; 15.7% cost savings over Naive."],
        ["Extreme Tail Surge SKUs (Top 5% Volume)", "Account for 74.5% of squared forecast error; payday and marketing surges.", "Tiered Conformal Quantile Buffer (Applying high-volume offset +4.58 units)", "Restores surge coverage to 91.20%; eliminates recurring stockouts during promotional peaks."]
    ]
}
