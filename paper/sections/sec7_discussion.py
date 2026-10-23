# -*- coding: utf-8 -*-
"""
Section 7: Discussion and Managerial Implications
"""

SECTION_7_TEXT = """
# 7. Discussion and Managerial Implications

The empirical findings presented in Section 6 provide critical insights for both operations research theory and enterprise supply chain engineering. In this section, we synthesize the broader implications of our results, examine why statistical accuracy decouples from decision value, detail the methodological lessons uncovered during our audits, and outline actionable heuristics for retail practitioners.

## 7.1 The Decoupling of Point Accuracy and Operational Value
For decades, supply chain management literature has operated under the working assumption that improving demand forecasting accuracy directly translates into inventory cost reductions. Our findings decisively challenge this premise in asymmetric retail environments.

As established in Table 6.1 and confirmed via non-parametric Wilcoxon hypothesis testing ($p = 0.3430$), Quantile TFT ($P_{50}$) and Huber TFT are statistically indistinguishable in point forecasting accuracy: WAPE is 28.09% vs. 27.99%, and MAE is 12.43 vs. 12.39 units. Under traditional forecasting benchmarks, an operations manager would judge these two models as functionally equivalent, or marginally favor the Huber model due to its superior RMSE (29.54 vs. 35.23 units).

Yet, when deployed into an identical discrete-event replenishment simulator under identical parameter tuning, the Quantile TFT policy achieves an operational cost reduction of **$\\$8,251** (95% bootstrap CI $[-\\$19,105, -\\$2,126]$, $p = 0.0010$). Conversely, a Seasonal Naive heuristic with a 48.0% worse MAE (18.41 vs. 12.43 units) achieves a total cost within bootstrap noise of the deep learning model at low service targets.

### The Operational Mechanism
Why does point accuracy decouple from inventory performance? The answer lies in the interaction between loss functions and asymmetric replenishment dynamics:
1. **The Flaw of Conditional Expectation in Supply Chains:** When trained via symmetric $L_2$ or Huber loss, single-point models estimate the conditional expectation $\\mathbb{E}[Y \\mid X]$. Because supermarket grocery demand is punctuated by intermittent, heavy-tailed spikes (where 20 SKUs generate 74.5% of squared error), the conditional mean is pulled upward to balance large positive errors. On normal transaction days, the point forecast over-estimates baseline demand, leading to chronic over-ordering.
2. **The Inability to Absorb Multi-Day Lead Times:** Despite pulling upward, the conditional mean does not carry explicit variance information over the 7-day transit pipeline. When a true promotional surge arrives, the point forecast is insufficient to cover the demand shock. Downstream optimizers are forced to inflate the order-up-to multiplier ($\\beta \\to 5.0$), locking up working capital in static cycle stock.
3. **The Advantage of Functional Quantile Specialization:** By utilizing Pinball loss, Quantile TFT separates baseline tracking from risk buffering. The median head ($P_{50}$) ignores outlier spikes, tracking baseline demand with minimal inventory holding ($-\\$4,400$ in holding savings). Concurrently, the upper quantile head ($P_{90}$) isolates surge volatility, dynamically scaling safety stock ($P_{90} - P_{50}$) only when promotional flags or calendar indicators signal impending spikes ($-\\$3,200$ in stockout savings).

**Managerial Principle 1:** *Point forecasting error metrics (MAE, RMSE, WAPE) are unreliable proxies for inventory health in asymmetric retail environments. Deep learning architectures must be evaluated on downstream operational decision loss rather than statistical loss alone.*

## 7.2 Why Distribution-Free Quantile Spreads Outperform Parametric Residuals
In traditional supply chain practice, safety stock sizing relies on post-hoc parametric modeling: practitioners fit a historical standard deviation of forecast residuals ($\\hat{\\sigma}_{\\text{res}}$) or an empirical error percentile ($\\hat{q}_{0.90}$) for every active SKU across the catalog (Silver et al., 1998).

Our experimental results demonstrate that **dynamic quantile spreads derived directly from transformer heads outperform parametric residual models while eliminating post-hoc parameter fitting**:
1. **Elimination of $N$ Fitted Parameters:** In our 363-pair cohort, standard residual buffering requires estimating and maintaining 363 pair-specific residual standard deviations or quantiles. In enterprise catalogs with 50,000 SKUs, maintaining these residual statistics across non-stationary regimes introduces severe operational overhead and estimation error.
2. **Dynamic Responsiveness vs. Static Historical Windows:** A rolling standard deviation $\\hat{\\sigma}_{\\text{res}}$ is backward-looking: it reflects average errors over the past 28 days. If a marketing promotion or holiday is scheduled for tomorrow, a historical residual variance cannot anticipate the surge. In contrast, Quantile TFT's upper head ($P_{90}$) ingests future calendar features and promotional schedules, dynamically widening the safety buffer specifically for high-risk days and contracting it immediately after (as illustrated in Figure 4.3 / `quantile_safety_stock_distribution.png`).
3. **Protection via Conformal Guarantees:** By augmenting the model-derived spread with two scalar Mondrian conformal offsets ($+0.1785$ and $+4.5766$), the enterprise obtains finite-sample coverage guarantees across the catalog without fitting series-specific variances.

**Managerial Principle 2:** *Deriving dynamic safety buffers from deep quantile spreads replaces thousands of static, backward-looking residual variance parameters with a forward-looking, calendar-aware uncertainty envelope.*

## 7.3 Methodological Lessons for Data-Driven Supply Chain Research
The accounting waterfall audit documented in Table 5.1 provides cautionary lessons for the machine learning and operations research communities:

1. **Beware of Initial Inventory Subsidization:** Simulating inventory policies with initial stock proportional to target levels ($I_0 = \\gamma S$) is widespread in literature. As proved in Table 5.1, this convention shifted absolute simulation costs by **$\\$28,920**—completely confounding policy comparisons. Researchers must adopt policy-independent starting states ($I_0 = 2\\bar{D}_{\\text{pre}}$, prefilled pipeline) and prove lead-time invariance ($t < L$) to ensure fair evaluation.
2. **Beware of Test Lookahead in Replenishment Baselines:** Allowing cycle stock targets to scale with realized test mean demand ($S = s + \\beta \\bar{D}_{\\text{test}}$) masked **$\\$7,827** in true operating costs. In production, future sales are unknown; policy targets must scale strictly with causal trailing statistics ($\\bar{D}_{\\text{pre}}$).
3. **Beware of Truncated Horizon Boundary Distortion:** Charging fixed order fees for orders placed toward the end of finite simulations without crediting arriving stock penalizes proactive ordering. Causal arriving-order fee accounting (the W1 rule) resolves this distortion and proves exact cent-level cost reconciliation.

**Managerial Principle 3:** *Reported cost savings from machine learning inventory models are highly vulnerable to simulation leakage and accounting artifacts. Rigorous peer review requires invariant initial conditions, causal parameter definitions, and open-source audit ledgers.*

## 7.4 Actionable Decision Rules for Enterprise Retailers
Based on our multi-horizon experiments, Pareto frontiers, and economic sensitivity grid, we propose a concrete decision matrix for retail category managers:

**Table 7.1. Enterprise Retail Replenishment Policy Decision Matrix**

| Product Segment & Cost Regime | Demand & Operational Characteristics | Recommended Replenishment Strategy | Expected Operational Benefit |
| :--- | :--- | :--- | :--- |
| **Low-Margin, Non-Perishable ($c_p \\le \\$2.00$)** | Low stockout penalties; high holding sensitivity; steady baseline sales. | **Trailing Mean Base-Stock Policy** (Lean Cycle Stock, $\\beta \\approx 1.2$, Zero Safety Buffer) | Matches deep model costs within 0.6%; zero neural network inference overhead. |
| **Moderate Velocity Dry Goods ($c_p \\in [2.5, 4.0]$)** | Standard commercial contracts; 90–92% target fill rate; moderate promo elasticity. | **Quantile TFT with Raw Spread** ($P_{50}$ base orders + raw $[P_{90} - P_{50}]$ buffer) | Delivers 92–94% fill rates with minimal holding capital; eliminates residual fitting. |
| **High-Velocity & Perishable ($c_p \\ge \\$4.00$, SLA $\\ge 95\\%$)** | Severe stockout penalties; high spoilage cost ($h = \\$0.15$); heavy-tailed promo spikes. | **Conformalized Quantile TFT** ($P_{50}$ orders + Mondrian Calibrated $P_{90}$ buffer) | **Sole policy reaching $\\ge 95\\%$ fill without holding explosion; $15.7\\%$ cost savings over Naive.** |
| **Extreme Tail Surge SKUs (Top 5% Volume)** | Account for 74.5% of squared forecast error; payday and marketing surges. | **Tiered Conformal Quantile Buffer** (Applying high-volume offset $\\hat{c}_{\\text{hvol}} = +4.58$) | Restores surge coverage to 91.20%; eliminates recurring stockouts during promotional peaks. |
"""
