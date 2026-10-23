# -*- coding: utf-8 -*-
"""
Clean Section 5: Methodological Safeguards: The Locked Invariant Simulation Protocol
Contains the Lead-Time Invariance Theorem and the Table 3 Accounting Waterfall.
"""

SECTION_5_PARAGRAPHS = [
    ("5. METHODOLOGICAL SAFEGUARDS: THE LOCKED SIMULATION PROTOCOL", "h1"),
    ("In this section, we document the methodological safeguards formulated to ensure that our simulation framework is completely leak-free, reproducible, and mathematically sound. We establish the Lead-Time Invariance Theorem, define causal arriving-order accounting, and present the Methodological Accounting Waterfall (Table 3), which quantifies the distortion introduced by common simulation conventions.", "p"),
    
    ("5.1 Policy-Independent Fair Initialization (I_0 and Pipeline Steady State)", "h2"),
    ("A common flaw in applied machine learning supply chain literature is the unintentional leakage of test-window realizations into operational policy parameters. In preliminary implementations of (s, S) policies, order-up-to base stock targets are frequently defined as S = s + beta * D̄_test, where D̄_test represents the mean customer sales realized over the test period itself. In our locked specification, all baseline demand estimates utilized in downstream policies are restricted to the causal trailing 28-day window immediately preceding the evaluation split: D̄_{i, pre} = (1/28) * sum_{d=1..28} D_{i, t_start - d}. No future demand, promotional realization, or sales aggregate from the evaluation window is accessible to the replenishment policy.", "p"),
    ("In finite-horizon discrete-event inventory simulations, starting conditions exert an enormous influence on reported service levels and costs. In many legacy studies, starting on-hand inventory is initialized proportional to the policy's own order-up-to target (e.g., I_0 = 0.70 * S) with an empty pipeline vector (O_0 = 0). This convention introduces severe experimental bias: policies with larger cycle stock targets beta receive substantially higher initial inventory for free. Over short evaluation windows (e.g., 14 to 15 days), policies with inflated base stocks can satisfy customer demand throughout the first week without placing replenishment orders, artificially masking stockouts and subsidizing inefficient policies.", "p"),
    ("To eliminate starting-state bias, we establish Policy-Independent Fair Initialization:", "p"),
    ("1. Initial On-Hand Stock: For every store-item pair i, starting physical inventory is set independently of policy parameters: I_{i,0} = 2.0 * D̄_{i, pre}.", "bullet"),
    ("2. Pipeline Steady-State Pre-Fill: All L = 7 transit pipeline slots are pre-filled with steady-state historical mean demand: O_{i, -k} = D̄_{i, pre} for all k in {1, 2, ..., 7}.", "bullet"),
    ("Under this formulation, every policy begins from an identical, realistic operating state representing a smooth, moving supply chain.", "p"),

    ("5.2 Invariance Theorem on Truncated Horizons (t < L)", "h2"),
    ("Under positive supplier lead times, initial conditions create a mathematically invariant operational window:", "p"),
    ("Theorem 1 (Lead-Time Invariance on Days 0 to 6): Let supplier lead time be L = 7 days. Under policy-independent fair initialization (I_{i,0} = 2.0 * D̄_{i, pre}, O_{i, -k} = D̄_{i, pre} for k in {1, ..., 7}), any replenishment order O_{i,t} >= 0 dispatched at decision epoch t >= 0 arrives at epoch t + L = t + 7 >= 7. Consequently, for all calendar days t in {0, 1, ..., 6}, physical on-hand inventory, customer fulfillment, lost sales stockouts, warehouse holding costs, and stockout penalties are strictly identical across all possible replenishment policies.", "p"),
    ("Proof: By induction on t in {0, 1, ..., 6}. At t = 0, initial on-hand stock is identical across all policies. Realized demand D_{i,0} is determined externally by customer behavior. Thus, sales and stockout penalties are invariant across policies. For any t in {1, 2, ..., 6}, physical stock transitions as I_{i,t} = max(0, I_{i, t-1} - D_{i, t-1}) + O_{i, t - L}. Because t <= 6 and L = 7, the pipeline arrival index is t - L <= -1. The arriving orders originate strictly from the pre-filled steady-state pipeline, which is invariant across policies. No order placed at decision epoch tau >= 0 can arrive prior to day tau + 7 >= 7. Therefore, physical stock I_{i,t} is invariant across all policies for all t in {0, 1, ..., 6}.", "p"),
    ("To verify Theorem 1, unit tests were executed across all 15 policy configurations in our unified simulator. Maximum absolute differences across all policies on Days 0 to 6 confirmed zero variance: Difference in Holding Cost = $0.00; Difference in Stockout Penalties = $0.00; Difference in Physical Sales = 0.00 units. Across all 363 active test pairs, the cumulative operating cost accrued over Days 0 to 6 is constant across all evaluated policies: Holding Cost (0..6) = $23,637.13; Stockout Penalties (0..6) = $22,120.24; Invariant Cost Sum = $45,757.37.", "p"),

    ("5.3 Causal Arriving-Order Cost Accounting (W1 Rule)", "h2"),
    ("Because days 0 to 6 are invariant, the primary evaluation window where policies actively diverge is Days 7 to 14 (8 decision days). This introduces a critical accounting boundary question: how should fixed order dispatch fees ($8.00 per order) be allocated?", "p"),
    ("Under lead time L = 7 days: orders placed on days 0 to 7 arrive on days 7 to 14, physically delivering inventory that serves customer demand inside the primary evaluation window. Conversely, orders placed on days 8 to 14 arrive on days 15 to 21, delivering stock outside the evaluation window. In legacy simulation accounting, order fees were assessed on all orders placed during days 0 to 14, while demand fulfillment was truncated at day 14. This penalizes proactive policies by charging them order fees for trailing replenishment orders whose revenue benefits are never credited.", "p"),
    ("To establish causal alignment, we define The W1 Arriving-Order Fee Rule: Primary Operating Cost equals Holding(7..14) + Stockout(7..14) + Fees for orders placed on days 0 to 7. Under the W1 rule, the fee accounting window matches the physical delivery window. Furthermore, because orders placed on days 0 to 7 arrive inside the primary window, the difference between full-window (Days 0–14) and primary-window (Days 7–14) cost is proven to be strictly: Full Cost minus Primary Cost equals Holding(0..6) + Stockout(0..6) = $45,757.37 (exact to the cent across ALL policies). Robustness tables report full-window fees across all 15 days, confirming identical policy rankings.", "p"),

    ("5.4 Pre-Specified Validation Selection (Zero Test Leakage)", "h2"),
    ("To guarantee zero test-window leakage, all parameter tuning is frozen on the validation partition (July 17–31, 2017). The policy variant (safety buffer type and static vs. rolling formulation) that achieves the minimum validation fitness under the 95.0% service penalty barrier is locked. Its optimal parameters [alpha*, beta*] are transferred directly to the test partition (August 1–15, 2017) for one-shot forward evaluation without feedback or re-tuning.", "p")
]

TABLE_3_DATA = {
    "box_title": "TABLE 3 PLACEHOLDER & SPECIFICATION: Methodological Accounting Waterfall Ledger",
    "items": [
        ("Table Number", "Table 3 (Table 5.1 in Text)"),
        ("Table Title", "Methodological Accounting Waterfall Ledger: Tracking Simulation Conventions"),
        ("Evaluated Policy", "Fixed Quantile Conformal Policy (Static, alpha = 0.10, beta = 3.4879; August 1–15, 2017)"),
        ("Key Finding Shown", "Test lookahead in cycle stock masked +$7,827 in true cost; unfair initial inventory shifted absolute cost by -$28,920.")
    ],
    "headers": ["Step", "Methodological Intervention", "Reported Cost ($)", "Fill Rate (%)", "Delta Cost ($)", "Methodological Diagnostic"],
    "rows": [
        ["1", "Original Notebook (Test lookahead in D̄; I_0 = 0.70 * S)", "136,378.10", "96.41%", "Baseline", "Flawed baseline: lookahead bias + unequal stock subsidization"],
        ["2", "Causal D̄_pre (Trailing 28-day historical mean)", "144,205.38", "95.30%", "+7,827.28", "Lookahead bias was masking $7.8k in true replenishment costs"],
        ["3", "Causal sigma_pre (Historical residual variance)", "144,205.38", "95.30%", "+0.00", "Quantile spread is model-derived; unaffected by parametric sigma"],
        ["4", "Policy-Independent Fair I_0 (2 * D̄_pre, prefilled pipe)", "115,285.42", "95.26%", "-28,919.96", "Unequal initial inventory subsidized high-beta policies by $28.9k"],
        ["5", "Validation-Derived Conformal Offsets (+0.18 / +4.58)", "115,286.91", "95.26%", "+1.49", "Conformal calibration preserves operational cost efficiency"],
        ["6", "Causal Arriving-Order Fee Rule (W1 Rule, orders t <= 7)", "110,702.91", "95.26%", "-4,584.00", "Eliminates horizon truncation distortion on non-arriving orders"]
    ]
}
