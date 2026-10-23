# -*- coding: utf-8 -*-
"""
Section 5: Methodological Safeguards: The Locked Invariant Simulation Protocol
"""

SECTION_5_TEXT = """
# 5. Methodological Safeguards: The Locked Invariant Simulation Protocol

In this section, we document the methodological safeguards formulated to ensure that our simulation framework is completely leak-free, reproducible, and mathematically sound. We establish the Lead-Time Invariance Theorem, define causal arriving-order accounting, and present the Methodological Accounting Waterfall (Table 3), which quantifies the distortion introduced by common simulation conventions.

## 5.1 Elimination of Lookahead Bias in Policy Formulation
A common flaw in applied machine learning supply chain literature is the unintentional leakage of test-window realizations into operational policy parameters. In preliminary implementations of $(s, S)$ policies, order-up-to base stock targets are frequently defined as $S = s + \\beta \\bar{D}_{\\text{test}}$, where $\\bar{D}_{\\text{test}}$ represents the mean customer sales realized over the test period itself. 

In our locked specification, we enforce strict temporal causality. All baseline demand estimates utilized in downstream policies are restricted to the causal trailing 28-day window immediately preceding the evaluation split:
$$\\bar{D}_{i, \\text{pre}} = \\frac{1}{28} \\sum_{d=1}^{28} D_{i, t_{\\text{start}} - d}$$
No future demand, promotional realization, or sales aggregate from the evaluation window is accessible to the replenishment policy.

## 5.2 Policy-Independent Fair Initialization
In finite-horizon discrete-event inventory simulations, starting conditions exert an enormous influence on reported service levels and costs. In many legacy studies, starting on-hand inventory is initialized proportional to the policy's own order-up-to target (e.g., $I_{i,0} = 0.70 \\cdot S_i$) with an empty pipeline vector ($\\mathbf{O}_{i,0} = \\mathbf{0}$).

This convention introduces severe experimental bias: policies with larger cycle stock targets $\\beta$ receive substantially higher initial inventory for free. Over short evaluation windows (e.g., 14 to 15 days), policies with inflated base stocks can satisfy customer demand throughout the first week without placing replenishment orders, artificially masking stockouts and subsidizing inefficient policies.

To eliminate starting-state bias, we establish **Policy-Independent Fair Initialization**:
1. **Initial On-Hand Stock:** For every store-item pair $i$, starting physical inventory is set independently of policy parameters $\\alpha$ and $\\beta$:
   $$I_{i,0} = 2.0 \\times \\bar{D}_{i, \\text{pre}}$$
2. **Pipeline Steady-State Pre-Fill:** All $L = 7$ transit pipeline slots are pre-filled with steady-state historical mean demand:
   $$O_{i, -k} = \\bar{D}_{i, \\text{pre}} \\quad \\forall k \\in \\{1, 2, \\dots, 7\\}$$

Under this formulation, every policy begins from an identical, realistic operating state representing a smooth, moving supply chain.

## 5.3 Mathematical Proof of Lead-Time Invariance (Days 0 to 6)
Under positive supplier lead times, initial conditions create a mathematically invariant operational window:

**Theorem 1 (Lead-Time Invariance on Days 0 to 6).**  
*Let supplier lead time be $L = 7$ days. Under policy-independent fair initialization ($I_{i,0} = 2.0 \\bar{D}_{i, \\text{pre}}$, $O_{i, -k} = \\bar{D}_{i, \\text{pre}}$ for $k \\in \\{1, \\dots, 7\\}$), any replenishment order $O_{i,t} \\ge 0$ dispatched at decision epoch $t \\ge 0$ arrives at epoch $t + L = t + 7 \\ge 7$. Consequently, for all calendar days $t \\in \\{0, 1, \\dots, 6\\}$, physical on-hand inventory $I_{i,t}$, customer fulfillment $Y_{i,t}$, lost sales stockouts $\\text{Stockout}_{i,t}$, warehouse holding costs, and stockout penalties are strictly identical across all possible replenishment policies.*

*Proof.* By induction on $t \\in \\{0, 1, \\dots, 6\\}$:  
At $t = 0$, $I_{i,0} = 2.0 \\bar{D}_{i, \\text{pre}}$ is identical across all policies. Realized demand $D_{i,0}$ is determined externally by customer behavior. Thus, sales $Y_{i,0} = \\min(I_{i,0}, D_{i,0})$ and stockout penalties $\\text{Stockout}_{i,0} = \\max(0, D_{i,0} - I_{i,0})$ are invariant across policies.  
For any $t \\in \\{1, 2, \\dots, 6\\}$, physical stock transitions as:
$$I_{i,t} = \\max(0, I_{i, t-1} - D_{i, t-1}) + O_{i, t - L}$$
Because $t \\le 6$ and $L = 7$, the pipeline arrival index is $t - L \\le -1$. The arriving orders $O_{i, t - L}$ originate strictly from the pre-filled steady-state pipeline ($O_{i, -k} = \\bar{D}_{i, \\text{pre}}$), which is invariant across policies. No order $O_{i,\\tau}$ placed at decision epoch $\\tau \\ge 0$ can arrive prior to day $\\tau + 7 \\ge 7$. Therefore, physical stock $I_{i,t}$ is invariant across all policies for all $t \\in \\{0, 1, \\dots, 6\\}$. $\\blacksquare$

### Empirical Verification of the Invariance Proof
To verify Theorem 1, unit tests were executed across all 15 policy configurations in our unified simulator (`unified_simulator.py`). Evaluating maximum absolute differences across all policies on Days 0 to 6 confirmed:
* Difference in Holding Cost: $\\$0.000000$ ($0.00$)
* Difference in Stockout Penalties: $\\$0.000000$ ($0.00$)
* Difference in Physical Sales: $0.000000$ units

Across all 363 active test pairs, the cumulative operating cost accrued over Days 0 to 6 is constant across all evaluated policies:
$$\\text{HoldingCost}(0..6) = \\$23,637.13$$
$$\\text{StockoutCost}(0..6) = \\$22,120.24$$
$$\\text{Invariant Cost Sum} = \\$23,637.13 + \\$22,120.24 = \\mathbf{\\$45,757.37}$$

## 5.4 Causal Arriving-Order Fee Accounting (The W1 Rule)
Because days 0 to 6 are invariant, the primary evaluation window where policies actively diverge is **Days 7 to 14** (8 decision days). This introduces a critical accounting boundary question: *how should fixed order dispatch fees ($K = \\$8.00$) be allocated?*

Under lead time $L = 7$:
* Orders placed on days $t \\in \\{0, 1, \\dots, 7\\}$ arrive on days $t + 7 \\in \\{7, 8, \\dots, 14\\}$, physically delivering inventory that serves customer demand inside the primary evaluation window.
* Orders placed on days $t \\in \\{8, 9, \\dots, 14\\}$ arrive on days $t + 7 \\in \\{15, 16, \\dots, 21\\}$, delivering stock outside the evaluation window.

In legacy simulation accounting, order fees were assessed on all orders placed during days 0 to 14, while demand fulfillment was truncated at day 14. This penalizes proactive policies by charging them order fees for trailing replenishment orders whose revenue benefits are never credited.

To establish causal alignment, we define **The W1 Arriving-Order Fee Rule**:
$$\\text{Primary Operating Cost} = \\text{Holding}(7..14) + \\text{Stockout}(7..14) + \\sum_{i=1}^N \\sum_{t=0}^7 K \\cdot \\mathbb{I}(O_{i,t} > 0)$$
Under the W1 rule, the fee accounting window matches the physical delivery window. Furthermore, because orders placed on days 0 to 7 arrive inside the primary window, the difference between full-window (Days 0–14) and primary-window (Days 7–14) cost is proven to be strictly:
$$\\text{Full Cost} - \\text{Primary Cost} = \\text{Holding}(0..6) + \\text{Stockout}(0..6) = \\mathbf{\\$45,757.37} \\quad (\\text{exact to the cent across ALL policies})$$
Robustness tables report full-window fees across all 15 days, confirming identical policy rankings.

## 5.5 Pre-Specified Validation-Only Tuning (Zero Test Leakage)
To guarantee zero test-window leakage, all parameter tuning is frozen on the validation partition (July 17–31, 2017). The policy variant (safety buffer type and static vs. rolling formulation) that achieves the minimum validation fitness:
$$\\text{Fitness} = \\text{Cost}_{\\text{val}} + 50,000 \\cdot \\left[ \\max(0, 95.0 - \\text{SL}_{\\text{val}}) \\right]^2$$
is locked. Its optimal parameters $(\\alpha^*, \\beta^*)$ are transferred directly to the test partition (August 1–15, 2017) for one-shot forward evaluation without feedback or re-tuning.

## 5.6 Table 3: The Methodological Accounting Waterfall Ledger
To quantify how simulation conventions distort reported supply chain performance, we track a single policy (Quantile Conformal, static, $\\alpha = 0.10, \\beta = 3.4879$) across six progressive methodological corrections in Table 5.1.

**Table 5.1 (Table 3 in Manuscript). Methodological Accounting Waterfall Ledger**  
*Evaluating a Fixed Policy Across Progressive Simulation Conventions (August 1–15, 2017)*

| Step | Methodological Intervention | Reported Cost ($) | Fill Rate (%) | $\\Delta$ Cost ($) | Methodological Diagnostic |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Original Unaudited Notebook** (Test lookahead in $\\bar{D}$; $I_0 = 0.70 \\cdot S$) | 136,378.10 | 96.41 | Baseline | Flawed baseline: lookahead bias + unequal stock subsidization |
| **2** | **Causal $\\bar{D}_{\\text{pre}}$** (Trailing 28-day historical mean) | 144,205.38 | 95.30 | +7,827.28 | **Lookahead bias was masking $\\$7.8\\text{k}$ in true replenishment costs** |
| **3** | **Causal $\\hat{\\sigma}_{\\text{pre}}$** (Historical residual variance) | 144,205.38 | 95.30 | +0.00 | Quantile spread is model-derived; unaffected by parametric $\\sigma$ |
| **4** | **Policy-Independent Fair $I_0$** ($2\\bar{D}_{\\text{pre}}$, prefilled pipe) | 115,285.42 | 95.26 | -28,919.96 | **Unequal initial inventory subsidized high-$\\beta$ policies by $\\$28.9\\text{k}$** |
| **5** | **Validation-Derived Conformal Offsets** (+0.1785 / +4.5766) | 115,286.91 | 95.26 | +1.49 | Conformal calibration preserves operational cost efficiency |
| **6** | **Causal Arriving-Order Fee Rule** (Fees for orders arriving $t \\le 14$) | 110,702.91 | 95.26 | -4,584.00 | Eliminates horizon truncation distortion on non-arriving orders |

Table 5.1 delivers a profound methodological insight: **lookahead bias in cycle stock sizing alone masked $\\$7,827$ in operating costs**, an amount roughly equal to the entire performance delta between Quantile and Huber transformers. Furthermore, unfair initial inventory subsidization distorted absolute costs by nearly $\\$29,000$. This confirms that reported gains in empirical supply chain studies are meaningless unless evaluation protocols are strictly audited.
"""
