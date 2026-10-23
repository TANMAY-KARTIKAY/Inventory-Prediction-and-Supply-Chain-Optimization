# -*- coding: utf-8 -*-
"""
Section 3: Problem Formulation and Supply Chain Environment
"""

SECTION_3_TEXT = """
# 3. Problem Formulation and Supply Chain Environment

In this section, we formulate the mathematical model governing the multi-item, multi-period retail inventory system. We formalize the discrete-event operational transitions, the dynamic order-up-to control policy, the asymmetric cost accounting structure, and the constrained optimization objective.

## 3.1 System Topology and Temporal Chronology
Consider a decentralized retail supermarket enterprise managing an active set of $N$ store-item combinations indexed by $i \\in \\{1, 2, \\dots, N\\}$ ($N = 363$ active test pairs in our empirical deployment). Replenishment decisions are executed in discrete daily review intervals indexed by $t \\in \\{0, 1, \\dots, T-1\\}$, spanning an evaluation horizon of $T = 15$ calendar days (August 1 to August 15, 2017).

Physical goods are replenished from regional distribution centers to retail stores with a deterministic, positive supplier transit lead time of $L = 7$ days. On each calendar day $t$, the operational sequence of events proceeds strictly in the following causal order:

1. **Pipeline Arrival:** Purchase orders dispatched $L$ days prior (at decision epoch $t - L$) physically arrive at store $i$, and are immediately checked in to replenish on-hand physical stock.
2. **Customer Demand Realization:** Uncorrelated customer demand $D_{i,t} \\ge 0$ arrives at the store during retail operating hours.
3. **Fulfillment and Lost Sales Accounting:** Customer demand is satisfied from available on-hand inventory. Unmet demand is lost immediately, incurring contractual stockout penalties.
4. **End-of-Day Inventory State Transition:** Ending on-hand physical inventory is logged, and holding costs (including perishability penalties) are assessed on remaining stock.
5. **Replenishment Review and Order Placement:** The inventory position is evaluated, and the replenishment policy determines whether to dispatch a new purchase order $O_{i,t} \\ge 0$, which will arrive at epoch $t + L$.

## 3.2 Dynamic $(s_t, S_t)$ Order-Up-To Control Policy
Let $I_{i,t} \\ge 0$ denote the physical on-hand inventory of store-item pair $i$ at the beginning of day $t$ prior to demand realization. Let $\\mathbf{O}_{i,t} = [O_{i, t-1}, O_{i, t-2}, \\dots, O_{i, t-L}]^T \\in \\mathbb{R}_+^L$ denote the vector of in-transit replenishment orders currently in the pipeline.

The effective **Inventory Position** $\\text{IP}_{i,t}$ represents total operational capital committed to the SKU:
$$\\text{IP}_{i,t} = I_{i,t} + \\sum_{k=1}^L O_{i, t-k}$$

The store executes a non-stationary, dynamic order-up-to $(s_{i,t}, S_{i,t})$ policy parameterized by two continuous control multipliers $\\boldsymbol{\\theta} = [\\alpha, \\beta]^T \\in \\mathbb{R}_+^2$:

### A. Reorder Point Formulation ($s_{i,t}$)
The reorder threshold $s_{i,t}$ defines the minimum inventory position required to protect the store against stockouts over the full lead-time duration $[t+1, t+L]$:
$$s_{i,t} = \\sum_{h=1}^L \\hat{y}_{i, t+h}^{P_{50}} + \\text{SS}_{i,t}(\\alpha)$$
where $\\hat{y}_{i, t+h}^{P_{50}}$ represents the multi-horizon median demand prediction issued at epoch $t$ for horizon day $t+h$, and $\\text{SS}_{i,t}(\\alpha)$ is the dynamic safety stock buffer scaled by multiplier $\\alpha \\in [0.0, 1.5]$.

### B. Order-Up-To Base Stock Formulation ($S_{i,t}$)
When inventory position breaches the reorder threshold ($\\text{IP}_{i,t} \\le s_{i,t}$), the store raises its inventory position to target level $S_{i,t}$:
$$S_{i,t} = s_{i,t} + \\beta \\cdot \\bar{D}_{i, \\text{pre}}$$
where $\\beta \\in [0.5, 5.0]$ scales the operational cycle stock, and $\\bar{D}_{i, \\text{pre}}$ represents the causal trailing mean customer demand evaluated over the 28 historical days immediately preceding the evaluation window:
$$\\bar{D}_{i, \\text{pre}} = \\frac{1}{28} \\sum_{d=1}^{28} D_{i, t_{\\text{start}} - d}$$

Crucially, defining cycle stock relative to historical causal mean $\\bar{D}_{i, \\text{pre}}$ ensures that no future realized test demand leaks into policy execution.

### C. Replenishment Dispatch Decision ($O_{i,t}$)
The physical purchase order dispatched at the conclusion of decision epoch $t$ is given by:
$$O_{i,t} = \\begin{cases} S_{i,t} - \\text{IP}_{i,t}, & \\text{if } \\text{IP}_{i,t} \\le s_{i,t} \\\\ 0, & \\text{if } \\text{IP}_{i,t} > s_{i,t} \\end{cases}$$

## 3.3 State Transitions and Lost Sales Fulfillment
Upon demand realization $D_{i,t}$, physical retail sales $Y_{i,t}$ and lost sales stockouts $\\text{Stockout}_{i,t}$ are computed as:
$$Y_{i,t} = \\min(I_{i,t}, D_{i,t})$$
$$\\text{Stockout}_{i,t} = \\max(0, D_{i,t} - I_{i,t})$$

Ending physical inventory $I_{i,t}^+$ carried overnight to epoch $t+1$ transitions as:
$$I_{i, t+1} = \\max(0, I_{i,t} - D_{i,t}) + O_{i, t - L + 1}$$
where pipeline orders advance by one day, and orders placed at epoch $t - L + 1$ transition into physical on-hand availability at epoch $t+1$.

## 3.4 The Asymmetric Cost Accounting Objective
Total operating expenses accrued across the supply chain comprise three distinct economic cost components: holding costs, stockout penalties, and purchase order dispatch fees.

### A. Warehouse Holding and Perishability Obsolescence Costs
Ending physical inventory incurs a daily holding rate $h_i$ that accounts for physical warehouse storage, cost of capital, and biological spoilage risk:
$$h_i = \\begin{cases} \\$0.15 \\text{ per unit-day}, & \\text{if item } i \\text{ is Perishable} \\\\ \\$0.10 \\text{ per unit-day}, & \\text{if item } i \\text{ is Dry Grocery} \\end{cases}$$
$$\\text{HoldingCost}_i = \\sum_{t \\in \\mathcal{T}} h_i \\cdot I_{i,t}^+$$
where $\\mathcal{T}$ represents the evaluated scoring window. Oversampling perishable items (51.0% in our cohort vs. 23.2% in the population) enables direct empirical examination of how asymmetric holding rates penalize over-buffering.

### B. Stockout and Service Penalty Costs
Under a lost-sales environment, unmet customer demand incurs an immediate financial penalty $c_p = \\$2.50$ per unit (varied from $\\$1.00$ to $\\$10.00$ in sensitivity analysis):
$$\\text{StockoutCost}_i = \\sum_{t \\in \\mathcal{T}} c_p \\cdot \\text{Stockout}_{i,t}$$

### C. Fixed Purchase Order Dispatch Fees
Every replenishment event that generates a non-zero purchase order ($O_{i,t} > 0$) incurs a fixed administrative, picking, and logistics transport fee of $K = \\$8.00$ per purchase order line:
$$\\text{OrderCost}_i = \\sum_{t \\in \\mathcal{T}_{\\text{order}}} K \\cdot \\mathbb{I}(O_{i,t} > 0)$$
where $\\mathbb{I}(\\cdot)$ is the indicator function. The exact temporal indexing of $\\mathcal{T}_{\\text{order}}$ is defined under the causal arriving-order rule in Section 5.

### D. Total Supply Chain Operating Cost
Total enterprise cost across all $N$ SKUs is the unweighted sum of the three cost components:
$$J(\\alpha, \\beta) = \\sum_{i=1}^N \\left[ \\text{HoldingCost}_i(\\alpha, \\beta) + \\text{StockoutCost}_i(\\alpha, \\beta) + \\text{OrderCost}_i(\\alpha, \\beta) \\right]$$

## 3.5 Customer Fill Rate and Constrained Optimization Formulation
Commercial retail contracts enforce strict Service-Level Agreements (SLAs). In high-volume enterprise retail, service performance is measured via the aggregate quantity-based **Customer Fill Rate** (also denoted as Service Level, $\\text{SL}$), defined as total units fulfilled divided by total units demanded:
$$\\text{SL}(\\alpha, \\beta) = \\frac{\\sum_{i=1}^N \\sum_{t \\in \\mathcal{T}} Y_{i,t}(\\alpha, \\beta)}{\\sum_{i=1}^N \\sum_{t \\in \\mathcal{T}} D_{i,t}} \\times 100\\%$$

The operational challenge is to discover policy control parameters $\\boldsymbol{\\theta}^* = [\\alpha^*, \\beta^*]^T$ that minimize total operating cost subject to satisfying an enterprise service constraint of $\\text{SL} \\ge 95.0\\%$:
$$\\min_{\\alpha \\in [0.0, 1.5], \\beta \\in [0.5, 5.0]} J(\\alpha, \\beta) \\quad \\text{subject to} \\quad \\text{SL}(\\alpha, \\beta) \\ge 95.0\\%$$

Because the customer fill rate is evaluated through non-linear discrete-event simulation, analytical Lagrange multipliers cannot be solved directly. To enforce SLA compliance within derivative-free metaheuristic algorithms (PSO and GWO), we formulate an exterior quadratic barrier penalty function:
$$\\text{Fitness}(\\alpha, \\beta) = J(\\alpha, \\beta) + \\lambda \\cdot \\left[ \\max\\left(0, 95.0 - \\text{SL}(\\alpha, \\beta)\\right) \\right]^2$$
where the penalty scale parameter $\\lambda = 50,000.0$ imposes an insurmountable financial barrier on any policy that violates the 95.0% threshold. For example, a policy that achieves a 94.0% fill rate (a 1.0% SLA breach) incurs an immediate financial penalty of $\\$50,000 \\times (1.0)^2 = \\$50,000$, instantly rendering the solution unviable.
"""
