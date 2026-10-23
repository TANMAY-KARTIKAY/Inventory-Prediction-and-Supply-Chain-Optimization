# -*- coding: utf-8 -*-
"""
Clean Section 3: Problem Formulation and Supply Chain Environment
All mathematical equations formatted cleanly in plain Unicode typography with equation numbers.
"""

SECTION_3_PARAGRAPHS = [
    ("3. PROBLEM FORMULATION & SUPPLY CHAIN ENVIRONMENT", "h1"),
    ("In this section, we formulate the mathematical model governing the multi-item, multi-period retail inventory system. We formalize the discrete-event operational transitions, the dynamic order-up-to control policy, the asymmetric cost accounting structure, and the constrained optimization objective.", "p"),
    
    ("3.1 System Topology and Temporal Chronology", "h2"),
    ("Consider a decentralized retail supermarket enterprise managing an active set of N store-item combinations indexed by i in {1, 2, ..., N} (N = 363 active test pairs in our empirical deployment). Replenishment decisions are executed in discrete daily review intervals indexed by t in {0, 1, ..., T-1}, spanning an evaluation horizon of T = 15 calendar days (August 1 to August 15, 2017).", "p"),
    ("Physical goods are replenished from regional distribution centers to retail stores with a deterministic, positive supplier transit lead time of L = 7 days. On each calendar day t, the operational sequence of events proceeds strictly in the following causal order:", "p"),
    ("1. Pipeline Arrival: Purchase orders dispatched L days prior (at decision epoch t - L) physically arrive at store i, and are immediately checked in to replenish on-hand physical stock.", "bullet"),
    ("2. Customer Demand Realization: Customer demand D_{i,t} arrives at the store during retail operating hours.", "bullet"),
    ("3. Fulfillment and Lost Sales Accounting: Customer demand is satisfied from available on-hand inventory. Unmet demand is lost immediately, incurring contractual stockout penalties.", "bullet"),
    ("4. End-of-Day Inventory State Transition: Ending on-hand physical inventory is logged, and holding costs (including perishability penalties) are assessed on remaining stock.", "bullet"),
    ("5. Replenishment Review and Order Placement: The inventory position is evaluated, and the replenishment policy determines whether to dispatch a new purchase order O_{i,t} >= 0, which will arrive at epoch t + L.", "bullet"),

    ("3.2 Dynamic (s_t, S_t) Order-Up-To Control Policy", "h2"),
    ("Let I_{i,t} >= 0 denote physical on-hand inventory of store-item pair i at the beginning of day t prior to demand realization. Let O_{i,t} = [O_{i, t-1}, O_{i, t-2}, ..., O_{i, t-L}] denote the vector of in-transit replenishment orders currently in the pipeline.", "p"),
    ("The effective Inventory Position IP_{i,t} represents total operational capital committed to the SKU:", "p"),
    ("IP_{i,t} = I_{i,t} + sum_{k=1..L} O_{i, t-k}", "eq", "Eq. 1"),
    ("The store executes a non-stationary, dynamic order-up-to (s_{i,t}, S_{i,t}) policy parameterized by two continuous control multipliers theta = [alpha, beta]:", "p"),
    ("The reorder threshold s_{i,t} defines the minimum inventory position required to protect the store against stockouts over the full lead-time duration [t+1, t+L]:", "p"),
    ("s_{i,t} = sum_{h=1..L} P50_{i, t+h} + SS_{i,t}(alpha)", "eq", "Eq. 2"),
    ("where P50_{i, t+h} represents the multi-horizon median demand prediction issued at epoch t for horizon day t+h, and SS_{i,t}(alpha) is the dynamic safety stock buffer scaled by multiplier alpha in [0.0, 1.5].", "p"),
    ("When the inventory position breaches the reorder threshold (IP_{i,t} <= s_{i,t}), the store raises its inventory position to target base-stock level S_{i,t}:", "p"),
    ("S_{i,t} = s_{i,t} + beta * D̄_{i, pre}", "eq", "Eq. 3"),
    ("where beta in [0.5, 5.0] scales the operational cycle stock, and D̄_{i, pre} represents the causal trailing mean customer demand evaluated over the 28 historical days immediately preceding the evaluation window:", "p"),
    ("D̄_{i, pre} = (1 / 28) * sum_{d=1..28} D_{i, t_start - d}", "eq", "Eq. 4"),
    ("Crucially, defining cycle stock relative to historical causal mean D̄_{i, pre} ensures that no future realized test demand leaks into policy execution.", "p"),
    ("The physical purchase order dispatched at the conclusion of decision epoch t is given by:", "p"),
    ("O_{i,t} = S_{i,t} - IP_{i,t}   if IP_{i,t} <= s_{i,t};   otherwise O_{i,t} = 0", "eq", "Eq. 5"),

    ("3.3 State Transitions and Lost Sales Fulfillment", "h2"),
    ("Upon demand realization D_{i,t}, physical retail sales Y_{i,t} and lost sales stockouts Stockout_{i,t} are computed as:", "p"),
    ("Y_{i,t} = min(I_{i,t}, D_{i,t})", "eq", "Eq. 6"),
    ("Stockout_{i,t} = max(0, D_{i,t} - I_{i,t})", "eq", "Eq. 7"),
    ("Ending physical inventory I_{i,t}^+ carried overnight to epoch t+1 transitions as:", "p"),
    ("I_{i, t+1} = max(0, I_{i,t} - D_{i,t}) + O_{i, t - L + 1}", "eq", "Eq. 8"),
    ("where pipeline orders advance by one day, and orders placed at epoch t - L + 1 transition into physical on-hand availability at epoch t+1.", "p"),

    ("3.4 The Asymmetric Cost Accounting Objective", "h2"),
    ("Total operating expenses accrued across the supply chain comprise three distinct economic cost components: holding costs, stockout penalties, and purchase order dispatch fees.", "p"),
    ("1. Warehouse Holding and Perishability Obsolescence Costs: Ending physical inventory incurs a daily holding rate h_i that accounts for physical warehouse storage, cost of capital, and biological spoilage risk: h_i = $0.15 per unit-day for perishable fresh foods, and h_i = $0.10 per unit-day for dry groceries.", "p"),
    ("HoldingCost_i = sum_{t in T} h_i * I_{i,t}^+", "eq", "Eq. 9"),
    ("2. Stockout and Service Penalty Costs: Under a lost-sales environment, unmet customer demand incurs an immediate financial penalty c_p = $2.50 per unit (varied from $1.00 to $10.00 in sensitivity analysis):", "p"),
    ("StockoutCost_i = sum_{t in T} c_p * Stockout_{i,t}", "eq", "Eq. 10"),
    ("3. Fixed Purchase Order Dispatch Fees: Every replenishment event that generates a non-zero purchase order (O_{i,t} > 0) incurs a fixed administrative, picking, and logistics transport fee of K = $8.00 per purchase order line:", "p"),
    ("OrderCost_i = sum_{t in T_order} K * Indicator(O_{i,t} > 0)", "eq", "Eq. 11"),
    ("Total enterprise operating cost is the unweighted sum of all three cost components across all N store-item pairs: J(alpha, beta) = sum_{i=1..N} [ HoldingCost_i + StockoutCost_i + OrderCost_i ].", "p"),

    ("3.5 Customer Fill Rate and Constrained Optimization Formulation", "h2"),
    ("In high-volume enterprise retail, customer satisfaction is measured via the aggregate quantity-based Customer Fill Rate (Service Level, SL), defined as total physical units fulfilled divided by total units demanded:", "p"),
    ("SL(alpha, beta) = [ sum_{i,t} Y_{i,t}(alpha, beta) / sum_{i,t} D_{i,t} ] * 100%", "eq", "Eq. 12"),
    ("The operational challenge is to discover policy control parameters theta* = [alpha*, beta*] that minimize total operating cost subject to satisfying an enterprise service constraint of SL >= 95.0%:", "p"),
    ("min J(alpha, beta)   subject to   SL(alpha, beta) >= 95.0%,   alpha in [0.0, 1.5],   beta in [0.5, 5.0]", "eq", "Eq. 13"),
    ("To enforce SLA compliance within derivative-free metaheuristic algorithms (PSO and GWO), we formulate an exterior quadratic barrier penalty function:", "p"),
    ("Fitness(alpha, beta) = J(alpha, beta) + 50,000 * [ max(0, 95.0 - SL(alpha, beta)) ]^2", "eq", "Eq. 14"),
    ("where the penalty scale parameter lambda = 50,000.0 imposes an insurmountable financial barrier on any policy that violates the 95.0% threshold. For example, a policy that achieves a 94.0% fill rate (a 1.0% SLA breach) incurs an immediate penalty of $50,000 * (1.0)^2 = $50,000, instantly rendering the candidate solution non-viable.", "p")
]
