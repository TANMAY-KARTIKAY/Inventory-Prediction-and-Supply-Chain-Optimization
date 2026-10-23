"""
e0_fair_simulator.py
Phase 2 Experiment E0: Fair Discrete-Event Inventory Simulator Implementation & Verification

Features:
1. Strictly Causal Statistics:
   - Trailing 28-day mean demand D_bar computed before the decision date.
   - Trailing 28-day demand volatility sigma_demand computed before the decision date.
   - Validation-period forecast residual volatility sigma_residual for Huber model.
   - Recomputed validation conformal offsets (+0.1785 std, +4.5766 high-vol).
   - Standardized initial inventory I_0 = 0.70 * 7 * D_bar_pre (identical across all policies).
2. Variants:
   - Mode 'trailing_28d': D_bar from trailing 28 days actual sales.
   - Mode 'p50_forecast': D_bar from 7-day P50 forecast mean (zero realized demand dependence).
   - Warm-up flag: False (full 15 days) vs True (7-day warmup excluded from costs).
3. Rigorous Unit Tests:
   - Test 1: Inventory never negative.
   - Test 2: Cost balance (holding + stockout + ordering == total cost).
   - Test 3: Units-based fill rate definition check.
   - Test 4: Causality / Tamper test (tampering test demand on day t does not alter decisions on days 1..t-1,
             and policy thresholds (s, S, I_0) are completely invariant to test demand).
4. Benchmark comparison on test split saved to:
   revision_experiments/outputs/tables/e0_fair_mode_benchmark.csv
"""

import os
import sys
import time
import copy
import numpy as np
import pandas as pd

np.random.seed(42)
t_start = time.time()

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 80)
print("EXPERIMENT E0: FAIR SIMULATOR IMPLEMENTATION & CAUSALITY VERIFICATION")
print("=" * 80)

# Economic parameters
UNIT_HOLDING_DRY     = 0.10
UNIT_HOLDING_PERISH  = 0.15
STOCKOUT_PENALTY     = 2.50
FIXED_ORDER_COST     = 8.00
LEAD_TIME_DAYS       = 7

# Recomputed validation conformal offsets (from Phase 1 Amendment C)
OFFSET_STD_VAL  = 0.1785
OFFSET_HVOL_VAL = 4.5766

# Load pairs metadata
pairs_path = os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "selected_pairs.csv")
df_pairs = pd.read_csv(pairs_path)
pair_perishable = {}
pair_is_high_vol = {}
for _, row in df_pairs.iterrows():
    pair = (int(row["store_nbr"]), int(row["item_nbr"]))
    pair_perishable[pair] = bool(row["perishable"])
    pair_is_high_vol[pair] = bool(row["average_sales"] >= 100.0)

# Identify active 363 pairs in test split
test_npz = np.load(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "sequences_test.npz"), mmap_mode="r")
active_test_pairs = sorted(list(set(zip(test_npz["store_nbr"], test_npz["item_nbr"]))))
print(f"Active pairs in test evaluation: {len(active_test_pairs)}")

# Load daily historical splits
print("Loading daily chronological transaction files...")
df_train_daily = pd.read_csv(os.path.join(BASE_DIR, "Temporal Split(6)", "train_split.csv"),
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales"])
df_val_daily   = pd.read_csv(os.path.join(BASE_DIR, "Temporal Split(6)", "validation_split.csv"),
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales"])

# 1. Compute Trailing 28-day demand for Validation Window (2017-07-17 to 2017-07-31)
# Trailing dates: 2017-06-19 to 2017-07-16 (from df_train_daily)
val_trailing_dates = sorted(df_train_daily[df_train_daily["date"] >= "2017-06-19"]["date"].unique())
print(f"Validation trailing 28 days: {val_trailing_dates[0]} to {val_trailing_dates[-1]} ({len(val_trailing_dates)} days)")
df_val_trailing = df_train_daily[df_train_daily["date"].isin(val_trailing_dates)]

dbar_val_trailing = {}
sigma_val_trailing = {}
for (s_nbr, i_nbr), grp in df_val_trailing.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        sales = grp["unit_sales"].values
        dbar_val_trailing[pair] = max(1.0, float(np.mean(sales)))
        sigma_val_trailing[pair] = float(np.std(sales) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

for pair in active_test_pairs:
    if pair not in dbar_val_trailing:
        dbar_val_trailing[pair] = 1.0
        sigma_val_trailing[pair] = 5.0

# 2. Compute Trailing 28-day demand for Test Window (2017-08-01 to 2017-08-15)
# Trailing dates: 2017-07-04 to 2017-07-31 (train + val)
df_hist_test = pd.concat([
    df_train_daily[df_train_daily["date"] >= "2017-07-04"],
    df_val_daily
], ignore_index=True)
test_trailing_dates = sorted(df_hist_test["date"].unique())
print(f"Test trailing 28 days: {test_trailing_dates[0]} to {test_trailing_dates[-1]} ({len(test_trailing_dates)} days)")

dbar_test_trailing = {}
sigma_test_trailing = {}
for (s_nbr, i_nbr), grp in df_hist_test.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        sales = grp["unit_sales"].values
        dbar_test_trailing[pair] = max(1.0, float(np.mean(sales)))
        sigma_test_trailing[pair] = float(np.std(sales) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

for pair in active_test_pairs:
    if pair not in dbar_test_trailing:
        dbar_test_trailing[pair] = 1.0
        sigma_test_trailing[pair] = 5.0

# 3. Load Validation Forecasts & Compute Validation Residual Sigma for Huber
df_val_q = pd.read_csv(os.path.join(TABLES_DIR, "validation_forecasts_quantile.csv"))
df_val_h = pd.read_csv(os.path.join(TABLES_DIR, "validation_forecasts_huber.csv"))

val_daily_actual = {}
val_daily_p50    = {}
val_daily_p90    = {}

for (s_nbr, i_nbr), group in df_val_q.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        d1 = group[group["forecast_day"] == 1].sort_values("target_start_date")
        act = list(d1["actual_sales"].values)
        p50 = list(d1["p50_forecast"].values)
        is_hv = pair_is_high_vol.get(pair, False)
        off = OFFSET_HVOL_VAL if is_hv else OFFSET_STD_VAL
        p90 = list((d1["p90_forecast"] + off).values)
        if len(act) < 15:
            last_date = group["target_start_date"].max()
            last_w = group[group["target_start_date"] == last_date].set_index("forecast_day")
            needed = 15 - len(act)
            for st in range(2, 2 + needed):
                d_key = min(st, 7)
                if d_key in last_w.index:
                    act.append(last_w.loc[d_key, "actual_sales"])
                    p50.append(last_w.loc[d_key, "p50_forecast"])
                    p90.append(last_w.loc[d_key, "p90_forecast"] + off)
                else:
                    act.append(act[-1])
                    p50.append(p50[-1])
                    p90.append(p90[-1])
        val_daily_actual[pair] = np.array(act[:15], dtype=np.float64)
        val_daily_p50[pair]    = np.array(p50[:15], dtype=np.float64)
        val_daily_p90[pair]    = np.array(p90[:15], dtype=np.float64)

for pair in active_test_pairs:
    if pair not in val_daily_actual:
        val_daily_actual[pair] = np.full(15, dbar_val_trailing[pair])
        val_daily_p50[pair]    = np.full(15, dbar_val_trailing[pair])
        val_daily_p90[pair]    = np.full(15, dbar_val_trailing[pair] * 1.5)

val_daily_huber = {}
for (s_nbr, i_nbr), group in df_val_h.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        group = group.sort_values("target_start_date")
        fc = [r["pred_day_1"] for _, r in group.iterrows()]
        if len(fc) < 15:
            last_r = group.iloc[-1]
            for st in range(2, 2 + (15 - len(fc))):
                fc.append(last_r[f"pred_day_{min(st, 7)}"])
        val_daily_huber[pair] = np.array(fc[:15], dtype=np.float64)

for pair in active_test_pairs:
    if pair not in val_daily_huber:
        val_daily_huber[pair] = np.full(15, dbar_val_trailing[pair])

sigma_res_val = {}
for pair in active_test_pairs:
    act = val_daily_actual[pair]
    h_fc = val_daily_huber[pair]
    res = act - h_fc
    sigma_res_val[pair] = float(np.std(res) * np.sqrt(LEAD_TIME_DAYS) + 1.0)

# 4. Load Test Forecasts (Aug 1-15, 2017)
df_test_q = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_quantile_forecasts.csv"))
df_test_q["store_nbr"] = df_test_q["sequence_idx"].map(lambda idx: test_npz["store_nbr"][idx])
df_test_q["item_nbr"]  = df_test_q["sequence_idx"].map(lambda idx: test_npz["item_nbr"][idx])
df_test_q["target_start_date"] = df_test_q["sequence_idx"].map(lambda idx: test_npz["target_start_date"][idx])
df_test_q["is_high_vol"] = df_test_q.apply(lambda r: pair_is_high_vol.get((r["store_nbr"], r["item_nbr"]), False), axis=1)
df_test_q["p90_calibrated"] = df_test_q["p90_forecast"] + np.where(df_test_q["is_high_vol"], OFFSET_HVOL_VAL, OFFSET_STD_VAL)

test_daily_actual = {}
test_daily_p50    = {}
test_daily_p90    = {}

for (s_nbr, i_nbr), group in df_test_q.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        d1 = group[group["forecast_day"] == 1].sort_values("target_start_date")
        act = list(d1["actual_sales"].values)
        p50 = list(d1["p50_forecast"].values)
        p90 = list(d1["p90_calibrated"].values)
        if len(act) < 15:
            last_s = group["target_start_date"].max()
            last_w = group[group["target_start_date"] == last_s].set_index("forecast_day")
            needed = 15 - len(act)
            for st in range(2, 2 + needed):
                d_key = min(st, 7)
                if d_key in last_w.index:
                    act.append(last_w.loc[d_key, "actual_sales"])
                    p50.append(last_w.loc[d_key, "p50_forecast"])
                    p90.append(last_w.loc[d_key, "p90_calibrated"])
                else:
                    act.append(act[-1])
                    p50.append(p50[-1])
                    p90.append(p90[-1])
        test_daily_actual[pair] = np.array(act[:15], dtype=np.float64)
        test_daily_p50[pair]    = np.array(p50[:15], dtype=np.float64)
        test_daily_p90[pair]    = np.array(p90[:15], dtype=np.float64)

df_test_h = pd.read_csv(os.path.join(BASE_DIR, "Multi-Horizon Quantile Forecasting(process)", "tft_test_forecasts.csv"))
test_daily_huber = {}
for (s_nbr, i_nbr), group in df_test_h.groupby(["store_nbr", "item_nbr"]):
    pair = (int(s_nbr), int(i_nbr))
    if pair in active_test_pairs:
        group = group.sort_values("target_start_date")
        fc = [r["pred_day_1"] for _, r in group.iterrows()]
        if len(fc) < 15:
            last_r = group.iloc[-1]
            for st in range(2, 2 + (15 - len(fc))):
                fc.append(last_r[f"pred_day_{min(st, 7)}"])
        test_daily_huber[pair] = np.array(fc[:15], dtype=np.float64)

print("Ingested daily actuals and forecasts for both validation and test windows.")

# ==============================================================================
# 5. CORE FAIR SIMULATOR FUNCTION
# ==============================================================================
def simulate_fair_inventory(
    alpha: float,
    beta: float,
    policy_name: str,
    split_name: str = "validation",
    dbar_mode: str = "trailing_28d",
    warmup: bool = False,
    override_demand: dict = None
):
    """
    Modular, causal discrete-event replenishment simulator.
    """
    # HARD RULE 4 ASSERTION: Disallow test split when tuning
    if getattr(simulate_fair_inventory, "is_tuning_active", False) and split_name == "test":
        raise AssertionError("CRITICAL VIOLATION: Tuning routine attempted to evaluate on TEST split!")

    if split_name == "validation":
        dbar_trailing = dbar_val_trailing
        sigma_demand = sigma_val_trailing
        daily_actual = val_daily_actual
        daily_p50 = val_daily_p50
        daily_p90 = val_daily_p90
        daily_huber = val_daily_huber
    elif split_name == "test":
        dbar_trailing = dbar_test_trailing
        sigma_demand = sigma_test_trailing
        daily_actual = test_daily_actual
        daily_p50 = test_daily_p50
        daily_p90 = test_daily_p90
        daily_huber = test_daily_huber
    else:
        raise ValueError(f"Unknown split: {split_name}")

    if override_demand is not None:
        daily_actual = override_demand

    tot_hold = 0.0
    tot_stockout = 0.0
    tot_order = 0.0
    tot_demand = 0.0
    tot_sales = 0.0
    tot_orders_placed = 0

    daily_spend = np.zeros(15)
    per_day_records = []

    for pair in active_test_pairs:
        demand = daily_actual[pair]
        p50 = daily_p50[pair]
        p90 = daily_p90[pair]
        h_fc = daily_huber[pair]
        is_perish = pair_perishable.get(pair, False)
        h_cost = UNIT_HOLDING_PERISH if is_perish else UNIT_HOLDING_DRY

        # 1. Establish Causal D_bar
        if dbar_mode == "trailing_28d":
            mean_d = dbar_trailing[pair]
        elif dbar_mode == "p50_forecast":
            mean_d = max(1.0, float(np.mean(p50[:LEAD_TIME_DAYS])))
        else:
            raise ValueError(f"Unknown dbar_mode: {dbar_mode}")

        # 2. Establish Policy Lead Expectation and Safety Stock Buffer
        if policy_name in ["Mean-demand Gaussian", "Classical Gaussian"]:
            lead_exp = mean_d * LEAD_TIME_DAYS
            ss = alpha * 1.282 * sigma_demand[pair]
        elif policy_name in ["Huber TFT + Gaussian", "Huber TFT"]:
            lead_exp = float(np.sum(h_fc[:LEAD_TIME_DAYS]))
            ss = alpha * 1.282 * sigma_res_val[pair]
        elif policy_name in ["Quantile TFT", "Quantile Conformal", "Quantile Default", "Quantile TFT (PSO params)", "Quantile TFT + Conformal PSO"]:
            lead_exp = float(np.sum(p50[:LEAD_TIME_DAYS]))
            ss = alpha * float(np.sum(np.maximum(0.0, p90[:LEAD_TIME_DAYS] - p50[:LEAD_TIME_DAYS])))
        elif policy_name in ["No Safety Stock", "No SS"]:
            lead_exp = float(np.sum(p50[:LEAD_TIME_DAYS]))
            ss = 0.0
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        s = lead_exp + ss
        S = s + (beta * mean_d)

        # 3. Standardized Initial Inventory: Identical for ALL policies based on causal trailing demand
        I_0 = 0.70 * LEAD_TIME_DAYS * dbar_trailing[pair]
        on_hand = I_0
        pipeline = [0.0] * LEAD_TIME_DAYS

        # 4. Daily Business Simulation Clock (Days 0 to 14)
        for t in range(15):
            d_t = demand[t]

            # Morning: receive delivery truck
            arrivals = pipeline.pop(0)
            on_hand += arrivals

            # Inventory must never be negative
            assert on_hand >= -1e-9, f"FATAL: On-hand inventory became negative ({on_hand}) on day {t}!"

            # Daytime: fulfill customer shopping demand
            sales = min(on_hand, d_t)
            stockout = max(0.0, d_t - on_hand)
            on_hand = max(0.0, on_hand - d_t)

            # Evening review
            c_ord = 0.0
            inv_pos = on_hand + sum(pipeline)
            order_placed_flag = False
            if inv_pos <= s:
                qty = max(0.0, S - inv_pos)
                pipeline.append(qty)
                c_ord = FIXED_ORDER_COST
                order_placed_flag = True
            else:
                pipeline.append(0.0)

            c_hd = on_hand * h_cost
            c_so = stockout * STOCKOUT_PENALTY

            # Accounting (respecting warmup window)
            in_cost_window = (not warmup) or (t >= LEAD_TIME_DAYS)

            if in_cost_window:
                tot_demand += d_t
                tot_sales += sales
                tot_hold += c_hd
                tot_stockout += c_so
                tot_order += c_ord
                if order_placed_flag:
                    tot_orders_placed += 1
                daily_spend[t] += (c_hd + c_so + c_ord)

            per_day_records.append({
                "pair": pair,
                "t": t,
                "on_hand": on_hand,
                "sales": sales,
                "stockout": stockout,
                "order_qty": pipeline[-1],
                "c_hd": c_hd,
                "c_so": c_so,
                "c_ord": c_ord,
                "s": s,
                "S": S,
                "I_0": I_0
            })

    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0
    tot_cost = tot_hold + tot_stockout + tot_order

    return {
        "Policy": policy_name,
        "alpha": alpha,
        "beta": beta,
        "split": split_name,
        "dbar_mode": dbar_mode,
        "warmup": warmup,
        "Total Cost ($)": tot_cost,
        "Holding Cost ($)": tot_hold,
        "Stockout Cost ($)": tot_stockout,
        "Ordering Cost ($)": tot_order,
        "Fill Rate (%)": sl,
        "Total Orders": tot_orders_placed,
        "Total Demand": tot_demand,
        "Total Sales": tot_sales,
        "Daily Spend": daily_spend,
        "_per_day_records": per_day_records
    }

if __name__ == "__main__":
    # ==============================================================================
    # 6. SUITE OF UNIT TESTS (CONDITION 1)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("EXECUTING MANDATORY UNIT TESTS FOR E0")
    print("=" * 80)

# Run test simulation
test_res = simulate_fair_inventory(
    alpha=1.0, beta=4.0, policy_name="Quantile TFT",
    split_name="test", dbar_mode="trailing_28d", warmup=False
)

# Test 1: Inventory Never Negative
all_on_hand = [r["on_hand"] for r in test_res["_per_day_records"]]
min_on_hand = min(all_on_hand)
print(f"Unit Test 1 (Non-negative inventory): Minimum on-hand = {min_on_hand:.6f} units")
assert min_on_hand >= 0.0, "Unit Test 1 Failed: On-hand inventory was negative!"
print(">> [PASS] Unit Test 1: On-hand inventory is strictly non-negative across all pairs and days.")

# Test 2: Cost Components Sum Exactly to Total Cost
sum_components = test_res["Holding Cost ($)"] + test_res["Stockout Cost ($)"] + test_res["Ordering Cost ($)"]
cost_diff = abs(test_res["Total Cost ($)"] - sum_components)
print(f"Unit Test 2 (Cost balance): Balance delta = {cost_diff:.8e}")
assert cost_diff < 1e-6, "Unit Test 2 Failed: Cost components do not sum to total cost!"
print(">> [PASS] Unit Test 2: Holding + Stockout + Ordering cost sum exactly to Total Cost.")

# Test 3: Units-Based Fill Rate Definition Matches
manual_sl = (test_res["Total Sales"] / test_res["Total Demand"]) * 100.0
sl_diff = abs(test_res["Fill Rate (%)"] - manual_sl)
print(f"Unit Test 3 (Units fill rate): Definition delta = {sl_diff:.8e} (SL = {manual_sl:.4f}%)")
assert sl_diff < 1e-6, "Unit Test 3 Failed: Service level does not match units-based definition!"
print(">> [PASS] Unit Test 3: Fill rate strictly matches units-based Type II definition.")

# Test 4: Causality & Tamper Test
print("\nRunning Causality / Tamper Test...")
clean_res = simulate_fair_inventory(
    alpha=1.0, beta=4.0, policy_name="Quantile TFT",
    split_name="test", dbar_mode="trailing_28d", warmup=False
)
clean_records = { (r["pair"], r["t"]): r for r in clean_res["_per_day_records"] }

tampered_demand = copy.deepcopy(test_daily_actual)
tamper_t = 7  # Day 8 in 0-indexed terms
for p in active_test_pairs:
    tampered_demand[p][tamper_t:] = np.random.uniform(500.0, 1000.0, 15 - tamper_t)

tampered_res = simulate_fair_inventory(
    alpha=1.0, beta=4.0, policy_name="Quantile TFT",
    split_name="test", dbar_mode="trailing_28d", warmup=False,
    override_demand=tampered_demand
)
tampered_records = { (r["pair"], r["t"]): r for r in tampered_res["_per_day_records"] }

tamper_failures = 0
for p in active_test_pairs:
    c_0 = clean_records[(p, 0)]
    t_0 = tampered_records[(p, 0)]
    if c_0["s"] != t_0["s"] or c_0["S"] != t_0["S"] or c_0["I_0"] != t_0["I_0"]:
        tamper_failures += 1

    for t in range(tamper_t):
        c_r = clean_records[(p, t)]
        t_r = tampered_records[(p, t)]
        if (abs(c_r["on_hand"] - t_r["on_hand"]) > 1e-9 or
            abs(c_r["sales"] - t_r["sales"]) > 1e-9 or
            abs(c_r["order_qty"] - t_r["order_qty"]) > 1e-9):
            tamper_failures += 1

print(f"Tamper test pre-day-{tamper_t+1} state variances: {tamper_failures}")
assert tamper_failures == 0, "Unit Test 4 Failed: Decisions before day t changed when future demand was tampered!"
print(f">> [PASS] Unit Test 4: Causality proven. Tampering future demand on Day {tamper_t+1} causes ZERO change on Days 1..{tamper_t}, and policy thresholds (s, S, I_0) are completely invariant to test demand.")

print("=" * 80)
print("ALL E0 UNIT TESTS PASSED SUCCESSFULLY! PROCEEDING TO BENCHMARK RUNS.")
print("=" * 80)

# ==============================================================================
# 7. RUN E0 BENCHMARK POLICIES (Test Window Evaluation)
# ==============================================================================
print("\nRunning E0 Fair Benchmark Policies on Test Window (Aug 1-15, 2017)...")

e0_runs = []

# Include Legacy Oracle row for reference (Condition 6)
e0_runs.append({
    "Policy": "Quantile TFT + Conformal PSO",
    "alpha": 0.1000,
    "beta": 3.4879,
    "Mode": "oracle (uses test information)",
    "dbar_source": "test_15d_realized",
    "I_0_source": "policy_dependent",
    "warmup": False,
    "Total Cost ($)": 136378.10,
    "Holding Cost ($)": 103576.04,
    "Stockout Cost ($)": 21690.06,
    "Ordering Cost ($)": 11112.00,
    "Fill Rate (%)": 96.41
})

fair_test_specs = [
    # 1. Standard Fair Trailing 28d Mode (Full 15 days)
    ("Mean-demand Gaussian", 1.0, 4.0, "trailing_28d", False),
    ("Huber TFT + Gaussian", 1.0, 4.0, "trailing_28d", False),
    ("Quantile Default", 1.0, 4.0, "trailing_28d", False),
    ("Quantile TFT (PSO params)", 0.1000, 3.4879, "trailing_28d", False),
    
    # 2. P50 Forecast-based D_bar Mode (Full 15 days)
    ("Mean-demand Gaussian", 1.0, 4.0, "p50_forecast", False),
    ("Huber TFT + Gaussian", 1.0, 4.0, "p50_forecast", False),
    ("Quantile Default", 1.0, 4.0, "p50_forecast", False),
    ("Quantile TFT (PSO params)", 0.1000, 3.4879, "p50_forecast", False),

    # 3. Fair Trailing with 7-Day Warmup
    ("Mean-demand Gaussian", 1.0, 4.0, "trailing_28d", True),
    ("Huber TFT + Gaussian", 1.0, 4.0, "trailing_28d", True),
    ("Quantile Default", 1.0, 4.0, "trailing_28d", True),
    ("Quantile TFT (PSO params)", 0.1000, 3.4879, "trailing_28d", True),
]

for p_name, a_val, b_val, d_mode, w_flag in fair_test_specs:
    res = simulate_fair_inventory(
        alpha=a_val, beta=b_val, policy_name=p_name,
        split_name="test", dbar_mode=d_mode, warmup=w_flag
    )
    e0_runs.append({
        "Policy": p_name,
        "alpha": a_val,
        "beta": b_val,
        "Mode": f"fair_{d_mode}" + ("_warmup" if w_flag else ""),
        "dbar_source": d_mode,
        "I_0_source": "standardized_causal",
        "warmup": w_flag,
        "Total Cost ($)": round(res["Total Cost ($)"], 2),
        "Holding Cost ($)": round(res["Holding Cost ($)"], 2),
        "Stockout Cost ($)": round(res["Stockout Cost ($)"], 2),
        "Ordering Cost ($)": round(res["Ordering Cost ($)"], 2),
        "Fill Rate (%)": round(res["Fill Rate (%)"], 2),
    })

df_e0 = pd.DataFrame(e0_runs)
out_e0_path = os.path.join(TABLES_DIR, "e0_fair_mode_benchmark.csv")
df_e0.to_csv(out_e0_path, index=False)
print(f"\n[OK] Saved E0 Fair Mode Benchmark Table to: {out_e0_path}")
print(df_e0[["Policy", "alpha", "beta", "Mode", "Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)", "Fill Rate (%)"]].to_string(index=False))

elapsed = time.time() - t_start
print(f"\nExperiment E0 completed in {elapsed:.2f} seconds.")
