import time
import numpy as np
import pandas as pd

# Load e0 module to get preprocessed data
from e0_fair_simulator import (
    active_test_pairs, pair_perishable, pair_is_high_vol,
    dbar_val_trailing, sigma_val_trailing, sigma_res_val,
    val_daily_actual, val_daily_p50, val_daily_p90, val_daily_huber,
    dbar_test_trailing, sigma_test_trailing,
    test_daily_actual, test_daily_p50, test_daily_p90, test_daily_huber,
    simulate_fair_inventory,
    UNIT_HOLDING_DRY, UNIT_HOLDING_PERISH, STOCKOUT_PENALTY, FIXED_ORDER_COST, LEAD_TIME_DAYS
)

# Convert dictionary data into 2D NumPy arrays of shape (N_pairs, 15)
N = len(active_test_pairs)
val_demand_mat = np.array([val_daily_actual[p] for p in active_test_pairs], dtype=np.float64) # (N, 15)
val_p50_mat    = np.array([val_daily_p50[p] for p in active_test_pairs], dtype=np.float64)
val_p90_mat    = np.array([val_daily_p90[p] for p in active_test_pairs], dtype=np.float64)
val_huber_mat  = np.array([val_daily_huber[p] for p in active_test_pairs], dtype=np.float64)
val_dbar_vec   = np.array([dbar_val_trailing[p] for p in active_test_pairs], dtype=np.float64) # (N,)
val_sigma_vec  = np.array([sigma_val_trailing[p] for p in active_test_pairs], dtype=np.float64)
val_sig_res_vec= np.array([sigma_res_val[p] for p in active_test_pairs], dtype=np.float64)

h_cost_vec     = np.array([UNIT_HOLDING_PERISH if pair_perishable[p] else UNIT_HOLDING_DRY for p in active_test_pairs], dtype=np.float64)

def simulate_vectorized(alpha, beta, policy_name="Quantile TFT", split="val"):
    if split == "val":
        demand_mat = val_demand_mat
        p50_mat = val_p50_mat
        p90_mat = val_p90_mat
        huber_mat = val_huber_mat
        dbar_vec = val_dbar_vec
        sigma_vec = val_sigma_vec
        sig_res_vec = val_sig_res_vec
    
    # Thresholds
    if policy_name in ["Mean-demand Gaussian", "Classical Gaussian"]:
        lead_exp = dbar_vec * LEAD_TIME_DAYS
        ss = alpha * 1.282 * sigma_vec
    elif policy_name in ["Huber TFT + Gaussian", "Huber TFT"]:
        lead_exp = np.sum(huber_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = alpha * 1.282 * sig_res_vec
    elif policy_name in ["Quantile TFT", "Quantile Conformal", "Quantile Default"]:
        lead_exp = np.sum(p50_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = alpha * np.sum(np.maximum(0.0, p90_mat[:, :LEAD_TIME_DAYS] - p50_mat[:, :LEAD_TIME_DAYS]), axis=1)
    elif policy_name in ["No Safety Stock", "No SS"]:
        lead_exp = np.sum(p50_mat[:, :LEAD_TIME_DAYS], axis=1)
        ss = np.zeros(N, dtype=np.float64)

    s = lead_exp + ss
    S = s + (beta * dbar_vec)

    # Initial inventory
    on_hand = 0.70 * LEAD_TIME_DAYS * dbar_vec
    pipeline = np.zeros((N, LEAD_TIME_DAYS), dtype=np.float64)

    tot_hold = 0.0
    tot_stockout = 0.0
    tot_order = 0.0
    tot_sales = 0.0
    tot_demand = float(np.sum(demand_mat))

    for t in range(15):
        d_t = demand_mat[:, t]
        arrivals = pipeline[:, 0]
        on_hand += arrivals
        
        # Sales & stockout
        sales = np.minimum(on_hand, d_t)
        stockout = np.maximum(0.0, d_t - on_hand)
        on_hand = np.maximum(0.0, on_hand - d_t)

        tot_sales += float(np.sum(sales))
        tot_stockout += float(np.sum(stockout * STOCKOUT_PENALTY))
        tot_hold += float(np.sum(on_hand * h_cost_vec))

        # Review & ordering
        inv_pos = on_hand + np.sum(pipeline[:, 1:], axis=1)
        order_mask = (inv_pos <= s)
        order_qty = np.where(order_mask, np.maximum(0.0, S - inv_pos), 0.0)

        tot_order += float(np.sum(order_mask) * FIXED_ORDER_COST)

        # Roll pipeline left and append new order
        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = order_qty

    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0
    tot_cost = tot_hold + tot_stockout + tot_order
    return tot_cost, sl, tot_hold, tot_stockout, tot_order

# Compare speed and accuracy
t0 = time.time()
cost_v, sl_v, h_v, s_v, o_v = simulate_vectorized(0.1, 3.4879, "Quantile TFT", "val")
dur_v = time.time() - t0

t0 = time.time()
res_iter = simulate_fair_inventory(0.1, 3.4879, "Quantile TFT", "validation", "trailing_28d", False)
dur_i = time.time() - t0

print(f"Vectorized: Cost = ${cost_v:,.2f}, SL = {sl_v:.2f}% ({dur_v*1000:.2f} ms)")
print(f"Iterative:  Cost = ${res_iter['Total Cost ($)']:,.2f}, SL = {res_iter['Fill Rate (%)']:.2f}% ({dur_i*1000:.2f} ms)")
print(f"Cost Delta = ${abs(cost_v - res_iter['Total Cost ($)']):.8e}")
print(f"SL Delta   = {abs(sl_v - res_iter['Fill Rate (%)']):.8e}")
assert abs(cost_v - res_iter["Total Cost ($)"]) < 1e-4, "Mismatch between vectorized and iterative simulator!"
print(f"Speedup: {dur_i / dur_v:.1f}x faster!")
