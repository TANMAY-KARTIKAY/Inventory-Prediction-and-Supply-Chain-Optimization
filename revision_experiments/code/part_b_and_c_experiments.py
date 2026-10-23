"""
part_b_and_c_experiments.py
Part B: Like-for-like paired bootstrap (B=2,000, pairs resampled)
Part C: Descriptive test matched fill-rate across expanded grid (alpha up to 4.0/3.0, strictly inside achievable range)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t_start = time.time()

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("EXECUTING PART B (BOOTSTRAP B=2000) & PART C (MATCHED FILL RATE)")
print("=" * 80)

# Load 3D matrices instantly
data_3d = np.load(os.path.join(TABLES_DIR, "simulation_3d_matrices.npz"))

test_roll_q_p50     = data_3d["test_roll_q_p50"]
test_roll_q_p90_raw = data_3d["test_roll_q_p90_raw"]
test_roll_q_p90_cal = data_3d["test_roll_q_p90_cal"]
test_roll_huber     = data_3d["test_roll_huber"]
test_roll_sn        = data_3d["test_roll_sn"]
test_roll_mean      = data_3d["test_roll_mean"]

test_demand_mat = data_3d["test_demand_mat"]
test_dbar_vec   = data_3d["test_dbar_vec"]
test_sigma_vec  = data_3d["test_sigma_vec"]
h_cost_vec      = data_3d["h_cost_vec"]
is_hvol_vec     = data_3d["is_hvol_vec"]

sig_res_p50 = data_3d["sig_res_p50"]
sig_res_hub = data_3d["sig_res_hub"]
sig_res_sn  = data_3d["sig_res_sn"]
sig_res_men = data_3d["sig_res_men"]

q90_res_p50 = data_3d["q90_res_p50"]
q90_res_hub = data_3d["q90_res_hub"]
q90_res_sn  = data_3d["q90_res_sn"]
q90_res_men = data_3d["q90_res_men"]

N_PAIRS = len(h_cost_vec)
LEAD_TIME_DAYS = 7
FIXED_ORDER_COST = 8.00
STOCKOUT_PENALTY = 2.50

# Fast simulator
def simulate_fast(forecaster, buffer_type, alpha, beta, return_per_pair=False):
    dbar = test_dbar_vec
    dm = test_demand_mat

    pipeline = np.tile(dbar[:, None], (1, LEAD_TIME_DAYS)).copy()
    on_hand = 2.0 * dbar.copy()

    pair_hold = np.zeros(N_PAIRS, dtype=np.float64)
    pair_stockout = np.zeros(N_PAIRS, dtype=np.float64)
    pair_order_primary = np.zeros(N_PAIRS, dtype=np.float64)
    pair_sales = np.zeros(N_PAIRS, dtype=np.float64)
    pair_demand = np.zeros(N_PAIRS, dtype=np.float64)

    for t in range(15):
        d_t = dm[:, t]
        arrivals = pipeline[:, 0]
        on_hand += arrivals

        sales = np.minimum(on_hand, d_t)
        stockout = np.maximum(0.0, d_t - on_hand)
        on_hand = np.maximum(0.0, on_hand - d_t)

        inv_pos = on_hand + np.sum(pipeline[:, 1:], axis=1)

        # Forecast slice
        if forecaster == "Quantile":
            fc_slice = test_roll_q_p50[:, t, :]
        elif forecaster == "Huber":
            fc_slice = test_roll_huber[:, t, :]
        elif forecaster == "SeasonalNaive":
            fc_slice = test_roll_sn[:, t, :]
        elif forecaster == "Mean":
            fc_slice = test_roll_mean[:, t, :]
        else:
            raise ValueError(f"Unknown forecaster {forecaster}")

        l_exp_t = np.sum(fc_slice, axis=1)

        # Safety stock
        if buffer_type == "none":
            ss_t = np.zeros(N_PAIRS, dtype=np.float64)
        elif buffer_type == "gaussian_demand":
            ss_t = alpha * 1.282 * test_sigma_vec
        elif buffer_type == "gaussian_residual":
            sig = sig_res_hub if forecaster == "Huber" else (sig_res_sn if forecaster == "SeasonalNaive" else sig_res_p50)
            ss_t = alpha * 1.282 * sig
        elif buffer_type == "empirical_residual":
            q_val = q90_res_hub if forecaster == "Huber" else (q90_res_sn if forecaster == "SeasonalNaive" else q90_res_p50)
            ss_t = alpha * q_val
        elif buffer_type == "raw_quantile":
            ss_t = alpha * np.sum(np.maximum(0.0, test_roll_q_p90_raw[:, t, :] - test_roll_q_p50[:, t, :]), axis=1)
        elif buffer_type == "conformal_quantile":
            ss_t = alpha * np.sum(np.maximum(0.0, test_roll_q_p90_cal[:, t, :] - test_roll_q_p50[:, t, :]), axis=1)
        else:
            raise ValueError(f"Unknown buffer {buffer_type}")

        s_t = l_exp_t + ss_t
        S_t = s_t + (beta * dbar)

        order_mask = (inv_pos <= s_t)
        order_qty = np.where(order_mask, np.maximum(0.0, S_t - inv_pos), 0.0)

        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = order_qty

        pair_sales += sales
        pair_demand += d_t
        pair_stockout += stockout * STOCKOUT_PENALTY
        pair_hold += on_hand * h_cost_vec
        if t <= 7:
            pair_order_primary += np.where(order_mask, FIXED_ORDER_COST, 0.0)

    pair_cost = pair_hold + pair_stockout + pair_order_primary

    if return_per_pair:
        return {"cost": pair_cost, "sales": pair_sales, "demand": pair_demand, "hold": pair_hold, "stockout": pair_stockout, "order": pair_order_primary}

    tot_demand = float(np.sum(pair_demand))
    tot_sales  = float(np.sum(pair_sales))
    tot_cost   = float(np.sum(pair_cost))
    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0
    return tot_cost, sl

# Load tuned parameters from Part A's comprehensive table
df_a4 = pd.read_csv(os.path.join(TABLES_DIR, "item_a4_comprehensive_forecaster_evaluation.csv"))

def get_best_params(forecaster, buffer_name, form="rolling"):
    sub = df_a4[(df_a4["Forecaster"] == forecaster) & (df_a4["Buffer"] == buffer_name) & (df_a4["Form"] == form)]
    row = sub.iloc[0]
    return float(row["Alpha*"]), float(row["Beta*"])

a_q_conf, b_q_conf = get_best_params("Quantile", "conformal_quantile", "rolling")
a_q_none, b_q_none = get_best_params("Quantile", "none", "rolling")
a_q_raw,  b_q_raw  = get_best_params("Quantile", "raw_quantile", "rolling")
a_h_none, b_h_none = get_best_params("Huber", "none", "rolling")
a_h_gauss,b_h_gauss= get_best_params("Huber", "gaussian_residual", "rolling")
a_sn_tune,b_sn_tune= get_best_params("SeasonalNaive", "gaussian_residual", "rolling")
a_mn_tune,b_mn_tune= get_best_params("Mean", "gaussian_demand", "rolling")

# Pairwise vector evaluations
p_q_conf = simulate_fast("Quantile", "conformal_quantile", a_q_conf, b_q_conf, return_per_pair=True)
p_q_none = simulate_fast("Quantile", "none", a_q_none, b_q_none, return_per_pair=True)
p_q_raw  = simulate_fast("Quantile", "raw_quantile", a_q_raw, b_q_raw, return_per_pair=True)
p_h_none = simulate_fast("Huber", "none", a_h_none, b_h_none, return_per_pair=True)
p_h_gaus = simulate_fast("Huber", "gaussian_residual", a_h_gauss, b_h_gauss, return_per_pair=True)
p_sn     = simulate_fast("SeasonalNaive", "gaussian_residual", a_sn_tune, b_sn_tune, return_per_pair=True)
p_mn     = simulate_fast("Mean", "gaussian_demand", a_mn_tune, b_mn_tune, return_per_pair=True)

# ------------------------------------------------------------------------------
# PART B: PAIRED BOOTSTRAP (B=2000, STORE-ITEM PAIRS RESAMPLED)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART B: PAIRED BOOTSTRAP OVER STORE-ITEM PAIRS (B = 2,000)")
print("=" * 80)

np.random.seed(42)
B = 2000

tot_test_dem = p_q_conf["demand"]
top20_indices = set(np.argsort(tot_test_dem)[-20:])
all_indices = np.arange(N_PAIRS)
excl_indices = np.array([i for i in range(N_PAIRS) if i not in top20_indices])
hvol_indices = np.where(is_hvol_vec)[0]

SUBSETS = [
    ("All Pairs (N=363)", all_indices),
    ("Excluding Top 20 (N=343)", excl_indices),
    ("High-Volume Only (N=20)", hvol_indices)
]

COMPARISONS = [
    ("Quantile Conformal vs Quantile None", p_q_conf, p_q_none),
    ("Quantile Conformal vs Quantile Raw",  p_q_conf, p_q_raw),
    ("Quantile Conformal vs Huber None",    p_q_conf, p_h_none),
    ("Quantile Conformal vs Huber Tuned",   p_q_conf, p_h_gaus),
    ("Quantile Conformal vs Seasonal Naive",p_q_conf, p_sn),
    ("Quantile Conformal vs Mean-Demand",   p_q_conf, p_mn)
]

bootstrap_records = []

for sub_name, sub_idx in SUBSETS:
    n_sub = len(sub_idx)
    boot_resamples = np.random.choice(sub_idx, size=(B, n_sub), replace=True)
    active_comps = [("Quantile Conformal vs Quantile None", p_q_conf, p_q_none)] if "High-Volume" in sub_name else COMPARISONS
    
    for comp_name, p1, p2 in active_comps:
        c1_boot = np.sum(p1["cost"][boot_resamples], axis=1)
        c2_boot = np.sum(p2["cost"][boot_resamples], axis=1)
        delta_c = c1_boot - c2_boot
        
        s1_boot = np.sum(p1["sales"][boot_resamples], axis=1)
        d1_boot = np.sum(p1["demand"][boot_resamples], axis=1)
        sl1_boot = (s1_boot / np.maximum(1e-6, d1_boot)) * 100.0
        
        s2_boot = np.sum(p2["sales"][boot_resamples], axis=1)
        d2_boot = np.sum(p2["demand"][boot_resamples], axis=1)
        sl2_boot = (s2_boot / np.maximum(1e-6, d2_boot)) * 100.0
        
        delta_sl = sl1_boot - sl2_boot
        
        mean_dc = float(np.mean(delta_c))
        ci_dc_low = float(np.percentile(delta_c, 2.5))
        ci_dc_high = float(np.percentile(delta_c, 97.5))
        win_rate = float(np.mean(delta_c < 0.0) * 100.0)
        
        mean_dsl = float(np.mean(delta_sl))
        ci_dsl_low = float(np.percentile(delta_sl, 2.5))
        ci_dsl_high = float(np.percentile(delta_sl, 97.5))
        
        bootstrap_records.append({
            "Sample Subset": sub_name,
            "Comparison (Policy 1 vs Policy 2)": comp_name,
            "Mean Delta Cost ($)": round(mean_dc, 2),
            "95% CI Delta Cost ($)": f"[{ci_dc_low:+.2f}, {ci_dc_high:+.2f}]",
            "P1 Cheaper Win Rate (%)": round(win_rate, 2),
            "Mean Delta Fill Rate (%)": round(mean_dsl, 2),
            "95% CI Delta Fill Rate (%)": f"[{ci_dsl_low:+.2f}%, {ci_dsl_high:+.2f}%]"
        })

df_boot = pd.DataFrame(bootstrap_records)
df_boot.to_csv(os.path.join(TABLES_DIR, "part_b_paired_bootstrap.csv"), index=False)
print("Part B Paired Bootstrap Results:")
print(df_boot.to_string(index=False))

# ------------------------------------------------------------------------------
# PART C: MATCHED FILL RATE ACROSS EXPANDED GRID
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PART C: MATCHED FILL RATE ACROSS EXPANDED GRID")
print("=" * 80)

POLICIES_GRID = [
    ("Quantile Conformal", "Quantile", "conformal_quantile", np.linspace(0.0, 3.0, 31)),
    ("Quantile Raw",       "Quantile", "raw_quantile",       np.linspace(0.0, 3.0, 31)),
    ("Quantile None (a=0)","Quantile", "none",               np.array([0.0])),
    ("Huber Gaussian",     "Huber",    "gaussian_residual",  np.linspace(0.0, 4.0, 41)),
    ("Mean-Demand Gauss",  "Mean",     "gaussian_demand",    np.linspace(0.0, 4.0, 41)),
    ("Seasonal Naive Gauss","SeasonalNaive", "gaussian_residual", np.linspace(0.0, 4.0, 41))
]

beta_grid = np.linspace(0.5, 5.0, 46)

grid_eval_records = []
frontiers = {}

for pol_label, f_n, b_n, a_vals in POLICIES_GRID:
    print(f"Evaluating test grid for {pol_label} ({len(a_vals) * len(beta_grid)} points)...")
    costs = []
    sls = []
    for a_v in a_vals:
        for b_v in beta_grid:
            c, sl = simulate_fast(f_n, b_n, a_v, b_v)
            costs.append(c)
            sls.append(sl)
            grid_eval_records.append({
                "Policy": pol_label,
                "Alpha": a_v,
                "Beta": b_v,
                "Test Cost ($)": c,
                "Test Fill Rate (%)": sl
            })
    
    df_pol = pd.DataFrame({"cost": costs, "sl": sls}).sort_values("sl")
    df_pol["sl_bin"] = np.round(df_pol["sl"], 2)
    frontier_df = df_pol.groupby("sl_bin")["cost"].min().reset_index().rename(columns={"sl_bin": "sl"}).sort_values("sl")
    frontiers[pol_label] = (frontier_df["sl"].values, frontier_df["cost"].values)

df_grid_eval = pd.DataFrame(grid_eval_records)
df_grid_eval.to_csv(os.path.join(TABLES_DIR, "part_c_test_grid_evaluations.csv"), index=False)

TARGET_SLS = [92.0, 93.0, 94.0, 95.0]
matched_records = []

for target_sl in TARGET_SLS:
    for pol_label, f_n, b_n, _ in POLICIES_GRID:
        sl_arr, c_arr = frontiers[pol_label]
        min_sl = float(np.min(sl_arr))
        max_sl = float(np.max(sl_arr))
        
        if min_sl <= target_sl <= max_sl:
            interp_c = float(np.interp(target_sl, sl_arr, c_arr))
            status = "Interpolated"
            cost_str = f"{interp_c:.2f}"
        else:
            interp_c = np.nan
            status = "N/A (Outside Range)"
            cost_str = "N/A"
            
        matched_records.append({
            "Target Test Fill Rate (%)": target_sl,
            "Policy": pol_label,
            "Descriptive Cost ($)": cost_str,
            "Achievable Min SL (%)": round(min_sl, 2),
            "Achievable Max SL (%)": round(max_sl, 2),
            "Status": status
        })

df_matched = pd.DataFrame(matched_records)
df_matched.to_csv(os.path.join(TABLES_DIR, "part_c_matched_fill_rate_descriptive.csv"), index=False)
print("\nPart C Matched Fill Rate (Descriptive Test-Grid Oracle):")
print(df_matched.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
colors = {
    "Quantile Conformal": "#1f77b4",
    "Quantile Raw": "#17becf",
    "Quantile None (a=0)": "#7f7f7f",
    "Huber Gaussian": "#d62728",
    "Mean-Demand Gauss": "#ff7f0e",
    "Seasonal Naive Gauss": "#2ca02c"
}

for pol_label, (sl_arr, c_arr) in frontiers.items():
    plt.plot(sl_arr, c_arr, marker="o", markersize=3, label=pol_label, color=colors.get(pol_label, "black"))

for target_sl in TARGET_SLS:
    plt.axvline(target_sl, color="gray", linestyle="--", alpha=0.5)

plt.xlabel("Test Fill Rate (%)", fontsize=12)
plt.ylabel("Test Total Cost ($)", fontsize=12)
plt.title("Part C: Descriptive Test-Grid Frontier Across Expanded (Alpha, Beta) Bounds", fontsize=13, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "part_c_matched_fill_rate_descriptive.png"), dpi=300)
plt.close()
print("Saved Part C plot to outputs/figures/part_c_matched_fill_rate_descriptive.png")

print(f"\nPart B & C completed in {time.time() - t_start:.2f} seconds.")
