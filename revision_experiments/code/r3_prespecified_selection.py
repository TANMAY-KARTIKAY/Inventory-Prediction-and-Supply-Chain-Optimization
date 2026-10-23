"""
r3_prespecified_selection.py
R3. Pre-specified selection:
- For each forecaster, select (buffer, static/rolling) with lowest validation fitness (cost + penalty).
- Restricted selection: restricted to buffers using no validation-fitted statistics
  (none, trailing-sigma gaussian_demand, raw_quantile, conformal_quantile).
- Evaluates chosen variants once on test (Primary: days 7-14; Robustness: days 0-14).
- Reruns paired bootstrap (B=2,000; all 363 pairs, and excluding top 20) between validation-selected variants,
  reporting cost and fill differences with 95% CIs and win rates.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
CODE_DIR = os.path.join(BASE_DIR, "revision_experiments", "code")
sys.path.insert(0, CODE_DIR)

from unified_simulator import (
    simulate_inventory,
    get_tuning_fitness,
    N_PAIRS,
    is_hvol_vec
)

np.random.seed(42)
t_start = time.time()

# 1. Load A4 Comprehensive Evaluation Table
a4_path = os.path.join(TABLES_DIR, "item_a4_comprehensive_forecaster_evaluation.csv")
df_a4 = pd.read_csv(a4_path)

# Evaluate all variants on both Primary (days 7-14) and Robustness (days 0-14)
records_all = []
for _, r in df_a4.iterrows():
    f = r["Forecaster"]
    b = r["Buffer"]
    form = r["Form"]
    is_roll = (form == "rolling")
    a = float(r["Alpha*"])
    beta = float(r["Beta*"])
    
    # Validation evaluations
    fit_val_prim, c_v_prim, sl_v_prim = get_tuning_fitness(a, beta, f, b, split="val", is_rolling=is_roll, score_days=(7, 15))
    fit_val_rob, c_v_rob, sl_v_rob   = get_tuning_fitness(a, beta, f, b, split="val", is_rolling=is_roll, score_days=(0, 15))
    
    # Test evaluations
    c_t_prim, sl_t_prim, h_t_prim, so_t_prim, o_t_prim, _ = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(7, 15), accounting_mode="arriving_orders"
    )
    c_t_rob, sl_t_rob, h_t_rob, so_t_rob, o_t_rob_prim, o_t_rob_all = simulate_inventory(
        f, b, a, beta, split="test", is_rolling=is_roll, score_days=(0, 15), accounting_mode="arriving_orders"
    )
    
    records_all.append({
        "Forecaster": f,
        "Buffer": b,
        "Form": form,
        "Alpha*": a,
        "Beta*": beta,
        # Primary Window (Days 7 to 14)
        "Val Fitness (Primary) ($)": round(fit_val_prim, 2),
        "Val Cost (Primary) ($)": round(c_v_prim, 2),
        "Val Fill Rate (Primary) (%)": round(sl_v_prim, 2),
        "Test Cost (Primary) ($)": round(c_t_prim, 2),
        "Test Holding (Primary) ($)": round(h_t_prim, 2),
        "Test Stockout (Primary) ($)": round(so_t_prim, 2),
        "Test Ordering (Primary) ($)": round(o_t_prim, 2),
        "Test Fill Rate (Primary) (%)": round(sl_t_prim, 2),
        # Robustness Window (Days 0 to 14)
        "Val Fitness (Robustness) ($)": round(fit_val_rob, 2),
        "Val Cost (Robustness) ($)": round(c_v_rob, 2),
        "Val Fill Rate (Robustness) (%)": round(sl_v_rob, 2),
        "Test Cost (Robustness) ($)": round(c_t_rob, 2),
        "Test Holding (Robustness) ($)": round(h_t_rob, 2),
        "Test Stockout (Robustness) ($)": round(so_t_rob, 2),
        "Test Ordering (Robustness) ($)": round(o_t_rob_prim, 2),
        "Test Ordering All Fees ($)": round(o_t_rob_all, 2),
        "Test Fill Rate (Robustness) (%)": round(sl_t_rob, 2),
        "Misses 95% on Test (Robustness)?": bool(sl_t_rob < 95.0),
        "Misses 95% on Test (Primary)?": bool(sl_t_prim < 95.0)
    })

df_all_variants = pd.DataFrame(records_all)
df_all_variants.to_csv(os.path.join(TABLES_DIR, "r3_all_variants_val_vs_test.csv"), index=False)
print("Saved r3_all_variants_val_vs_test.csv")

# ------------------------------------------------------------------------------
# 2. PRE-SPECIFIED SELECTION ACROSS FORECASTERS
# ------------------------------------------------------------------------------
FORECASTERS = ["Quantile", "Huber", "SeasonalNaive", "Mean"]
RESTRICTED_BUFFERS = ["none", "gaussian_demand", "raw_quantile", "conformal_quantile"]

selection_records = []

for sel_type in ["All Buffers", "Restricted (No Val Residual Stats)"]:
    sub_df = df_all_variants if sel_type == "All Buffers" else df_all_variants[df_all_variants["Buffer"].isin(RESTRICTED_BUFFERS)]
    for f in FORECASTERS:
        f_grp = sub_df[sub_df["Forecaster"] == f]
        # Pre-specified rule: choose variant with lowest validation fitness
        best_row = f_grp.sort_values("Val Fitness (Robustness) ($)").iloc[0]
        selection_records.append({
            "Selection Regime": sel_type,
            "Forecaster": f,
            "Selected Buffer": best_row["Buffer"],
            "Selected Form": best_row["Form"],
            "Alpha*": best_row["Alpha*"],
            "Beta*": best_row["Beta*"],
            "Val Fitness ($)": best_row["Val Fitness (Robustness) ($)"],
            "Val Cost ($)": best_row["Val Cost (Robustness) ($)"],
            "Val Fill Rate (%)": best_row["Val Fill Rate (Robustness) (%)"],
            "Test Cost (Primary 7-14) ($)": best_row["Test Cost (Primary) ($)"],
            "Test Fill Rate (Primary 7-14) (%)": best_row["Test Fill Rate (Primary) (%)"],
            "Test Cost (Robustness 0-14) ($)": best_row["Test Cost (Robustness) ($)"],
            "Test Holding ($)": best_row["Test Holding (Robustness) ($)"],
            "Test Stockout ($)": best_row["Test Stockout (Robustness) ($)"],
            "Test Ordering ($)": best_row["Test Ordering (Robustness) ($)"],
            "Test Fill Rate (Robustness 0-14) (%)": best_row["Test Fill Rate (Robustness) (%)"],
            "Misses 95% on Test?": best_row["Misses 95% on Test (Robustness)?"]
        })

df_selection = pd.DataFrame(selection_records)
df_selection.to_csv(os.path.join(TABLES_DIR, "r3_prespecified_selection_summary.csv"), index=False)
print("\nPRE-SPECIFIED VALIDATION SELECTION SUMMARY:")
print(df_selection.to_string(index=False))


# ------------------------------------------------------------------------------
# 3. PAIRED BOOTSTRAP (B = 2,000) BETWEEN VALIDATION-SELECTED VARIANTS
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RUNNING PAIRED BOOTSTRAP (B = 2,000) FOR VALIDATION-SELECTED POLICIES")
print("=" * 80)

# Pre-compute per-pair results on test for all policies of interest
# Primary Window scoring (days 7 to 14) and Robustness (days 0 to 14)
def get_policy_pair_data(f, b, a, beta, is_roll, score_days=(7, 15)):
    cost_p, sales_p, dem_p, hold_p, so_p, ord_p, _ = simulate_inventory(
        forecaster=f,
        buffer_type=b,
        alpha=a,
        beta=beta,
        split="test",
        is_rolling=is_roll,
        accounting_mode="arriving_orders",
        score_days=score_days,
        return_per_pair=True
    )
    return cost_p, sales_p, dem_p

# Identify Top-20 volume pairs
sim_inputs = np.load(os.path.join(TABLES_DIR, "simulation_inputs.npz"))
test_dbar = sim_inputs["test_dbar_vec"]
top20_indices = set(np.argsort(test_dbar)[-20:])
non_top20_mask = np.array([idx not in top20_indices for idx in range(N_PAIRS)])

# Extract pre-specified winning variants (under All Buffers regime)
sel_dict = {}
for _, r in df_selection[df_selection["Selection Regime"] == "All Buffers"].iterrows():
    sel_dict[r["Forecaster"]] = (r["Selected Buffer"], r["Selected Form"] == "rolling", float(r["Alpha*"]), float(r["Beta*"]))

# Also extract Quantile Conformal and Raw for direct comparisons
q_conf = ("conformal_quantile", True, 0.2333, 1.0807)
q_raw  = ("raw_quantile", True, 0.2350, 1.0984)
q_none = ("none", True, 0.0000, 3.9731)

COMPARISONS = [
    ("Quantile Selected vs Huber Selected",
     ("Quantile", sel_dict["Quantile"][0], sel_dict["Quantile"][2], sel_dict["Quantile"][3], sel_dict["Quantile"][1]),
     ("Huber", sel_dict["Huber"][0], sel_dict["Huber"][2], sel_dict["Huber"][3], sel_dict["Huber"][1])),
    
    ("Quantile Selected vs Seasonal Naive Selected",
     ("Quantile", sel_dict["Quantile"][0], sel_dict["Quantile"][2], sel_dict["Quantile"][3], sel_dict["Quantile"][1]),
     ("SeasonalNaive", sel_dict["SeasonalNaive"][0], sel_dict["SeasonalNaive"][2], sel_dict["SeasonalNaive"][3], sel_dict["SeasonalNaive"][1])),
     
    ("Quantile Selected vs Mean Selected",
     ("Quantile", sel_dict["Quantile"][0], sel_dict["Quantile"][2], sel_dict["Quantile"][3], sel_dict["Quantile"][1]),
     ("Mean", sel_dict["Mean"][0], sel_dict["Mean"][2], sel_dict["Mean"][3], sel_dict["Mean"][1])),
     
    ("Quantile Conformal vs Quantile None",
     ("Quantile", q_conf[0], q_conf[2], q_conf[3], q_conf[1]),
     ("Quantile", q_none[0], q_none[2], q_none[3], q_none[1])),
     
    ("Quantile Conformal vs Quantile Raw",
     ("Quantile", q_conf[0], q_conf[2], q_conf[3], q_conf[1]),
     ("Quantile", q_raw[0], q_raw[2], q_raw[3], q_raw[1])),
     
    ("Conformal vs None (High-Volume Pairs Only)",
     ("Quantile", q_conf[0], q_conf[2], q_conf[3], q_conf[1]),
     ("Quantile", q_none[0], q_none[2], q_none[3], q_none[1]))
]

B = 2000
boot_records = []

for comp_name, pol_a, pol_b in COMPARISONS:
    # Run on both Primary (days 7-14) and Robustness (days 0-14)
    for score_window, w_label in [((7, 15), "Primary (Days 7-14)"), ((0, 15), "Robustness (Days 0-14)")]:
        cost_a, sales_a, dem_a = get_policy_pair_data(pol_a[0], pol_a[1], pol_a[2], pol_a[3], pol_a[4], score_days=score_window)
        cost_b, sales_b, dem_b = get_policy_pair_data(pol_b[0], pol_b[1], pol_b[2], pol_b[3], pol_b[4], score_days=score_window)
        
        # Subsets: All Pairs and Excl Top 20 (or High-Vol only)
        if "High-Volume" in comp_name:
            subsets = [("High-Volume Pairs Only (N=50)", is_hvol_vec)]
        else:
            subsets = [
                ("All Pairs (N=363)", np.ones(N_PAIRS, dtype=bool)),
                ("Excluding Top 20 Volume Pairs (N=343)", non_top20_mask)
            ]
            
        for sub_label, mask in subsets:
            c_a = cost_a[mask]
            s_a = sales_a[mask]
            d_a = dem_a[mask]
            
            c_b = cost_b[mask]
            s_b = sales_b[mask]
            d_b = dem_b[mask]
            
            n_sub = len(c_a)
            boot_idx = np.random.randint(0, n_sub, size=(B, n_sub))
            
            b_cost_diff = np.zeros(B)
            b_sl_diff = np.zeros(B)
            
            for b_i in range(B):
                idx = boot_idx[b_i]
                tot_c_a = np.sum(c_a[idx])
                tot_c_b = np.sum(c_b[idx])
                b_cost_diff[b_i] = tot_c_a - tot_c_b
                
                sl_a = (np.sum(s_a[idx]) / np.maximum(1e-6, np.sum(d_a[idx]))) * 100.0
                sl_b = (np.sum(s_b[idx]) / np.maximum(1e-6, np.sum(d_b[idx]))) * 100.0
                b_sl_diff[b_i] = sl_a - sl_b
                
            mean_c_diff = float(np.mean(b_cost_diff))
            ci_c_lower  = float(np.percentile(b_cost_diff, 2.5))
            ci_c_upper  = float(np.percentile(b_cost_diff, 97.5))
            
            mean_sl_diff = float(np.mean(b_sl_diff))
            ci_sl_lower  = float(np.percentile(b_sl_diff, 2.5))
            ci_sl_upper  = float(np.percentile(b_sl_diff, 97.5))
            
            p_cheaper = float(np.mean(b_cost_diff < 0.0) * 100.0)
            
            boot_records.append({
                "Comparison": comp_name,
                "Scoring Window": w_label,
                "Cohort": sub_label,
                "Mean Cost Delta ($)": round(mean_c_diff, 2),
                "Cost Delta 95% CI ($)": f"[{ci_c_lower:.2f}, {ci_c_upper:.2f}]",
                "Mean Fill Delta (%)": round(mean_sl_diff, 2),
                "Fill Delta 95% CI (%)": f"[{ci_sl_lower:.2f}, {ci_sl_upper:.2f}]",
                "Share Pol A Cheaper (%)": round(p_cheaper, 1)
            })

df_boot = pd.DataFrame(boot_records)
df_boot.to_csv(os.path.join(TABLES_DIR, "r3_paired_bootstrap_selected.csv"), index=False)
print("Saved r3_paired_bootstrap_selected.csv")
print("\nSAMPLE BOOTSTRAP RESULTS:")
print(df_boot.head(10).to_string(index=False))
print(f"\nR3 Execution Completed in {time.time() - t_start:.2f} seconds.")
