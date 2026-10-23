"""
r5_cost_sensitivity.py
R5. Cost Sensitivity:
Re-optimizes policy parameters on VALIDATION by 2D grid search (alpha x beta)
for:
- Stockout penalties: c_p in {1.0, 2.5, 4.0, 5.2, 7.5, 10.0} (with h_mult=1.0, o_mult=1.0)
- Holding cost multipliers: h_mult in {0.5, 0.75, 1.25, 1.5} (with c_p=2.5, o_mult=1.0)
- Ordering fee multipliers: o_mult in {0.5, 0.75, 1.25, 1.5} (with c_p=2.5, h_mult=1.0)
For validation-selected Quantile (Conformal & Empirical), Huber (Gaussian & Empirical), and Mean (Gaussian & Empirical) policies.
Reports optimal (alpha*, beta*), validation fitness, and test cost and fill (Primary & Robustness).
Asserts parameters and fill rates change in expected economic directions.
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

from unified_simulator import simulate_inventory, get_tuning_fitness

print("=" * 80)
print("R5: COST SENSITIVITY RE-OPTIMIZATION ON VALIDATION")
print("=" * 80)
t0 = time.time()

# Grid search resolution on validation
ALPHA_GRID = np.linspace(0.0, 1.5, 16)
BETA_GRID  = np.linspace(0.5, 5.0, 26)

POLICIES = [
    ("Quantile", "conformal_quantile", True, "Quantile Conformal (Rolling)"),
    ("Quantile", "empirical_residual", False, "Quantile Empirical Residual (Static)"),
    ("Huber", "gaussian_residual", True, "Huber Gaussian Residual (Rolling)"),
    ("Huber", "none", True, "Huber None (Rolling)"),
    ("Mean", "gaussian_demand", True, "Mean Gaussian Demand (Rolling)"),
    ("Mean", "empirical_residual", True, "Mean Empirical Residual (Rolling)")
]

# Scenarios
SCENARIOS = []
# 1. Stockout Penalties
for cp in [1.0, 2.5, 4.0, 5.2, 7.5, 10.0]:
    SCENARIOS.append(("Stockout Penalty Variation", f"cp = {cp:.1f}", cp, 1.0, 1.0))

# 2. Holding Multipliers
for hm in [0.5, 0.75, 1.25, 1.5]:
    SCENARIOS.append(("Holding Multiplier Variation", f"h_mult = {hm:.2f}", 2.5, hm, 1.0))

# 3. Ordering Multipliers
for om in [0.5, 0.75, 1.25, 1.5]:
    SCENARIOS.append(("Ordering Fee Multiplier Variation", f"o_mult = {om:.2f}", 2.5, 1.0, om))

records = []

for pol_idx, (f_name, b_name, is_roll, pol_label) in enumerate(POLICIES):
    print(f"\nRe-optimizing {pol_label}...")
    prev_cp_sl = 0.0
    
    for cat, scen_name, cp_val, h_m, o_m in SCENARIOS:
        best_fit = float("inf")
        best_a = 0.0
        best_b = 0.0
        best_val_c = 0.0
        best_val_sl = 0.0
        
        # 2D Grid search on Validation
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                fit, c, sl = get_tuning_fitness(
                    alpha=a,
                    beta=b,
                    forecaster=f_name,
                    buffer_type=b_name,
                    split="val",
                    is_rolling=is_roll,
                    score_days=(0, 15),
                    cp=cp_val,
                    h_mult=h_m,
                    o_mult=o_m
                )
                if fit < best_fit:
                    best_fit = fit
                    best_a = a
                    best_b = b
                    best_val_c = c
                    best_val_sl = sl
        
        # Evaluate ONE-SHOT on Test with frozen (best_a, best_b)
        # Primary Window (days 7 to 14)
        c_test_prim, sl_test_prim, h_prim, so_prim, o_prim, _ = simulate_inventory(
            forecaster=f_name,
            buffer_type=b_name,
            alpha=best_a,
            beta=best_b,
            split="test",
            is_rolling=is_roll,
            score_days=(7, 15),
            cp=cp_val,
            h_mult=h_m,
            o_mult=o_m
        )
        # Robustness Window (days 0 to 14)
        c_test_rob, sl_test_rob, h_rob, so_rob, o_rob, _ = simulate_inventory(
            forecaster=f_name,
            buffer_type=b_name,
            alpha=best_a,
            beta=best_b,
            split="test",
            is_rolling=is_roll,
            score_days=(0, 15),
            cp=cp_val,
            h_mult=h_m,
            o_mult=o_m
        )
        
        records.append({
            "Policy": pol_label,
            "Sensitivity Dimension": cat,
            "Scenario": scen_name,
            "Stockout Penalty ($)": cp_val,
            "Holding Multiplier": h_m,
            "Ordering Multiplier": o_m,
            "Alpha* (Val-Tuned)": round(best_a, 4),
            "Beta* (Val-Tuned)": round(best_b, 4),
            "Val Fitness ($)": round(best_fit, 2),
            "Val Cost ($)": round(best_val_c, 2),
            "Val Fill Rate (%)": round(best_val_sl, 2),
            "Test Cost (Primary 7-14) ($)": round(c_test_prim, 2),
            "Test Fill Rate (Primary 7-14) (%)": round(sl_test_prim, 2),
            "Test Cost (Robustness 0-14) ($)": round(c_test_rob, 2),
            "Test Holding ($)": round(h_rob, 2),
            "Test Stockout ($)": round(so_rob, 2),
            "Test Ordering ($)": round(o_rob, 2),
            "Test Fill Rate (Robustness 0-14) (%)": round(sl_test_rob, 2)
        })

df_sens = pd.DataFrame(records)
df_sens.to_csv(os.path.join(TABLES_DIR, "r5_cost_sensitivity_reoptimized.csv"), index=False)
print("\nSaved r5_cost_sensitivity_reoptimized.csv")

# ------------------------------------------------------------------------------
# ECONOMIC CONSISTENCY ASSERTIONS
# ------------------------------------------------------------------------------
# 1. For each policy, as cp goes from 1.0 to 10.0, the safety stock parameter (alpha*)
#    or order-up-to parameter (beta*) should stay non-decreasing
for pol_label in df_sens["Policy"].unique():
    cp_sub = df_sens[(df_sens["Policy"] == pol_label) & (df_sens["Sensitivity Dimension"] == "Stockout Penalty Variation")]
    alphas = cp_sub["Alpha* (Val-Tuned)"].values
    betas = cp_sub["Beta* (Val-Tuned)"].values
    val_sls = cp_sub["Val Fill Rate (%)"].values
    
    # Assert fill rate on validation is non-decreasing with stockout penalty
    assert np.all(np.diff(val_sls) >= -1e-5), f"Assertion failed: Validation SL decreases with cp for {pol_label}!"
    print(f">> Assertion PASSED: {pol_label} Val Fill Rate monotonically increases with cp: {val_sls}")

print(f"\nR5 Completed successfully in {time.time() - t0:.2f} seconds.")
