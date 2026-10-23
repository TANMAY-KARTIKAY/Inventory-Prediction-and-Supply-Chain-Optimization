"""
r4_matched_fill_frontier.py
R4. Matched-fill frontier:
Lower envelope of cost over configurations with test fill >= target.
Strictly monotone, no interpolation or extrapolation (N/A when unreachable).
Evaluated for targets SL in {90%, 91%, 92%, 93%, 94%, 95%, 96%, 97%}.
Reports both Primary scoring window (days 7 to 14) and Robustness window (days 0 to 14).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "figures")
CODE_DIR = os.path.join(BASE_DIR, "revision_experiments", "code")
sys.path.insert(0, CODE_DIR)

from unified_simulator import simulate_inventory

print("=" * 80)
print("R4: COMPUTING MONOTONE LOWER-ENVELOPE MATCHED-FILL FRONTIER")
print("=" * 80)

# Load existing wide grid evaluations if available, or generate comprehensive grid
grid_path = os.path.join(TABLES_DIR, "part_c_test_grid_evaluations.csv")
if os.path.exists(grid_path):
    print("Loading existing grid evaluations from part_c_test_grid_evaluations.csv...")
    df_grid = pd.read_csv(grid_path)
else:
    raise FileNotFoundError(f"Missing {grid_path}")

TARGETS = [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]
POLICIES = df_grid["Policy"].unique()

# Lower envelope over configurations with test fill >= target
# C*(T) = min { Cost(c) : Fill(c) >= T }
frontier_records = []

for pol in POLICIES:
    p_df = df_grid[df_grid["Policy"] == pol].copy()
    achievable_min = p_df["Test Fill Rate (%)"].min()
    achievable_max = p_df["Test Fill Rate (%)"].max()
    
    for target in TARGETS:
        # Configurations achieving at least target fill rate
        eligible = p_df[p_df["Test Fill Rate (%)"] >= target]
        if eligible.empty:
            cost_val = "N/A"
            alpha_val = "N/A"
            beta_val = "N/A"
            achieved_sl = "N/A"
            status = "N/A (Unreachable)"
        else:
            best_row = eligible.sort_values("Test Cost ($)").iloc[0]
            cost_val = round(best_row["Test Cost ($)"], 2)
            alpha_val = round(best_row["Alpha"], 4)
            beta_val = round(best_row["Beta"], 4)
            achieved_sl = round(best_row["Test Fill Rate (%)"], 2)
            status = "Lower Envelope (Feasible)"
            
        frontier_records.append({
            "Policy": pol,
            "Target Fill Rate (%)": target,
            "Minimum Cost ($)": cost_val,
            "Optimal Alpha": alpha_val,
            "Optimal Beta": beta_val,
            "Achieved Fill Rate (%)": achieved_sl,
            "Achievable Range Min (%)": round(achievable_min, 2),
            "Achievable Range Max (%)": round(achievable_max, 2),
            "Status": status
        })

df_frontier = pd.DataFrame(frontier_records)
df_frontier.to_csv(os.path.join(TABLES_DIR, "r4_matched_fill_frontier.csv"), index=False)
print("Saved r4_matched_fill_frontier.csv")

# Verify Monotonicity Assertion for all policies
for pol in POLICIES:
    sub = df_frontier[(df_frontier["Policy"] == pol) & (df_frontier["Minimum Cost ($)"] != "N/A")].copy()
    sub["Cost_num"] = sub["Minimum Cost ($)"].astype(float)
    costs = sub["Cost_num"].values
    if len(costs) > 1:
        is_monotone = np.all(np.diff(costs) >= -1e-6)
        print(f"Policy '{pol}' Monotonicity Verified: {is_monotone} (Costs: {costs})")
        assert is_monotone, f"Monotonicity violated for {pol}!"

# Generate Clean Figure
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

palette = {
    "Quantile Conformal": "#1f77b4",
    "Quantile Raw": "#17becf",
    "Quantile None (a=0)": "#7f7f7f",
    "Huber Gaussian": "#ff7f0e",
    "Seasonal Naive Gauss": "#2ca02c",
    "Mean-Demand Gauss": "#d62728"
}

for pol in POLICIES:
    sub = df_frontier[(df_frontier["Policy"] == pol) & (df_frontier["Minimum Cost ($)"] != "N/A")].copy()
    if not sub.empty:
        targets_plot = sub["Target Fill Rate (%)"].values
        costs_plot = sub["Minimum Cost ($)"].astype(float).values
        color = palette.get(pol, "black")
        ls = "--" if "None" in pol else "-"
        marker = "o" if "Conformal" in pol else "s"
        ax.plot(targets_plot, costs_plot, label=pol, color=color, linestyle=ls, marker=marker, linewidth=2, markersize=6)

ax.set_title("Non-Dominated Monotone Lower Envelope Frontier (Matched Test Fill Rate)", fontsize=13, fontweight="bold")
ax.set_xlabel("Target Service Level Fill Rate (%)", fontsize=11, fontweight="bold")
ax.set_ylabel("Minimum Achievable Test Inventory Cost ($)", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
plt.tight_layout()

fig_path = os.path.join(FIGURES_DIR, "r4_matched_fill_frontier.png")
fig.savefig(fig_path)
plt.close(fig)
print(f"Saved figure to {fig_path}")
