"""
unified_simulator.py
Locked, single unified inventory simulator module for Round 3 experiments.
Used by tuning, test, waterfall, bootstrap, matched-fill frontier, and sensitivity.
Contains zero top-level execution side-effects.
"""

import os
import numpy as np

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
TABLES_DIR = os.path.join(BASE_DIR, "revision_experiments", "outputs", "tables")

# Economic Cost Parameters
UNIT_HOLDING_DRY    = 0.10
UNIT_HOLDING_PERISH = 0.15
STOCKOUT_PENALTY    = 2.50
FIXED_ORDER_COST    = 8.00
LEAD_TIME_DAYS      = 7

# Exact Validation Conformal Offsets
OFFSET_STD_VAL  = 0.1785
OFFSET_HVOL_VAL = 4.5766

# Hardcoded Legacy Offsets
OFFSET_STD_LEG  = 0.18
OFFSET_HVOL_LEG = 4.51

# Load Precomputed 3D Matrices and Inputs
_data_3d = np.load(os.path.join(TABLES_DIR, "simulation_3d_matrices.npz"))
_sim_inputs = np.load(os.path.join(TABLES_DIR, "simulation_inputs.npz"))

val_demand_mat  = _data_3d["val_demand_mat"]
test_demand_mat = _data_3d["test_demand_mat"]
val_dbar_vec    = _data_3d["val_dbar_vec"]
test_dbar_vec   = _data_3d["test_dbar_vec"]
val_sigma_vec   = _data_3d["val_sigma_vec"]
test_sigma_vec  = _data_3d["test_sigma_vec"]

h_cost_vec      = _data_3d["h_cost_vec"]
is_hvol_vec     = _data_3d["is_hvol_vec"]
N_PAIRS         = len(h_cost_vec)

# 3D Forecast arrays: shape (N_PAIRS, 15, 7)
val_roll_q_p50      = _data_3d["val_roll_q_p50"]
test_roll_q_p50     = _data_3d["test_roll_q_p50"]
val_roll_q_p90_raw  = _data_3d["val_roll_q_p90_raw"]
test_roll_q_p90_raw = _data_3d["test_roll_q_p90_raw"]
val_roll_q_p90_cal  = _data_3d["val_roll_q_p90_cal"]
test_roll_q_p90_cal = _data_3d["test_roll_q_p90_cal"]

val_roll_huber      = _data_3d["val_roll_huber"]
test_roll_huber     = _data_3d["test_roll_huber"]
val_roll_sn         = _data_3d["val_roll_sn"]
test_roll_sn        = _data_3d["test_roll_sn"]
val_roll_mean       = _data_3d["val_roll_mean"]
test_roll_mean      = _data_3d["test_roll_mean"]

# Residual metrics (strictly validation-derived)
val_sig_res_p50     = _data_3d["sig_res_p50"]
val_sig_res_hub     = _data_3d["sig_res_hub"]
val_sig_res_sn      = _data_3d["sig_res_sn"]
val_sig_res_men     = _data_3d["sig_res_men"]

val_q90_res_p50     = _data_3d["q90_res_p50"]
val_q90_res_hub     = _data_3d["q90_res_hub"]
val_q90_res_sn      = _data_3d["q90_res_sn"]
val_q90_res_men     = _data_3d["q90_res_men"]


# ------------------------------------------------------------------------------
# DISPATCH REGISTRIES (EXPLICIT, STRICT ERROR ON UNKNOWN)
# ------------------------------------------------------------------------------
def _get_fc_quantile(split: str, is_rolling: bool, t: int, k_mult: float):
    mat = val_roll_q_p50 if split == "val" else test_roll_q_p50
    return mat[:, t, :] if is_rolling else mat[:, 0, :]

def _get_fc_huber(split: str, is_rolling: bool, t: int, k_mult: float):
    mat = val_roll_huber if split == "val" else test_roll_huber
    fc = mat[:, t, :] if is_rolling else mat[:, 0, :]
    return fc * k_mult

def _get_fc_seasonal_naive(split: str, is_rolling: bool, t: int, k_mult: float):
    mat = val_roll_sn if split == "val" else test_roll_sn
    return mat[:, t, :] if is_rolling else mat[:, 0, :]

def _get_fc_mean(split: str, is_rolling: bool, t: int, k_mult: float):
    mat = val_roll_mean if split == "val" else test_roll_mean
    return mat[:, t, :] if is_rolling else mat[:, 0, :]

FORECASTER_REGISTRY = {
    "Quantile": _get_fc_quantile,
    "Huber": _get_fc_huber,
    "SeasonalNaive": _get_fc_seasonal_naive,
    "Mean": _get_fc_mean,
}

def _buf_none(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    return np.zeros(N_PAIRS, dtype=np.float64)

def _buf_gaussian_demand(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    sig = val_sigma_vec if split == "val" else test_sigma_vec
    return alpha * 1.282 * sig

def _buf_gaussian_residual(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    if forecaster == "Huber":
        sig = val_sig_res_hub
    elif forecaster == "SeasonalNaive":
        sig = val_sig_res_sn
    elif forecaster == "Mean":
        sig = val_sig_res_men
    elif forecaster == "Quantile":
        sig = val_sig_res_p50
    else:
        raise ValueError(f"Unknown forecaster for gaussian_residual: {forecaster}")
    return alpha * 1.282 * sig

def _buf_empirical_residual(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    if forecaster == "Huber":
        q = val_q90_res_hub
    elif forecaster == "SeasonalNaive":
        q = val_q90_res_sn
    elif forecaster == "Mean":
        q = val_q90_res_men
    elif forecaster == "Quantile":
        q = val_q90_res_p50
    else:
        raise ValueError(f"Unknown forecaster for empirical_residual: {forecaster}")
    return alpha * q

def _buf_raw_quantile(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    p50_mat = val_roll_q_p50 if split == "val" else test_roll_q_p50
    p90_mat = val_roll_q_p90_raw if split == "val" else test_roll_q_p90_raw
    p50_slice = p50_mat[:, t, :] if is_rolling else p50_mat[:, 0, :]
    p90_slice = p90_mat[:, t, :] if is_rolling else p90_mat[:, 0, :]
    return alpha * np.sum(np.maximum(0.0, p90_slice - p50_slice), axis=1)

def _buf_conformal_quantile(forecaster: str, alpha: float, split: str, is_rolling: bool, t: int, override_offsets):
    p50_mat = val_roll_q_p50 if split == "val" else test_roll_q_p50
    p50_slice = p50_mat[:, t, :] if is_rolling else p50_mat[:, 0, :]
    
    if override_offsets is not None:
        off_s, off_h = override_offsets
        off_v = np.where(is_hvol_vec, off_h, off_s)
        raw_mat = val_roll_q_p90_raw if split == "val" else test_roll_q_p90_raw
        raw_slice = raw_mat[:, t, :] if is_rolling else raw_mat[:, 0, :]
        p90_slice = raw_slice + off_v[:, None]
    else:
        cal_mat = val_roll_q_p90_cal if split == "val" else test_roll_q_p90_cal
        p90_slice = cal_mat[:, t, :] if is_rolling else cal_mat[:, 0, :]
        
    return alpha * np.sum(np.maximum(0.0, p90_slice - p50_slice), axis=1)

BUFFER_REGISTRY = {
    "none": _buf_none,
    "gaussian_demand": _buf_gaussian_demand,
    "gaussian_residual": _buf_gaussian_residual,
    "empirical_residual": _buf_empirical_residual,
    "empirical_residual_quantile": _buf_empirical_residual,
    "raw_quantile": _buf_raw_quantile,
    "conformal_quantile": _buf_conformal_quantile,
}


# ------------------------------------------------------------------------------
# CORE VECTORIZED SIMULATOR
# ------------------------------------------------------------------------------
def simulate_inventory(
    forecaster: str,
    buffer_type: str,
    alpha: float,
    beta: float,
    split: str = "val",
    is_rolling: bool = True,
    accounting_mode: str = "arriving_orders",  # "arriving_orders" (primary) or "all_orders"
    k_mult: float = 1.0,
    override_offsets: tuple = None,
    init_mode: str = "primary",  # "primary" (fair) or "legacy"
    score_days: tuple = (7, 15), # (7, 15) Primary (days 7-14), or (0, 15) Robustness
    cp: float = STOCKOUT_PENALTY,
    h_mult: float = 1.0,
    o_mult: float = 1.0,
    return_per_pair: bool = False,
    return_full_sl: bool = False
):
    if forecaster not in FORECASTER_REGISTRY:
        raise ValueError(f"Unknown forecaster: '{forecaster}'. Valid: {list(FORECASTER_REGISTRY.keys())}")
    if buffer_type not in BUFFER_REGISTRY:
        raise ValueError(f"Unknown buffer_type: '{buffer_type}'. Valid: {list(BUFFER_REGISTRY.keys())}")
    if accounting_mode not in ["arriving_orders", "all_orders"]:
        raise ValueError(f"Unknown accounting_mode: '{accounting_mode}'")

    dm = val_demand_mat if split == "val" else test_demand_mat
    dbar = val_dbar_vec if split == "val" else test_dbar_vec

    if init_mode == "legacy":
        mean_d_realized = np.mean(dm, axis=1)
        p50_0 = test_roll_q_p50[:, 0, :]
        p90_0 = test_roll_q_p90_raw[:, 0, :] + np.where(is_hvol_vec, OFFSET_HVOL_LEG, OFFSET_STD_LEG)[:, None]
        ss_leg = alpha * np.sum(np.maximum(0.0, p90_0 - p50_0), axis=1)
        s_leg = np.sum(p50_0, axis=1) + ss_leg
        S_leg = s_leg + (beta * mean_d_realized)
        on_hand = S_leg * 0.70
        pipeline = np.zeros((N_PAIRS, LEAD_TIME_DAYS), dtype=np.float64)
    else:
        # Primary Fair Initialization: prefilled pipeline with dbar_pre, on_hand = 2 * dbar_pre
        pipeline = np.tile(dbar[:, None], (1, LEAD_TIME_DAYS)).copy()
        on_hand = 2.0 * dbar.copy()

    pair_hold          = np.zeros(N_PAIRS, dtype=np.float64)
    pair_stockout      = np.zeros(N_PAIRS, dtype=np.float64)
    pair_order_primary = np.zeros(N_PAIRS, dtype=np.float64)
    pair_order_all     = np.zeros(N_PAIRS, dtype=np.float64)
    pair_sales         = np.zeros(N_PAIRS, dtype=np.float64)
    pair_demand        = np.zeros(N_PAIRS, dtype=np.float64)
    pair_sales_full    = np.zeros(N_PAIRS, dtype=np.float64)
    pair_demand_full   = np.zeros(N_PAIRS, dtype=np.float64)

    eff_h_cost = h_cost_vec * h_mult
    eff_order_fee = FIXED_ORDER_COST * o_mult

    start_t, end_t = score_days

    for t in range(15):
        d_t = dm[:, t]
        arrivals = pipeline[:, 0]
        on_hand += arrivals

        sales = np.minimum(on_hand, d_t)
        stockout = np.maximum(0.0, d_t - on_hand)
        on_hand = np.maximum(0.0, on_hand - d_t)

        inv_pos = on_hand + np.sum(pipeline[:, 1:], axis=1)

        if init_mode == "legacy":
            s_t = s_leg
            S_t = S_leg
        else:
            fc_slice = FORECASTER_REGISTRY[forecaster](split, is_rolling, t, k_mult)
            l_exp_t = np.sum(fc_slice, axis=1)
            ss_t = BUFFER_REGISTRY[buffer_type](forecaster, alpha, split, is_rolling, t, override_offsets)
            s_t = l_exp_t + ss_t
            S_t = s_t + (beta * dbar)

        order_mask = (inv_pos <= s_t)
        order_qty = np.where(order_mask, np.maximum(0.0, S_t - inv_pos), 0.0)

        pipeline[:, :-1] = pipeline[:, 1:]
        pipeline[:, -1] = order_qty

        # W1 Fee rule: orders placed on days 0 to 7 arrive inside days 7 to 14
        # Un-gated by scoring window
        if t <= 7:
            pair_order_primary += np.where(order_mask, eff_order_fee, 0.0)
        pair_order_all += np.where(order_mask, eff_order_fee, 0.0)

        # Track full window sales and demand for sl_full (0-14)
        pair_sales_full  += sales
        pair_demand_full += d_t

        # Scoring Window Accumulation
        if start_t <= t < end_t:
            pair_sales    += sales
            pair_demand   += d_t
            pair_stockout += stockout * cp
            pair_hold     += on_hand * eff_h_cost

    chosen_order = pair_order_primary if accounting_mode == "arriving_orders" else pair_order_all
    pair_cost = pair_hold + pair_stockout + chosen_order

    tot_demand_f = float(np.sum(pair_demand_full))
    tot_sales_f  = float(np.sum(pair_sales_full))
    sl_full = (tot_sales_f / tot_demand_f) * 100.0 if tot_demand_f > 0 else 100.0

    if return_per_pair:
        if return_full_sl:
            return pair_cost, pair_sales, pair_demand, pair_hold, pair_stockout, chosen_order, pair_order_all, pair_sales_full, pair_demand_full
        return pair_cost, pair_sales, pair_demand, pair_hold, pair_stockout, chosen_order, pair_order_all

    tot_demand = float(np.sum(pair_demand))
    tot_sales  = float(np.sum(pair_sales))
    tot_cost   = float(np.sum(pair_cost))
    sl = (tot_sales / tot_demand) * 100.0 if tot_demand > 0 else 100.0

    if return_full_sl:
        return (
            tot_cost,
            sl,
            float(np.sum(pair_hold)),
            float(np.sum(pair_stockout)),
            float(np.sum(chosen_order)),
            float(np.sum(pair_order_all)),
            sl_full
        )

    return (
        tot_cost,
        sl,
        float(np.sum(pair_hold)),
        float(np.sum(pair_stockout)),
        float(np.sum(chosen_order)),
        float(np.sum(pair_order_all))
    )

def get_tuning_fitness(
    alpha: float,
    beta: float,
    forecaster: str,
    buffer_type: str,
    split: str = "val",
    is_rolling: bool = True,
    accounting_mode: str = "arriving_orders",
    k_mult: float = 1.0,
    score_days: tuple = (7, 15),
    cp: float = STOCKOUT_PENALTY,
    h_mult: float = 1.0,
    o_mult: float = 1.0,
    override_offsets: tuple = None,
    sl_eval_mode: str = "primary"
):
    res = simulate_inventory(
        forecaster=forecaster,
        buffer_type=buffer_type,
        alpha=alpha,
        beta=beta,
        split=split,
        is_rolling=is_rolling,
        accounting_mode=accounting_mode,
        k_mult=k_mult,
        override_offsets=override_offsets,
        score_days=score_days,
        cp=cp,
        h_mult=h_mult,
        o_mult=o_mult,
        return_full_sl=True
    )
    tot_cost, sl_prim, _, _, _, _, sl_full = res
    sl = sl_prim if sl_eval_mode == "primary" else sl_full
    pen = 0.0
    if sl < 95.0:
        pen = 50000.0 * ((95.0 - sl) ** 2)
    return tot_cost + pen, tot_cost, sl

