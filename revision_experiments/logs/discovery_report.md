# PHASE 0: DISCOVERY REPORT

**Generated:** 2026-10-12  
**Working Directory:** `C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments`  
**Status:** Read-Only Audit Complete. Awaiting User Approval to Proceed to Phase 1.  

---

## 1. Inventory Policy & Discrete-Event Simulator (Item a)

* **Source File & Location:**
  * File: `C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\MHQF optm.ipynb`
  * Function Names:
    * `simulate_inventory(alpha=1.0, beta=4.0, policy_mode="quantile")` (Cell 1, lines 34–98)
    * `run_benchmark_policy(policy_name, alpha_val, beta_val)` (Cell 4, lines 17–89)

* **Dynamic (s, S) Threshold Calculations:**
  * **Reorder Point ($s_t$):**
    $$s_t = \text{lead\_exp} + ss$$
    * For *Quantile Policy*: $\text{lead\_exp} = \sum_{k=0}^{6} P_{50}[k]$, and safety buffer:
      $$ss = \alpha \cdot \sum_{k=0}^{6} \max(0, P_{90}[k] - P_{50}[k])$$
    * For *Classical Gaussian*: $\text{lead\_exp} = 7 \cdot \bar{D}$, and safety buffer:
      $$ss = \alpha \cdot 1.282 \cdot \left(\sigma_{\text{demand}} \cdot \sqrt{7} + 1.0\right)$$
    * For *Huber TFT + Gaussian*: $\text{lead\_exp} = \sum_{k=0}^{6} \hat{y}_{\text{huber}}[k]$, and safety buffer:
      $$ss = \alpha \cdot 1.282 \cdot \left(\sigma_{\text{residual}} \cdot \sqrt{7} + 1.0\right)$$
  * **Order-Up-To Level ($S_t$):**
    $$S_t = s_t + (\beta \cdot \bar{D})$$
    where $\bar{D} = \max(1.0, \text{mean}(D_{\text{actual}}))$.

* **Role of Decision Variables:**
  * $\alpha$: Multiplier directly scaling the safety stock buffer ($ss$).
  * $\beta$: Multiplier specifying target cycle stock in units of average daily demand (days of demand cover).

* **Estimation of $\sigma_{\text{demand}}$:**
  * In Cell 1 (`simulate_inventory`): Estimated as `np.std(demand) * np.sqrt(7) + 1.0`, where `demand` is the 15-day actual sales vector of the test period.
  * In Cell 4 (`run_benchmark_policy` for Huber): Estimated as `np.std(demand - h_fc) * np.sqrt(7) + 1.0` (standard deviation of forecast residuals over the 15-day test period).

* **Definition of Customer Fill Rate (Service Level):**
  * Defined strictly as **Units-Based Fill Rate (Type II Service Level)**:
    $$\text{Service Level (\%)} = \frac{\sum \text{Customer Sales Fulfilled}}{\sum \text{Customer Demand Realized}} \times 100.0$$
    It is **not** order-based or cycle service level (Type I).

* **Initial Inventory Setting ($I_0 = 0.70 \cdot S$):**
  * Implemented as `on_hand = S * 0.70` on Day 1.
  * *Critical Observation:* Because $S = s + \beta \bar{D}$ and $s$ depends on $\alpha$ and the forecaster, **$I_0$ is policy-dependent and parameter-dependent**. Policies with higher $\alpha$ or $\beta$ begin the simulation with substantially higher initial on-hand stock on Day 1.

* **Costing of Orders Placed Late in the Window:**
  * Supplier delivery lead time is fixed at $L = 7$ business days.
  * Review occurs each evening: if $\text{Inventory Position} \le s$, an order $Q_t = S - \text{IP}$ is appended to the pipeline, and the fixed fee of **$8.00** (`FIXED_ORDER_COST`) is immediately charged.
  * *Boundary Effect:* Orders dispatched on Day 9 through Day 15 ($t \ge 8$ in 0-indexed terms) **never arrive** before the 15-day simulation terminates ($8 + 7 = 15$). The $8.00 dispatch cost is accrued in full, but the units arrive after day 15 and cannot fulfill demand within the evaluation horizon.

---

## 2. Metaheuristic Implementations & Fitness Function (Item b)

* **Source File & Location:**
  * `MHQF optm.ipynb` (Cell 2: lines 1–85 for PSO; Cell 3: lines 1–110 for GWO)
* **Fitness Function:**
  ```python
  def pso_objective(position):
      alpha_val, beta_val = position[0], position[1]
      res = simulate_inventory(alpha=alpha_val, beta=beta_val, policy_mode="quantile")
      cost = res["total_cost"]
      sl = res["service_level"]
      penalty = 0.0
      if sl < 95.0:
          penalty = 50000.0 * ((95.0 - sl) ** 2)
      fitness = cost + penalty
      return fitness, res
  ```
  * Formulation: $\text{Fitness}(\alpha, \beta) = \text{Total Cost} + 50000 \cdot \max(0, 95.0 - \text{Fill Rate})^2$
* **Parameter Search Bounds:**
  * $\alpha \in [0.1, 1.5]$ (`BOUNDS_MIN[0] = 0.1`, `BOUNDS_MAX[0] = 1.5`)
  * $\beta \in [0.5, 5.0]$ (`BOUNDS_MIN[1] = 0.5`, `BOUNDS_MAX[1] = 5.0`)
* **Agent Budget & Parameters:**
  * Swarm / Pack Size: $N = 25$ agents
  * Search Iterations: $T = 20$ iterations
  * Total Objective Evaluations: $25 \times 20 = 500$ evaluations per run
  * PSO Hyperparameters: Inertia $W = 0.72$, Cognitive $C_1 = 1.49$, Social $C_2 = 1.49$
  * GWO Hyperparameters: Pack hierarchy ($\alpha, \beta, \delta, \omega$), exploration parameter $a$ linearly decaying from $2.0 \to 0.0$ over 20 iterations.
  * Random Seed: Fixed at `np.random.seed(42)`.

---

## 3. Inventory of Saved Forecast Files & Checkpoints (Item c)

* **Saved Forecast CSV Files Found Across Repository:**
  1. `Multi-Horizon Quantile Forecasting(process)\tft_quantile_forecasts.csv` (679,056 bytes)
     * Model: Quantile Temporal Fusion Transformer (P10, P50, P90)
     * Split: **Test split only** (3,237 sequences $\times$ 7 days = 22,659 rows)
     * Columns: `sequence_idx`, `forecast_day`, `p10_forecast`, `p50_forecast`, `p90_forecast`, `actual_sales`
     * Units: **Original physical sales units** (inverted via `expm1(StandardScaler.inverse_transform)`).
  2. `Multi-Horizon Quantile Forecasting(process)\tft_test_forecasts.csv` (443,528 bytes)
     * Identical copies in: `MODLE TESTING\`, `Phase_4_Inventory_Optimization\`
     * Model: Champion Single-Point Huber TFT
     * Split: **Test split only** (3,237 rows, wide format)
     * Columns: `store_nbr`, `item_nbr`, `target_start_date`, `target_end_date`, `actual_day_1`..`7`, `pred_day_1`..`7`
     * Units: **Original physical sales units**.
  3. `TFT\optuna tft 10%\tft_test_forecasts.csv` (421,941 bytes)
     * Model: 10% subsample tuning trial; test split only; 22,659 rows.
* **Validation Forecasts Audit:**
  * **DO SAVED VALIDATION FORECAST CSV FILES EXIST?**
    * **NO.** Neither `tft_val_forecasts.csv` nor `tft_quantile_val_forecasts.csv` exists anywhere in the repository. Both existing forecast CSV files represent exclusively the test split (August 1 to August 15, 2017).
* **Saved Checkpoint Files Available for Offline Inference (Rule 1):**
  1. `tft_quantile_champion.pth` (1,106,095 bytes in `Multi-Horizon Quantile Forecasting(process)` and `TFT\Multi-Horizon Quantile Forecasting`)
     * State dict: Full model weights, optimizer, best validation loss ($0.0873$).
  2. `tft_best_champion.pth` (1,093,347 bytes in `MODLE TESTING`, `Phase_4_Inventory_Optimization`, `TFT\opt 100%`)
     * State dict: Full model weights of Champion Huber TFT ($228,503$ parameters).
  3. `patchtst_best_model.pth` (688,430 bytes in `patchtst`)
     * State dict: Full model weights of PatchTST benchmark ($~170,000$ parameters).
* **Sequence Data Containers (`.npz`):**
  * `sequences_validation.npz` (10,669,274 bytes, 3,299 sequences) is fully intact and available in `Multi-Horizon Quantile Forecasting(process)` and `OUTPUT\outputs`.
  * `sequences_test.npz` (10,468,890 bytes, 3,237 sequences) is fully intact.
  * `target_sales_scaler.joblib` (852 bytes) is fully intact.

---

## 4. Conformal Calibration Implementation Audit (Item d)

* **Source File & Location:**
  * `MHQF optm.ipynb` (Cell 0, lines 26–39)
* **Audit Findings:**
  * The calibration code in `MHQF optm.ipynb` **applies hardcoded constants** derived from validation non-conformity scoring:
    ```python
    OFFSET_HIGH = 4.51
    OFFSET_STD  = 0.18
    df_q["p90_calibrated"] = df_q["p90_forecast"] + np.where(df_q["is_high_vol"], OFFSET_HIGH, OFFSET_STD)
    ```
  * **Calibration Split:** Stated as the Validation Split (July 17 to July 31, 2017).
  * **Nonconformity Score Formula:**
    $$R_{i, t} = \max\left(0, y_{i, t} - \hat{y}_{i, t, 0.90}\right)$$
  * **Tier Definitions:**
    * High-Volume Volatile Tier: SKUs with historical training `average_sales >= 100` units/day in `selected_pairs.csv` (52 active pairs in test split).
    * Standard Catalog Tier: SKUs with `average_sales < 100` units/day (311 active pairs in test split).
  * **Units:** Offsets are added directly to `p90_forecast`, which is in **original physical units**. Thus, $\delta_{\text{high}} = +4.51$ units and $\delta_{\text{std}} = +0.18$ units are in physical sales units.
  * **Note on Artifact:** The script that computed the exact quantile non-conformity values (+4.51 and +0.18) on validation was executed interactively; the raw array of validation non-conformity scores was not saved as a standalone CSV.

---

## 5. Ledger Construction & Parameter Audit (Item e)

Verified from `MHQF optm.ipynb` (Cell 4, lines 92–102) and `supply_chain_optimization_results.csv`:

| Ledger Policy Name | Forecasting Engine | Safety Stock Buffer Formula | $\alpha$ Value | $\beta$ Value | Optimizer Run? | Reported Total Cost | Reported Fill Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical Gaussian** | Historical Mean Demand | $ss = \alpha \cdot 1.282 \cdot (\sigma_d \sqrt{7} + 1.0)$ | $1.000$ (Default) | $4.000$ (Default) | **NO** (Untuned) | $142,167.01 | 99.54% |
| **Huber TFT + Gaussian** | Huber Champion TFT Point | $ss = \alpha \cdot 1.282 \cdot (\sigma_{\text{res}} \sqrt{7} + 1.0)$ | $1.000$ (Default) | $4.000$ (Default) | **NO** (Untuned) | $150,496.63 | 98.59% |
| **Quantile Default** | Quantile TFT ($P_{50}$) | $ss = 1.0 \cdot \sum \max(0, P_{90}^{\text{cal}} - P_{50})$ | $1.000$ (Default) | $4.000$ (Default) | **NO** (Untuned) | $204,328.24 | 99.38% |
| **Quantile TFT + Conformal PSO** | Quantile TFT ($P_{50}$) | $ss = \alpha \cdot \sum \max(0, P_{90}^{\text{cal}} - P_{50})$ | $0.100$ (Tuned) | $3.488$ (Tuned) | **YES** (Tuned on Test) | $136,378.10 | 96.41% |

* **Audit Verdict:** The reviewer's critique is completely confirmed:
  * Only the Quantile TFT policy received parameter optimization.
  * The baseline policies were evaluated strictly under arbitrary default parameters ($\alpha=1.0, \beta=4.0$).
  * The optimization was executed directly against test demand.

---

## 6. Analysis of the 363 Active Test Pairs (Item f)

* **Total Selected Cohort:** Exactly **400 pairs** in `selected_pairs.csv`.
* **Pairs in `sequences_test.npz`:** Exactly **363 pairs**.
* **Why 37 Pairs Were Excluded:**
  * **35 pairs** ceased sales activity prior to the test window (e.g., Store 47, Item 617763 last active 2017-05-07; Store 39, Item 813770 last active 2017-04-02; Store 39, Item 897348 last active 2017-07-16). These products were discontinued before August 1, 2017.
  * **2 pairs** were present in `test_split.csv` but had insufficient rows to complete a single 7-day forecast horizon:
    * Store 46, Item 159156: 5 observations (active only Aug 1–Aug 5).
    * Store 6, Item 159156: 3 observations (active only Aug 1–Aug 3).
  * A valid sequence requires 28 lookback days + 7 continuous forecast horizon days. Any pair with fewer than 7 days in the test split cannot form a sequence.
  * Thus, exactly **363 pairs** had active, complete test sequences.

---

## 7. Model Feature Counts & `onpromotion` Temporal Role (Item g)

* **Source File:** `sequence_generation_multivariate_report.json` and `generate_sequences_multivariate(8).py`
* **Input Feature Counts:**
  * **Past Features ($X_{\text{past}}$):** **18 channels** ($T = 28$ lookback steps)
    * `unit_sales_scaled`, `onpromotion`, `promotion_missing`, `was_observed`, `day_of_week_sin/cos` (2), `month_sin/cos` (2), `day_of_month_sin/cos` (2), `week_of_year_sin/cos` (2), `time_idx`, `is_holiday`, `is_event`, `is_additional`, `is_work_day`, `is_transfer`.
  * **Future Known Features ($X_{\text{future}}$):** **14 channels** ($T = 7$ horizon steps)
    * `day_of_week_sin/cos` (2), `month_sin/cos` (2), `day_of_month_sin/cos` (2), `week_of_year_sin/cos` (2), `time_idx`, `is_holiday`, `is_event`, `is_additional`, `is_work_day`, `is_transfer`.
  * **Static Metadata Embeddings ($X_{\text{static}}$):** **9 entity identifiers**
    * `store_nbr` (cardinality 53, dim 16), `item_nbr` (cardinality 122, dim 16), `family` (cardinality 32, dim 8), `class` (cardinality 65, dim 8), `perishable` (cardinality 2, dim 4), `city` (cardinality 22, dim 8), `state` (cardinality 16, dim 8), `type` (cardinality 5, dim 4), `cluster` (cardinality 17, dim 8).
* **Is `onpromotion` a Future (Decoder) Input?**
  * **NO.** In the final multivariate sequence generator, `onpromotion` is strictly present in **$X_{\text{past}}$ (Past Covariate)** and is **absent from $X_{\text{future}}$**. Future inputs consist exclusively of deterministic calendar cycles, trend indices, and public holidays.

---

## 8. Missing Files & Unanswered Questions for Phase 1/2 (Item h)

### Missing Files Identified on Disk:
1. `validation_forecasts.csv`: Neither Huber Champion TFT nor Quantile TFT has saved validation forecasts on disk. Both model checkpoints (`tft_best_champion.pth` and `tft_quantile_champion.pth`) and `sequences_validation.npz` (3,299 sequences) are present on disk. Offline inference with gradients disabled (under Rule 1) will be run in Phase 2 to produce validation forecasts for policy tuning.
2. `conformal_calibration_raw_scores.csv`: The validation non-conformity residuals that produced $\delta = +0.18$ and $+4.51$ were not saved as a standalone table.

### Ambiguities & Methodological Choices to Confirm:
1. **Validation Window Alignment for Policy Tuning (Phase 2):**
   * Validation sequences (`sequences_validation.npz`) cover target dates July 17 to July 31, 2017 ($15$ days) across 367 pairs.
   * For Phase 2 experiments (E1, E2, E3), we will construct the continuous 15-day actual sales vector and forecast vector on validation, exactly mirroring the simulator logic.
2. **Initial Inventory $I_0$ in Fair Mode (E0):**
   * Legacy mode sets $I_0 = 0.70 \cdot S$ (which gives policies with higher order-up-to levels more initial free stock).
   * In Fair Mode, we define $I_0 = 0.70 \cdot (7 \cdot \bar{D}_{\text{pre-window}})$, identical across all policies.

---
**PHASE 0 STATUS: COMPLETE.**
*Awaiting your explicit reply "APPROVED" before proceeding to Phase 1 (Reproduction).*
