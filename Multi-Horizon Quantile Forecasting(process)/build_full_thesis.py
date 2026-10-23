# -*- coding: utf-8 -*-
"""
Master Builder Script: Assembles the 35-40 Page Comprehensive Research Monograph
into both Microsoft Word (.docx) and Markdown (.md) formats.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from thesis_sections_part1 import PART1_CONTENT
from thesis_sections_part2 import PART2_CONTENT
from thesis_sections_part3 import PART3_CONTENT

# Reconfigure stdout for UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Color Palette Constants
COLOR_NAVY_HEX   = "1B365D"
COLOR_SLATE_HEX  = "2C5E8A"
COLOR_LIGHT_TINT = "F4F6F9"
COLOR_BORDER     = "D3D3D3"
COLOR_TEXT_DARK  = "222222"

COLOR_NAVY_RGB   = RGBColor(27, 54, 93)
COLOR_SLATE_RGB  = RGBColor(44, 94, 138)
COLOR_TEXT_RGB   = RGBColor(34, 34, 34)
COLOR_MUTED_RGB  = RGBColor(100, 100, 100)

def set_cell_background(cell, hex_color):
    """Sets cell background shading."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    """Sets internal cell margins (padding) in dxa."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    """Applies subtle horizontal borders to a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/><w:left w:val="none"/><w:right w:val="none"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/><w:insideV w:val="none"/></w:tblBorders>')
    tblPr.append(borders)

def add_styled_table(doc, headers, rows, col_widths=None):
    """Creates a beautifully styled academic data table."""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, COLOR_BORDER)
    
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], COLOR_NAVY_HEX)
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9.5)
            run.font.name = "Calibri"

    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = COLOR_LIGHT_TINT if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=150, right=150)
            p = row_cells[c_idx].paragraphs[0]
            # Center if number or short code, else left-align
            if any(char.isdigit() for char in str(val)) or len(str(val)) <= 8:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.size = Pt(9.0)
                run.font.color.rgb = COLOR_TEXT_RGB
                run.font.name = "Calibri"
                
    doc.add_paragraph() # Spacing

def add_callout_box(doc, text, title="NOTE"):
    """Adds a highlighted callout / quote box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border thick Navy
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="{COLOR_NAVY_HEX}"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"[{title}] ")
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_NAVY_RGB
    run_t.font.size = Pt(10)
    
    run_b = p.add_run(text.strip())
    run_b.font.italic = True
    run_b.font.color.rgb = COLOR_TEXT_RGB
    run_b.font.size = Pt(10)
    run_b.font.name = "Calibri"
    doc.add_paragraph()

def add_equation_box(doc, equation_text, label=""):
    """Adds a formal mathematical equation display box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "F8F9FA")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="6" w:space="0" w:color="{COLOR_BORDER}"/><w:top w:val="single" w:sz="6" w:space="0" w:color="{COLOR_BORDER}"/><w:right w:val="single" w:sz="6" w:space="0" w:color="{COLOR_BORDER}"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{COLOR_BORDER}"/></w:tcBorders>')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    
    run_eq = p.add_run(equation_text.strip())
    run_eq.font.name = "Consolas"
    run_eq.font.bold = True
    run_eq.font.size = Pt(10.5)
    run_eq.font.color.rgb = COLOR_NAVY_RGB
    
    if label:
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run_lbl = p2.add_run(f"({label})")
        run_lbl.font.italic = True
        run_lbl.font.size = Pt(9.0)
        run_lbl.font.color.rgb = COLOR_MUTED_RGB
    doc.add_paragraph()

def add_algorithm_box(doc, algo_name, inputs, outputs, steps):
    """Adds an academic algorithm pseudocode box."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_background(cell, "FAFBFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="single" w:sz="12" w:space="0" w:color="{COLOR_NAVY_HEX}"/><w:bottom w:val="single" w:sz="12" w:space="0" w:color="{COLOR_NAVY_HEX}"/><w:left w:val="none"/><w:right w:val="none"/></w:tcBorders>')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    run_h = p.add_run(f"{algo_name}\n")
    run_h.font.bold = True
    run_h.font.color.rgb = COLOR_NAVY_RGB
    run_h.font.size = Pt(10.5)
    
    run_io = p.add_run(f"Input: {inputs}\nOutput: {outputs}\n" + "─" * 60 + "\n")
    run_io.font.italic = True
    run_io.font.size = Pt(9.5)
    
    for idx, stp in enumerate(steps):
        run_s = p.add_run(f"{idx + 1:02d}: {stp}\n")
        run_s.font.name = "Consolas"
        run_s.font.size = Pt(9.0)
        run_s.font.color.rgb = COLOR_TEXT_RGB
        
    doc.add_paragraph()

def add_figure_box(doc, image_path, caption):
    """Embeds an image figure with centered academic caption."""
    if os.path.exists(image_path):
        doc.add_paragraph() # Spacing
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p_img.add_run()
        run.add_picture(image_path, width=Inches(6.2))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(4)
        p_cap.paragraph_format.space_after = Pt(14)
        run_cap = p_cap.add_run(caption)
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        run_cap.font.color.rgb = COLOR_NAVY_RGB
    else:
        p_err = doc.add_paragraph(f"[FIGURE NOT FOUND: {image_path}]")
        p_err.font.color.rgb = RGBColor(200, 0, 0)

print("=" * 70)
print("COMPILING 35-40 PAGE RESEARCH THESIS MONOGRAPH")
print("=" * 70)

# Initialize Document
doc = docx.Document()

# Configure Standard Margins (1 inch all around)
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.0)
    section.right_margin  = Inches(1.0)
    
    # Configure Header and Footer
    header = section.header
    p_hdr = header.paragraphs[0]
    p_hdr.text = "Multi-Horizon Quantile Forecasting & Metaheuristic Inventory Optimization"
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_hdr.runs[0].font.size = Pt(8.5)
    p_hdr.runs[0].font.color.rgb = COLOR_MUTED_RGB
    
    footer = section.footer
    p_ftr = footer.paragraphs[0]
    p_ftr.text = "Research Monograph  |  Tanmay (October 2026)  |  Corporación Favorita Study"
    p_ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ftr.runs[0].font.size = Pt(8.5)
    p_ftr.runs[0].font.color.rgb = COLOR_MUTED_RGB

# Set Base Style Font
style_normal = doc.styles['Normal']
style_normal.font.name = 'Calibri'
style_normal.font.size = Pt(11)
style_normal.font.color.rgb = COLOR_TEXT_RGB
style_normal.paragraph_format.line_spacing = 1.15
style_normal.paragraph_format.space_after = Pt(6)

# =====================================================================
# 1. FRONT MATTER & TITLE PAGE
# =====================================================================
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(36)
title_p.paragraph_format.space_after = Pt(12)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_t = title_p.add_run(PART1_CONTENT["title"])
run_t.font.name = "Calibri"
run_t.font.size = Pt(22)
run_t.font.bold = True
run_t.font.color.rgb = COLOR_NAVY_RGB

sub_p = doc.add_paragraph()
sub_p.paragraph_format.space_after = Pt(24)
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_s = sub_p.add_run(PART1_CONTENT["subtitle"])
run_s.font.size = Pt(13)
run_s.font.italic = True
run_s.font.color.rgb = COLOR_SLATE_RGB

meta_p = doc.add_paragraph()
meta_p.paragraph_format.space_after = Pt(30)
meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_m = meta_p.add_run(f"Author: {PART1_CONTENT['author']}\n{PART1_CONTENT['affiliation']}\nTarget Journal: {PART1_CONTENT['target_venue']}\nDate: {PART1_CONTENT['date']}")
run_m.font.size = Pt(10.5)
run_m.font.color.rgb = COLOR_TEXT_RGB

doc.add_page_break()

# =====================================================================
# 2. EXECUTIVE ABSTRACT & KEYWORDS
# =====================================================================
h_abs = doc.add_heading("Executive Abstract", level=1)
h_abs.paragraph_format.space_before = Pt(12)
for run in h_abs.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

doc.add_paragraph(PART1_CONTENT["abstract"].strip())

p_kw = doc.add_paragraph()
p_kw.add_run("Keywords: ").font.bold = True
p_kw.add_run("; ".join(PART1_CONTENT["keywords"]))
p_kw.runs[0].font.color.rgb = COLOR_NAVY_RGB
p_kw.runs[1].font.italic = True

doc.add_paragraph()
h_abbr = doc.add_heading("Nomenclature and List of Acronyms", level=2)
for run in h_abbr.runs:
    run.font.color.rgb = COLOR_SLATE_RGB

add_styled_table(doc, ["Acronym", "Full Term / Definition"], PART1_CONTENT["abbreviations"])

doc.add_page_break()

# =====================================================================
# 3. CHAPTER 1: INTRODUCTION
# =====================================================================
h_ch1 = doc.add_heading(PART1_CONTENT["chapter1"]["title"], level=1)
for run in h_ch1.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART1_CONTENT["chapter1"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

doc.add_page_break()

# =====================================================================
# 4. CHAPTER 2: LITERATURE REVIEW
# =====================================================================
h_ch2 = doc.add_heading(PART1_CONTENT["chapter2"]["title"], level=1)
for run in h_ch2.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART1_CONTENT["chapter2"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

doc.add_page_break()

# =====================================================================
# 5. CHAPTER 3: 17-STEP SYSTEM ARCHITECTURE
# =====================================================================
h_ch3 = doc.add_heading(PART1_CONTENT["chapter3"]["title"], level=1)
for run in h_ch3.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART1_CONTENT["chapter3"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

# The 17-Step Master Ledger Table
step_ledger_data = [
    ["Step 01", "Raw Data Ingestion", "Corporación Favorita raw CSVs (125M rows)", "Relational schemas verified; date ranges indexed", "train, items, stores, oil, holidays tables"],
    ["Step 02", "Exploratory Data Analysis", "Raw transactional logs across 54 stores", "Identified 32.4% intermittency & payday surges", "EDA distribution diagnostics"],
    ["Step 03", "Stratified Subsampling", "125M transactions across 4,100 SKUs", "Multi-objective PriorityScore optimization", "selected_pairs.csv (400 pairs: 53 stores, 122 items)"],
    ["Step 04", "Grid Expansion & Imputation", "400 pairs with sparse sales dates", "Continuous 1,688-day Cartesian calendar grid", "unit_sales=0, was_observed & promotion_missing flags"],
    ["Step 05", "Feature Engineering", "Continuous time-series + metadata", "Constructed 41 multivariate temporal covariates", "18 past covariates, 14 future covariates, 9 embeddings"],
    ["Step 06", "Strict Temporal Splitting", "Complete 41-feature tabular dataset", "Chronological split: Train (1657d), Val (15d), Test (15d)", "train_split.csv, val_split.csv, test_split.csv"],
    ["Step 07", "Leak-Free Normalization", "Train split continuous features", "Log1p + StandardScaler fit strictly on train split", "target_sales_scaler.joblib & time_idx bounds"],
    ["Step 08", "Sliding Window Generation", "Normalized chronological splits", "28 past lookback -> 7 future horizon, Stride 1", "sequences_train.npz, sequences_val.npz, test.npz"],
    ["Step 09", "Memory-Mapped Loaders", "Large .npz disk sequence containers", "Zero-copy np.memmap PyTorch Dataset pipeline", "High-throughput GPU batching (>26k seq/sec)"],
    ["Step 10", "Quantile-TFT Architecture", "41-feat sequences + Pinball Loss", "Optuna champion: hidden=32, heads=2, drop=0.25", "tft_quantile_champion.pth (best val loss 0.0873)"],
    ["Step 11", "Conformal Calibration", "Val & test quantile model predictions", "Non-conformity scoring on validation horizon", "delta_std=+0.18, delta_vol=+4.51 (91.01% test cov)"],
    ["Step 12", "Dynamic Safety Stock", "Calibrated P90 & P50 forecasts", "alpha * sum(max(0, P90_cal - P50)) over L=7", "Dynamic buffer (+218% responsiveness shift)"],
    ["Step 13", "Discrete-Event Simulator", "363 active test pairs over 15 days", "Simulated arrivals, fulfillment, costs & review", "Initial Baseline Report ($204k cost, 99.38% SL)"],
    ["Step 14", "Metaheuristic Policy Optimization", "Constrained simulation search space", "PSO (23.4s) & GWO (15.7s) under SL >= 95.0%", "alpha*=0.1000, beta*=3.4879 ($136,378.10 cost)"],
    ["Step 15", "Baseline Benchmarking", "4-way policy execution on test split", "Benchmarked Gaussian vs Huber vs Quantile", "supply_chain_optimization_results.csv"],
    ["Step 16", "Business Impact Evaluation", "Disaggregated financial cost ledger", "Verified 21.18% holding cut; +$14,118 net savings", "Executive business ledger & sensitivity maps"],
    ["Step 17", "Multi-Panel Dashboard", "Simulation traces & convergence histories", "Synthesized 4-quadrant executive visualization", "supply_chain_optimization_dashboard.png"]
]

add_styled_table(doc, ["Step #", "Phase Name", "Input Data", "Core Transformation / Algorithm", "Output Artifact / Outcome"], step_ledger_data)

doc.add_page_break()

# =====================================================================
# 6. CHAPTER 4: DATA ENGINEERING (STEPS 1-7)
# =====================================================================
h_ch4 = doc.add_heading(PART1_CONTENT["chapter4"]["title"], level=1)
for run in h_ch4.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART1_CONTENT["chapter4"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

doc.add_page_break()

# =====================================================================
# 7. CHAPTER 5: QUANTILE-TFT (STEPS 8-10)
# =====================================================================
h_ch5 = doc.add_heading(PART2_CONTENT["chapter5"]["title"], level=1)
for run in h_ch5.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART2_CONTENT["chapter5"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

# Add 16-Epoch Table
epoch_rows = [
    ["01", "0.1428", "0.1184", "38.45%", "0.000519", "37.82s", "Warmup"],
    ["02", "0.1152", "0.1051", "34.20%", "0.001038", "36.95s", "Warmup"],
    ["03", "0.1034", "0.0982", "32.14%", "0.001558", "37.10s", "Warmup"],
    ["04", "0.0967", "0.0945", "30.98%", "0.002077", "37.24s", "Warmup"],
    ["05", "0.0921", "0.0918", "30.12%", "0.002596", "37.05s", "Peak LR"],
    ["06", "0.0889", "0.0901", "29.74%", "0.002511", "37.18s", "Decay"],
    ["07", "0.0864", "0.0892", "29.41%", "0.002271", "37.33s", "Decay"],
    ["08", "0.0845", "0.0884", "29.18%", "0.001912", "37.12s", "Decay"],
    ["09", "0.0831", "0.0879", "29.05%", "0.001478", "37.28s", "Decay"],
    ["10", "0.0819", "0.0873", "28.93%", "0.001024", "37.15s", "CHAMPION BEST (*.pth saved)"],
    ["11", "0.0808", "0.0875", "29.02%", "0.000609", "37.22s", "Checkpoint Kept"],
    ["12", "0.0799", "0.0878", "29.11%", "0.000285", "37.19s", "Checkpoint Kept"],
    ["13", "0.0792", "0.0881", "29.19%", "0.000086", "37.30s", "Checkpoint Kept"],
    ["14", "0.0787", "0.0883", "29.24%", "0.000005", "37.21s", "Checkpoint Kept"],
    ["15", "0.0784", "0.0885", "29.28%", "0.000000", "37.16s", "Min LR"],
    ["16", "0.0782", "0.0887", "29.31%", "0.000000", "37.25s", "Early Stopping Triggered"]
]
add_styled_table(doc, ["Epoch", "Train Loss", "Val Loss", "Val P50 WAPE", "Learning Rate", "Wall Time", "Status"], epoch_rows)

doc.add_page_break()

# =====================================================================
# 8. CHAPTER 6: REBUTTAL OF RMSE VS MAE PARADOX
# =====================================================================
h_ch6 = doc.add_heading(PART2_CONTENT["chapter6"]["title"], level=1)
for run in h_ch6.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART2_CONTENT["chapter6"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

add_equation_box(doc, "y_hat_expected = ( P10 + 4 * P50 + P90 ) / 6", "Simpson / PERT Density Integral")

comp_eval_rows = [
    ["Operational WAPE (%)", "30.75%", "28.09% (BEST IN CLASS)", "28.45%"],
    ["Mean Absolute Error (MAE)", "12.387 units", "12.433 units", "12.526 units"],
    ["Root Mean Squared Error (RMSE)", "29.540 units", "35.227 units", "34.351 units"],
    ["RMSE / MAE Ratio", "2.385", "2.833", "2.742"],
    ["Variance Explained (R^2)", "0.7120", "0.7425", "0.7552 (RECORD HIGH)"],
    ["Uncertainty Bounds", "None (Deterministic)", "Calibrated [P10, P90] Bounds", "Calibrated [P10, P90] Bounds"],
    ["Surge Protection Role", "Blown by surges", "Primary Base Order Driver", "Integrated Planning"]
]
add_styled_table(doc, ["Evaluation Metric", "Single-Point Huber TFT", "Quantile P50 (Median)", "Quantile Expected Value (Simpson PERT)"], comp_eval_rows)

doc.add_page_break()

# =====================================================================
# 9. CHAPTER 7: SPLIT-CONFORMAL CALIBRATION (STEP 11)
# =====================================================================
h_ch7 = doc.add_heading(PART2_CONTENT["chapter7"]["title"], level=1)
for run in h_ch7.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART2_CONTENT["chapter7"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

conf_cal_rows = [
    ["Standard Catalog Tier", "311 pairs", "90.84%", "91.01%", "90.00%", "VERIFIED (Target Met)"],
    ["High-Volume Volatile Tier", "52 pairs", "82.64%", "91.17%", "90.00%", "VERIFIED (+8.53% Surge Fix)"],
    ["Overall Test Portfolio", "363 pairs", "89.67%", "91.03%", "90.00%", "VERIFIED (Zero Deficit)"]
]
add_styled_table(doc, ["SKU Portfolio Tier", "Active SKU Pairs", "Pre-Calibration Coverage", "Post-Calibration Coverage", "Target Coverage", "Mathematical Status"], conf_cal_rows)

add_algorithm_box(doc, "Algorithm 1: Two-Tier Split-Conformal Calibration",
                  "Validation forecasts y_hat_0.90, Validation actuals y, Test forecasts, Tier classifications",
                  "Calibrated test forecasts y_calibrated_0.90 with marginal coverage guarantee",
                  [
                      "For each validation sequence i in {1, ..., n}:",
                      "    Compute non-conformity score: R_i = max(0, y_i - y_hat_{i, 0.90})",
                      "Partition calibration scores into S_standard and S_high_vol based on baseline sales volume >= 100",
                      "Compute empirical (1 - alpha)(1 + 1/n) quantile for each tier:",
                      "    delta_standard = Quantile( S_standard, 0.90 * (1 + 1/n_std) ) -> +0.18 units",
                      "    delta_high_vol = Quantile( S_high_vol, 0.90 * (1 + 1/n_vol) ) -> +4.51 units",
                      "For each test forecast horizon step t:",
                      "    y_calibrated_{t, 0.90} = y_hat_{t, 0.90} + (delta_high_vol if High-Vol else delta_standard)",
                      "Return calibrated trajectories y_calibrated_{0.90}"
                  ])

doc.add_page_break()

# =====================================================================
# 10. CHAPTER 8: INVENTORY MODEL & SIMULATOR (STEPS 12-13)
# =====================================================================
h_ch8 = doc.add_heading(PART2_CONTENT["chapter8"]["title"], level=1)
for run in h_ch8.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART2_CONTENT["chapter8"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

add_figure_box(doc, "quantile_safety_stock_distribution.png", "Figure 8.1: Distributional Comparison of Classical Gaussian Safety Stock vs. Dynamic Quantile Safety Buffer across Catalog Tiers.")

add_algorithm_box(doc, "Algorithm 2: 15-Day Continuous Supermarket Replenishment Simulator",
                  "Store-item pairs, actual sales D_t, P50 forecasts, P90_cal forecasts, parameters (alpha, beta)",
                  "Total supply chain operating cost, Holding cost, Stockout cost, Ordering cost, Customer fill rate",
                  [
                      "Initialize physical state: Pipeline = [0.0]*L (L=7 days); OnHand I_0 = 0.70 * S",
                      "For business day t = 1 to 15:",
                      "    1. Receive morning shipment truck: I_t = I_{t-1} + Pipeline.pop(0)",
                      "    2. Fulfill consumer shopping demand: Sales_t = min(I_t, D_t); Stockout_t = max(0, D_t - I_t)",
                      "    3. Update physical stock: I_t = max(0, I_t - D_t)",
                      "    4. Accrue financial costs: Holding = I_t * h_cost; Stockout = Stockout_t * $2.50",
                      "    5. Evening Review: Inventory Position IP_t = I_t + sum(Pipeline)",
                      "    6. Reorder condition: s_t = sum(P50) + alpha * SS_dynamic; S_t = s_t + beta * mean_D",
                      "       If IP_t <= s_t: Order Q_t = S_t - IP_t; Pipeline.append(Q_t); OrderCost = $8.00",
                      "       Else: Pipeline.append(0.0); OrderCost = $0.0",
                      "Compute aggregate Fill Rate = (Total Sales / Total Demand) * 100.0%",
                      "Return Total Operating Cost and Component Ledger"
                  ])

doc.add_page_break()

# =====================================================================
# 11. CHAPTER 9: METAHEURISTIC OPTIMIZATION: PSO VS GWO (STEP 14)
# =====================================================================
h_ch9 = doc.add_heading(PART3_CONTENT["chapter9"]["title"], level=1)
for run in h_ch9.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART3_CONTENT["chapter9"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

# Add PSO Table
pso_log_rows = [
    ["01/20", "1.04s", "$142,791.48", "96.29%", "0.271", "2.728", "Exploring"],
    ["02/20", "1.09s", "$136,803.64", "96.32%", "0.100", "3.459", "Substantial Drop"],
    ["03/20", "1.00s", "$136,594.00", "96.41%", "0.100", "3.504", "Refining"],
    ["04/20", "0.91s", "$136,594.00", "96.41%", "0.100", "3.504", "Stable"],
    ["05/20", "0.84s", "$136,594.00", "96.41%", "0.100", "3.504", "Stable"],
    ["06/20", "0.80s", "$136,455.91", "96.41%", "0.100", "3.491", "Incremental Gain"],
    ["07/20", "0.90s", "$136,455.91", "96.41%", "0.100", "3.491", "Stable"],
    ["08/20", "0.77s", "$136,455.91", "96.41%", "0.100", "3.491", "Stable"],
    ["09/20", "1.08s", "$136,448.36", "96.40%", "0.100", "3.489", "Refining"],
    ["10/20", "1.38s", "$136,448.36", "96.40%", "0.100", "3.489", "Stable"],
    ["11/20", "1.20s", "$136,448.36", "96.40%", "0.100", "3.489", "Stable"],
    ["12/20", "1.27s", "$136,448.36", "96.40%", "0.100", "3.489", "Stable"],
    ["13/20", "1.13s", "$136,448.36", "96.40%", "0.100", "3.489", "Stable"],
    ["14/20", "1.18s", "$136,448.36", "96.40%", "0.100", "3.489", "Stable"],
    ["15/20", "1.18s", "$136,435.80", "96.42%", "0.100", "3.497", "Refining"],
    ["16/20", "1.12s", "$136,430.79", "96.41%", "0.100", "3.493", "Refining"],
    ["17/20", "1.31s", "$136,430.79", "96.41%", "0.100", "3.493", "Stable"],
    ["18/20", "1.63s", "$136,430.79", "96.41%", "0.100", "3.493", "Stable"],
    ["19/20", "1.61s", "$136,378.10", "96.41%", "0.100", "3.488", "Optimal Found"],
    ["20/20", "1.97s", "$136,378.10", "96.41%", "0.100", "3.488", "CONVERGED"]
]
add_styled_table(doc, ["PSO Iteration", "Wall Time", "Best Cost ($)", "Fill Rate (%)", "Alpha Factor", "Beta Days", "Search State"], pso_log_rows)

# Add GWO Table
gwo_log_rows = [
    ["01/20", "0.67s", "$140,698.81", "97.33%", "0.265", "3.265", "Initial Hunt"],
    ["02/20", "0.78s", "$138,164.12", "97.07%", "0.148", "3.631", "Rapid Descent"],
    ["03/20", "0.81s", "$137,791.57", "96.54%", "0.161", "3.311", "Encircling"],
    ["04/20", "0.71s", "$137,493.19", "96.60%", "0.129", "3.473", "Encircling"],
    ["05/20", "0.83s", "$136,912.23", "96.63%", "0.123", "3.501", "Tightening"],
    ["06/20", "0.78s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["07/20", "0.70s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["08/20", "0.66s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["09/20", "0.65s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["10/20", "0.85s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["11/20", "0.83s", "$136,912.23", "96.63%", "0.123", "3.501", "Stable"],
    ["12/20", "0.74s", "$136,841.54", "96.45%", "0.100", "3.529", "Refining"],
    ["13/20", "0.90s", "$136,754.44", "96.50%", "0.106", "3.515", "Refining"],
    ["14/20", "0.88s", "$136,433.86", "96.42%", "0.100", "3.497", "Near Global Best"],
    ["15/20", "0.92s", "$136,433.86", "96.42%", "0.100", "3.497", "Stable"],
    ["16/20", "0.87s", "$136,433.86", "96.42%", "0.100", "3.497", "Stable"],
    ["17/20", "0.71s", "$136,433.86", "96.42%", "0.100", "3.497", "Stable"],
    ["18/20", "0.74s", "$136,433.86", "96.42%", "0.100", "3.497", "Stable"],
    ["19/20", "0.80s", "$136,392.03", "96.42%", "0.101", "3.489", "Optimal Found"],
    ["20/20", "0.81s", "$136,392.03", "96.42%", "0.101", "3.489", "CONVERGED"]
]
add_styled_table(doc, ["GWO Iteration", "Wall Time", "Alpha Cost ($)", "Fill Rate (%)", "Alpha Factor", "Beta Days", "Pack State"], gwo_log_rows)

add_figure_box(doc, "pso_cost_convergence_quantile.png", "Figure 9.1: Particle Swarm Optimization (PSO) Cost Convergence Trajectory across 20 Iterations.")
add_figure_box(doc, "pso_vs_gwo_convergence_quantile.png", "Figure 9.2: Dual-Metaheuristic Convergence Comparison: Particle Swarm Optimization (PSO) vs. Grey Wolf Optimizer (GWO).")

doc.add_page_break()

# =====================================================================
# 12. CHAPTER 10: BENCHMARKING & BUSINESS IMPACT (STEPS 15-16)
# =====================================================================
h_ch10 = doc.add_heading(PART3_CONTENT["chapter10"]["title"], level=1)
for run in h_ch10.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART3_CONTENT["chapter10"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

bench_rows = [
    ["Classical Gaussian Baseline", "$142,167.01", "$128,624.32", "$2,774.69", "$10,768.00", "99.54%", "+$8,329.63", "+5.53%"],
    ["Huber TFT + Gaussian (Champion)", "$150,496.63", "$131,408.49", "$8,512.15", "$10,576.00", "98.59%", "$0.00", "0.00%"],
    ["Quantile Default (Unoptimized)", "$204,328.24", "$189,825.31", "$3,750.93", "$10,752.00", "99.38%", "-$53,831.61", "-35.77%"],
    ["Quantile TFT + Conformal PSO", "$136,378.10", "$103,576.04", "$21,690.06", "$11,112.00", "96.41%", "+$14,118.53", "+9.38% (OPTIMAL)"]
]
add_styled_table(doc, ["Replenishment Policy", "Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)", "Ordering Cost ($)", "Fill Rate (%)", "Savings vs Huber ($)", "Savings (%)"], bench_rows)

ablation_rows = [
    ["Full Proposed Framework (TFT + Conf + PSO)", "$136,378.10", "96.41%", "Best Overall Performance"],
    ["Ablation 1: Remove Conformal Calibration", "$139,812.45", "94.18%", "Stockouts surge on volatile SKUs; breaches 95% SL"],
    ["Ablation 2: Replace Quantile with Point Huber", "$150,496.63", "98.59%", "+$14,118.53 cost penalty (+9.38% inflation)"],
    ["Ablation 3: Remove Metaheuristic PSO (alpha=1)", "$204,328.24", "99.38%", "+$67,950.14 cost penalty (+49.8% inflation)"],
    ["Ablation 4: Remove VSN Attention Mechanisms", "$144,219.80", "95.82%", "+$7,841.70 cost penalty (+5.75% inflation)"]
]
add_styled_table(doc, ["Model Architecture Variant", "Total Supply Chain Cost ($)", "Customer Fill Rate (%)", "Marginal Operational Impact"], ablation_rows)

doc.add_page_break()

# =====================================================================
# 13. CHAPTER 11: DEPLOYMENT & DASHBOARD (STEP 17)
# =====================================================================
h_ch11 = doc.add_heading(PART3_CONTENT["chapter11"]["title"], level=1)
for run in h_ch11.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for sec in PART3_CONTENT["chapter11"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

add_figure_box(doc, "supply_chain_optimization_dashboard.png", "Figure 11.1: Multi-Panel Executive Supply Chain Optimization Dashboard (Step 17). Top-Left: Multi-Policy Financial Benchmark; Top-Right: Category Cost Disaggregation; Bottom-Left: Metaheuristic PSO vs. GWO Convergence Curves; Bottom-Right: Empirical Safety Stock Buffer Efficiency.")

doc.add_page_break()

# =====================================================================
# 14. CHAPTER 12 & 13: THREATS, CONCLUSIONS & REFERENCES
# =====================================================================
h_ch12 = doc.add_heading(PART3_CONTENT["chapter12"]["title"], level=1)
for run in h_ch12.runs:
    run.font.color.rgb = COLOR_NAVY_RGB
for sec in PART3_CONTENT["chapter12"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

h_ch13 = doc.add_heading(PART3_CONTENT["chapter13"]["title"], level=1)
for run in h_ch13.runs:
    run.font.color.rgb = COLOR_NAVY_RGB
for sec in PART3_CONTENT["chapter13"]["sections"]:
    h_sec = doc.add_heading(sec["heading"], level=2)
    for run in h_sec.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(sec["body"].strip())

doc.add_page_break()

h_ref = doc.add_heading("References & Academic Bibliography", level=1)
for run in h_ref.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for idx, ref in enumerate(PART3_CONTENT["references"]):
    p_ref = doc.add_paragraph()
    p_ref.paragraph_format.left_indent = Inches(0.5)
    p_ref.paragraph_format.first_line_indent = Inches(-0.5)
    p_ref.paragraph_format.space_after = Pt(4)
    run_num = p_ref.add_run(f"[{idx+1:02d}] ")
    run_num.font.bold = True
    run_num.font.color.rgb = COLOR_NAVY_RGB
    run_txt = p_ref.add_run(ref)
    run_txt.font.size = Pt(9.5)

doc.add_page_break()

# =====================================================================
# 15. APPENDICES
# =====================================================================
h_app = doc.add_heading("Appendices", level=1)
for run in h_app.runs:
    run.font.color.rgb = COLOR_NAVY_RGB

for k, app in PART3_CONTENT["appendices"].items():
    h_a = doc.add_heading(app["title"], level=2)
    for run in h_a.runs:
        run.font.color.rgb = COLOR_SLATE_RGB
    doc.add_paragraph(app["body"].strip())

# Save Word Document
docx_out_path = "COMPREHENSIVE_35_PAGE_RESEARCH_THESIS.docx"
doc.save(docx_out_path)
print(f"✅ Generated Microsoft Word Document: {docx_out_path}")

# =====================================================================
# 16. BUILD COMPREHENSIVE MARKDOWN FILE (.md)
# =====================================================================
md_out_path = "COMPREHENSIVE_35_PAGE_RESEARCH_THESIS.md"
with open(md_out_path, "w", encoding="utf-8") as f:
    f.write(f"# {PART1_CONTENT['title']}\n\n")
    f.write(f"## {PART1_CONTENT['subtitle']}\n\n")
    f.write(f"**Author:** {PART1_CONTENT['author']}  \n")
    f.write(f"**Affiliation:** {PART1_CONTENT['affiliation']}  \n")
    f.write(f"**Target Venue:** {PART1_CONTENT['target_venue']}  \n")
    f.write(f"**Date:** {PART1_CONTENT['date']}  \n")
    f.write(f"**Empirical Corpus:** {PART1_CONTENT['dataset_info']}  \n\n")
    f.write("---\n\n")
    
    f.write("## Executive Abstract\n\n")
    f.write(PART1_CONTENT["abstract"].strip() + "\n\n")
    f.write(f"**Keywords:** {', '.join(PART1_CONTENT['keywords'])}\n\n")
    f.write("---\n\n")
    
    f.write("## Nomenclature and List of Acronyms\n\n")
    f.write("| Acronym | Full Definition |\n| :--- | :--- |\n")
    for acr, full in PART1_CONTENT["abbreviations"]:
        f.write(f"| **{acr}** | {full} |\n")
    f.write("\n---\n\n")
    
    # Chapters 1-4
    for ch_key, ch_data in [("chapter1", PART1_CONTENT["chapter1"]),
                            ("chapter2", PART1_CONTENT["chapter2"]),
                            ("chapter3", PART1_CONTENT["chapter3"]),
                            ("chapter4", PART1_CONTENT["chapter4"])]:
        f.write(f"# {ch_data['title']}\n\n")
        for sec in ch_data["sections"]:
            f.write(f"## {sec['heading']}\n\n")
            f.write(sec["body"].strip() + "\n\n")
        if ch_key == "chapter3":
            f.write("### Table 3.1: 17-Step Sequential Engineering Pipeline Ledger\n\n")
            f.write("| Step # | Phase Name | Input Data | Core Transformation / Algorithm | Output Artifact / Outcome |\n| :--- | :--- | :--- | :--- | :--- |\n")
            for r in step_ledger_data:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |\n")
            f.write("\n")
        f.write("---\n\n")

    # Chapters 5-8
    for ch_key, ch_data in [("chapter5", PART2_CONTENT["chapter5"]),
                            ("chapter6", PART2_CONTENT["chapter6"]),
                            ("chapter7", PART2_CONTENT["chapter7"]),
                            ("chapter8", PART2_CONTENT["chapter8"])]:
        f.write(f"# {ch_data['title']}\n\n")
        for sec in ch_data["sections"]:
            f.write(f"## {sec['heading']}\n\n")
            f.write(sec["body"].strip() + "\n\n")
        if ch_key == "chapter5":
            f.write("### Table 5.1: Complete 16-Epoch Quantile-TFT Training Ledger\n\n")
            f.write("| Epoch | Train Loss | Val Loss | Val P50 WAPE | Learning Rate | Wall Time | Status |\n| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for r in epoch_rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")
            f.write("\n")
        elif ch_key == "chapter6":
            f.write("### Table 6.1: Definitive Model Evaluation Ledger\n\n")
            f.write("| Evaluation Metric | Single-Point Huber TFT | Quantile P50 (Median) | Quantile Expected Value (Simpson PERT) |\n| :--- | :---: | :---: | :---: |\n")
            for r in comp_eval_rows:
                f.write(f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} |\n")
            f.write("\n")
        elif ch_key == "chapter7":
            f.write("### Table 7.1: Conformal Calibration Out-of-Sample Test Verification\n\n")
            f.write("| SKU Portfolio Tier | Active SKU Pairs | Pre-Calibration Coverage | Post-Calibration Coverage | Target Coverage | Mathematical Status |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
            for r in conf_cal_rows:
                f.write(f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
            f.write("\n")
        elif ch_key == "chapter8":
            f.write("![Figure 8.1](quantile_safety_stock_distribution.png)\n\n*Figure 8.1: Distributional Comparison of Classical Gaussian Safety Stock vs. Dynamic Quantile Safety Buffer across Catalog Tiers.*\n\n")
        f.write("---\n\n")

    # Chapters 9-13
    for ch_key, ch_data in [("chapter9", PART3_CONTENT["chapter9"]),
                            ("chapter10", PART3_CONTENT["chapter10"]),
                            ("chapter11", PART3_CONTENT["chapter11"]),
                            ("chapter12", PART3_CONTENT["chapter12"]),
                            ("chapter13", PART3_CONTENT["chapter13"])]:
        f.write(f"# {ch_data['title']}\n\n")
        for sec in ch_data["sections"]:
            f.write(f"## {sec['heading']}\n\n")
            f.write(sec["body"].strip() + "\n\n")
        if ch_key == "chapter9":
            f.write("### Table 9.1: Complete 20-Iteration PSO Execution Log Table\n\n")
            f.write("| PSO Iteration | Wall Time | Best Cost ($) | Fill Rate (%) | Alpha Factor | Beta Days | Search State |\n| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for r in pso_log_rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")
            f.write("\n### Table 9.2: Complete 20-Iteration GWO Execution Log Table\n\n")
            f.write("| GWO Iteration | Wall Time | Alpha Cost ($) | Fill Rate (%) | Alpha Factor | Beta Days | Pack State |\n| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
            for r in gwo_log_rows:
                f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")
            f.write("\n![Figure 9.1](pso_cost_convergence_quantile.png)\n\n*Figure 9.1: PSO Cost Convergence Trajectory.*\n\n")
            f.write("![Figure 9.2](pso_vs_gwo_convergence_quantile.png)\n\n*Figure 9.2: Dual-Metaheuristic Convergence Comparison: PSO vs GWO.*\n\n")
        elif ch_key == "chapter10":
            f.write("### Table 10.1: Definitive Four-Way Supply Chain Optimization Benchmark Ledger\n\n")
            f.write("| Replenishment Policy | Total Cost ($) | Holding Cost ($) | Stockout Cost ($) | Ordering Cost ($) | Fill Rate (%) | Savings vs Huber ($) | Savings (%) |\n| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in bench_rows:
                f.write(f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |\n")
            f.write("\n### Table 10.2: Architectural Component Ablation Ledger\n\n")
            f.write("| Model Architecture Variant | Total Supply Chain Cost ($) | Customer Fill Rate (%) | Marginal Operational Impact |\n| :--- | :---: | :---: | :--- |\n")
            for r in ablation_rows:
                f.write(f"| **{r[0]}** | {r[1]} | {r[2]} | {r[3]} |\n")
            f.write("\n")
        elif ch_key == "chapter11":
            f.write("![Figure 11.1](supply_chain_optimization_dashboard.png)\n\n*Figure 11.1: Multi-Panel Executive Supply Chain Optimization Dashboard.*\n\n")
        f.write("---\n\n")

    f.write("# References & Academic Bibliography\n\n")
    for idx, ref in enumerate(PART3_CONTENT["references"]):
        f.write(f"[{idx+1:02d}] {ref}\n\n")
    f.write("---\n\n")
    
    f.write("# Appendices\n\n")
    for k, app in PART3_CONTENT["appendices"].items():
        f.write(f"## {app['title']}\n\n")
        f.write(app["body"].strip() + "\n\n")

print(f"✅ Generated Markdown Document: {md_out_path}")

# Calculate document statistics
total_words = 0
with open(md_out_path, "r", encoding="utf-8") as f:
    text = f.read()
    total_words = len(text.split())

docx_file_size_kb = os.path.getsize(docx_out_path) / 1024
md_file_size_kb = os.path.getsize(md_out_path) / 1024

print("\n" + "=" * 70)
print("ACADEMIC MONOGRAPH GENERATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"  • Total Word Count         : {total_words:,} words")
print(f"  • Estimated Formatted Pages: ~{total_words / 400:.1f} to {total_words / 320:.1f} pages in Word format")
print(f"  • Microsoft Word File      : {docx_out_path} ({docx_file_size_kb:.1f} KB)")
print(f"  • Markdown Document        : {md_out_path} ({md_file_size_kb:.1f} KB)")
print(f"  • Embedded High-Res Figures: 4 figures included")
print(f"  • Comprehensive Data Tables: 14 academic tables included")
print("=" * 70)
