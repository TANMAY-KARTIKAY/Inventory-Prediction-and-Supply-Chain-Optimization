# -*- coding: utf-8 -*-
"""
generate_clean_manuscript.py
Complete script to generate manuscript.docx and manuscript.md.
Eliminates all raw LaTeX code from running text.
Inserts prominent Figure and Table callout boxes.
Embeds high-resolution images directly from disk.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)"
PAPER_DIR = os.path.join(BASE_DIR, "paper")
MD_PATH = os.path.join(PAPER_DIR, "manuscript.md")
DOCX_PATH = os.path.join(PAPER_DIR, "manuscript.docx")

# Available image paths
IMG_SYSARCHI = os.path.join(BASE_DIR, "sysarchi.png")
IMG_SAFETY_STOCK = os.path.join(BASE_DIR, r"Multi-Horizon Quantile Forecasting(process)\quantile_safety_stock_distribution.png")
IMG_FITNESS_SURFACE = os.path.join(BASE_DIR, r"revision_experiments\outputs\figures\e2_fitness_surface_validation.png")
IMG_PARETO_FRONTIER = os.path.join(BASE_DIR, r"revision_experiments\outputs\figures\r4_matched_fill_frontier.png")
IMG_CONVERGENCE = os.path.join(BASE_DIR, r"Multi-Horizon Quantile Forecasting(process)\pso_vs_gwo_convergence_quantile.png")
IMG_DASHBOARD = os.path.join(BASE_DIR, r"Phase_4_Inventory_Optimization\supply_chain_optimization_dashboard.png")

# Colors
NAVY_HEX = "1B365D"
SLATE_HEX = "2C5E8A"
BOX_BG_HEX = "F0F4F8"
TBL_BG_HEX = "F8FAFC"
BORDER_HEX = "D1D5DB"

NAVY_RGB = RGBColor(27, 54, 93)
SLATE_RGB = RGBColor(44, 94, 138)
TEXT_RGB = RGBColor(34, 34, 34)
MUTED_RGB = RGBColor(90, 90, 90)

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/><w:left w:val="none"/><w:right w:val="none"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/><w:insideV w:val="none"/></w:tblBorders>')
    tblPr.append(borders)

def add_callout_box(doc, box_title, items, border_color=SLATE_HEX, bg_color=BOX_BG_HEX):
    """Creates a clean academic indicator/callout box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/><w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/><w:right w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/></w:tcBorders>')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(box_title)
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(10)
    r_title.font.bold = True
    r_title.font.color.rgb = NAVY_RGB
    
    for label, val in items:
        p2 = cell.add_paragraph()
        p2.paragraph_format.space_before = Pt(1)
        p2.paragraph_format.space_after = Pt(2)
        r_lbl = p2.add_run(label + ": ")
        r_lbl.font.name = "Calibri"
        r_lbl.font.size = Pt(9)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = SLATE_RGB
        r_val = p2.add_run(val)
        r_val.font.name = "Calibri"
        r_val.font.size = Pt(9)
        r_val.font.color.rgb = TEXT_RGB
        
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(4)

def add_academic_table(doc, headers, rows):
    """Renders a fully styled, publication-grade data table."""
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, BORDER_HEX)
    
    hdr_row = tbl.rows[0]
    for c_idx, h_text in enumerate(headers):
        c = hdr_row.cells[c_idx]
        set_cell_background(c, NAVY_HEX)
        set_cell_margins(c, top=90, bottom=90, left=110, right=110)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        r.font.name = "Calibri"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    for r_idx, r_cells in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        bg = TBL_BG_HEX if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell_text in enumerate(r_cells):
            c = row.cells[c_idx]
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=110, right=110)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(cell_text)
            r.font.name = "Calibri"
            r.font.size = Pt(8)
            r.font.color.rgb = TEXT_RGB
            
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(6)

def add_styled_heading(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.name = "Georgia"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = NAVY_RGB
    elif level == 2:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = SLATE_RGB
    elif level == 3:
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = TEXT_RGB

def add_body_paragraph(doc, text, bold_prefix=None, indent=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    if indent:
        p.paragraph_format.left_indent = Inches(0.25)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = NAVY_RGB
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = TEXT_RGB

def add_equation_box(doc, eq_text, eq_num=None):
    """Adds a clean, centered mathematical formula block."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(5)
    r_eq = p.add_run(eq_text)
    r_eq.font.name = "Calibri"
    r_eq.font.size = Pt(9.5)
    r_eq.font.bold = True
    r_eq.font.color.rgb = NAVY_RGB
    if eq_num:
        r_num = p.add_run(f"    ({eq_num})")
        r_num.font.name = "Calibri"
        r_num.font.size = Pt(9)
        r_num.font.italic = True
        r_num.font.color.rgb = MUTED_RGB

print("Generator core initialized.")
