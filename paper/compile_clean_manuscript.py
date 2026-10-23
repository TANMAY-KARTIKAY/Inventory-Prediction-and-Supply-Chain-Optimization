# -*- coding: utf-8 -*-
"""
Master Manuscript Compiler for Clean Research Paper
Assembles all clean sections into:
1. manuscript.docx (Publication-grade Word Document with embedded images, clean tables, and placeholder callout boxes)
2. manuscript.md (Clean GitHub-Flavored Markdown manuscript free of raw LaTeX clutter)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)\paper"
sys.path.append(os.path.join(BASE_DIR, "clean_sections"))

import sec1_intro
import sec2_lit
import sec3_problem
import sec4_methodology
import sec5_safeguards
import sec6_results
import sec7_discussion
import sec8_limitations_conclusion

# Professional Typography Colors
COLOR_PRIMARY = RGBColor(27, 54, 93)       # #1B365D Deep Navy
COLOR_SECONDARY = RGBColor(43, 84, 126)    # #2B547E Steel Blue
COLOR_DARK_TEXT = RGBColor(34, 34, 34)     # #222222 Body Charcoal
COLOR_MUTED_TEXT = RGBColor(100, 100, 100) # #646464 Metadata Gray

HEX_PRIMARY = "1B365D"
HEX_LIGHT_BG = "F4F7FA"
HEX_BORDER = "CCCCCC"
HEX_ZEBRA = "F9FAFB"
HEX_WHITE = "FFFFFF"

def sanitize_text(s):
    """Remove control characters and invalid XML characters."""
    if not isinstance(s, str):
        s = str(s)
    return "".join(c for c in s if ord(c) >= 32 or c in "\t\n\r")

def unpack_item(item):
    """Unpacks a tuple/list into (text, style, eq_label)."""
    if len(item) == 3:
        return item[0], item[1], item[2]
    elif len(item) == 2:
        return item[0], item[1], None
    return str(item), "p", None

def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    """Sets internal padding (in dxa) for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_cell_border_left_thick(cell, color_hex="1B365D", sz="28"):
    """Adds a thick left accent border to a callout cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

def set_table_row_bottom_border(cell, color_hex="CCCCCC", sz="4"):
    """Adds light bottom border to standard table cells."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

def add_callout_box(doc, box_title, items, is_table=True):
    """Creates an academic placeholder / indicator callout box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    
    set_cell_background(cell, HEX_LIGHT_BG)
    set_cell_border_left_thick(cell, HEX_PRIMARY, sz="28")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run_icon = p.add_run("📊 " if is_table else "🖼️ ")
    run_icon.font.name = "Segoe UI Emoji"
    run_icon.font.size = Pt(11)
    
    clean_title = box_title.replace("PLACEHOLDER & SPECIFICATION: ", "").replace("PLACEHOLDER & SPECIFICATION", "").strip()
    run_title = p.add_run(sanitize_text(clean_title.upper()))
    run_title.font.name = "Arial"
    run_title.font.size = Pt(10)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    
    for label, val in items:
        # Skip internal dev filepaths in presentation callouts
        if label in ["Target Image Filename", "Image Location"]:
            continue
        p_item = cell.add_paragraph()
        p_item.paragraph_format.space_before = Pt(1)
        p_item.paragraph_format.space_after = Pt(2)
        p_item.paragraph_format.line_spacing = 1.15
        
        r_lbl = p_item.add_run(f"• {sanitize_text(label)}: ")
        r_lbl.font.name = "Calibri"
        r_lbl.font.size = Pt(9.5)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = COLOR_DARK_TEXT
        
        r_val = p_item.add_run(sanitize_text(val))
        r_val.font.name = "Calibri"
        r_val.font.size = Pt(9.5)
        r_val.font.color.rgb = COLOR_DARK_TEXT
        
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def add_academic_table(doc, headers, rows, title=None):
    """Creates a publication-grade academic table in docx."""
    col_count = len(headers)
    tbl = doc.add_table(rows=len(rows) + 1, cols=col_count)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = True
    
    # Header Row
    hdr_cells = tbl.rows[0].cells
    for i, title_text in enumerate(headers):
        cell = hdr_cells[i]
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(sanitize_text(title_text))
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    # Repeat header row & cantSplit
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    
    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        trPr_row = row._tr.get_or_add_trPr()
        trPr_row.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        bg_hex = HEX_ZEBRA if (r_idx % 2 == 1) else HEX_WHITE
        
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg_hex)
            set_table_row_bottom_border(cell, HEX_BORDER, sz="4")
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            
            p = cell.paragraphs[0]
            val_str = sanitize_text(str(val))
            
            if any(val_str.startswith(char) for char in ["$", "-", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "["]) and not val_str.startswith("Rank"):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            
            r = p.add_run(val_str)
            r.font.name = "Calibri"
            r.font.size = Pt(9.0)
            r.font.color.rgb = COLOR_DARK_TEXT
            if "Rank 1" in val_str or "Dominates" in val_str or "Champion" in val_str:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
                
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_figure_with_image(doc, fig_spec, img_path):
    """Renders the figure indicator box, embeds the verified PNG image, and adds caption."""
    add_callout_box(doc, fig_spec["box_title"], fig_spec["items"], is_table=False)
    
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(6)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(6.0))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(12)
        p_cap.paragraph_format.left_indent = Inches(0.4)
        p_cap.paragraph_format.right_indent = Inches(0.4)
        
        caption_text = dict(fig_spec["items"]).get("Caption", "")
        r_cap = p_cap.add_run(sanitize_text(caption_text))
        r_cap.font.name = "Calibri"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_DARK_TEXT
    else:
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.0)
        set_cell_background(cell, "FFF3CD")
        set_cell_margins(cell, top=200, bottom=200, left=200, right=200)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[FIGURE IMAGE PLACEHOLDER: Insert image {os.path.basename(img_path)} here]")
        r.font.name = "Arial"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = RGBColor(133, 100, 4)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_paragraph_formatted(doc, text, style_type, eq_label=None):
    """Formats headings, body paragraphs, bullet points, and equations."""
    p = doc.add_paragraph()
    text = sanitize_text(text)
    
    if style_type == "title":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(20)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        
    elif style_type == "author":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(14)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.color.rgb = COLOR_MUTED_TEXT
        
    elif style_type == "abstract_h":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        
    elif style_type == "abstract":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.right_indent = Inches(0.4)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.color.rgb = COLOR_DARK_TEXT
        
    elif style_type == "keywords":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.right_indent = Inches(0.4)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(18)
        r_lbl = p.add_run("Keywords: ")
        r_lbl.font.name = "Times New Roman"
        r_lbl.font.size = Pt(9.5)
        r_lbl.font.bold = True
        r_lbl.font.color.rgb = COLOR_PRIMARY
        r_val = p.add_run(text)
        r_val.font.name = "Times New Roman"
        r_val.font.size = Pt(9.5)
        r_val.font.italic = True
        r_val.font.color.rgb = COLOR_DARK_TEXT
        
    elif style_type == "h1":
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        
    elif style_type == "h2":
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = COLOR_SECONDARY
        
    elif style_type == "h3":
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = COLOR_DARK_TEXT
        
    elif style_type == "p":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.color.rgb = COLOR_DARK_TEXT
        
    elif style_type == "bullet":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(f"•  {text}")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_DARK_TEXT
        
    elif style_type == "eq":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.right_indent = Inches(0.5)
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(5)
        eq_text = f"{text}       ({eq_label})" if eq_label else text
        r = p.add_run(eq_text)
        r.font.name = "Cambria Math"
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        
    elif style_type == "ref":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.color.rgb = COLOR_DARK_TEXT

def render_paragraphs_docx(doc, p_list):
    """Renders a list of paragraph items into docx."""
    for item in p_list:
        text, style, eq_label = unpack_item(item)
        add_paragraph_formatted(doc, text, style, eq_label)

def build_docx(target_docx):
    """Builds complete publication manuscript.docx."""
    print("Initializing Document...")
    doc = docx.Document()
    
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    img_sysarchi = os.path.join(BASE_DIR, "sysarchi.png")
    img_fig2 = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\quantile_safety_stock_distribution.png"
    img_fig3 = r"C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments\outputs\figures\e2_fitness_surface_validation.png"
    img_fig4 = r"C:\Users\tanma\OneDrive\Desktop\project(1)\revision_experiments\outputs\figures\r4_matched_fill_frontier.png"
    img_fig5 = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Multi-Horizon Quantile Forecasting(process)\pso_vs_gwo_convergence_quantile.png"
    img_fig_d1 = r"C:\Users\tanma\OneDrive\Desktop\project(1)\Phase_4_Inventory_Optimization\supply_chain_optimization_dashboard.png"

    # FRONT MATTER
    print("Writing Front Matter...")
    add_paragraph_formatted(doc, sec1_intro.TITLE, "title")
    author_line = "  •  ".join([f"{a['name']} ({a['affiliation']})" for a in sec1_intro.AUTHORS])
    add_paragraph_formatted(doc, author_line, "author")
    aff_lines = " | ".join([f"({i+1}) {aff}" for i, aff in enumerate(sec1_intro.AFFILIATIONS)])
    add_paragraph_formatted(doc, aff_lines, "author")
    add_paragraph_formatted(doc, "ABSTRACT", "abstract_h")
    add_paragraph_formatted(doc, sec1_intro.ABSTRACT.strip(), "abstract")
    add_paragraph_formatted(doc, ", ".join(sec1_intro.KEYWORDS), "keywords")
        
    # SECTION 1
    print("Writing Section 1...")
    render_paragraphs_docx(doc, sec1_intro.SECTION_1_PARAGRAPHS)
        
    # SECTION 2
    print("Writing Section 2...")
    render_paragraphs_docx(doc, sec2_lit.SECTION_2_PARAGRAPHS)
    add_callout_box(doc, sec2_lit.TABLE_2_1_DATA["box_title"], sec2_lit.TABLE_2_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec2_lit.TABLE_2_1_DATA["headers"], sec2_lit.TABLE_2_1_DATA["rows"])
    
    # SECTION 3
    print("Writing Section 3...")
    render_paragraphs_docx(doc, sec3_problem.SECTION_3_PARAGRAPHS)
        
    # SECTION 4
    print("Writing Section 4...")
    sec4_list = sec4_methodology.SECTION_4_PARAGRAPHS
    idx_42 = [i for i, item in enumerate(sec4_list) if item[0].startswith("4.2")][0]
    idx_45 = [i for i, item in enumerate(sec4_list) if item[0].startswith("4.5")][0]
    
    # 4.1 Data Pipeline
    render_paragraphs_docx(doc, sec4_list[:idx_42])
    add_figure_with_image(doc, sec4_methodology.FIG_1_SPEC, img_sysarchi)
    
    # 4.2 to 4.4
    render_paragraphs_docx(doc, sec4_list[idx_42:idx_45])
    add_figure_with_image(doc, sec4_methodology.FIG_2_SPEC, img_fig2)
    
    # 4.5 Metaheuristics
    render_paragraphs_docx(doc, sec4_list[idx_45:])
    add_figure_with_image(doc, sec4_methodology.FIG_3_SPEC, img_fig3)
    
    # SECTION 5
    print("Writing Section 5...")
    render_paragraphs_docx(doc, sec5_safeguards.SECTION_5_PARAGRAPHS)
    add_callout_box(doc, sec5_safeguards.TABLE_3_DATA["box_title"], sec5_safeguards.TABLE_3_DATA["items"], is_table=True)
    add_academic_table(doc, sec5_safeguards.TABLE_3_DATA["headers"], sec5_safeguards.TABLE_3_DATA["rows"])
    
    # SECTION 6
    print("Writing Section 6...")
    sec6_list = sec6_results.SECTION_6_PARAGRAPHS
    h2_indices = [i for i, item in enumerate(sec6_list) if item[1] == 'h2']
    
    # 6.1 Point forecasts (h2_indices[0])
    render_paragraphs_docx(doc, sec6_list[:h2_indices[1]])
    add_callout_box(doc, sec6_results.TABLE_1_DATA["box_title"], sec6_results.TABLE_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_1_DATA["headers"], sec6_results.TABLE_1_DATA["rows"])
    
    # 6.2 Conformal (h2_indices[1])
    render_paragraphs_docx(doc, sec6_list[h2_indices[1]:h2_indices[2]])
    add_callout_box(doc, sec6_results.TABLE_2_DATA["box_title"], sec6_results.TABLE_2_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_2_DATA["headers"], sec6_results.TABLE_2_DATA["rows"])
    
    # 6.3 Downstream Replenishment (h2_indices[2])
    render_paragraphs_docx(doc, sec6_list[h2_indices[2]:h2_indices[3]])
    add_callout_box(doc, sec6_results.TABLE_4_DATA["box_title"], sec6_results.TABLE_4_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_4_DATA["headers"], sec6_results.TABLE_4_DATA["rows"])
    
    # 6.4 Factorial Safety Stock (h2_indices[3])
    render_paragraphs_docx(doc, sec6_list[h2_indices[3]:h2_indices[4]])
    add_callout_box(doc, sec6_results.TABLE_6_DATA["box_title"], sec6_results.TABLE_6_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_6_DATA["headers"], sec6_results.TABLE_6_DATA["rows"])
    
    # 6.5 Bootstrap (h2_indices[4])
    render_paragraphs_docx(doc, sec6_list[h2_indices[4]:h2_indices[5]])
    add_callout_box(doc, sec6_results.TABLE_5_DATA["box_title"], sec6_results.TABLE_5_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_5_DATA["headers"], sec6_results.TABLE_5_DATA["rows"])
    
    # 6.6 Pareto Frontier (h2_indices[5])
    render_paragraphs_docx(doc, sec6_list[h2_indices[5]:h2_indices[6]])
    add_callout_box(doc, sec6_results.TABLE_7_DATA["box_title"], sec6_results.TABLE_7_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_7_DATA["headers"], sec6_results.TABLE_7_DATA["rows"])
    add_figure_with_image(doc, sec6_results.FIG_4_SPEC, img_fig4)
    
    # 6.7 Economic Stress Testing (h2_indices[6])
    render_paragraphs_docx(doc, sec6_list[h2_indices[6]:])
    add_callout_box(doc, sec6_results.TABLE_8_DATA["box_title"], sec6_results.TABLE_8_DATA["items"], is_table=True)
    add_academic_table(doc, sec6_results.TABLE_8_DATA["headers"], sec6_results.TABLE_8_DATA["rows"])
    add_figure_with_image(doc, sec6_results.FIG_5_SPEC, img_fig5)
    
    # SECTION 7
    print("Writing Section 7...")
    render_paragraphs_docx(doc, sec7_discussion.SECTION_7_PARAGRAPHS)
    add_callout_box(doc, sec7_discussion.TABLE_7_1_DATA["box_title"], sec7_discussion.TABLE_7_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec7_discussion.TABLE_7_1_DATA["headers"], sec7_discussion.TABLE_7_1_DATA["rows"])
    
    # SECTION 8 & 9
    print("Writing Section 8 & 9...")
    render_paragraphs_docx(doc, sec8_limitations_conclusion.SECTION_8_9_PARAGRAPHS)
        
    # REFERENCES
    print("Writing References...")
    add_paragraph_formatted(doc, "REFERENCES", "h1")
    render_paragraphs_docx(doc, sec8_limitations_conclusion.REFERENCES_DATA)
        
    # APPENDICES
    print("Writing Appendices...")
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[0][0], "h1")
    
    # Appendix A
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[1][0], "h2")
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[2][0], "p")
    add_callout_box(doc, sec8_limitations_conclusion.TABLE_A_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_A_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec8_limitations_conclusion.TABLE_A_1_DATA["headers"], sec8_limitations_conclusion.TABLE_A_1_DATA["rows"])
    
    # Appendix B
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[3][0], "h2")
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[4][0], "p")
    add_callout_box(doc, sec8_limitations_conclusion.TABLE_B_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_B_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec8_limitations_conclusion.TABLE_B_1_DATA["headers"], sec8_limitations_conclusion.TABLE_B_1_DATA["rows"])
    
    # Appendix C
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[5][0], "h2")
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[6][0], "p")
    add_callout_box(doc, sec8_limitations_conclusion.TABLE_C_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_C_1_DATA["items"], is_table=True)
    add_academic_table(doc, sec8_limitations_conclusion.TABLE_C_1_DATA["headers"], sec8_limitations_conclusion.TABLE_C_1_DATA["rows"])
    
    # Appendix D
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[7][0], "h2")
    add_paragraph_formatted(doc, sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[8][0], "p")
    add_figure_with_image(doc, sec8_limitations_conclusion.FIG_D_1_SPEC, img_fig_d1)
    
    print(f"Saving {target_docx}...")
    try:
        doc.save(target_docx)
        print("Docx compiled successfully to manuscript.docx!")
    except PermissionError:
        alt_docx = os.path.join(BASE_DIR, "manuscript_clean.docx")
        print(f"manuscript.docx is open in Word. Saving to {alt_docx} instead...")
        doc.save(alt_docx)
        print(f"Docx compiled successfully to {alt_docx}!")

def format_md_table(headers, rows):
    """Formats markdown table."""
    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(str(c).replace("\n", " ") for c in r) + " |")
    return "\n".join(md)

def format_md_callout(box_title, items, is_table=True):
    """Formats markdown callout blockquote."""
    clean_title = box_title.replace("PLACEHOLDER & SPECIFICATION: ", "").replace("PLACEHOLDER & SPECIFICATION", "").strip()
    lines = []
    lines.append(f"> [!NOTE] **{clean_title}**")
    for lbl, val in items:
        if lbl in ["Target Image Filename", "Image Location"]:
            continue
        lines.append(f"> - **{lbl}:** {val}")
    return "\n".join(lines)

def build_markdown(target_md):
    """Builds clean publication manuscript.md."""
    print("Building Markdown Manuscript...")
    lines = []
    
    # FRONT MATTER
    lines.append(f"# {sec1_intro.TITLE}\n")
    author_line = "  •  ".join([f"{a['name']} ({a['affiliation']})" for a in sec1_intro.AUTHORS])
    lines.append(f"**{author_line}**\n")
    aff_lines = " | ".join([f"({i+1}) {aff}" for i, aff in enumerate(sec1_intro.AFFILIATIONS)])
    lines.append(f"*{aff_lines}*\n")
    lines.append("### ABSTRACT\n")
    lines.append(f"> {sec1_intro.ABSTRACT.strip()}\n")
    lines.append(f"**Keywords:** *{', '.join(sec1_intro.KEYWORDS)}*\n\n---\n")
            
    def append_paragraphs(p_list):
        for item in p_list:
            text, style, eq_label = unpack_item(item)
            if style == "h1":
                lines.append(f"\n# {text}\n")
            elif style == "h2":
                lines.append(f"\n## {text}\n")
            elif style == "h3":
                lines.append(f"\n### {text}\n")
            elif style == "p":
                lines.append(f"\n{text}\n")
            elif style == "bullet":
                lines.append(f"- {text}")
            elif style == "eq":
                if eq_label:
                    lines.append(f"\n$$\n{text} \\quad \\text{{({eq_label})}}\n$$\n")
                else:
                    lines.append(f"\n$$\n{text}\n$$\n")
            elif style == "ref":
                lines.append(f"- {text}")

    # SECTION 1
    append_paragraphs(sec1_intro.SECTION_1_PARAGRAPHS)
    
    # SECTION 2
    append_paragraphs(sec2_lit.SECTION_2_PARAGRAPHS)
    lines.append("\n" + format_md_callout(sec2_lit.TABLE_2_1_DATA["box_title"], sec2_lit.TABLE_2_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec2_lit.TABLE_2_1_DATA["headers"], sec2_lit.TABLE_2_1_DATA["rows"]) + "\n")
    
    # SECTION 3
    append_paragraphs(sec3_problem.SECTION_3_PARAGRAPHS)
    
    # SECTION 4
    sec4_list = sec4_methodology.SECTION_4_PARAGRAPHS
    idx_42 = [i for i, item in enumerate(sec4_list) if item[0].startswith("4.2")][0]
    idx_45 = [i for i, item in enumerate(sec4_list) if item[0].startswith("4.5")][0]
    
    append_paragraphs(sec4_list[:idx_42])
    lines.append("\n" + format_md_callout(sec4_methodology.FIG_1_SPEC["box_title"], sec4_methodology.FIG_1_SPEC["items"], False) + "\n")
    lines.append("\n![Figure 1](sysarchi.png)\n*Figure 1. End-to-End Closed-Loop Framework Architecture.*\n")
    
    append_paragraphs(sec4_list[idx_42:idx_45])
    lines.append("\n" + format_md_callout(sec4_methodology.FIG_2_SPEC["box_title"], sec4_methodology.FIG_2_SPEC["items"], False) + "\n")
    lines.append("\n![Figure 2](C:/Users/tanma/OneDrive/Desktop/project(1)/Multi-Horizon Quantile Forecasting(process)/quantile_safety_stock_distribution.png)\n*Figure 2. Empirical Dynamic Safety Stock Sizing.*\n")
    
    append_paragraphs(sec4_list[idx_45:])
    lines.append("\n" + format_md_callout(sec4_methodology.FIG_3_SPEC["box_title"], sec4_methodology.FIG_3_SPEC["items"], False) + "\n")
    lines.append("\n![Figure 3](C:/Users/tanma/OneDrive/Desktop/project(1)/revision_experiments/outputs/figures/e2_fitness_surface_validation.png)\n*Figure 3. Validation Fitness Landscape and Algorithmic Convergence.*\n")
    
    # SECTION 5
    append_paragraphs(sec5_safeguards.SECTION_5_PARAGRAPHS)
    lines.append("\n" + format_md_callout(sec5_safeguards.TABLE_3_DATA["box_title"], sec5_safeguards.TABLE_3_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec5_safeguards.TABLE_3_DATA["headers"], sec5_safeguards.TABLE_3_DATA["rows"]) + "\n")
    
    # SECTION 6
    sec6_list = sec6_results.SECTION_6_PARAGRAPHS
    h2_indices = [i for i, item in enumerate(sec6_list) if item[1] == 'h2']
    
    # 6.1
    append_paragraphs(sec6_list[:h2_indices[1]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_1_DATA["box_title"], sec6_results.TABLE_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_1_DATA["headers"], sec6_results.TABLE_1_DATA["rows"]) + "\n")
    
    # 6.2
    append_paragraphs(sec6_list[h2_indices[1]:h2_indices[2]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_2_DATA["box_title"], sec6_results.TABLE_2_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_2_DATA["headers"], sec6_results.TABLE_2_DATA["rows"]) + "\n")
    
    # 6.3
    append_paragraphs(sec6_list[h2_indices[2]:h2_indices[3]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_4_DATA["box_title"], sec6_results.TABLE_4_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_4_DATA["headers"], sec6_results.TABLE_4_DATA["rows"]) + "\n")
    
    # 6.4
    append_paragraphs(sec6_list[h2_indices[3]:h2_indices[4]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_6_DATA["box_title"], sec6_results.TABLE_6_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_6_DATA["headers"], sec6_results.TABLE_6_DATA["rows"]) + "\n")
    
    # 6.5
    append_paragraphs(sec6_list[h2_indices[4]:h2_indices[5]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_5_DATA["box_title"], sec6_results.TABLE_5_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_5_DATA["headers"], sec6_results.TABLE_5_DATA["rows"]) + "\n")
    
    # 6.6
    append_paragraphs(sec6_list[h2_indices[5]:h2_indices[6]])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_7_DATA["box_title"], sec6_results.TABLE_7_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_7_DATA["headers"], sec6_results.TABLE_7_DATA["rows"]) + "\n")
    lines.append("\n" + format_md_callout(sec6_results.FIG_4_SPEC["box_title"], sec6_results.FIG_4_SPEC["items"], False) + "\n")
    lines.append("\n![Figure 4](C:/Users/tanma/OneDrive/Desktop/project(1)/revision_experiments/outputs/figures/r4_matched_fill_frontier.png)\n*Figure 4. Non-Dominated Matched-Fill Pareto Frontier.*\n")
    
    # 6.7
    append_paragraphs(sec6_list[h2_indices[6]:])
    lines.append("\n" + format_md_callout(sec6_results.TABLE_8_DATA["box_title"], sec6_results.TABLE_8_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec6_results.TABLE_8_DATA["headers"], sec6_results.TABLE_8_DATA["rows"]) + "\n")
    lines.append("\n" + format_md_callout(sec6_results.FIG_5_SPEC["box_title"], sec6_results.FIG_5_SPEC["items"], False) + "\n")
    lines.append("\n![Figure 5](C:/Users/tanma/OneDrive/Desktop/project(1)/Multi-Horizon Quantile Forecasting(process)/pso_vs_gwo_convergence_quantile.png)\n*Figure 5. Nature-Inspired Swarm Convergence Dynamics.*\n")
    
    # SECTION 7
    append_paragraphs(sec7_discussion.SECTION_7_PARAGRAPHS)
    lines.append("\n" + format_md_callout(sec7_discussion.TABLE_7_1_DATA["box_title"], sec7_discussion.TABLE_7_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec7_discussion.TABLE_7_1_DATA["headers"], sec7_discussion.TABLE_7_1_DATA["rows"]) + "\n")
    
    # SECTION 8 & 9
    append_paragraphs(sec8_limitations_conclusion.SECTION_8_9_PARAGRAPHS)
    
    # REFERENCES
    lines.append("\n# REFERENCES\n")
    append_paragraphs(sec8_limitations_conclusion.REFERENCES_DATA)
    
    # APPENDICES
    lines.append(f"\n# {sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[0][0]}\n")
    
    # Appendix A
    lines.append(f"\n## {sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[1][0]}\n")
    lines.append(f"\n{sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[2][0]}\n")
    lines.append("\n" + format_md_callout(sec8_limitations_conclusion.TABLE_A_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_A_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec8_limitations_conclusion.TABLE_A_1_DATA["headers"], sec8_limitations_conclusion.TABLE_A_1_DATA["rows"]) + "\n")
    
    # Appendix B
    lines.append(f"\n## {sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[3][0]}\n")
    lines.append(f"\n{sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[4][0]}\n")
    lines.append("\n" + format_md_callout(sec8_limitations_conclusion.TABLE_B_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_B_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec8_limitations_conclusion.TABLE_B_1_DATA["headers"], sec8_limitations_conclusion.TABLE_B_1_DATA["rows"]) + "\n")
    
    # Appendix C
    lines.append(f"\n## {sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[5][0]}\n")
    lines.append(f"\n{sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[6][0]}\n")
    lines.append("\n" + format_md_callout(sec8_limitations_conclusion.TABLE_C_1_DATA["box_title"], sec8_limitations_conclusion.TABLE_C_1_DATA["items"], True) + "\n")
    lines.append("\n" + format_md_table(sec8_limitations_conclusion.TABLE_C_1_DATA["headers"], sec8_limitations_conclusion.TABLE_C_1_DATA["rows"]) + "\n")
    
    # Appendix D
    lines.append(f"\n## {sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[7][0]}\n")
    lines.append(f"\n{sec8_limitations_conclusion.APPENDICES_PARAGRAPHS[8][0]}\n")
    lines.append("\n" + format_md_callout(sec8_limitations_conclusion.FIG_D_1_SPEC["box_title"], sec8_limitations_conclusion.FIG_D_1_SPEC["items"], False) + "\n")
    lines.append("\n![Figure D.1](C:/Users/tanma/OneDrive/Desktop/project(1)/Phase_4_Inventory_Optimization/supply_chain_optimization_dashboard.png)\n*Figure D.1. Enterprise Retail Replenishment Decision Dashboard.*\n")
    
    with open(target_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved to {target_md}!")

if __name__ == "__main__":
    out_docx = os.path.join(BASE_DIR, "manuscript.docx")
    out_md = os.path.join(BASE_DIR, "manuscript.md")
    build_docx(out_docx)
    build_markdown(out_md)
    print("Compilation complete for both formats!")
