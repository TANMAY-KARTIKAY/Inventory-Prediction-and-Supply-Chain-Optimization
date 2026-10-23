# -*- coding: utf-8 -*-
"""
Clean Academic Paper Builder:
Generates manuscript.docx and manuscript.md following the exact requested outline,
completely removing raw LaTeX syntax, embedding high-resolution figures directly,
and providing prominent indicator callout boxes for every table and image.
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

# Color palette
COLOR_NAVY_HEX = "1B365D"
COLOR_SLATE_HEX = "2C5E8A"
COLOR_LIGHT_BG = "F4F6F9"
COLOR_BOX_BG = "EBF2FA"
COLOR_BORDER = "CCCCCC"
COLOR_NAVY_RGB = RGBColor(27, 54, 93)
COLOR_SLATE_RGB = RGBColor(44, 94, 138)
COLOR_TEXT_RGB = RGBColor(34, 34, 34)
COLOR_MUTED_RGB = RGBColor(100, 100, 100)

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

print("Starting clean paper builder...")
