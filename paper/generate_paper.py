# -*- coding: utf-8 -*-
"""
Builder script to generate the complete ~8,000-word journal paper:
'Bridging Prediction and Inventory Optimization in Retail Supply Chains: 
 Conformalized Temporal Fusion Transformers with Guaranteed Coverage'

Generates:
1. paper/manuscript.md (Full GitHub-flavored Markdown manuscript with LaTeX equations)
2. paper/manuscript.docx (Fully formatted Microsoft Word publication draft)
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

PAPER_DIR = r"C:\Users\tanma\OneDrive\Desktop\project(1)\paper"
MD_PATH = os.path.join(PAPER_DIR, "manuscript.md")
DOCX_PATH = os.path.join(PAPER_DIR, "manuscript.docx")

print(f"Target Markdown: {MD_PATH}")
print(f"Target Word Doc: {DOCX_PATH}")
