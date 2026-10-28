"""
scripts/build_manuscript_docx.py

Parses manuscript.md and compiles a professionally formatted IEEE-style Word (.docx) manuscript.
Applies Times New Roman font, standardized IEEE heading styles, professional table borders and shading,
and proper hanging indents for the references section.
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in twips (1/20 pt)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_table_borders(table):
    """Applies clean horizontal academic borders (booktabs style)."""
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    
    top = OxmlElement('w:top')
    top.set(qn('w:val'), 'single')
    top.set(qn('w:sz'), '12')  # 1.5 pt
    top.set(qn('w:space'), '0')
    top.set(qn('w:color'), '333333')
    tblBorders.append(top)
    
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '12')
    bottom.set(qn('w:space'), '0')
    bottom.set(qn('w:color'), '333333')
    tblBorders.append(bottom)
    
    insideH = OxmlElement('w:insideH')
    insideH.set(qn('w:val'), 'single')
    insideH.set(qn('w:sz'), '4')  # 0.5 pt
    insideH.set(qn('w:space'), '0')
    insideH.set(qn('w:color'), 'CCCCCC')
    tblBorders.append(insideH)

    for side in ['left', 'right', 'insideV']:
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:val'), 'none')
        tblBorders.append(node)

    tblPr.append(tblBorders)


def create_manuscript_docx(input_md="manuscript.md", output_docx="manuscript.docx"):
    print(f"Reading {input_md}...")
    with open(input_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    # Configure Margins: 0.75 in (IEEE Standard)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    in_references = False
    in_code_block = False
    code_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip('\r\n')

        # Code block handler
        if line.startswith("```"):
            if in_code_block:
                # Flush code block
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.left_indent = Inches(0.2)
                p.paragraph_format.right_indent = Inches(0.2)
                run = p.add_run("\n".join(code_lines))
                run.font.name = 'Consolas'
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Skip horizontal dividers
        if line.strip() == "---":
            i += 1
            continue

        # Title: # Title
        if line.startswith("# "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(12)
            run = p.add_run(title_text)
            run.font.size = Pt(20)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
            i += 1
            continue

        # Author metadata lines
        if line.startswith("**Author") or line.startswith("**Institution") or line.startswith("**Target Venue"):
            clean_meta = re.sub(r"\*\*(.*?)\*\*", r"\1", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(clean_meta)
            run.font.size = Pt(10)
            if "Author" in line:
                run.font.bold = True
            else:
                run.font.italic = True
            i += 1
            continue

        # Abstract
        if line.startswith("### Abstract"):
            i += 1
            # Next line usually contains the abstract text
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                abstract_text = lines[i].strip()
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.right_indent = Inches(0.25)
                
                run_bold = p.add_run("Abstract— ")
                run_bold.font.bold = True
                run_bold.font.italic = True
                run_bold.font.size = Pt(9.5)
                
                run_text = p.add_run(abstract_text)
                run_text.font.italic = True
                run_text.font.size = Pt(9.5)
                i += 1
            continue

        # Index Terms / Keywords
        if line.startswith("**Index Terms"):
            clean_terms = re.sub(r"\*\*Index Terms[—\-–]\*\*", "", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(14)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.right_indent = Inches(0.25)
            
            run_bold = p.add_run("Index Terms— ")
            run_bold.font.bold = True
            run_bold.font.italic = True
            run_bold.font.size = Pt(9.5)
            
            run_text = p.add_run(clean_terms)
            run_text.font.italic = True
            run_text.font.size = Pt(9.5)
            i += 1
            continue

        # Major Headings: ## I. INTRODUCTION
        if line.startswith("## "):
            heading_text = line[3:].strip()
            if "References" in heading_text:
                in_references = True

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if not in_references else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(heading_text.upper())
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
            i += 1
            continue

        # Subheadings: ### A. Subheading
        if line.startswith("### "):
            subheading_text = line[4:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(subheading_text)
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.italic = True
            i += 1
            continue

        # Table Caption (e.g. TABLE I: ...)
        if line.startswith("TABLE "):
            table_caption = line.strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(table_caption)
            run.font.size = Pt(9)
            run.font.bold = True
            i += 1
            continue

        # Markdown Table Detection
        if line.startswith("|") and line.endswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_str = lines[i].strip()
                # Skip divider line |:---|:---|
                if not re.match(r"^\|(\s*[-:]+[-|\s:]*)\|$", row_str):
                    cols = [c.strip() for c in row_str.strip('|').split('|')]
                    table_rows.append(cols)
                i += 1

            if table_rows:
                num_cols = len(table_rows[0])
                table = doc.add_table(rows=len(table_rows), cols=num_cols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(table)

                for r_idx, row_data in enumerate(table_rows):
                    is_header = (r_idx == 0)
                    for c_idx, cell_value in enumerate(row_data[:num_cols]):
                        cell = table.cell(r_idx, c_idx)
                        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
                        if is_header:
                            set_cell_background(cell, "F2F2F2")

                        p_cell = cell.paragraphs[0]
                        p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_header or c_idx > 0) else WD_ALIGN_PARAGRAPH.LEFT
                        p_cell.paragraph_format.space_before = Pt(1)
                        p_cell.paragraph_format.space_after = Pt(1)
                        
                        # Clean bold/italic markdown
                        clean_text = cell_value.replace("**", "").replace("*", "")
                        run_cell = p_cell.add_run(clean_text)
                        run_cell.font.name = 'Times New Roman'
                        run_cell.font.size = Pt(8.5)
                        if is_header:
                            run_cell.font.bold = True

                # Space after table
                p_spacer = doc.add_paragraph()
                p_spacer.paragraph_format.space_before = Pt(0)
                p_spacer.paragraph_format.space_after = Pt(6)
            continue

        # References handler (hanging indent)
        if in_references and re.match(r"^\[\d+\]", line):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)

            # Strip bold formatting
            clean_ref = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            
            # Format title italic if inside quotes or asterisks
            parts = re.split(r'(".*?"|\*.*?\*)', clean_ref)
            for part in parts:
                if part.startswith('*') and part.endswith('*'):
                    r = p.add_run(part.strip('*'))
                    r.font.italic = True
                else:
                    r = p.add_run(part)
                r.font.size = Pt(9)
            i += 1
            continue

        # Regular Body Paragraphs
        if line.strip():
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.15

            # Figure caption style
            if line.startswith("*Fig."):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(10)
                run = p.add_run(line.strip('*'))
                run.font.size = Pt(9)
                run.font.italic = True
                i += 1
                continue

            # Standard text with inline formatting
            # Parse bold (**text**) and italic (*text*) and math ($...$)
            tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|\$.*?\$)', line)
            for tok in tokens:
                if tok.startswith('**') and tok.endswith('**'):
                    r = p.add_run(tok[2:-2])
                    r.font.bold = True
                elif tok.startswith('*') and tok.endswith('*'):
                    r = p.add_run(tok[1:-1])
                    r.font.italic = True
                elif tok.startswith('$') and tok.endswith('$'):
                    r = p.add_run(tok[1:-1])
                    r.font.italic = True
                else:
                    r = p.add_run(tok)
                r.font.size = Pt(10)

        i += 1

    print(f"Saving compiled Word document to {output_docx}...")
    doc.save(output_docx)
    print(f"Successfully generated {output_docx} (Size: {os.path.getsize(output_docx)} bytes)")


if __name__ == "__main__":
    create_manuscript_docx()
