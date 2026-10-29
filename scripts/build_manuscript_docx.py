"""
scripts/build_manuscript_docx.py

Compiles manuscript.md into a publication-ready IEEE-styled Microsoft Word (.docx) document.
Customized to user specifications:
1. Font: Calibri family across the entire document (body, headings, tables, captions, references).
2. Headings: Left-aligned towards the sides like a normal academic paper (not centered).
3. Diagrams & Graphs: All 14 figures (architecture diagrams, flows, ROC/PR curves, injection curves, heatmaps, ablation, confusion matrices, networks, and subgraphs) embedded directly as high-resolution images with captions.
4. Mathematical Formulas: Block equations rendered natively as Office Math (OMML) with Cambria Math typography and right-aligned numbering; inline math converted to clean mathematical Unicode symbols.
5. Strictly zero em dashes throughout.
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn


def strip_em_dashes(text: str) -> str:
    """Removes all traces of em dashes, replacing them with appropriate punctuation."""
    text = text.replace('\u2014', ' - ')
    text = text.replace('—', ' - ')
    text = text.replace('&mdash;', ' - ')
    return text


def clean_inline_math(text: str) -> str:
    """Converts inline LaTeX math into clean mathematical Unicode typography."""
    replacements = [
        (r'\\cos', 'cos'),
        (r'\\max', 'max'),
        (r'\\exp', 'exp'),
        (r'\\theta', 'θ'),
        (r'\\tau', 'τ'),
        (r'\\delta', 'δ'),
        (r'\\Delta', 'Δ'),
        (r'\\sigma', 'σ'),
        (r'\\ge', '≥'),
        (r'\\le', '≤'),
        (r'\\pm', '±'),
        (r'\\in', '∈'),
        (r'\\to', '→'),
        (r'\\sim', '~'),
        (r'\\cap', '∩'),
        (r'\\cup', '∪'),
        (r'\\mathcal\{P\}', 'P'),
        (r'\\mathcal\{K\}', 'K'),
        (r'\\mathcal\{N\}', 'N'),
        (r'\\mathbb\{I\}', 'I'),
        (r'\\mathbb\{R\}\^\{384\}', 'R³⁸⁴'),
        (r'\\mathbb\{R\}', 'R'),
        (r'\\text\{retrieval\}', 'retrieval'),
        (r'\\text\{sem\}', 'sem'),
        (r'\\text\{temp\}', 'temp'),
        (r'\\text\{cross\}', 'cross'),
        (r'\\text\{ROC-AUC\}', 'ROC-AUC'),
        (r'\\text\{PR-AUC\}', 'PR-AUC'),
        (r'\\text\{F1\}', 'F1'),
        (r'\\text\{IO\}', 'IO'),
        (r'\\text\{s\}', 's'),
        (r'\\text\{h\}', 'h'),
        (r'\\text\{where \}', 'where '),
        (r'\\text\{', ''),
        (r'\}', ''),
        (r'\\,', ' '),
        (r'\\\|', ' ‖ '),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return text


def create_omml_equation(equation_text: str, eq_num: str = ""):
    """Constructs an Office Math (OMML) XML element for native Word equation rendering."""
    clean_eq = clean_inline_math(equation_text)
    clean_eq = clean_eq.replace('\\sum', '∑')
    clean_eq = clean_eq.strip('$').strip()

    num_part = f'<m:r><m:t>          ({eq_num})</m:t></m:r>' if eq_num else ''

    omml_xml = (
        '<m:oMathPara xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
        '<m:oMath>'
        f'<m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>{clean_eq}</m:t></m:r>'
        f'{num_part}'
        '</m:oMath>'
        '</m:oMathPara>'
    )
    return parse_xml(omml_xml)


def set_cell_background(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tc_pr.append(shd)


def set_cell_margins(cell, top=70, bottom=70, left=100, right=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    
    top = OxmlElement('w:top')
    top.set(qn('w:val'), 'single')
    top.set(qn('w:sz'), '12')  # 1.5 pt
    top.set(qn('w:space'), '0')
    top.set(qn('w:color'), '1E293B')
    tblBorders.append(top)
    
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '12')
    bottom.set(qn('w:space'), '0')
    bottom.set(qn('w:color'), '1E293B')
    tblBorders.append(bottom)
    
    insideH = OxmlElement('w:insideH')
    insideH.set(qn('w:val'), 'single')
    insideH.set(qn('w:sz'), '4')  # 0.5 pt
    insideH.set(qn('w:space'), '0')
    insideH.set(qn('w:color'), 'CBD5E1')
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

    # Standard Margins: 0.75 in
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Normal Style: CALIBRI
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)  # Slate-900

    in_references = False
    i = 0
    total_figures_embedded = 0
    total_equations_rendered = 0

    while i < len(lines):
        raw_line = lines[i].rstrip('\r\n')
        line = strip_em_dashes(raw_line)

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
            run.font.name = 'Calibri'
            run.font.size = Pt(21)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            i += 1
            continue

        # Author Metadata Lines
        if line.startswith("**Author") or line.startswith("**Institution") or line.startswith("**Target Venue"):
            clean_meta = re.sub(r"\*\*(.*?)\*\*", r"\1", line).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(clean_meta)
            run.font.name = 'Calibri'
            run.font.size = Pt(10)
            if "Author" in line:
                run.font.bold = True
            else:
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            i += 1
            continue

        # Abstract
        if line.startswith("### Abstract"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                abstract_text = strip_em_dashes(lines[i].strip())
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.right_indent = Inches(0.25)
                
                run_bold = p.add_run("Abstract: ")
                run_bold.font.name = 'Calibri'
                run_bold.font.bold = True
                run_bold.font.size = Pt(9.5)
                run_bold.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                
                run_text = p.add_run(abstract_text)
                run_text.font.name = 'Calibri'
                run_text.font.size = Pt(9.5)
                run_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                i += 1
            continue

        # Index Terms / Keywords
        if line.startswith("**Index Terms:"):
            clean_terms = line.replace("**Index Terms:**", "").strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(14)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.right_indent = Inches(0.25)
            
            run_bold = p.add_run("Index Terms: ")
            run_bold.font.name = 'Calibri'
            run_bold.font.bold = True
            run_bold.font.size = Pt(9.5)
            
            run_text = p.add_run(clean_terms)
            run_text.font.name = 'Calibri'
            run_text.font.italic = True
            run_text.font.size = Pt(9.5)
            run_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            i += 1
            continue

        # Major Section Headings: ## I. INTRODUCTION (LEFT-ALIGNED LIKE NORMAL PAPER)
        if line.startswith("## "):
            heading_text = line[3:].strip()
            if "References" in heading_text:
                in_references = True

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT  # Left-aligned as requested
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(heading_text.upper())
            run.font.name = 'Calibri'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            i += 1
            continue

        # Subheadings: ### A. Subheading (LEFT-ALIGNED)
        if line.startswith("### "):
            subheading_text = line[4:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT  # Left-aligned
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(subheading_text)
            run.font.name = 'Calibri'
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            i += 1
            continue

        # Image Embedding Handler: ![caption](image_path)
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", line.strip())
        if img_match:
            caption_text = img_match.group(1)
            img_rel_path = img_match.group(2)
            img_abs_path = os.path.abspath(img_rel_path)

            if os.path.exists(img_abs_path):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(10)
                p_img.paragraph_format.space_after = Pt(4)
                p_img.paragraph_format.keep_with_next = True
                
                # Auto-scale width to fit IEEE page margins (max 6.2 inches)
                doc.add_picture(img_abs_path, width=Inches(6.0))
                total_figures_embedded += 1
                print(f"[{total_figures_embedded}] Embedded figure: {img_rel_path}")

                # Check if next line is a duplicate caption line
                if i + 1 < len(lines) and lines[i+1].strip().startswith("*Fig."):
                    i += 1
                    caption_text = lines[i].strip().strip('*')

                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(2)
                p_cap.paragraph_format.space_after = Pt(12)
                run_cap = p_cap.add_run(caption_text)
                run_cap.font.name = 'Calibri'
                run_cap.font.size = Pt(9)
                run_cap.font.italic = True
                run_cap.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            else:
                print(f"Warning: Figure image not found at {img_abs_path}")
            i += 1
            continue

        # Standalone Figure Caption (*Fig. X...*)
        if line.startswith("*Fig."):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(line.strip('*'))
            run.font.name = 'Calibri'
            run.font.size = Pt(9)
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            i += 1
            continue

        # Block Equations: $$...$$
        if line.startswith("$$") and line.endswith("$$"):
            eq_content = line[2:-2].strip()
            num_match = re.search(r'\\quad\s*\((\d+)\)$', eq_content)
            eq_num = num_match.group(1) if num_match else ""
            if eq_num:
                eq_content = re.sub(r'\\quad\s*\(\d+\)$', '', eq_content).strip()

            p_eq = doc.add_paragraph()
            p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_eq.paragraph_format.space_before = Pt(6)
            p_eq.paragraph_format.space_after = Pt(6)

            try:
                omml_element = create_omml_equation(eq_content, eq_num)
                p_eq._p.append(omml_element)
                total_equations_rendered += 1
            except Exception as e:
                clean_text = clean_inline_math(eq_content)
                run_eq = p_eq.add_run(clean_text)
                run_eq.font.name = 'Cambria Math'
                run_eq.font.size = Pt(10)
                if eq_num:
                    r_num = p_eq.add_run(f"          ({eq_num})")
                    r_num.font.name = 'Calibri'
                    r_num.font.size = Pt(10)
            i += 1
            continue

        # Table Caption (TABLE I: ...)
        if line.startswith("TABLE "):
            table_caption = line.strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(table_caption)
            run.font.name = 'Calibri'
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            i += 1
            continue

        # Markdown Table Detection
        if line.startswith("|") and line.endswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row_str = strip_em_dashes(lines[i].strip())
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
                        set_cell_margins(cell, top=65, bottom=65, left=90, right=90)
                        if is_header:
                            set_cell_background(cell, "F1F5F9")  # Slate-100 header

                        p_cell = cell.paragraphs[0]
                        p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_header or c_idx > 0) else WD_ALIGN_PARAGRAPH.LEFT
                        p_cell.paragraph_format.space_before = Pt(1)
                        p_cell.paragraph_format.space_after = Pt(1)
                        
                        clean_text = cell_value.replace("**", "").replace("*", "")
                        clean_text = clean_inline_math(clean_text)
                        run_cell = p_cell.add_run(clean_text)
                        run_cell.font.name = 'Calibri'
                        run_cell.font.size = Pt(8.5)
                        if is_header:
                            run_cell.font.bold = True
                            run_cell.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                        else:
                            run_cell.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

                p_spacer = doc.add_paragraph()
                p_spacer.paragraph_format.space_before = Pt(0)
                p_spacer.paragraph_format.space_after = Pt(6)
            continue

        # References Section (Hanging Indent)
        if in_references and re.match(r"^\[\d+\]", line):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)

            clean_ref = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            clean_ref = clean_ref.replace('–', '-')
            parts = re.split(r'(".*?"|\*.*?\*)', clean_ref)
            for part in parts:
                if part.startswith('*') and part.endswith('*'):
                    r = p.add_run(part.strip('*'))
                    r.font.italic = True
                else:
                    r = p.add_run(part)
                r.font.name = 'Calibri'
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            i += 1
            continue

        # Regular Body Paragraphs
        if line.strip():
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.line_spacing = 1.15

            tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|\$.*?\$)', line)
            for tok in tokens:
                if not tok:
                    continue
                if tok.startswith('**') and tok.endswith('**'):
                    r = p.add_run(clean_inline_math(tok[2:-2]))
                    r.font.name = 'Calibri'
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                elif tok.startswith('*') and tok.endswith('*'):
                    r = p.add_run(clean_inline_math(tok[1:-1]))
                    r.font.name = 'Calibri'
                    r.font.italic = True
                    r.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                elif tok.startswith('$') and tok.endswith('$'):
                    math_content = clean_inline_math(tok[1:-1])
                    r = p.add_run(math_content)
                    r.font.name = 'Cambria Math'
                    r.font.italic = True
                else:
                    r = p.add_run(tok)
                    r.font.name = 'Calibri'
                    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                r.font.size = Pt(10)

        i += 1

    print(f"Total Figures Embedded: {total_figures_embedded}")
    print(f"Total OMML Equations Rendered: {total_equations_rendered}")
    print(f"Saving compiled Word document to {output_docx}...")
    doc.save(output_docx)
    print(f"Successfully generated {output_docx} (Size: {os.path.getsize(output_docx)} bytes)")


if __name__ == "__main__":
    create_manuscript_docx()
