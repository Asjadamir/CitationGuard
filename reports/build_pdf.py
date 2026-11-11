"""Render reports/phase4_report.md to PDF with ReportLab.

Run from the repo root: python reports/build_pdf.py
Handles the Markdown subset used in the report: headings, paragraphs,
bullet / numbered lists, pipe tables, **bold**, *italic* and `code`.
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "phase4_report.md"
OUT = ROOT / "reports" / "phase4_report.pdf"
ACCENT = colors.HexColor("#2F6F8F")

# Figures inserted after the section whose heading starts with the key.
FIGURES = {
    "3.1": ("phase4_retrieval_comparison.png", 13.5),
    "3.3": ("phase4_pipeline_comparison.png", 15.0),
}

base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("t", parent=base["Title"], fontSize=15, leading=18, spaceAfter=4, textColor=ACCENT),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=3, textColor=ACCENT, keepWithNext=1),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10, leading=12, spaceBefore=5, spaceAfter=2, keepWithNext=1),
    "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=8.8, leading=11.2, spaceAfter=3, alignment=TA_LEFT),
    "cell": ParagraphStyle("c", parent=base["BodyText"], fontSize=7.6, leading=9.2),
    "cellh": ParagraphStyle("ch", parent=base["BodyText"], fontSize=7.6, leading=9.2, textColor=colors.white,
                            fontName="Helvetica-Bold"),
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", text)
    return text


def table(lines: list[str], width: float):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), STYLES["cellh" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    # Column widths proportional to the longest text in each column.
    # Column widths follow the body text; headers only set a floor so they don't break mid-word.
    body = rows[1:] or rows
    lens = [max(max(len(r[j]) for r in body), max(len(w) for w in rows[0][j].split()) + 2) for j in range(len(rows[0]))]
    lens = [min(max(n, 6), 60) for n in lens]
    widths = [width * n / sum(lens) for n in lens]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F4F7")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C5CDD5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def bullet_list(items: list[str], numbered: bool):
    flow = [ListItem(Paragraph(inline(i), STYLES["body"]), leftIndent=12) for i in items]
    kw = {"bulletType": "1", "bulletFormat": "%s.", "bulletFontSize": 8.5} if numbered else {"bulletType": "bullet", "start": "•", "bulletFontSize": 8}
    return ListFlowable(flow, leftIndent=12, **kw)


def figure(name: str, width_cm: float):
    path = ROOT / "figures" / name
    img = Image(str(path))
    scale = width_cm * cm / img.imageWidth
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return img


def build() -> None:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="CitationGuard — Phase 4: Evaluation and Analysis", author="CitationGuard team")
    width = doc.width
    story = []
    lines = SRC.read_text(encoding="utf-8").splitlines()
    pending_fig = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), STYLES["title"]))
        elif ln.startswith("## ") or ln.startswith("### "):
            if pending_fig:
                story.append(figure(*pending_fig))
                pending_fig = None
            level = "h2" if ln.startswith("## ") else "h3"
            heading = ln.lstrip("#").strip()
            for key, fig in FIGURES.items():
                if heading.startswith(key):
                    pending_fig = fig
            # keep a heading with the paragraph that follows it
            story.append(Paragraph(inline(heading), STYLES[level]))
        elif ln.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            story += [table(block, width), Spacer(1, 4)]
            continue
        elif re.match(r"\s*(- |\d+\. )", ln):
            numbered = bool(re.match(r"\s*\d+\. ", ln))
            items = []
            while i < len(lines) and re.match(r"\s*(- |\d+\. )", lines[i]):
                item = re.sub(r"^\s*(- |\d+\. )", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and not re.match(r"\s*(- |\d+\. )", lines[i]):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            story.append(bullet_list(items, numbered))
            continue
        else:
            para = [ln]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||- |\d+\. )", lines[i]):
                para.append(lines[i])
                i += 1
            story.append(Paragraph(inline(" ".join(para)), STYLES["body"]))
            continue
        i += 1
    if pending_fig:
        story.append(figure(*pending_fig))

    def footer(canvas, doc_):
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"CitationGuard · Phase 4 · page {doc_.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
