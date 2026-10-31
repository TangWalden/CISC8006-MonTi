"""Render the English evidence report without loading an ML runtime or data.

Requires ReportLab and pypdf in the isolated document runtime, not the ML env.
The editable Markdown is the prose authority; existing figures remain unchanged.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

import reportlab
import pypdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "21caa7531058f9149b814ae1c276ab7b654412a9"
REPOSITORY = "https://github.com/TangWalden/CISC8006-MonTi"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_target(value, source):
    if "://" in value:
        return value
    path_part, separator, fragment = value.partition("#")
    path = (source.parent / path_part).resolve()
    if not path.exists():
        raise ValueError("Broken report reference: " + value)
    relative = path.relative_to(ROOT).as_posix()
    kind = "tree" if path.is_dir() else "blob"
    return (REPOSITORY + "/" + kind + "/" + SNAPSHOT + "/" + quote(relative)
            + ("#" + fragment if separator else ""))


def inline(value, source):
    # This report uses a deliberately small Markdown subset.
    pieces = []
    cursor = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", value):
        pieces.append(escape(value[cursor:match.start()]))
        href = escape(public_target(match.group(2), source), quote=True)
        pieces.append('<link href="' + href + '" color="#153D61">'
                      + escape(match.group(1)) + '</link>')
        cursor = match.end()
    pieces.append(escape(value[cursor:]))
    text = "".join(pieces)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.5">\1</font>', text)


def styles():
    base = getSampleStyleSheet()
    result = {}
    specs = {
        "title": dict(fontName="Helvetica-Bold", fontSize=22, leading=27,
                      spaceAfter=13, keepWithNext=True),
        "heading": dict(fontName="Helvetica-Bold", fontSize=15, leading=19,
                        spaceBefore=5, spaceAfter=10, keepWithNext=True),
        "body": dict(fontName="Times-Roman", fontSize=10.6, leading=13.6,
                     spaceAfter=8, alignment=TA_JUSTIFY),
        "meta": dict(fontName="Helvetica", fontSize=9, leading=12,
                     spaceAfter=8, alignment=TA_LEFT),
        "caption": dict(fontName="Times-Italic", fontSize=9, leading=11.5,
                        spaceBefore=4, spaceAfter=9),
        "reference": dict(fontName="Times-Roman", fontSize=9.4, leading=12,
                          spaceAfter=6, alignment=TA_LEFT),
        "cell": dict(fontName="Times-Roman", fontSize=9.6, leading=12,
                     alignment=TA_LEFT),
        "numeric": dict(fontName="Times-Roman", fontSize=9.6, leading=12,
                        alignment=TA_CENTER),
        "cell_header": dict(fontName="Helvetica-Bold", fontSize=9, leading=11.5,
                            alignment=TA_CENTER),
    }
    for name, spec in specs.items():
        result[name] = ParagraphStyle(name, parent=base["Normal"], textColor=colors.black, **spec)
    return result


def render_table(lines, width, source, sty):
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError("Inconsistent report table columns")
    if count == 2:
        proportions = [0.75, 0.25]
    elif count == 3:
        proportions = [0.24, 0.35, 0.41]
        if rows[0][0] == "Stratum":
            raise ValueError("Stratum table should have four columns")
    elif count == 4 and rows[0][0] == "Stratum":
        proportions = [0.48, 0.18, 0.16, 0.18]
    elif count == 4 and rows[0][0] == "Generator":
        proportions = [0.19, 0.20, 0.26, 0.35]
    elif count == 4:
        proportions = [0.23, 0.25, 0.25, 0.27]
    else:
        proportions = [1.0 / count] * count
    widths = [width * item for item in proportions]
    cells = []
    for row_index, row in enumerate(rows):
        cells.append([
            Paragraph(inline(value, source), sty["cell_header"] if row_index == 0 else
                      sty["cell"] if column == 0 or count == 3 else sty["numeric"])
            for column, value in enumerate(row)
        ])
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D9D9D9")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EBEF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(doc.leftMargin, 28, "CISC8006 report draft | 31 October 2026")
    canvas.drawRightString(A4[0] - doc.rightMargin, 28, str(doc.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="docs/REPORT_DRAFT.md")
    parser.add_argument("--output", default="docs/MonTi_Report_Draft.pdf")
    parser.add_argument("--manifest", default="evidence/report_draft_build.json")
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    manifest = (ROOT / args.manifest).resolve()
    for path in (source, output, manifest):
        path.relative_to(ROOT)
    prose = source.read_text(encoding="utf-8")
    if any(ord(character) > 127 for character in prose):
        raise ValueError("Use ASCII report prose to avoid font/encoding ambiguity")
    sty = styles()
    width = A4[0] - 104
    story = []
    inputs = {source, Path(__file__).resolve()}
    lines = prose.splitlines()
    index = 0
    references = False
    paragraph_count = 0
    page_breaks = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            page_breaks += 1
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:], source), sty["title"]))
            index += 1
            continue
        if line.startswith("## "):
            references = line == "## References and evidence index"
            story.append(Paragraph(inline(line[3:], source), sty["heading"]))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.extend([Spacer(1, 3), render_table(table_lines, width, source, sty), Spacer(1, 11)])
            continue
        picture = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if picture:
            path = (source.parent / picture.group(2)).resolve()
            inputs.add(path)
            image = Image(str(path))
            scale = width / image.imageWidth
            image.drawWidth = width
            image.drawHeight = image.imageHeight * scale
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            together = [image]
            if index < len(lines) and lines[index].startswith("Figure "):
                together.append(Paragraph(inline(lines[index], source), sty["caption"]))
                index += 1
            story.append(KeepTogether(together))
            continue
        body = [line[2:] if references and line.startswith("- ") else line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "<!--", "![", "- ")):
            body.append(lines[index].strip())
            index += 1
        text = " ".join(body)
        if references:
            style = sty["reference"]
        elif paragraph_count < 3:
            style = sty["meta"]
        else:
            style = sty["body"]
        story.append(Paragraph(inline(text, source), style))
        paragraph_count += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=52, rightMargin=52,
        topMargin=48, bottomMargin=48,
        title="MonTi Component Reproduction and Validation Diagnostics",
        author="Codex prepared draft for Junyi Tang and Yifan Fan",
        subject="Incomplete CISC8006 course report with audited results and pending improvement work",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer,
                   canvasmaker=partial(Canvas, invariant=1))
    reader = PdfReader(str(output))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    if len(page_texts) != page_breaks + 1:
        raise ValueError("Unexpected page overflow: expected %d pages, got %d" % (page_breaks + 1, len(page_texts)))
    if any(re.search(r"[\u3400-\u9fff]", text) for text in page_texts):
        raise ValueError("Non-English CJK text in PDF extraction")
    if not all(len(text.strip()) > 100 for text in page_texts):
        raise ValueError("Suspiciously empty report page")
    checked_links = 0
    for match in re.finditer(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", prose):
        value = match.group(2)
        public_target(value, source)
        checked_links += 1
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    numerical_inputs = [
        "course/frozen_protocol.json", "course/configs/baseline.json",
        "evidence/reproduction_summary.json", "evidence/baseline_audit.json",
        "evidence/run_registry.json", "evidence/data/partition_audit.json",
        "evidence/improvement_pilot_summary.json", "evidence/improvement_pilot_audit.json",
        "evidence/results_first_diagnosis_v1/summary.json", "evidence/results_first_diagnosis_audit_v1.json",
        "evidence/one_edge_diagnosis_v2/summary.json", "evidence/one_edge_diagnosis_audit_v2.json",
        "evidence/one_edge_endpoint_attribution_v1.json", "evidence/score_role_replay_v2/summary.json",
        "evidence/score_role_replay_audit_v2.json", "evidence/score_role_replay_recount_after_ledger_update.json",
        "evidence/research_cost_ledger.json", "evidence/publication_history_map.json",
        "evidence/outcome_v2_synthetic_checks/attempt1.json",
        "evidence/outcome_v2_synthetic_checks/review_and_receipt.json",
        "docs/figures/figure_provenance.json", "docs/figures/results_first/provenance.json",
        "environment/runtime.json", "evidence/course-guide.txt",
    ]
    for relative in numerical_inputs:
        committed = subprocess.check_output(["git", "show", SNAPSHOT + ":" + relative], cwd=str(ROOT))
        if hashlib.sha256(committed).hexdigest() != sha256(ROOT / relative):
            raise ValueError("Evidence snapshot mismatch: " + relative)
        inputs.add(ROOT / relative)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "monti_report_draft_build_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered PDF draft; separate visual and factual QA required before publication",
        "source_tree_head_before_build": source_commit,
        "numerical_evidence_snapshot": SNAPSHOT,
        "command": "bundled-python -m course.build_report_draft",
        "runtime": {"python": sys.version, "reportlab": reportlab.Version, "pypdf": pypdf.__version__},
        "numerical_input_files_matched_to_snapshot": len(numerical_inputs),
        "inputs": {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in sorted(inputs)},
        "output": {str(output.relative_to(ROOT).as_posix()): sha256(output)},
        "pages": len(reader.pages),
        "checked_reference_links": checked_links,
        "pdf_timestamp_policy": "ReportLab invariant timestamps for repeatable bytes; actual build UTC is recorded here",
        "scope": "Document preparation from saved evidence only; no ML imports, training, inference or GPU work",
        "course_submission_status": "draft incomplete; owner direction, full improvement/control evaluation, genuine human review and final freeze remain pending",
    }
    manifest.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "built", "pages": len(reader.pages), "references_checked": checked_links,
                      "output": str(output.relative_to(ROOT)), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
