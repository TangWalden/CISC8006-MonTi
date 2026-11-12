"""Render the staged MonTi report from saved evidence with no ML imports."""
import argparse
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
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
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DATA_SNAPSHOT = '70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980'
REPOSITORY = 'https://github.com/TangWalden/CISC8006-MonTi'
INPUTS = [
    'results/downstream_cooperation/latest.json',
    'results/downstream_cooperation/seed_metrics.csv',
    'results/downstream_cooperation/case_counts.csv',
    'results/downstream_cooperation/quota_by_case.csv',
    'results/downstream_cooperation/provenance.json',
    'results/downstream_cooperation/audit_saved.json',
    'results/downstream_cooperation/audit_tensors.json',
    'evidence/downstream_cooperation_v1/execution_protocol.json',
    'evidence/downstream_cooperation_v1/execution_source_manifest.json',
    'evidence/downstream_cooperation_v1/completed_summary.json',
    'evidence/downstream_cooperation_v1/completed_main_guardian.json',
    'evidence/reproduction_summary.json',
    'evidence/research_cost_ledger.json',
]


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def git(*args): return subprocess.check_output(['git'] + list(args), cwd=ROOT)


def target(value, source, prose_commit):
    if '://' in value: return value
    path_part, separator, fragment = value.partition('#')
    path = (source.parent / path_part).resolve()
    if not path.exists(): raise ValueError('Missing report reference: ' + value)
    relative = path.relative_to(ROOT).as_posix()
    version = DATA_SNAPSHOT if relative in INPUTS else prose_commit
    return REPOSITORY + ('/tree/' if path.is_dir() else '/blob/') + version + '/' + quote(relative) + ('#' + fragment if separator else '')


def inline(text, source, prose_commit):
    pieces = []; offset = 0
    for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
        pieces.append(escape(text[offset:match.start()]))
        pieces.append('<link href="' + escape(target(match[2], source, prose_commit), quote=True)
                      + '" color="#165577">' + escape(match[1]) + '</link>')
        offset = match.end()
    pieces.append(escape(text[offset:])); value = ''.join(pieces)
    value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
    return re.sub(r'`([^`]+)`', r'<font name="Courier" size="7.8">\1</font>', value)


def styles():
    common = dict(textColor=colors.HexColor('#172536'), splitLongWords=True)
    return {
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=22, leading=27, spaceAfter=17, **common),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=15, leading=19, spaceBefore=5, spaceAfter=10, keepWithNext=True, **common),
        'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=8, spaceAfter=6, keepWithNext=True, **common),
        'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=10.7, leading=14.1, spaceAfter=8, **common),
        'bullet': ParagraphStyle('bullet', fontName='Times-Roman', fontSize=10.5, leading=13.5, leftIndent=12, firstLineIndent=-9, spaceAfter=5, **common),
        'header': ParagraphStyle('header', fontName='Helvetica-Bold', fontSize=8.8, leading=11.4, alignment=TA_LEFT, **common),
        'cell': ParagraphStyle('cell', fontName='Times-Roman', fontSize=9.7, leading=12.1, alignment=TA_LEFT, **common),
    }


def table(lines, width, sty, source, prose_commit):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-+:?', value) for value in row)]
    n = len(rows[0])
    if any(len(row) != n for row in rows): raise ValueError('Table columns differ.')
    if n == 2: proportions = [.66, .34]
    elif n == 3: proportions = [.42, .25, .33]
    elif n == 4: proportions = [.37, .21, .21, .21]
    elif n == 5: proportions = [.36, .16, .16, .16, .16]
    else: proportions = [1 / n] * n
    cells = [[Paragraph(inline(value, source, prose_commit), sty['header'] if i == 0 else sty['cell'])
              for value in row] for i, row in enumerate(rows)]
    item = Table(cells, colWidths=[width * p for p in proportions], repeatRows=1, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5EFF4')),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#5283A0')),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#CCD8DF')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return item


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#CCD8DF'))
    canvas.line(doc.leftMargin, 38, A4[0] - doc.rightMargin, 38)
    canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#506173'))
    canvas.drawString(doc.leftMargin, 25, 'MonTi role specialization - Development report - 12 November 2026')
    canvas.drawRightString(A4[0] - doc.rightMargin, 25, str(doc.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='docs/STAGED_REPORT.md')
    parser.add_argument('--output', default='docs/MonTi_Staged_Report.pdf')
    parser.add_argument('--manifest', default='evidence/staged_report_build.json')
    args = parser.parse_args()
    source, output, manifest = [(ROOT / name).resolve() for name in (args.source, args.output, args.manifest)]
    for path in (source, output, manifest): path.relative_to(ROOT)
    prose_commit = git('rev-parse', 'HEAD').decode().strip()
    if git('show', prose_commit + ':' + source.relative_to(ROOT).as_posix()) != source.read_bytes():
        raise ValueError('Commit the reviewed report prose before rendering.')
    for name in INPUTS:
        if git('show', DATA_SNAPSHOT + ':' + name) != (ROOT / name).read_bytes():
            raise ValueError('Numerical evidence changed: ' + name)
    prose = source.read_text(encoding='utf-8')
    if any(ord(char) > 127 for char in prose): raise ValueError('Use ASCII prose for stable standard PDF fonts.')
    sty = styles(); width = A4[0] - 104; story = []
    lines = prose.splitlines(); i = 0; breaks = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line == '<!-- pagebreak -->': story.append(PageBreak()); breaks += 1; i += 1; continue
        if line.startswith('# '): story.append(Paragraph(inline(line[2:], source, prose_commit), sty['title'])); i += 1; continue
        if line.startswith('## '): story.append(Paragraph(inline(line[3:], source, prose_commit), sty['h2'])); i += 1; continue
        if line.startswith('### '): story.append(Paragraph(inline(line[4:], source, prose_commit), sty['h3'])); i += 1; continue
        if line.startswith('|'):
            group = []
            while i < len(lines) and lines[i].strip().startswith('|'): group.append(lines[i]); i += 1
            story.extend([Spacer(1, 3), table(group, width, sty, source, prose_commit), Spacer(1, 11)]); continue
        if line.startswith('- '): story.append(Paragraph('- ' + inline(line[2:], source, prose_commit), sty['bullet'])); i += 1; continue
        group = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '- ', '<!--')):
            group.append(lines[i].strip()); i += 1
        story.append(Paragraph(inline(' '.join(group), source, prose_commit), sty['body']))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=52, rightMargin=52,
        topMargin=48, bottomMargin=53, title='Upstream and Downstream Role Specialization for MonTi',
        author='MonTi CISC8006 project - prepared with Codex assistance',
        subject='Three-seed controlled validation study of a staged graph injection extension')
    doc.build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=partial(Canvas, invariant=1))
    reader = pypdf.PdfReader(str(output)); texts = [page.extract_text() or '' for page in reader.pages]
    if len(texts) != breaks + 1: raise ValueError('Page overflow: expected %d, got %d' % (breaks + 1, len(texts)))
    if any(len(text.strip()) < 100 for text in texts): raise ValueError('Suspicious empty page.')
    if any(re.search(r'[\u3400-\u9fff]', text) for text in texts): raise ValueError('Unexpected CJK text.')
    for literal in ('53.6486', '57.0344', '58.1436', '4.4950'):
        if literal not in '\n'.join(texts): raise ValueError('Missing key report value: ' + literal)
    refs = list(re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', prose))
    for match in refs: target(match[2], source, prose_commit)
    inputs = INPUTS + [source.relative_to(ROOT).as_posix(), Path(__file__).relative_to(ROOT).as_posix()]
    data = dict(schema='monti_staged_report_build_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='PDF rendered; separate visual and factual review required', numerical_evidence_snapshot=DATA_SNAPSHOT,
        editorial_source_commit=prose_commit, scientific_execution_source='9e013cb182b5af71a9d0b719b5a939deda3c88e2',
        pages=len(texts), expected_pages=breaks + 1, reference_links_checked=len(refs),
        inputs={name: digest(ROOT / name) for name in inputs}, output={output.relative_to(ROOT).as_posix(): digest(output)},
        runtime=dict(python=sys.version, reportlab=reportlab.Version, pypdf=pypdf.__version__),
        scope='Report preparation from immutable saved evidence; no model, ML runtime, dataset/checkpoint loading, inference or GPU work.',
        link_policy='Scientific records use the immutable evidence snapshot. Editorial links use the committed reviewed prose snapshot.',
        pdf_timestamp_policy='Invariant ReportLab PDF metadata; actual build UTC recorded separately.')
    manifest.write_bytes((json.dumps(data, indent=2, ensure_ascii=True) + '\n').encode())
    print(json.dumps(dict(status='rendered', pages=len(texts), output=str(output.relative_to(ROOT)), sha256=digest(output))))


if __name__ == '__main__': main()
