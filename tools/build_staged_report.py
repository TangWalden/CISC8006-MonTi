"""Render the staged MonTi report from saved evidence with no ML imports."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import re
import subprocess
import statistics
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
FINAL_INPUTS = ['results/final_staged/' + name for name in (
    'final_public.json', 'latest.json', 'seed_metrics.csv', 'case_counts.csv',
    'quota_by_case.csv', 'provenance.json', 'audit_saved.json', 'audit_tensors.json', 'export_receipt.json'
)] + ['evidence/final_staged_v1/protocol.json']
AUTHOR_INPUTS = ['results/author_staged/' + name for name in (
    'latest.json', 'seed_metrics.csv', 'case_counts.csv', 'quota_by_case.csv',
    'provenance.json', 'audit_saved.json', 'audit_tensors.json', 'export_receipt.json'
)] + ['evidence/author_staged_v1/protocol.json']
REFERENCE_SNAPSHOT = DATA_SNAPSHOT
REFERENCE_INPUTS = set(INPUTS)


def numerical_literal(value):
    value = float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite completed-study report value.')
    return format(value, '.4f')


def completed_study(study):
    """Validate saved publication metadata; never rebuild an audit or model."""
    prefix = ROOT / 'results' / study
    read = lambda name: json.loads((prefix / name).read_text(encoding='utf-8'))
    latest, provenance, receipt = [read(name) for name in ('latest.json', 'provenance.json', 'export_receipt.json')]
    protocol = json.loads((ROOT / 'evidence' / (study + '_v1') / 'protocol.json').read_text(encoding='utf-8'))
    author = study == 'author_staged'
    expected_schema = 'monti_author_staged_v1' if author else 'monti_final_staged_v1'
    origin = latest.get('scientific_provenance', latest.get('provenance'))
    if (protocol['schema'] != expected_schema or protocol['status'] != 'frozen'
            or protocol['owner_execution_authorized'] is not True or protocol['final_TEST_allowed'] is not True
            or latest.get('status') != 'completed' or latest.get('full_TEST_evaluated') is not True
            or latest['seeds'] != [0, 1, 2] or protocol['seeds'] != [0, 1, 2]
            or latest['arms'] != ['native_monti', 'quota_random', 'quota_learned']
            or len(set(protocol['train_groups'])) != 937 or len(protocol['train_groups']) != 937
            or len(set(protocol['test_groups'])) != 909 or len(protocol['test_groups']) != 909
            or protocol['cohort']['test_target_occurrences'] != 10515
            or not isinstance(origin, dict) or origin['source_commit'] != protocol['source_manifest']['source_commit']
            or origin['protocol_sha256'] != digest(ROOT / 'evidence' / (study + '_v1') / 'protocol.json')
            or provenance['scientific_provenance'] != origin or receipt['scientific_provenance'] != origin
            or receipt['status'] != 'completed'):
        raise ValueError('Only a completed, source-bound full registered TEST export is valid: ' + study)
    with (prefix / 'seed_metrics.csv').open(encoding='utf-8', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    arms = latest['arms']
    keys = {(int(row['seed']), row['arm']) for row in metrics}
    if (len(metrics) != 9 or keys != {(seed, arm) for seed in (0, 1, 2) for arm in arms}
            or any(int(row['groups']) != 909 or int(row['occurrences']) != 10515 for row in metrics)):
        raise ValueError('Full TEST seed/cohort metric table is incomplete: ' + study)
    mean_values = latest['gaga_mean_mr_percent']
    for arm in arms:
        mean = statistics.mean(float(row['gaga_mr_percent']) for row in metrics if row['arm'] == arm)
        if not math.isfinite(mean) or not math.isclose(mean, mean_values[arm], abs_tol=1e-9, rel_tol=1e-10):
            raise ValueError('Completed TEST mean differs from its nine-row seed table: ' + study)
    primary = latest['contrasts']['native_monti']
    if len(primary['seed_differences_pp']) != 3 or len(primary['nominal_paired_t95_pp']) != 2:
        raise ValueError('Completed primary contrast is malformed: ' + study)
    for filename, kind in (('audit_saved.json', 'saved_logit_recount'), ('audit_tensors.json', 'saved_tensor_geometry')):
        audit = read(filename)
        if (audit.get('schema') != 'monti_public_audit_summary_v1' or audit.get('public_summary') is not True
                or audit['kind'] != kind or audit['status'] != 'passed' or audit['scientific_provenance'] != origin
                or re.fullmatch('[0-9a-f]{64}', audit['full_raw_audit_sha256']) is None
                or re.fullmatch('[0-9a-f]{64}', audit['auditor_source_sha256']) is None
                or re.fullmatch('[0-9a-f]{40}', audit['audit_source_commit']) is None
                or provenance['full_raw_audit_sha256'][kind] != audit['full_raw_audit_sha256']):
            raise ValueError('Explicit passed compact audit summary/raw-audit binding differs: ' + study)
        if kind == 'saved_logit_recount' and audit['audit_source_commit'] != origin['source_commit']:
            raise ValueError('Saved-logit auditor source differs from its frozen study: ' + study)
    for name, wanted in receipt['output_sha256'].items():
        if not (prefix / name).is_file() or digest(prefix / name) != wanted:
            raise ValueError('Published export receipt byte binding differs: ' + study + '/' + name)
    if author and (protocol['author_upstream']['distinct_upstream_checkpoints'] != 1
            or latest['replication']['distinct_author_generators'] != 1
            or latest['replication']['author_configuration_seed'] != 3):
        raise ValueError('The author comparison must remain conditional on one released generator.')
    if not author:
        display = read('final_public.json')
        if display['status'] != 'completed' or display['scientific_provenance'] != origin or display['seeds'] != [0, 1, 2]:
            raise ValueError('Formal presentation export is not the completed matched study.')
        for arm in arms:
            if not math.isclose(display['arms'][arm]['mean_mr'], mean_values[arm], abs_tol=1e-9, rel_tol=1e-10):
                raise ValueError('Formal presentation export contains another mean.')
    literals = {study + '_' + arm + '_mean_mr': numerical_literal(mean_values[arm]) for arm in arms}
    literals.update({study + '_primary_delta_pp': numerical_literal(primary['mean_pp']),
        study + '_primary_ci_lower_pp': numerical_literal(primary['nominal_paired_t95_pp'][0]),
        study + '_primary_ci_upper_pp': numerical_literal(primary['nominal_paired_t95_pp'][1])})
    return origin, literals


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def git(*args): return subprocess.check_output(['git'] + list(args), cwd=ROOT)


def target(value, source, prose_commit):
    if '://' in value: return value
    path_part, separator, fragment = value.partition('#')
    path = (source.parent / path_part).resolve()
    if not path.exists(): raise ValueError('Missing report reference: ' + value)
    relative = path.relative_to(ROOT).as_posix()
    version = REFERENCE_SNAPSHOT if relative in REFERENCE_INPUTS else prose_commit
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


def footer(canvas, doc, text='MonTi role specialization - Development report - 12 November 2026'):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#CCD8DF'))
    canvas.line(doc.leftMargin, 38, A4[0] - doc.rightMargin, 38)
    canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#506173'))
    canvas.drawString(doc.leftMargin, 25, text)
    canvas.drawRightString(A4[0] - doc.rightMargin, 25, str(doc.page))
    canvas.restoreState()


def main():
    global REFERENCE_SNAPSHOT, REFERENCE_INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='docs/STAGED_REPORT.md')
    parser.add_argument('--output', default='docs/MonTi_Staged_Report.pdf')
    parser.add_argument('--manifest', default='evidence/staged_report_build.json')
    parser.add_argument('--evaluation', choices=('development', 'final'), default='development')
    parser.add_argument('--data-snapshot', default=DATA_SNAPSHOT, help='Committed numerical snapshot; final mode requires completed published final exports.')
    parser.add_argument('--include-author', action='store_true', help='Include the separately completed one-author-generator comparison; final mode only.')
    parser.add_argument('--date', default='12 November 2026', help='ASCII display date in the page footer.')
    parser.add_argument('--footer-descriptor', help='Optional ASCII footer descriptor; final mode retains its full registered TEST label.')
    args = parser.parse_args()
    if args.include_author and args.evaluation != 'final': raise ValueError('--include-author requires --evaluation final.')
    data_snapshot = git('rev-parse', args.data_snapshot + '^{commit}').decode().strip()
    numerical_inputs = INPUTS + (FINAL_INPUTS if args.evaluation == 'final' else []) + (AUTHOR_INPUTS if args.include_author else [])
    REFERENCE_SNAPSHOT, REFERENCE_INPUTS = data_snapshot, set(numerical_inputs)
    source, output, manifest = [(ROOT / name).resolve() for name in (args.source, args.output, args.manifest)]
    for path in (source, output, manifest): path.relative_to(ROOT)
    prose_commit = git('rev-parse', 'HEAD').decode().strip()
    if git('show', prose_commit + ':' + source.relative_to(ROOT).as_posix()) != source.read_bytes():
        raise ValueError('Commit the reviewed report prose before rendering.')
    for name in numerical_inputs:
        if git('show', data_snapshot + ':' + name) != (ROOT / name).read_bytes():
            raise ValueError('Numerical evidence changed: ' + name)
    scientific_sources = {}
    required_literals = dict(development_native='53.6486', development_random='57.0344', development_staged='58.1436', development_gain='4.4950')
    if args.evaluation == 'final':
        scientific_sources['formal'], required_literals = completed_study('final_staged')
        if args.include_author:
            scientific_sources['author'], author_literals = completed_study('author_staged'); required_literals.update(author_literals)
    else:
        pilot = json.loads((ROOT / 'results/downstream_cooperation/latest.json').read_text(encoding='utf-8'))
        scientific_sources['development'] = pilot['provenance']
    descriptor = 'Full registered TEST study' if args.evaluation == 'final' else 'Development report'
    if args.footer_descriptor: descriptor = descriptor + ' / ' + args.footer_descriptor if args.evaluation == 'final' else args.footer_descriptor
    footer_text = 'MonTi role specialization - ' + descriptor + ' - ' + args.date
    if any(ord(char) > 127 for char in footer_text): raise ValueError('Use ASCII footer descriptor/date for standard PDF fonts.')
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
        subject=('Full registered TEST study of staged MonTi; supplementary comparison conditional on one author generator' if args.include_author
            else 'Full registered TEST study of staged MonTi' if args.evaluation == 'final'
            else 'Three-seed controlled validation study of a staged graph injection extension'))
    page_footer = partial(footer, text=footer_text)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer, canvasmaker=partial(Canvas, invariant=1))
    reader = pypdf.PdfReader(str(output)); texts = [page.extract_text() or '' for page in reader.pages]
    if len(texts) != breaks + 1: raise ValueError('Page overflow: expected %d, got %d' % (breaks + 1, len(texts)))
    if any(len(text.strip()) < 100 for text in texts): raise ValueError('Suspicious empty page.')
    if any(re.search(r'[\u3400-\u9fff]', text) for text in texts): raise ValueError('Unexpected CJK text.')
    for field, literal in required_literals.items():
        if literal not in '\n'.join(texts): raise ValueError('Missing four-decimal report value for ' + field + ': ' + literal)
    refs = list(re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', prose))
    for match in refs: target(match[2], source, prose_commit)
    inputs = numerical_inputs + [source.relative_to(ROOT).as_posix(), Path(__file__).relative_to(ROOT).as_posix()]
    data = dict(schema='monti_staged_report_build_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='PDF rendered; separate visual and factual review required', numerical_evidence_snapshot=data_snapshot,
        evaluation=args.evaluation, includes_author_comparison=args.include_author, report_scope=descriptor, footer_date=args.date,
        editorial_source_commit=prose_commit, scientific_execution_source=scientific_sources['formal' if args.evaluation == 'final' else 'development']['source_commit'],
        scientific_execution_sources=scientific_sources, required_four_decimal_values=required_literals,
        public_audit_summary_policy='Explicit passed public summaries bind retained full raw-audit SHA256 values; this build does not rerun an audit.',
        pages=len(texts), expected_pages=breaks + 1, reference_links_checked=len(refs),
        inputs={name: digest(ROOT / name) for name in inputs}, output={output.relative_to(ROOT).as_posix(): digest(output)},
        runtime=dict(python=sys.version, reportlab=reportlab.Version, pypdf=pypdf.__version__),
        scope='Report preparation from immutable saved evidence; no model, ML runtime, dataset/checkpoint loading, inference or GPU work.',
        link_policy='Scientific records use the immutable evidence snapshot. Editorial links use the committed reviewed prose snapshot.',
        pdf_timestamp_policy='Invariant ReportLab PDF metadata; actual build UTC recorded separately.')
    manifest.write_bytes((json.dumps(data, indent=2, ensure_ascii=True) + '\n').encode())
    print(json.dumps(dict(status='rendered', pages=len(texts), output=str(output.relative_to(ROOT)), sha256=digest(output))))


if __name__ == '__main__': main()
