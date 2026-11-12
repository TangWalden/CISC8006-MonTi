"""Create a new editable Bento research talk, without touching prior decks.

Uses only byte-verified saved pilot evidence and an optional completed formal
public export. No models, browser, network, secrets or collaboration keys.
"""
import argparse
import csv
import datetime
import html
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


_spec = importlib.util.spec_from_file_location('_bento_base', Path(__file__).with_name('build_bento_slides.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
normalize, require, sha, DOC_RE = _base.normalize, _base.require, _base.sha, _base.DOC_RE
ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720
NAVY, BG, INK, ORANGE = '#132B38', '#FAF8F2', '#142D3A', '#ED7645'
GRAY, PALE, WHITE = '#637078', '#D9DDD9', '#FFFFFF'
FONT = 'Segoe UI, sans-serif'
REPO = 'https://github.com/TangWalden/CISC8006-MonTi'
PILOT_COMMIT = '70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980'
PAPER = 'https://arxiv.org/html/2412.18370v3'
ARMS = ('native_monti', 'quota_random', 'quota_learned')


def git_bytes(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=str(ROOT))


def source_url(ref, path):
    git_bytes(ref, path)
    return REPO + '/blob/' + ref + '/' + path


def text(key, value, x, y, w, h, size=28, color=INK, weight=400, **kwargs):
    return dict(id=key, type='text', x=x, y=y, w=w, h=h,
                html=html.escape(str(value)).replace('\n', '<br>'), fontFamily=FONT,
                fontSize=size, fontWeight=weight, color=color, lineHeight=1.16,
                valign='middle', **kwargs)


def shape(key, x, y, w, h, fill=BG, stroke='transparent', sw=0, kind='rect', **kwargs):
    return dict(id=key, type='shape', shape=kind, x=x, y=y, w=w, h=h,
                fill=fill, stroke=stroke, strokeWidth=sw, radius=0, **kwargs)


def rule(key, x, y, w, color=PALE):
    return shape(key, x, y, w, 2, color, kind='line')


def connect(key, src, dst, color=INK, width=3, arrow=False, dashed=False):
    item = shape(key, 0, 0, 1, 1, color, color, width, kind='line',
                 **{'from': dict(el=src, side='auto'), 'to': dict(el=dst, side='auto')})
    if arrow:
        item['lineEnd'] = 'arrow'
    if dashed:
        item['strokeStyle'] = 'dashed'
    return item


def label(key, value, x, y, w, color=GRAY):
    return text(key, value, x, y, w, 30, 21, color)


def base_slide(key, title, number, dark=False, transition='fade'):
    color = BG if dark else INK
    return dict(id=key, name=title.replace('\n', ' '), background=NAVY if dark else BG,
                transition=transition, elements=[text('title', title, 96, 62, 1088, 126, 46, color, 600),
                    text('page-number', '%02d' % number, 1124, 656, 60, 28, 20, '#9AA7AC' if dark else GRAY, 400, align='right')])


def notes(slide, script, sources, speaker=None):
    count = len(re.findall(r"\b[\w'-]+\b", script))
    require(60 <= count <= 100, 'Spoken script must contain 60–100 words: ' + slide['id'] + ' (%d).' % count)
    slide['notes'] = ('Planned speaker: ' + (speaker or 'Optional appendix; either presenter') +
                      '. This is a speaking plan, not a record of actual contribution or rehearsal.\n\nScript:\n' +
                      script + '\n\nSources:\n' + '\n'.join(sources))
    return count


def graph(x, y, scale=1, mode='joint', dark=False, future=False):
    """Schematic article graph; identical node IDs support native morph."""
    pos = {'target-0': (0, 38), 'target-1': (115, 0), 'target-2': (0, 220), 'target-3': (115, 180),
           'auxiliary-0': (270, 104), 'candidate-0': (420, 0), 'candidate-1': (535, 48),
           'candidate-2': (425, 150), 'candidate-3': (535, 240), 'candidate-4': (365, 275)}
    out = []
    context = [('target-0', 'target-1'), ('target-2', 'target-3'), ('candidate-0', 'candidate-1'),
               ('candidate-1', 'candidate-2'), ('candidate-2', 'candidate-4'), ('candidate-4', 'candidate-3')]
    for i, (a, b) in enumerate(context):
        out.append(connect('context-edge-' + str(i), a, b, '#657680' if dark else '#BEC5C5', 2))
    if not future:
        for i, target in enumerate(('target-0', 'target-3')):
            out.append(connect('target-edge-' + str(i), 'auxiliary-0', target, '#CED5D5' if dark else INK, 4))
        if mode in ('joint', 'complete'):
            for i, candidate in enumerate(('candidate-0', 'candidate-2', 'candidate-4')):
                out.append(connect('candidate-edge-' + str(i), 'auxiliary-0', candidate, ORANGE, 4))
    for key, (px, py) in pos.items():
        xx, yy = x + px * scale, y + py * scale
        if key.startswith('target'):
            size = 49 * scale
            out.append(shape(key, xx, yy, size, size, ORANGE, kind='ellipse'))
            out.append(text(key + '-label', 'T' + str(int(key[-1]) + 1), xx-4, yy, size+8, size, 21, NAVY, 600, align='center'))
        elif key.startswith('candidate'):
            size = 42 * scale
            out.append(shape(key, xx, yy, size, size, NAVY if dark else BG, '#B5C0C5' if dark else GRAY, 3, kind='ellipse'))
            out.append(text(key + '-label', 'C' + str(int(key[-1]) + 1), xx-3, yy+size+3, size+6, 26, 20, '#CBD3D5' if dark else GRAY, align='center'))
        else:
            size = 58 * scale
            out.append(shape(key, xx, yy, size, size, 'transparent' if future else ('#E8ECEB' if dark else NAVY),
                             '#B0BCC2' if future else 'transparent', 3 if future else 0,
                             strokeStyle='dashed' if future else 'solid'))
            out.append(text(key + '-label', 'U', xx, yy, size, size, 26, '#D9E1E3' if future else (NAVY if dark else BG), 600, align='center'))
    return out


def chart(key, categories, values, x, y, w, h, maximum=100, unit='%'):
    return dict(id=key, type='chart', preset='bar', x=x, y=y, w=w, h=h,
                option=dict(color=[ORANGE], textStyle=dict(fontFamily=FONT, color=INK),
                    grid=dict(left=65, right=20, top=40, bottom=70),
                    xAxis=dict(type='category', data=categories, axisLabel=dict(fontSize=22, color=INK),
                               axisLine=dict(show=True, lineStyle=dict(color='#BFC7C6'))),
                    yAxis=dict(type='value', min=0, max=maximum, axisLabel=dict(fontSize=20, color=GRAY, formatter='{value}' + unit),
                               splitLine=dict(show=True, lineStyle=dict(color='#E0E4E0'))),
                    series=[dict(type='bar', data=[float(v) for v in values], barWidth='45%',
                                 itemStyle=dict(color=ORANGE))],
                    tooltip=dict(trigger='item', formatter='{b}: {c}' + unit)))


def source_inputs(pilot_path, metrics_path, protocol_path):
    pilot_raw, metrics_raw = pilot_path.read_bytes(), metrics_path.read_bytes()
    provenance_path = pilot_path.with_name('provenance.json')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for path, raw in ((pilot_path, pilot_raw), (metrics_path, metrics_raw)):
        require(sha(raw) == provenance['output_sha256'][path.name], 'Pilot publication hash mismatch: ' + path.name)
        relative = path.resolve().relative_to(ROOT).as_posix()
        require(git_bytes(PILOT_COMMIT, relative) == raw, 'Pilot input differs from its cited snapshot: ' + relative)
    pilot = json.loads(pilot_raw)
    rows = list(csv.DictReader(io.StringIO(metrics_raw.decode('utf-8-sig'))))
    require(pilot['seeds'] == [3, 4, 5] and len(rows) == 12, 'Expected the completed three-seed development pilot.')
    for arm in ARMS:
        values = [float(next(row for row in rows if row['arm'] == arm and int(row['seed']) == s)['gaga_mr_percent']) for s in pilot['seeds']]
        _base.close(statistics.mean(values), pilot['gaga_mean_mr_percent'][arm])
        if arm != 'quota_learned':
            learned = [float(next(row for row in rows if row['arm'] == 'quota_learned' and int(row['seed']) == s)['gaga_mr_percent']) for s in pilot['seeds']]
            for a, b in zip([a-b for a, b in zip(learned, values)], pilot['contrasts'][arm]['seed_differences_pp']):
                _base.close(a, b)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_commit = subprocess.check_output(['git', 'rev-parse', 'd129b22'], cwd=str(ROOT), text=True).strip()
    require(git_bytes(protocol_commit, 'evidence/final_staged_v1/protocol.json') == protocol_path.read_bytes(),
            'Formal registration differs from its canonical published snapshot.')
    hashes = {str(path.resolve()): sha(path.read_bytes()) for path in (pilot_path, metrics_path, provenance_path, protocol_path)}
    return pilot, protocol, protocol_commit, hashes


def deck(pilot, formal, protocol_commit):
    pilot_url = source_url(PILOT_COMMIT, 'results/downstream_cooperation/latest.json')
    csv_url = source_url(PILOT_COMMIT, 'results/downstream_cooperation/seed_metrics.csv')
    formal_url = source_url(protocol_commit, 'evidence/final_staged_v1/protocol.json')
    method_url = source_url(PILOT_COMMIT, 'experiments/downstream_cooperation/README.md')
    source = [PAPER, method_url]
    metrics = pilot['gaga_mean_mr_percent']
    primary, random = pilot['contrasts']['native_monti'], pilot['contrasts']['quota_random']
    slides, word_counts = [], {}

    s = base_slide('cover', 'MonTi with two\nconstruction roles', 1, dark=True)
    s['elements'][0].update(x=96, y=112, w=675, h=180, fontSize=58)
    s['elements'] += [text('subtitle', 'Upstream planning and\ndownstream completion', 100, 312, 640, 100, 30, '#D9E1E3'),
        label('course', 'CISC8006 · GossipCop-S', 100, 606, 580, '#CAD4D6'),
        label('names', 'Junyi Tang  /  Yifan Fan', 100, 561, 580, '#CAD4D6'),
        shape('cover-upstream', 855, 227, 142, 112, '#E9EDE9'),
        text('cover-upstream-job', 'Features\n+ targets', 863, 239, 126, 88, 23, NAVY, 600, align='center'),
        text('cover-upstream-label', 'UPSTREAM', 843, 351, 166, 36, 21, BG, 600, align='center'),
        shape('cover-downstream', 1040, 420, 142, 112, ORANGE),
        text('cover-downstream-job', 'Candidate\nconnections', 1048, 432, 126, 88, 21, NAVY, 600, align='center'),
        text('cover-downstream-label', 'DOWNSTREAM', 1020, 546, 184, 36, 21, BG, 600, align='center'),
        connect('cover-flow', 'cover-upstream', 'cover-downstream', ORANGE, 4, arrow=True, dashed=True)]
    s['elements'][-1]['fx'] = dict(loop=dict(type='dash-march', distance=18, duration=3))
    word_counts[s['id']] = notes(s, 'This talk starts with MonTi, a model that constructs extra news articles and their connections to stress-test a fraud detector. We study a different construction workflow. A frozen upstream model commits the article features and target connections. A small downstream model then completes the connections to surrounding candidates. The question is whether this division of work helps under the same total budget. We will separate a completed development pilot from the registered full test.', source, 'Junyi Tang')
    slides.append(s)

    s = base_slide('group', 'A group is a fixed list of fraud targets', 2)
    s['elements'] += graph(125, 258, 1.08, future=True)
    s['elements'] += [text('all-fake', 'Targets are\nfake-news articles.', 885, 264, 295, 105, 30, INK, 600),
        text('candidate-role', 'Candidates provide\nsurrounding context.\nTheir labels may differ.', 885, 391, 295, 138, 26),
        label('diagram-note', 'Illustrative graph · U is a future injected article', 96, 623, 930)]
    word_counts[s['id']] = notes(s, 'A group is simply a list of target articles evaluated together. Here, the four orange circles are fake-news articles that the detector should identify. The outlined circles are surrounding candidate articles; they are not automatically benign. The square represents a new article that an attack may introduce. The attack does not get to replace the target list after seeing the outcome. This drawing explains the roles only; it is not an actual measured group.', source, 'Junyi Tang')
    slides.append(s)

    s = base_slide('original', 'MonTi learns features and connections together', 3)
    s['elements'] += [text('inputs', 'Target group\n+ candidate context', 96, 250, 270, 110, 28),
        shape('transformer', 400, 245, 220, 145, NAVY),
        text('transformer-label', 'MonTi\nTransformer', 414, 265, 192, 102, 28, BG, 600, align='center'),
        shape('input-anchor', 330, 290, 3, 3, 'transparent'),
        shape('output-anchor', 685, 315, 3, 3, 'transparent'),
        connect('input-flow', 'input-anchor', 'transformer', INK, 3, True),
        connect('output-flow', 'transformer', 'output-anchor', INK, 3, True)]
    s['elements'] += graph(755, 275, .67)
    s['elements'] += [text('joint-caption', 'One generator chooses new features\nand target / candidate connections.', 96, 477, 660, 104, 29),
        label('diagram-note', 'Schematic pipeline · graph connections are undirected', 96, 623, 1000)]
    word_counts[s['id']] = notes(s, 'MonTi is a learned graph-injection attack. It reads representations of the target group and selected surrounding candidates, mixes this context with a Transformer, and generates features for new articles. It also scores possible connections. Those connections can lead to targets or to candidates. Training uses a surrogate detector to learn constructions that cause more fake targets to be missed. The original idea already combines several targets and coordinated injected nodes; our proposal changes how construction work is divided.', [PAPER, method_url], 'Junyi Tang')
    slides.append(s)

    s = base_slide('question', 'Can a specialist complete\na frozen MonTi plan?', 4, dark=True, transition='morph')
    s['elements'] += [label('original-small', 'ORIGINAL', 96, 250, 300, '#BFCBD0'),
        text('original-line', 'Jointly choose features\nand all connections', 96, 293, 620, 100, 31, BG),
        rule('method-divider', 96, 418, 620, '#526874'),
        label('staged-small', 'PROPOSED WORKFLOW', 96, 451, 600, '#BFCBD0'),
        text('staged-line', 'Commit the upstream plan.\nTrain only its completion.', 96, 494, 640, 100, 31, BG)]
    s['elements'] += graph(827, 283, .55, mode='upstream', dark=True)
    word_counts[s['id']] = notes(s, 'The question is operational, rather than a claim about real criminal job roles. Instead of asking one generator to make every decision at once, can a specialist complete a plan that another model has already fixed? The upstream keeps its original weights. Its generated article features and the selected target connections become commitments. The downstream may choose only the remaining candidate connections. Both stages still use one shared node budget and one shared total edge budget.', source + [formal_url], 'Junyi Tang')
    slides.append(s)

    s = base_slide('roles', 'A fixed upstream and a trained downstream', 5, transition='morph')
    s['elements'] += graph(125, 258, 1.04, mode='complete')
    s['elements'] += [label('frozen-label', 'UPSTREAM · FROZEN', 886, 255, 298),
        text('frozen-copy', 'Article features\n+ target connections', 886, 292, 298, 100, 28, INK, 600),
        label('trained-label', 'DOWNSTREAM · TRAINED', 886, 425, 298, ORANGE),
        text('trained-copy', 'Candidate connections\n151,872 parameters', 886, 466, 298, 98, 28, INK, 600),
        label('diagram-note', 'Illustrative graph · a small specialist, not a second full MonTi', 96, 623, 1088)]
    word_counts[s['id']] = notes(s, 'In this workflow, the upstream produces the new article features and scores the possible edges. We freeze those features and construct the target-edge block within the agreed quota. The downstream is a smaller scorer with 151,872 trainable parameters. It receives the frozen construction context and selects candidate connections for the remaining quota. It does not generate another set of articles, change the upstream weights, or add a second budget. The dark and orange lines show two construction roles, not directed graph edges.', [method_url, formal_url], 'Junyi Tang')
    slides.append(s)

    s = base_slide('budget', 'The total budget stays fixed', 6)
    s['elements'] += [text('budget-left-title', '4 target edges + 4 candidate edges', 96, 218, 514, 70, 28, INK, 600),
        text('budget-right-title', '2 target edges + 6 candidate edges', 670, 218, 514, 70, 28, INK, 600),
        rule('budget-divider', 638, 255, 0, PALE)]
    s['elements'][-1].update(h=290)
    for side, bx, nt, nc in (('left', 96, 4, 4), ('right', 670, 2, 6)):
        u = side + '-u'
        for i in range(nt):
            tid = side + '-t-' + str(i)
            xx = bx + 42 + i * (410 / max(nt-1, 1))
            s['elements'] += [connect(side + '-te-' + str(i), u, tid, INK, 3),
                shape(tid, xx, 315, 36, 36, ORANGE, kind='ellipse')]
        for i in range(nc):
            cid = side + '-c-' + str(i)
            xx = bx + 22 + i * (452 / max(nc-1, 1))
            s['elements'] += [connect(side + '-ce-' + str(i), u, cid, ORANGE, 3),
                shape(cid, xx, 516, 34, 34, BG, GRAY, 3, kind='ellipse')]
        s['elements'] += [shape(u, bx+232, 408, 52, 52, NAVY),
            text(side+'-u-label', 'U', bx+232, 408, 52, 52, 25, BG, 600, align='center')]
    s['elements'] += [text('budget-message', 'Aim for half. Project to the closest feasible split.', 96, 584, 1088, 46, 29, INK, 600),
        label('diagram-note', 'Illustrative 8-edge examples · unique edges · no injected-to-injected edges', 96, 639, 1050)]
    word_counts[s['id']] = notes(s, 'The total structural node and edge budget is fixed, but a perfect half split is not always possible. In the first illustration, one new article has four available targets and at least four candidates, so eight edges can split four and four. In the second, only two targets are available. Duplicating a target edge is forbidden, so the same eight-edge budget becomes two target edges and six candidate edges. The actual algorithm chooses the closest feasible half split and reports the realized ratio.', [method_url, formal_url], 'Junyi Tang')
    slides.append(s)

    s = base_slide('train-transfer', 'GCN teaches the downstream.\nGAGA tests transfer.', 7)
    s['elements'] += [label('training-phase', 'TRAIN · FOUR EPOCHS', 96, 223, 520),
        shape('small-scorer', 96, 290, 280, 112, NAVY),
        text('small-scorer-label', 'Downstream scorer', 106, 312, 260, 66, 28, BG, 600, align='center'),
        shape('gcn', 486, 290, 238, 112, BG, INK, 3),
        text('gcn-label', 'Frozen GCN', 500, 312, 210, 66, 28, INK, 600, align='center'),
        connect('training-forward', 'small-scorer', 'gcn', INK, 3, True),
        shape('feedback', 236, 435, 365, 4, ORANGE, ORANGE, 3, kind='line', lineEnd='arrow', rotation=180),
        label('feedback-label', 'Training feedback', 306, 461, 370, ORANGE),
        shape('seal', 819, 290, 135, 112, '#E4E8E3'),
        text('seal-label', 'SEAL', 829, 312, 115, 66, 25, INK, 600, align='center'),
        connect('seal-flow', 'gcn', 'seal', INK, 3, True),
        shape('gaga', 1049, 290, 135, 112, NAVY),
        text('gaga-label', 'GAGA', 1059, 312, 115, 66, 25, BG, 600, align='center'),
        connect('evaluation-flow', 'seal', 'gaga', INK, 3, True),
        text('sealed-description', 'Final weights and graphs are sealed\nbefore GAGA evaluation.', 819, 465, 365, 106, 27),
        label('no-feedback', 'GAGA supplies evaluation only, with no training feedback.', 96, 623, 1088)]
    word_counts[s['id']] = notes(s, 'The downstream learns from a frozen GCN on the training groups. GCN supplies a differentiable training signal, while its own parameters stay unchanged. We train the downstream for four fixed epochs, then seal its final weights and the evaluation graphs. GAGA is a separate detector used to measure transfer after that commitment. There is no arrow back from GAGA: its evaluation outcomes do not train the scorer or select a checkpoint. This separation matters because learning and evaluation answer different questions.', [method_url, formal_url], 'Yifan Fan')
    slides.append(s)

    s = base_slide('comparison', 'Three methods share the same cases and budgets', 8)
    headers = ['Method', 'Target connections', 'Candidate connections', 'New training']
    values = [headers,
        ['Native MonTi', 'Native plan', 'Native plan', 'None'],
        ['Quota + random', 'Fixed quota block', 'Random choice', 'None'],
        ['Staged pipeline', 'Same quota block', 'Learned choice', 'Downstream only']]
    rows = []
    for i, values_row in enumerate(values):
        rows.append(dict(cells=[dict(html=html.escape(v), bold=(i == 0), color=BG if i == 0 else INK,
                                    bg=NAVY if i == 0 else (BG if i != 3 else '#F5E8DE')) for v in values_row]))
    s['elements'] += [dict(id='comparison-table', type='table', x=96, y=238, w=1088, h=292,
        header=True, columns=[dict(w=1.1), dict(w=1.35), dict(w=1.5), dict(w=1.2)], rows=rows,
        style=dict(headerBg=NAVY, headerColor=BG, borderColor='#D5DCD8', borderWidth=1,
                   cellPadX=18, cellPadY=21, fontSize=24, color=INK, radius=0)),
        text('shared-features', 'All three reuse the same upstream article features.', 96, 570, 1088, 49, 29),
        label('fairness', 'Same group list, node budget and total edge budget', 96, 627, 1050)]
    word_counts[s['id']] = notes(s, 'The main comparison has three methods. Native MonTi keeps its own connection plan. The random completion control uses the same projected quotas and frozen target block as the staged method, but picks candidate connections randomly. The staged method learns those candidate choices. All three reuse the upstream article features and the same total node and edge budgets. The completed pilot additionally retained an original-score completion arm in its saved evidence. This table presents the three methods registered for the full test.', [pilot_url, csv_url, formal_url], 'Yifan Fan')
    slides.append(s)

    s = base_slide('pilot-results', 'The pilot improved miss rate\nover our MonTi baseline', 9)
    vals = [metrics[a] for a in ARMS]
    s['elements'] += [label('mr-direction', 'GAGA miss rate (%) · higher means more missed fraud', 96, 206, 1088),
        chart('pilot-mr', ['Native MonTi', 'Quota + random', 'Staged'], vals, 96, 250, 790, 326),
        text('pilot-gain', '+%.2f' % primary['mean_pp'], 915, 305, 269, 100, 76, ORANGE, 600),
        text('pilot-gain-unit', 'percentage points\nvs native MonTi', 921, 414, 263, 92, 28),
        text('pilot-cohort', 'Completed development VALID pilot · 64 groups · seeds 3, 4, 5', 96, 594, 1088, 40, 24),
        text('mr-click', 'Miss rate (MR) · click for an example ↗', 96, 644, 750, 30, 21, GRAY, link='mr-example')]
    for i, value in enumerate(vals):
        s['elements'].append(text('pilot-value-'+str(i), '%.2f%%' % value,
            96+65+(790-85)/3*(i+.5)-65, 250+40+(100-value)/100*(326-110)-40, 130, 38, 25, INK, 600, align='center'))
    word_counts[s['id']] = notes(s, 'These are the completed development results on the fixed GAGA detector. Higher miss rate means more fake target articles escape detection. Our native MonTi baseline averages 53.65 percent, the random candidate completion reaches 57.03 percent, and the trained staged pipeline reaches 58.14 percent. The staged gain over native is 4.50 percentage points. These are our own baseline weights on 64 validation groups per seed, not the paper table or the formal test. Click the miss-rate label for a concrete counting example.', [pilot_url, csv_url], 'Yifan Fan')
    slides.append(s)

    s = base_slide('pilot-seeds', 'All three pilot seeds improved\nover native MonTi', 10)
    diffs = primary['seed_differences_pp']
    s['elements'] += [chart('pilot-seed-gains', ['Seed 3', 'Seed 4', 'Seed 5'], diffs, 96, 250, 675, 331, maximum=8, unit=' pp'),
        text('paired-heading', 'Paired mean: +%.2f pp' % primary['mean_pp'], 815, 280, 369, 60, 30, INK, 600),
        text('paired-ci', 'Nominal 95%% interval\n[+%.2f, +%.2f] pp' % tuple(primary['nominal_paired_t95_pp']), 815, 350, 369, 100, 27),
        rule('seed-divider', 815, 483, 369),
        text('random-contrast', 'Vs random: +%.2f pp\n95%% interval [%.2f, +%.2f] pp\nStill uncertain with three seeds.' %
             (random['mean_pp'], *random['nominal_paired_t95_pp']), 815, 513, 369, 132, 25),
        label('pilot-scope', 'Development VALID pilot · three paired training seeds', 96, 633, 700)]
    for i, value in enumerate(diffs):
        s['elements'].append(text('seed-value-'+str(i), '+%.2f' % value,
            96+65+(675-85)/3*(i+.5)-58, 250+40+(8-value)/8*(331-110)-40, 116, 38, 25, INK, 600, align='center'))
    word_counts[s['id']] = notes(s, 'The three paired pilot gains over native MonTi are 6.13, 4.38, and 2.98 percentage points. Their mean is positive, and the nominal paired interval is above zero. The harder question is whether learning candidate connections helps beyond random completion. That average difference is 1.11 points, but its interval crosses zero and one seed is negative. We therefore have encouraging development evidence for the full staged workflow, while the learning-specific advantage remains uncertain in this small pilot.', [pilot_url, csv_url], 'Yifan Fan')
    slides.append(s)

    if not formal['completed']:
        s = base_slide('formal-test', 'The full test uses\nthe same three-way comparison', 11)
        for x, value, caption in ((96, formal['protocol']['train_groups'], 'TRAIN groups'),
                                   (484, formal['protocol']['test_groups'], 'TEST groups'), (872, 3, 'training seeds')):
            s['elements'] += [text('formal-number-'+str(x), format(value, ','), x, 280, 310, 110, 76, INK, 600),
                text('formal-caption-'+str(x), caption, x, 399, 310, 44, 28)]
        s['elements'] += [text('formal-occurrences', '%s target occurrences per seed · 4 downstream epochs' % format(formal['protocol']['test_target_occurrences'], ','), 96, 515, 1088, 55, 28),
            label('formal-pending', 'Registered full TEST evaluation: results pending', 96, 607, 1088, ORANGE)]
        script = 'The formal evaluation is registered separately from the development pilot. It uses 937 training groups, 909 test groups, and seeds zero, one, and two, with 10,515 target occurrences per seed. It keeps the same three-way comparison and four fixed downstream epochs. Formal upstream checkpoints follow the frozen selection policy, rather than reusing the pilot checkpoints. This slide currently reports the registered design only. The final audited export will supply the full-test results, without substituting the pilot scores.'
    else:
        s = base_slide('formal-test', 'The registered full test: measured results', 11)
        fvals = [formal['arms'][a]['mean_mr'] for a in ARMS]
        s['elements'] += [chart('formal-mr', ['Native MonTi', 'Quota + random', 'Staged'], fvals, 96, 246, 790, 330),
            text('formal-gain', '%+.2f pp' % formal['primary']['mean_pp'], 920, 295, 264, 80, 48, ORANGE, 600),
            text('formal-ci', 'Vs native MonTi\n95%% interval\n[%.2f, %.2f] pp' % tuple(formal['primary']['ci95_pp']), 920, 399, 264, 133, 25),
            label('formal-scope', 'Formal TEST · 909 groups · seeds 0, 1, 2 · 10,515 occurrences per seed', 96, 614, 1088)]
        for i, value in enumerate(fvals):
            s['elements'].append(text('formal-value-'+str(i), '%.2f%%' % value, 96+65+(790-85)/3*(i+.5)-65,
                246+40+(100-value)/100*(330-110)-40, 130, 38, 25, INK, 600, align='center'))
        script = ('This slide uses only the completed formal public export, kept separate from the earlier pilot. The three methods are evaluated on the same 909 test groups and the registered seeds zero, one, and two. The displayed miss rates are the three-seed means. The paired difference and interval compare the staged workflow with our native MonTi baseline. These full-test results describe this registered setting and its fixed budgets; they do not establish a universal advantage or reproduce every original paper table.')
    word_counts[s['id']] = notes(s, script, [formal_url, 'Completed formal results, if attached: local results/final_staged/final_public.json; this new export is not assigned an invented publication URL.'], 'Yifan Fan')
    slides.append(s)

    s = base_slide('conclusion', 'Frozen upstream + trained downstream', 12, dark=True)
    s['elements'] += [text('conclusion-one', 'A reusable MonTi planner\nwith a specialized edge policy', 96, 246, 1088, 145, 40, BG, 600),
        text('conclusion-two', 'Development pilot: +%.2f pp over native.\nLearning vs random remains uncertain.' % primary['mean_pp'], 96, 436, 1088, 100, 30, '#D9E1E3'),
        label('conclusion-formal', 'Full-test results pending; the registered design is fixed.' if not formal['completed'] else
              'Formal TEST vs native: %+.2f pp; interval [%.2f, %.2f] pp.' % (formal['primary']['mean_pp'], *formal['primary']['ci95_pp']), 96, 575, 1088, '#D9E1E3'),
        label('repository', 'github.com/TangWalden/CISC8006-MonTi', 96, 631, 1040, '#CBD5D8')]
    s['elements'][-1]['html'] = '<a href="' + REPO + '" style="color:#CBD5D8">github.com/TangWalden/CISC8006-MonTi ↗</a>'
    word_counts[s['id']] = notes(s, 'Our proposal is a staged construction workflow: freeze an upstream MonTi plan, then train a smaller model to complete the candidate connections. It preserves the total attack budget and compares against both native MonTi and random completion. The development pilot supports further evaluation of the whole workflow, but does not settle whether learned completion reliably beats random choice. The report and repository retain the complete evidence, including less favorable controls. The registered full test remains a separate result, with its actual status stated explicitly.', [pilot_url, formal_url, REPO], 'Yifan Fan')
    slides.append(s)

    s = base_slide('mr-example', 'Miss rate: a four-target example', 13)
    s['elements'][1]['html'] = 'A1'
    s['stateOf'] = 'pilot-results'
    s['elements'] += [text('before-caption', 'Before attack', 96, 235, 470, 45, 29, INK, 600),
        text('after-caption', 'After attack', 692, 235, 470, 45, 29, INK, 600)]
    for prefix, bx, wrong in (('before', 96, 1), ('after', 692, 2)):
        for i in range(4):
            xx = bx + i*116
            s['elements'] += [shape(prefix+'-target-'+str(i), xx, 335, 64, 64, ORANGE, kind='ellipse'),
                text(prefix+'-target-label-'+str(i), 'T'+str(i+1), xx, 335, 64, 64, 24, NAVY, 600, align='center'),
                text(prefix+'-prediction-'+str(i), 'miss' if i < wrong else 'detect', xx-15, 418, 94, 36, 22, INK, 600 if i < wrong else 400, align='center')]
        s['elements'].append(text(prefix+'-mr', '%d / 4 = %d%% MR' % (wrong, wrong*25), bx, 507, 480, 60, 36, INK, 600))
    s['elements'] += [label('mr-distinction', 'All four articles are fake · total MR after attack = 50% · newly missed = 25%', 96, 596, 1088),
        text('return', '↖ Back to pilot results', 96, 644, 930, 30, 21, GRAY, link='pilot-results')]
    word_counts[s['id']] = notes(s, 'All four target articles are fake in this illustrative group. Before the attack, one is already missed, so the clean miss rate is one out of four, or 25 percent. After the attack, two are missed, giving a total miss rate of 50 percent. Only one of those misses is new, so the newly missed proportion is 25 percent. The paper-style total miss-rate metric includes existing clean misses. We retain every target in the denominator rather than filtering to favorable cases.', [pilot_url, PAPER])
    slides.append(s)

    s = base_slide('quota-evidence', 'Quota rule and evidence sources', 14)
    s['elements'][1]['html'] = 'A2'
    s['stateOf'] = 'conclusion'
    s['elements'] += [text('quota-formula', 'qT = clip(ceil(E / 2),\n                 max(B, E − B·C), min(B·T, E))\nqC = E − qT', 96, 215, 1088, 151, 30, INK, 600),
        label('quota-definitions', 'B injected articles · T targets · C candidates · E total edges', 96, 389, 1088),
        rule('evidence-divider', 96, 446, 1088),
        text('evidence-entries', '1. MonTi paper — original method\n2. Versioned pilot tables — measured development results\n3. Frozen full-test protocol — registered comparison', 96, 473, 1088, 143, 27),
        text('appendix-return', '↖ Back to conclusion', 96, 644, 930, 30, 21, GRAY, link='conclusion')]
    word_counts[s['id']] = notes(s, 'The formula starts from half of the total edge budget, rounded toward targets. It then projects into a feasible interval. Each new article needs at least one target connection; the total available target and candidate pairs impose the other bounds. The remaining edges go to candidates, without duplicate pairs or extra injected-to-injected edges in the completion methods. Three source categories support this talk: the original paper, the versioned completed pilot evidence, and the frozen full-test registration. Exact links are recorded here.', [PAPER, pilot_url, csv_url, formal_url])
    slides.append(s)
    slides[11]['elements'].append(text('appendix-link', 'Quota + evidence ↗', 920, 631, 264, 30, 21, '#CBD5D8', align='right', link='quota-evidence'))
    return dict(format='bento/slides', version=1, title='MonTi with two construction roles',
        size=dict(width=W, height=H), theme=dict(background=BG, color=INK, accent=ORANGE, fontFamily=FONT),
        meta=dict(author='CISC8006 project', subject='Staged graph-injection construction',
                  event='CISC8006', description='Research presentation draft. Planned speaking roles; not a contribution or rehearsal record.'),
        slides=slides), word_counts


def validate(doc):
    require(len(doc['slides']) == 14 and doc['size'] == dict(width=1280, height=720), 'Expected the fixed 14-slide canvas.')
    slide_ids = [s['id'] for s in doc['slides']]
    require(len(set(slide_ids)) == len(slide_ids), 'Duplicate slide ID.')
    for s in doc['slides']:
        ids = [e['id'] for e in s['elements']]
        require(len(ids) == len(set(ids)), 'Duplicate element ID: ' + s['id'])
        for e in s['elements']:
            for key in ('x', 'y', 'w', 'h'):
                require(type(e[key]) in (int, float) and math.isfinite(e[key]), 'Invalid geometry.')
            require(e['w'] >= 0 and e['h'] >= 0, 'Negative element dimensions.')
            if e.get('type') != 'shape' or not ('from' in e or 'to' in e):
                require(e['x'] >= 0 and e['y'] >= 0 and e['x']+e['w'] <= W+0.1 and e['y']+e['h'] <= H+0.1,
                        'Element outside the canvas: ' + s['id'] + '/' + e['id'])
            for end in ('from', 'to'):
                if end in e:
                    require(e[end]['el'] in ids, 'Connector target is missing.')
            if e.get('link'):
                require(e['link'] in slide_ids, 'Unresolved slide link.')
            if e['type'] == 'chart':
                require(all(type(v) in (int, float) for series in e['option']['series'] for v in series['data']), 'Chart data must be plain numbers.')
    require(not any(k in doc for k in ('docId', 'collab', 'readonly', 'template')), 'Fresh deck includes identity or collaboration state.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allow-pending', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT/'results/final_staged/final_public.json')
    parser.add_argument('--protocol', type=Path, default=ROOT/'evidence/final_staged_v1/protocol.json')
    parser.add_argument('--pilot', type=Path, default=ROOT/'results/downstream_cooperation/latest.json')
    parser.add_argument('--pilot-metrics', type=Path, default=ROOT/'results/downstream_cooperation/seed_metrics.csv')
    args = parser.parse_args()
    outputs = [p.resolve() for p in (args.output, args.json_output, args.manifest)]
    require(len(set(outputs)) == 3 and all(not p.exists() for p in outputs), 'Every output must be a distinct new file.')
    shell_raw = args.shell.read_bytes()
    shell = shell_raw.decode('utf-8')
    matches = list(DOC_RE.finditer(shell))
    require(len(matches) == 1, 'The official shell must contain exactly one bento-doc block.')
    match = matches[0]
    existing = json.loads(match.group(2)) if match.group(2).strip() else {}
    require(not existing.get('collab'), 'Refusing a shell containing collaboration state.')
    pilot, frozen, protocol_commit, inputs = source_inputs(args.pilot.resolve(), args.pilot_metrics.resolve(), args.protocol.resolve())
    formal_raw = json.loads(args.data.read_text(encoding='utf-8')) if args.data.exists() else None
    formal = normalize(formal_raw, args.allow_pending, frozen)
    if args.data.exists():
        inputs[str(args.data.resolve())] = sha(args.data.read_bytes())
    doc, word_counts = deck(pilot, formal, protocol_commit)
    for slide in doc['slides']:
        for element in slide['elements']:
            element.setdefault('rotation', 0)
            element.setdefault('opacity', 1)
            if element['type'] == 'text':
                element.setdefault('align', 'left')
    validate(doc)
    payload = json.dumps(doc, ensure_ascii=True, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    rendered = shell[:match.start(2)] + payload + shell[match.end(2):]
    rendered_match = DOC_RE.search(rendered)
    require(shell[:match.start(2)] == rendered[:rendered_match.start(2)] and shell[match.end(2):] == rendered[rendered_match.end(2):], 'Official shell changed outside its document block.')
    html_raw = rendered.encode('utf-8')
    json_raw = (json.dumps(doc, ensure_ascii=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    for path, raw in ((args.output, html_raw), (args.json_output, json_raw)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)
    manifest = dict(schema='monti_bento_story_build_v2', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='Built; awaiting root visual review', builder_sha256=sha(Path(__file__).read_bytes()),
        shell_sha256=sha(shell_raw), shell_outside_document_unchanged=True,
        schema_reference='https://bento.page/agents.md', schema_version_reviewed='1.2.13',
        input_sha256=inputs, pilot_snapshot=PILOT_COMMIT, formal_protocol_snapshot=protocol_commit,
        formal_completed=formal['completed'], main_slides=12, optional_appendices=2,
        spoken_script_words=word_counts, native_charts=sum(e['type']=='chart' for s in doc['slides'] for e in s['elements']),
        native_tables=sum(e['type']=='table' for s in doc['slides'] for e in s['elements']),
        output_sha256={str(args.output.resolve()):sha(html_raw), str(args.json_output.resolve()):sha(json_raw)},
        new_model_queries=0, new_training=0, new_GPU_hours=0, visual_review='Pending root inspection in Bento; no child browser was used.',
        contribution_scope='Planned speaking split: 180 seconds each over 12 main slides; no human rehearsal or actual authorship verified.')
    with args.manifest.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
        handle.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()), slides=14, charts=manifest['native_charts'],
                         formal_completed=formal['completed'], status=manifest['status']), ensure_ascii=True))


if __name__ == '__main__':
    main()
