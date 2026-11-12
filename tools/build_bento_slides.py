"""Build a key-free native Bento deck from the audited formal public export.

No network, ML runtime, predictions or collaboration credentials are used.
The official shell is preserved byte-for-byte outside its #bento-doc block.
Existing outputs are never overwritten. Missing formal results require the
explicit --allow-pending flag and produce a visibly pending deck, without
substituting any development-pilot score.

Expected final_public.json interface (MR units are percent, differences pp):
  status: completed; seeds: [0,1,2]
  arms: {native_monti|quota_random|quota_learned:
         {name, mean_mr, seed_mr:[three values] or {seed:value}}}
  primary, secondary_random: {mean_pp, ci95_pp:[lo,hi], seed_differences_pp}
  protocol: {train_groups, test_groups, epochs, ...}; cases: [...]
The demo consumes the same export. A failed/incomplete run never becomes a
completed score table. Successful completion does not imply positive efficacy.
"""
import argparse
import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


ARMS = ('native_monti', 'quota_random', 'quota_learned')
NAMES = ('Native MonTi', 'Random candidate edges', 'Staged pipeline')
FONT = 'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif'
BG, INK, MUTED, ACCENT, LIGHT = '#F7F5EF', '#17242B', '#53636B', '#D65A32', '#E4E6E1'
W, H = 1280, 720
REPO = 'https://github.com/TangWalden/CISC8006-MonTi'
DOC_RE = re.compile(r'(<script\b(?=[^>]*\bid=[\"\x27]bento-doc[\"\x27])[^>]*>)(.*?)(</script\s*>)', re.I | re.S)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected a finite numeric value.')
    return float(value)


def close(a, b):
    require(math.isclose(number(a), number(b), rel_tol=1e-8, abs_tol=1e-7), 'Formal summary and seed values differ.')


def ordered(value, seeds):
    out = [value[str(seed)] if str(seed) in value else value[seed] for seed in seeds] if isinstance(value, dict) else list(value)
    require(len(out) == len(seeds), 'Exactly one value per registered seed is required.')
    return [number(x) for x in out]


def normalize(raw, allow_pending=False, frozen_protocol=None):
    if raw is None or raw.get('status') != 'completed':
        require(allow_pending, 'Formal completed results are unavailable. Use --allow-pending only for an explicitly pending draft.')
        protocol = {}
        if frozen_protocol:
            require(frozen_protocol.get('schema') == 'monti_final_staged_v1'
                    and frozen_protocol.get('status') == 'frozen' and frozen_protocol.get('seeds') == [0, 1, 2],
                    'The pending deck accepts only the frozen formal registration.')
            cohort = frozen_protocol['cohort']
            protocol = dict(train_groups=cohort['train_groups'], test_groups=cohort['test_groups'],
                            epochs=frozen_protocol['settings']['downstream_epochs'],
                            test_target_occurrences=cohort['test_target_occurrences'])
        return dict(completed=False, status='Formal evaluation pending', seeds=[0, 1, 2], protocol=protocol, arms={})
    require(raw.get('metric_unit', 'percent') in ('percent', 'percentage'), 'mean_mr and seed_mr must use percent units.')
    seeds = raw['seeds']
    require(seeds == [0, 1, 2], 'The formal deck expects the registered seeds 0,1,2, not development seeds.')
    protocol = raw['protocol']
    require(type(protocol.get('test_groups')) is int and protocol['test_groups'] > 0,
            'A completed formal export must identify its TEST group count.')
    require(type(protocol.get('train_groups')) is int and protocol['train_groups'] > 0
            and type(protocol.get('epochs')) is int and protocol['epochs'] > 0,
            'A completed formal export must identify its actual TRAIN count and epochs.')
    if frozen_protocol:
        require(protocol['train_groups'] == frozen_protocol['cohort']['train_groups']
                and protocol['test_groups'] == frozen_protocol['cohort']['test_groups']
                and protocol['epochs'] == frozen_protocol['settings']['downstream_epochs'],
                'The completed export differs from the frozen formal cohort or training epochs.')
        protocol = dict(protocol)
        protocol.setdefault('test_target_occurrences', frozen_protocol['cohort']['test_target_occurrences'])
    arms = {}
    for key, label in zip(ARMS, NAMES):
        arm = raw['arms'][key]
        values = ordered(arm['seed_mr'], seeds)
        require(all(0 <= value <= 100 for value in values), 'MR percentages must lie between0and100.')
        mean = number(arm['mean_mr'])
        close(mean, statistics.mean(values))
        arms[key] = dict(name=str(arm.get('name') or label), mean_mr=mean, seed_mr=values)
    contrasts = {}
    for field, control in (('primary', 'native_monti'), ('secondary_random', 'quota_random')):
        given = raw[field]
        differences = ordered(given['seed_differences_pp'], seeds)
        expected = [a - b for a, b in zip(arms['quota_learned']['seed_mr'], arms[control]['seed_mr'])]
        for left, right in zip(differences, expected):
            close(left, right)
        close(given['mean_pp'], statistics.mean(differences))
        interval = [number(x) for x in given['ci95_pp']]
        require(len(interval) == 2 and interval[0] <= number(given['mean_pp']) <= interval[1], 'Invalid paired interval.')
        contrasts[field] = dict(mean_pp=number(given['mean_pp']), ci95_pp=interval, seed_differences_pp=differences)
    return dict(completed=True, status='Completed formal TEST evaluation', seeds=seeds,
                protocol=protocol, arms=arms, **contrasts)


def text(id, value, x, y, w, h, size=28, color=INK, weight=400, role=None, **extra):
    element = dict(id=id, type='text', x=x, y=y, w=w, h=h, rotation=0, opacity=1,
                   html=html.escape(str(value)).replace('\n', '<br>'), fontSize=size,
                   fontFamily=FONT, fontWeight=weight, color=color, align='left', valign='top', lineHeight=1.18)
    if role:
        element['role'] = role
    element.update(extra)
    return element


def box(id, x, y, w, h, fill=LIGHT, stroke='none', **extra):
    out = dict(id=id, type='shape', shape='rect', x=x, y=y, w=w, h=h,
               rotation=0, opacity=1, fill=fill, stroke=stroke, strokeWidth=1.5, radius=10)
    out.update(extra)
    return out


def line(id, x, y, w, color=MUTED, dashed=False):
    out = dict(id=id, type='shape', shape='line', x=x, y=y, w=w, h=1,
               rotation=0, opacity=1, fill=color, stroke=color, strokeWidth=3,
               radius=0, lineEnd='arrow')
    if dashed:
        out.update(strokeStyle='dashed', fx={'loop': {'type': 'dash-march', 'distance': 18, 'duration': 1.4}})
    return out


def path(id, x, y, w, h, d, color=MUTED):
    return dict(id=id, type='shape', shape='path', x=x, y=y, w=w, h=h, rotation=0, opacity=1,
                d=d, pathBox=[0, 0, w, h], fill='transparent', stroke=color,
                strokeWidth=3, radius=0, lineEnd='arrow')


def table(id, headers, body, x=96, y=220, w=1088, h=340, widths=None, font=23):
    rows = [headers] + body
    return dict(id=id, type='table', x=x, y=y, w=w, h=h, rotation=0, opacity=1, header=True,
                columns=[{'w': value} for value in (widths or [1] * len(headers))],
                rows=[{'cells': [{'html': html.escape(str(value))} for value in row]} for row in rows],
                style=dict(headerBg=INK, headerColor='#FFFFFF', zebra='#EDF0EB', borderColor='#CCD3CF',
                           borderWidth=1, cellPadX=16, cellPadY=12, fontSize=font, color=INK, radius=10))


def chart(id, values, labels, x=96, y=216, w=740, h=360, min_y=0, max_y=100):
    return dict(id=id, type='chart', x=x, y=y, w=w, h=h, rotation=0, opacity=1, preset='bar',
                option=dict(xAxis=dict(type='category', data=labels, axisLabel=dict(fontSize=17, color=INK)),
                            yAxis=dict(type='value', min=min_y, max=max_y,
                                       axisLabel=dict(fontSize=17, formatter='{value}%')),
                            grid=dict(left=65, right=20, top=30, bottom=60),
                            tooltip=dict(trigger='item', formatter='{b}: {c}'),
                            series=[dict(type='bar', data=[round(value, 8) for value in values],
                                         itemStyle=dict(color=ACCENT, borderRadius=6))]),
                fx={'enter': 'fade-up'})


def signed(value, digits=2):
    return ('+' if value > 0 else '') + format(value, '.' + str(digits) + 'f')


def fmt(value):
    return 'Pending' if value is None else str(value)


def source_urls(commit):
    base = REPO + '/blob/' + commit + '/'
    return dict(protocol=base + 'docs/FINAL_EVALUATION.md', registration=base + 'evidence/final_staged_v1/protocol.json',
                method=base + 'docs/METHOD.md', data=base + 'results/final_staged/final_public.json',
                provenance=base + 'results/final_staged/provenance.json',
                reproduction=base + 'docs/REPRODUCING.md')


def make_doc(data, commit, data_sha=None):
    sources = source_urls(commit)
    status = 'Formal TEST' if data['completed'] else 'Formal evaluation pending'
    slides = []
    notes_all = []
    notes_text = {
        1: 'Introduce the research question: separate a fixed target-facing planner from a trained candidate-edge policy. State whether formal results are available. Do not substitute pilot values.',
        2: 'Explain a group as a fixed list of known fraud targets. Candidate nodes provide context and possible endpoints. The picture is a schematic, not a measured case.',
        3: 'Describe native MonTi as a generator of auxiliary features and connections. Our baseline uses our reproduced checkpoints, not a number copied from a paper table or a verified author-best checkpoint.',
        4: 'Describe the two roles. Freeze upstream features and target connections. Train only the small downstream to select candidate edges. This compares the whole pipeline, not one causal component.',
        5: 'Explain the capacity limit. A half allocation is a request, projected to the feasible interval while preserving each original total node and edge budget. Point out changes to individual degrees and auxiliary-to-auxiliary policy.',
        6: 'Read the frozen formal counts and settings. Explain TRAIN-only GCN feedback and terminal checkpoints. Official GAGA scores only sealed final graphs. Never call a development pilot a formal TEST run.',
        7: 'Read the three comparable formal MR values if attached. Higher MR means more fraud targets escape detection, including targets already missed before attack. Use the definition link for a simple denominator example.',
        8: 'Read each paired seed difference and the reported interval. Seed is the replication unit. State a positive, inconclusive or negative result exactly as the saved final export reports.',
        9: 'Separate structural costs from training cost. Keep the random comparison visible. The complete pipeline changes several components; a gain over native does not isolate role division alone.',
        10: 'Conclude with the actual formal outcome or pending state. Demonstrate a saved case from the static viewer. It makes no model call. Invite the audience to inspect provenance and retained supplementary work.',
    }

    def add(id, title, elements, n=None, transition='fade', state_of=None, hidden=False, extra_notes=''):
        chrome = [text('running-title', 'MonTi / staged construction', 96, 42, 830, 24, 16, MUTED),
                  text('page-counter', '{{page}} / {{pages}}', 1080, 42, 104, 24, 16, MUTED),
                  text('slide-heading', title, 96, 92, 1088, 105, 44, INK, 700, 'title'),
                  text('evidence-footer', status + '. Evidence snapshot ' + commit[:8], 96, 661, 1088, 30, 15, MUTED)]
        speaker = 'Junyi Tang' if n and n <= 5 else 'Yifan Fan'
        plan = ('Proposed speaker: ' + speaker + '. Planned duration:36 seconds. This is a balanced speaking plan, not verified authorship, contribution or rehearsal.\n') if n else 'Appendix / optional drill-down.\n'
        notes = plan + (notes_text.get(n, '') if n else extra_notes) + '\nSources:\n' + '\n'.join(sources.values())
        out = dict(id=id, name=title, background=BG, transition=transition, notes=notes, elements=chrome + elements)
        if state_of:
            out['stateOf'] = state_of
        if hidden:
            out['hidden'] = True
        slides.append(out)
        notes_all.append(notes)

    cover_note = 'A fixed upstream planner with a trained downstream edge policy'
    cover = [text('cover-title', 'MonTi\nStaged attack construction', 96, 164, 1040, 230, 74, INK, 750, 'title', fx={'enter': 'fade-up'}),
             text('cover-subtitle', cover_note, 96, 423, 1020, 85, 30, MUTED, role='subtitle'),
             text('cover-team', 'Proposed presenters: Junyi Tang and Yifan Fan', 96, 554, 1080, 42, 24, MUTED),
             box('cover-role-rule', 96, 129, 110, 7, ACCENT)]
    add('cover', '', cover, 1)

    roles = []
    for row, (label, caption, fill) in enumerate([
            ('Targets', 'The fixed fraud-target list', ACCENT),
            ('Candidates', 'Original graph context and eligible endpoints', MUTED),
            ('Auxiliary nodes', 'New nodes generated under the budget', INK)]):
        y = 224 + row * 115
        roles.append(text('role-label-' + str(row), label, 96, y + 12, 245, 52, 30, INK, 650))
        for col in range(3):
            x = 365 + col * 80
            roles.append(box('role-node-%d-%d' % (row, col), x, y + 4, 46, 46, fill, shape='ellipse', radius=0))
        roles.append(text('role-caption-' + str(row), caption, 650, y + 9, 530, 72, 25, MUTED))
    roles.append(text('schematic-label', 'Role schematic only. The demo uses recorded cases.', 96, 599, 1080, 35, 20, MUTED))
    add('graph-task', 'The graph attack task', roles, 2)

    def pipeline(staged):
        e = [box('monTi-block', 96, 250 if not staged else 224, 352, 140, INK),
             text('monTi-name', 'Native MonTi' if not staged else 'Fixed upstream MonTi', 124, 278 if not staged else 246, 300, 45, 29, '#FFFFFF', 650),
             text('monTi-role', 'Features and connection choices' if not staged else 'Features and target connections', 124, 331 if not staged else 299, 295, 72, 22, '#FFFFFF'),
             box('target-output', 870, 224, 314, 120, LIGHT),
             text('target-output-label', 'Target connections', 895, 259, 270, 50, 27, INK, 650),
             box('candidate-output', 870, 430, 314, 120, LIGHT),
             text('candidate-output-label', 'Candidate connections', 895, 465, 270, 58, 26, INK, 650),
             line('target-flow', 470, 279, 378, ACCENT)]
        if staged:
            e += [box('downstream-block', 506, 425, 310, 130, ACCENT),
                  text('downstream-name', 'Trained downstream', 530, 449, 267, 47, 27, '#FFFFFF', 650),
                  text('downstream-role', '151,872 parameters', 530, 502, 267, 34, 22, '#FFFFFF'),
                  line('candidate-flow', 820, 490, 28, ACCENT, True),
                  text('fixed-output-label', 'Frozen upstream output', 520, 365, 310, 40, 22, MUTED),
                  path('downstream-input', 448, 334, 210, 91, 'M0 0 H35 V72 H210 V91', MUTED),
                  text('pipeline-limit', 'Only downstream parameters enter the optimizer.', 96, 601, 1088, 38, 24, MUTED)]
        else:
            e += [path('candidate-flow', 470, 350, 378, 140, 'M0 0 V140 H378', MUTED),
                  text('pipeline-limit', 'Our baseline evaluates our reproduced MonTi checkpoints.', 96, 601, 1088, 38, 24, MUTED)]
        return e
    add('native-pipeline', 'Native MonTi construction', pipeline(False), 3)
    add('staged-pipeline', 'Upstream planning and downstream completion', pipeline(True), 4, transition='morph')

    add('budget', 'The same total budget, a feasible split', [
        table('budget-table', ['Quantity', 'Rule'], [
            ['Auxiliary nodes B and total edges E', 'Preserve each original case budget'],
            ['Requested target quota', 'ceil(E / 2)'],
            ['Feasible target interval', '[max(B, E - B*C), min(B*T, E)]'],
            ['Remaining candidate edges', 'E minus the realized target quota']], h=328, widths=[1.05, 1.6], font=23),
        text('budget-limit', 'Individual degrees can change. Quota arms remove auxiliary-to-auxiliary edges.\nEvery actual ratio stays in the saved case table.', 96, 580, 1088, 65, 23, MUTED)], 5)

    p = data['protocol']
    protocol_body = [['Training groups', fmt(p.get('train_groups'))], ['TEST groups', fmt(p.get('test_groups'))],
                     ['Training seeds', ', '.join(map(str, data['seeds']))], ['Downstream epochs', fmt(p.get('epochs'))],
                     ['Optimization feedback', 'Frozen GCN on TRAIN only'], ['Final detector', 'Official fixed GAGA after sealing']]
    target_count = p.get('test_target_occurrences')
    protocol_note = ((format(target_count, ',') + ' TEST target occurrences per seed. ') if type(target_count) is int else '')
    protocol_note += 'Terminal checkpoints.\nNo result-driven extra seeds or training.'
    add('protocol', 'Frozen formal evaluation', [table('protocol-table', ['Setting', 'Registered value'], protocol_body,
        y=212, h=360, widths=[1, 1.65], font=22),
        text('protocol-boundary', protocol_note, 96, 590, 1088, 64, 22, MUTED)], 6)

    if data['completed']:
        values = [data['arms'][arm]['mean_mr'] for arm in ARMS]
        result_e = [chart('formal-bars', values, ['Native', 'Random', 'Staged']),
                    table('formal-values', ['Condition', 'MR'], [[label, format(value, '.2f') + '%'] for label, value in zip(['Native', 'Random', 'Staged'], values)],
                          x=872, y=248, w=312, h=250, widths=[1.15, 1], font=23)]
    else:
        result_e = [text('formal-pending', 'Formal results are not attached yet.', 96, 248, 1088, 80, 38, ACCENT, 650),
                    text('formal-no-substitution', 'The development pilot is a separate study.\nThis chart will use only the audited formal export.', 96, 355, 1020, 115, 29, MUTED)]
    result_e += [box('mr-link-hit', 96, 590, 630, 43, '#E9DDD0', link='mr-definition'),
                 text('mr-link-text', 'MR definition and denominator example', 113, 598, 600, 30, 21, INK, 600, link='mr-definition')]
    add('formal-results', 'Formal comparison on the same cases', result_e, 7)

    if data['completed']:
        diff = data['primary']['seed_differences_pp']; lo, hi = data['primary']['ci95_pp']
        seed_rows = [[str(seed), signed(value) + ' pp'] for seed, value in zip(data['seeds'], diff)]
        seed_rows += [['Mean', signed(data['primary']['mean_pp']) + ' pp'], ['Reported 95% interval', '[' + signed(lo) + ', ' + signed(hi) + '] pp']]
    else:
        seed_rows = [[str(seed), 'Pending'] for seed in data['seeds']] + [['Mean', 'Pending'], ['Reported 95% interval', 'Pending']]
    add('paired-results', 'Staged minus native, by seed', [table('seed-table', ['Training seed', 'Paired MR difference'], seed_rows,
        h=348, widths=[1, 1.7], font=25),
        text('seed-unit', 'Seed is the replication unit. Repeated target occurrences are correlated.', 96, 601, 1088, 42, 23, MUTED)], 8)

    comparison = 'Formal random comparison pending.'
    if data['completed']:
        c = data['secondary_random']; comparison = 'Staged minus random: ' + signed(c['mean_pp']) + ' pp\n95% interval [' + signed(c['ci95_pp'][0]) + ', ' + signed(c['ci95_pp'][1]) + '] pp'
    add('interpretation', 'What the comparison can establish', [
        text('interpretation-primary', 'The full pipeline changes role allocation and a trained edge policy.', 96, 220, 1088, 110, 36, INK, 650),
        text('interpretation-random', comparison, 96, 357, 1088, 96, 30, ACCENT, 600),
        text('interpretation-limit', 'Equal structural budgets still allow different per-node degrees.\nThe downstream adds training computation. A native gain does not isolate one causal component.', 96, 503, 1088, 123, 27, MUTED)], 9)

    conclusion = 'Formal evaluation is pending.'
    if data['completed']:
        lo, hi = data['primary']['ci95_pp']
        conclusion = ('A positive formal contrast against native MonTi.' if lo > 0 else
                      'The formal native comparison remains inconclusive.' if hi >= 0 else
                      'The staged pipeline underperforms native in this evaluation.')
    add('conclusion', 'Recorded evidence and a saved-case demo', [
        text('conclusion-text', conclusion, 96, 226, 1088, 118, 43, INK, 700),
        text('conclusion-demo', 'The static demo selects recorded cases and arms.\nIt displays budgets and saved graph edges without model calls.', 96, 382, 1088, 105, 31, MUTED),
        text('conclusion-records', 'Protocol, complete results, audit scope and supplementary attempts\nremain available in the project repository.', 96, 536, 1088, 85, 25, MUTED)], 10)

    add('mr-definition', 'Miss rate counts all target occurrences', [
        text('mr-equation', 'MR = incorrect fraud-target predictions / all fraud-target occurrences', 96, 218, 1088, 105, 34, INK, 650),
        table('mr-example', ['Illustrative group', 'Count'], [['Known fraud targets', '4'], ['Already missed before attack', '1'], ['Newly missed after attack', '1'], ['Total missed after attack', '2 / 4 = 50% MR']],
              y=344, h=245, widths=[1.8, 1], font=21),
        text('mr-example-scope', 'Definition example only, not a measured experimental case.', 96, 598, 1088, 34, 21, MUTED)],
        state_of='formal-results', transition='morph', extra_notes='Explain the complete denominator. All target nodes here have known fraud labels. Pre-existing misses count in MR but are not attack-induced flips.')
    appendix_rows = [['Evidence snapshot', commit[:16]], ['Final data SHA256', (data_sha or 'Not attached')[:24]],
                     ['Results and provenance', 'results/final_staged/'], ['Protocol and commands', 'docs/FINAL_EVALUATION.md']]
    add('audit-appendix', 'Evidence trail', [table('audit-table', ['Record', 'Location or identity'], appendix_rows,
        widths=[1.05, 1.7], h=300, font=23),
        text('audit-limit', 'Saved-record checks do not equal human technical review.\nPublic data omits raw model weights and private assets.', 96, 567, 1088, 80, 25, MUTED)],
        hidden=True, extra_notes='Use this appendix for provenance questions. Cite the exact formal data and audit records. Do not claim that automated validation is a human review.')
    add('limits-appendix', 'Scope and limitations', [
        text('limits-text', 'Our checkpoints are reproduced models. Author-best equivalence is unconfirmed.\n\nOriginal group lists and incidence-based budget semantics remain explicit.\n\nRepeated targets limit independent news-level interpretation.\n\nInstructor approval, actual contributions and rehearsal require human confirmation.',
             96, 222, 1088, 382, 28, MUTED)], hidden=True,
        extra_notes='Distinguish the formal evaluation from development pilots and the paper tables. Earlier positive, negative and failed studies remain in the supplementary index.')

    return dict(format='bento/slides', version=1, title='MonTi: Staged Attack Construction',
                docId='monti-staged-formal-20261112-v1',
                size=dict(width=W, height=H), theme=dict(background=BG, color=INK, accent=ACCENT, fontFamily=FONT),
                meta=dict(author='CISC8006 project', subject=data['status'], event='Proposed project presentation'),
                slides=slides)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell', required=True)
    parser.add_argument('--data', default='results/final_staged/final_public.json')
    parser.add_argument('--protocol', default='evidence/final_staged_v1/protocol.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--json-output')
    parser.add_argument('--manifest')
    parser.add_argument('--evidence-commit')
    parser.add_argument('--allow-pending', action='store_true')
    args = parser.parse_args(argv)
    shell_path, data_path, output = Path(args.shell).resolve(), Path(args.data).resolve(), Path(args.output).resolve()
    json_output = Path(args.json_output).resolve() if args.json_output else output.with_suffix('.json')
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    targets = [output, json_output] + ([manifest_path] if manifest_path else [])
    require(len(set(targets)) == len(targets) and all(not p.exists() for p in targets), 'Use new output paths and preserve prior revisions.')
    shell_bytes = shell_path.read_bytes(); shell = shell_bytes.decode('utf-8')
    matches = list(DOC_RE.finditer(shell))
    require(len(matches) == 1, 'The unchanged official shell must contain exactly one #bento-doc block.')
    existing = matches[0].group(2).strip()
    if existing:
        old = json.loads(existing)
        require(not old.get('collab'), 'Use the key-free official shell. A live owner/invite file must not enter this builder.')
    raw = json.loads(data_path.read_text(encoding='utf-8')) if data_path.is_file() else None
    protocol_path = Path(args.protocol).resolve()
    frozen_protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.is_file() else None
    data = normalize(raw, args.allow_pending, frozen_protocol)
    commit = args.evidence_commit
    if not commit:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(bool(re.fullmatch('[0-9a-f]{40}', commit)), 'Supply a complete versioned evidence commit.')
    data_bytes = data_path.read_bytes() if data_path.is_file() else None
    doc = make_doc(data, commit, sha(data_bytes) if data_bytes else None)
    require('collab' not in doc and 'readonly' not in doc and 'template' not in doc, 'The authored base must be editable and key-free.')
    payload = json.dumps(doc, ensure_ascii=True, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    match = matches[0]
    built = shell[:match.start(2)] + '\n' + payload + '\n' + shell[match.end(2):]
    require(built[:match.start(2)] == shell[:match.start(2)] and built.endswith(shell[match.end(2):]), 'Unexpected shell mutation.')
    output.parent.mkdir(parents=True, exist_ok=True); json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(built.encode('utf-8'))
    json_output.write_bytes((json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8'))
    receipt = dict(schema='monti_native_bento_build_v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   scientific_result_status=data['status'], formal_results_attached=data['completed'],
                   evidence_commit=commit, shell_sha256=sha(shell_bytes), builder_sha256=sha(Path(__file__).read_bytes()),
                   data_sha256=sha(data_bytes) if data_bytes else None,
                   frozen_protocol_sha256=sha(protocol_path.read_bytes()) if protocol_path.is_file() else None,
                   output_sha256={str(output): sha(output.read_bytes()), str(json_output): sha(json_output.read_bytes())},
                   slide_counts=dict(main=10, state=1, hidden_appendix=2, total=13),
                   format='bento/slides', public_collab_keys_present=False, proposed_main_talk_seconds=360,
                   proposed_speaker_seconds={'Junyi Tang': 180, 'Yifan Fan': 180},
                   human_rehearsal_or_contribution_verified=False, model_queries=0, GPU_hours=0,
                   validation_scope='Build-time structure and numerical consistency only. The actual Bento runtime validation and all-slide visual review remain required.')
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes((json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
