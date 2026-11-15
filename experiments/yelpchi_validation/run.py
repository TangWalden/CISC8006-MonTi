"""Execute frozen shared rewriter with only dataset/cohort binding adaptations."""
from common import shared
exec(shared('run.py',[
 ('n==571 and len({i for r in rr for i in r[\'node_ids\']})==240',"n==p['valid_occurrences'] and len({i for r in rr for i in r['node_ids']})==p['valid_unique']"),
 ("from course.dual_surrogate.runtime import load_classifier","from adapter import load_classifier"),
 ("oldrows={(r['seed'],r['group'],r['arm']):r for r in map(json.loads,(pathlib.Path(p['parent'])/'main/evaluation.jsonl').read_text().splitlines())}","oldrows={} # First original-victim scoring occurs only after all YelpChi plans seal."),
 ('if arm in oldnames:', 'if arm in oldnames and oldrows:')
]),globals())
