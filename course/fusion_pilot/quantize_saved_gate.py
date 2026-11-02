"""Apply the predeclared action grid to retained continuous gate outputs, no ML."""
import argparse
import hashlib
import json
import pathlib

GRID = (0., .25, .5, .75, 1.)


def read(path):
    return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    for name in ('continuous-outcomes', 'cards', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    records = read(args.continuous_outcomes)
    cards = {r['case_id']:r['card'] for r in read(args.cards)}
    if len(records) != 32 or {r['case_id'] for r in records} != set(cards):
        raise RuntimeError('The complete predeclared cohort is required.')
    rows = []
    for record in records:
        alpha = record['alpha_self']
        if type(alpha) not in (int,float) or not 0 <= alpha <= 1:
            raise RuntimeError('Invalid retained gate weight.')
        weight = min(GRID, key=lambda v:(abs(v-alpha),v))
        card = cards[record['case_id']]
        p = weight*card['self']['p_fraud']+(1-weight)*card['graph_context']['p_fraud']
        rows.append(dict(case_id=record['case_id'], label=record['label'],
                         continuous_alpha_self=alpha, alpha_self=weight,
                         p_fraud=p, predicted_fraud=p>=.5))
    benign = sum(r['label']==0 for r in rows);fraud=32-benign
    target=pathlib.Path(args.output);target.mkdir(parents=True,exist_ok=False)
    with (target/'outcomes.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    report=dict(status='completed',cases=32,alpha_grid=GRID,
        tie_policy='nearest, lower on exact tie; frozen before any gate outcomes',
        accuracy=sum(r['predicted_fraud']==bool(r['label']) for r in rows)/32,
        benign_false_positive_rate=sum(r['predicted_fraud'] and r['label']==0 for r in rows)/benign,
        fraud_recall=sum(r['predicted_fraud'] and r['label']==1 for r in rows)/fraud,
        new_model_calls=0,api_calls=0,gpu_hours=0,
        input_sha256={name:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                      for name,path in [('continuous_outcomes',args.continuous_outcomes),('cards',args.cards)]})
    (target/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
