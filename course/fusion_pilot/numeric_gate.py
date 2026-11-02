"""Same-numeric-information train-meta gate; fixed CPU fit, no API/graph query."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import time


def rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def features(row):
    card = row['card']
    graph = card['current_graph']
    missing = graph['neighbor_similarity_mean'] is None
    return [card['self']['p_fraud'], card['graph_context']['p_fraud'],
            math.log1p(graph['degree_nonself']),
            0.0 if missing else graph['neighbor_similarity_mean'],
            0.0 if missing else graph['neighbor_similarity_std'], float(missing)]


def main():
    parser = argparse.ArgumentParser()
    for name in ['meta-cards', 'meta-labels', 'probe-cards', 'probe-labels', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    started = time.monotonic()
    train, probe = rows(args.meta_cards), rows(args.probe_cards)
    train_labels = {r['case_id']: r['label'] for r in rows(args.meta_labels)}
    probe_labels = {r['case_id']: r['label'] for r in rows(args.probe_labels)}
    if len({r['case_id'] for r in train}) != len(train) or len(probe) != 32:
        raise RuntimeError('Unexpected duplicate/meta or probe cohort.')
    if {r['case_id'] for r in train} != set(train_labels) or {r['case_id'] for r in probe} != set(probe_labels):
        raise RuntimeError('Private supervision coverage mismatch.')
    x = torch.tensor([features(r) for r in train], dtype=torch.float32)
    y = torch.tensor([train_labels[r['case_id']] for r in train], dtype=torch.float32)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(.001)
    normalized = (x-mean)/scale
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history = []
    for epoch in range(100):
        alpha = model(normalized).squeeze(1)
        probability = alpha*x[:, 0]+(1-alpha)*x[:, 1]
        loss = torch.nn.functional.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    xp = torch.tensor([features(r) for r in probe], dtype=torch.float32)
    with torch.no_grad():
        continuous_alpha = model((xp-mean)/scale).squeeze(1)
        grid = torch.tensor([0., .25, .5, .75, 1.])
        alpha = grid[(continuous_alpha.unsqueeze(1)-grid).abs().argmin(1)]
        probability = alpha*xp[:, 0]+(1-alpha)*xp[:, 1]
        continuous_probability = continuous_alpha*xp[:, 0]+(1-continuous_alpha)*xp[:, 1]
    outcomes = [dict(case_id=r['case_id'], label=probe_labels[r['case_id']],
                     alpha_self=float(a), p_fraud=float(p), predicted_fraud=bool(p >= .5))
                for r, a, p in zip(probe, alpha.tolist(), probability.tolist())]
    with (output/'outcomes.jsonl').open('x') as stream:
        for row in outcomes:
            stream.write(json.dumps(row)+'\n')
    torch.save(dict(state_dict=model.state_dict(), mean=mean, scale=scale,
                    input_features=['p_self','p_graph','log1p_degree','cosine_mean','cosine_std','similarity_missing']), output/'gate.pickle')
    benign = sum(r['label'] == 0 for r in outcomes)
    fraud = len(outcomes)-benign
    result = dict(status='completed', training_meta_cases=len(train), probe_cases=len(probe),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        seed=20261103, epochs=100, optimizer_steps=100, architecture=[6,8,1],
        lr=.01, training_loss_history=history, calibration='uncalibrated',
        graph_meta_provenance='fixed graph branch saw original training labels; NOT out-of-fold',
        accuracy=sum(r['predicted_fraud'] == bool(r['label']) for r in outcomes)/len(outcomes),
        benign_false_positive_rate=sum(r['predicted_fraud'] and r['label'] == 0 for r in outcomes)/benign,
        fraud_recall=sum(r['predicted_fraud'] and r['label'] == 1 for r in outcomes)/fraud,
        action_grid=[0,.25,.5,.75,1], tie_policy='nearest grid value, lower on exact tie',
        continuous_secondary_accuracy=sum(bool(p >= .5) == bool(probe_labels[r['case_id']])
            for r,p in zip(probe,continuous_probability.tolist()))/len(probe),
        input_sha256={name:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for name,path in
            [('meta_cards',args.meta_cards),('meta_labels',args.meta_labels),('probe_cards',args.probe_cards),('probe_labels',args.probe_labels)]},
        elapsed_cpu_seconds=time.monotonic()-started, gpu_hours=0, graph_queries=0, api_calls=0,
        scope='Selected pilot comparator, not a final result or robustness guarantee')
    with (output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'training_loss_history'}))


if __name__ == '__main__':
    main()
