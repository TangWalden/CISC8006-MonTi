"""Post-baseline mechanism diagnosis restricted to train and validation groups."""
import datetime as dt
import json
import pathlib
import subprocess
import time
import traceback

import dgl
import numpy as np
import torch

from course.runner import ROOT, sha256, write_json
from modules.data_handler import DataHandler
from modules.experiment_handler import ExperimentHandler
from utils.utils import get_n_ids, get_complment_indices, set_seed


def main():
    output = ROOT / 'evidence/training_diagnostic_v1'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    write_json(output / 'registered.json', dict(
        purpose='train/validation-only mechanism diagnosis; no test inference',
        registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        command=['bash', 'course/run.sh', '-m', 'course.diagnose_training'],
        selection_seed=1729, groups_per_partition=64, attacker='first registered learned run, seed 0; not chosen by score'))
    try:
        config = json.loads((ROOT / 'course/configs/baseline.json').read_text())
        config.update(seed=1729, cuda_id=0, random_candidates=False)
        for key, default in dict(shared_parameters=False, without_candidates=False, pos_encoding=True, ada_budget=True).items():
            config.setdefault(key, default)
        set_seed(1729)
        torch.cuda.set_device(0)
        data = DataHandler(config)
        experiment = ExperimentHandler(config, data)
        registry = json.loads((ROOT / 'evidence/run_registry.json').read_text())
        run = next(r for r in registry if r['registration']['phase'] == 'reproduction' and
                   r['registration']['mode'] == 'learned' and r['registration']['seed'] == 0)
        checkpoint = ROOT / run['status']['checkpoint']
        assert sha256(checkpoint) == run['status']['checkpoint_sha256']
        experiment.model.load_state_dict(torch.load(str(checkpoint), map_location=experiment.device))
        experiment.model.eval()
        selection = np.random.RandomState(1729)
        rows, weightings = [], {}
        for name, groups in [('train', data.target_idx_train), ('valid', data.target_idx_valid)]:
            sizes = np.array([len(get_n_ids(data.target_assignment, group)) for group in groups])
            probabilities = sizes / sizes.sum()
            uniform = np.full(len(groups), 1 / len(groups))
            weightings[name] = dict(groups=len(groups), target_occurrences=int(sizes.sum()),
                                   size_quantiles=np.quantile(sizes, [0, .25, .5, .75, 1]).tolist(),
                                   total_variation_size_vs_uniform=float(np.abs(probabilities-uniform).sum()/2))
            subset = np.sort(selection.choice(groups, min(64, len(groups)), replace=False))
            for group in subset:
                ids = torch.tensor(get_n_ids(data.target_assignment, group), dtype=torch.long, device=data.graph.device)
                subgraph, targets = dgl.khop_in_subgraph(data.graph, ids, k=config['k_exp'], output_device=data.graph.device)
                src, dst = subgraph.out_edges(targets)
                subgraph.ndata['beta'] = torch.bincount(dst, minlength=subgraph.num_nodes()).float()
                pool = get_complment_indices(subgraph.num_nodes(), targets)
                with torch.no_grad():
                    _, chosen = experiment.model.select_candidates(subgraph, targets, True)
                    h = torch.nn.functional.normalize(subgraph.ndata['h'][chosen], dim=1)
                    similarity = ((h @ h.T).sum()-len(h))/(len(h)*(len(h)-1)) if len(h)>1 else torch.tensor(0.)
                def coverage(candidates):
                    selected_mask = torch.zeros(subgraph.num_nodes(), dtype=torch.bool, device=subgraph.device)
                    selected_mask[candidates.to(subgraph.device)] = True
                    covered = src[selected_mask[dst]].unique()
                    return float(len(covered) / len(targets))
                rows.append(dict(partition=name, target_group=int(group), target_size=len(targets),
                                 exploration_pool=len(pool), selected_candidates=len(chosen),
                                 selection_active=len(pool)>config['max_candidates'],
                                 selected_coverage=coverage(chosen), pool_coverage=coverage(pool),
                                 candidate_mean_cosine_similarity=float(similarity)))
        result = dict(status='completed', weighting=weightings, groups=rows,
                      attacker_checkpoint_sha256=sha256(checkpoint),
                      information_boundary='Only train/valid membership and graph topology; no test group diagnosis or new test predictions.',
                      limitation='post-hoc mechanism diagnosis, not a pre-registered claim test or independent reproduction')
    except Exception:
        result = dict(status='failed', traceback=traceback.format_exc())
        traceback.print_exc()
    result['elapsed_seconds'] = time.monotonic()-started
    result['gpu_wall_hours'] = result['elapsed_seconds']/3600
    write_json(output / 'result.json', result)
    print('DIAGNOSTIC', json.dumps(result), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
