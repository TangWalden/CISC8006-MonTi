"""Post-hoc saved-only TRAIN trajectory diagnosis; no ML imports or queries.

Overall VALID outcomes were known before this analysis was specified. Each row
is an online observation immediately before that group's optimizer update, not
a reevaluation of a fixed checkpoint. Group mean margins/losses do not retain
the underlying per-node distributions. All source/input hashes are frozen in
an external registration before this script parses any new outcome records.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import time


ARMS = ('legacy', 'source_fixed')
SEEDS = (3, 4, 5)
EPOCHS = tuple(range(1, 9))


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    return dict(min=min(values), q25=percentile(values,.25), median=percentile(values,.5),
                q75=percentile(values,.75), max=max(values), mean=statistics.fmean(values))


def budget_regime(row):
    if row['node_budget'] == 1:
        return 'one_node_one_edge' if row['edge_budget'] == 1 else 'one_node_multiple_edges'
    return 'multiple_nodes'


def aggregate(rows):
    n = sum(row['targets'] for row in rows)
    wrong = sum(row['wrong'] for row in rows)
    return dict(groups=len(rows), target_occurrences=n, wrong=wrong,
        occurrence_wrong_percent=100*wrong/n,
        group_equal_wrong_percent=100*statistics.fmean(row['wrong']/row['targets'] for row in rows),
        group_mean_softplus=statistics.fmean(row['loss'] for row in rows),
        occurrence_weighted_group_softplus=sum(row['targets']*row['loss'] for row in rows)/n,
        group_mean_attack_margin=statistics.fmean(row['margin'] for row in rows),
        occurrence_weighted_group_attack_margin=sum(row['targets']*row['margin'] for row in rows)/n,
        group_softplus_distribution=distribution([row['loss'] for row in rows]),
        group_mean_margin_distribution=distribution([row['margin'] for row in rows]),
        group_wrong_rate_distribution=distribution([100*row['wrong']/row['targets'] for row in rows]),
        all_wrong_groups=sum(row['wrong']==row['targets'] for row in rows),
        zero_wrong_groups=sum(row['wrong']==0 for row in rows),
        summed_injected_nodes=sum(row['node_budget'] for row in rows),
        summed_injected_edges=sum(row['edge_budget'] for row in rows))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registration',required=True)
    args=parser.parse_args()
    started,cpu_started=time.monotonic(),time.process_time()
    registration_path=pathlib.Path(args.registration).resolve()
    registration=read_json(registration_path)
    if registration['source_sha256'] != sha256(__file__):
        raise RuntimeError('Executed source differs from its before-read freeze.')
    manifest_path=pathlib.Path(registration['input_manifest_path'])
    if sha256(manifest_path)!=registration['input_manifest_sha256']:
        raise RuntimeError('Frozen input manifest changed.')
    manifest=read_json(manifest_path)
    root=pathlib.Path(registration['input_root']).resolve()
    output=pathlib.Path(registration['output_directory']).resolve()
    if output.exists():raise RuntimeError('A new immutable output directory is required.')
    hashes=[]
    for row in manifest['files']:
        relative=pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe input path.')
        path=(root/relative).resolve()
        if root not in path.parents or not path.is_file():raise RuntimeError('Missing/outside-root input.')
        actual=sha256(path)
        if actual!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise RuntimeError('Frozen input bytes changed: '+row['path'])
        hashes.append(dict(path=row['path'],sha256=actual,bytes=row['bytes']))
    # Only after every expected byte hash passes are outcomes parsed.
    protocol=read_json(root/'experiment_protocol_v1.json')
    train_groups=protocol['train_groups']
    if len(train_groups)!=256 or len(set(train_groups))!=256:
        raise RuntimeError('Frozen TRAIN cohort differs.')
    source_rows={};epoch_summaries=[];strata=[];result_checks=[]
    for seed in SEEDS:
        for arm in ARMS:
            run=root/'matrix_attempt1'/('%s_seed%d'%(arm,seed))
            result=read_json(run/'result.json')
            if (result['status']!='completed' or result['arm']!=arm or result['seed']!=seed or
                    result['optimizer_steps']!=2048 or result['epochs']!=8):
                raise RuntimeError('A terminal training receipt differs.')
            result_checks.append(dict(arm=arm,seed=seed,status=result['status'],optimizer_steps=result['optimizer_steps']))
            for epoch in EPOCHS:
                path=run/'raw'/('train_epoch_%03d.jsonl'%epoch)
                records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                if len(records)!=256 or {int(row['group']) for row in records}!=set(train_groups):
                    raise RuntimeError('TRAIN cohort or case count differs.')
                rows=[]
                for record in records:
                    if set(record['losses'])!={'gcn_base'}:raise RuntimeError('Non-GCN-only feedback.')
                    value=record['losses']['gcn_base']
                    row=dict(arm=arm,seed=seed,epoch=epoch,group=int(record['group']),
                        targets=int(record['targets']),node_budget=int(record['node_budget']),edge_budget=int(record['edge_budget']),
                        wrong=int(value['wrong']),loss=float(value['loss']),margin=float(value['mean_attack_margin']),
                        generation_seed=int(record['generation_seed']))
                    if (row['targets']<=0 or not 0<=row['wrong']<=row['targets'] or row['node_budget']<=0 or
                            row['edge_budget']<=0 or not all(math.isfinite(row[k]) for k in ('loss','margin')) or
                            abs(float(record['total_loss'])-row['loss'])>1e-12):
                        raise RuntimeError('Invalid online TRAIN observation.')
                    rows.append(row)
                if sum(row['targets'] for row in rows)!=3141:
                    raise RuntimeError('Expected all3141TRAIN occurrences.')
                receipt=read_json(run/('epoch_%03d.json'%epoch))
                if (receipt['epoch']!=epoch or receipt['groups']!=256 or receipt['optimizer_steps']!=epoch*256 or
                        not math.isclose(receipt['mean_loss'],statistics.fmean(row['loss'] for row in rows),abs_tol=1e-12,rel_tol=1e-12)):
                    raise RuntimeError('Epoch summary does not reconcile to online raw rows.')
                source_rows[arm,seed,epoch]={row['group']:row for row in rows}
                epoch_summaries.append(dict(arm=arm,seed=seed,epoch=epoch,**aggregate(rows)))
                for regime in ('one_node_one_edge','one_node_multiple_edges','multiple_nodes'):
                    selected=[row for row in rows if budget_regime(row)==regime]
                    if selected:strata.append(dict(arm=arm,seed=seed,epoch=epoch,budget_regime=regime,**aggregate(selected)))
    pairs=[];paired_epochs=[];paired_strata=[]
    for seed in SEEDS:
        for epoch in EPOCHS:
            selected=[]
            for group in train_groups:
                a,b=source_rows['legacy',seed,epoch][group],source_rows['source_fixed',seed,epoch][group]
                if any(a[key]!=b[key] for key in ('targets','node_budget','edge_budget','generation_seed')):
                    raise RuntimeError('Matched online TRAIN case keys/budgets differ.')
                row=dict(seed=seed,epoch=epoch,group=group,targets=a['targets'],node_budget=a['node_budget'],edge_budget=a['edge_budget'],
                    legacy_wrong=a['wrong'],source_fixed_wrong=b['wrong'],delta_wrong=b['wrong']-a['wrong'],
                    legacy_loss=a['loss'],source_fixed_loss=b['loss'],delta_loss=b['loss']-a['loss'],
                    legacy_mean_margin=a['margin'],source_fixed_mean_margin=b['margin'],delta_mean_margin=b['margin']-a['margin'])
                pairs.append(row);selected.append(row)
            def pair_aggregate(values):
                occurrences=sum(row['targets'] for row in values)
                return dict(groups=len(values),target_occurrences=occurrences,
                    delta_wrong=sum(row['delta_wrong'] for row in values),
                    delta_occurrence_wrong_pp=100*sum(row['delta_wrong'] for row in values)/occurrences,
                    group_mean_delta_softplus=statistics.fmean(row['delta_loss'] for row in values),
                    group_mean_delta_attack_margin=statistics.fmean(row['delta_mean_margin'] for row in values),
                    wrong_count_wins=sum(row['delta_wrong']>0 for row in values),
                    wrong_count_ties=sum(row['delta_wrong']==0 for row in values),
                    wrong_count_losses=sum(row['delta_wrong']<0 for row in values),
                    lower_softplus_groups=sum(row['delta_loss']<0 for row in values),
                    higher_softplus_groups=sum(row['delta_loss']>0 for row in values))
            paired_epochs.append(dict(seed=seed,epoch=epoch,**pair_aggregate(selected)))
            for regime in ('one_node_one_edge','one_node_multiple_edges','multiple_nodes'):
                values=[row for row in selected if budget_regime(row)==regime]
                if values:paired_strata.append(dict(seed=seed,epoch=epoch,budget_regime=regime,**pair_aggregate(values)))
    preselected=[row for row in pairs if row['group'] in (263,238)]
    terminal=[row for row in pairs if row['epoch']==8]
    extrema=[]
    for seed in SEEDS:
        rows=[row for row in terminal if row['seed']==seed]
        extrema.extend([dict(selection='posthoc_min_terminal_delta_wrong_tiebreak_group',**min(rows,key=lambda x:(x['delta_wrong'],x['group']))),
            dict(selection='posthoc_max_terminal_delta_wrong_tiebreak_group',**min(rows,key=lambda x:(-x['delta_wrong'],x['group']))),
            dict(selection='posthoc_max_terminal_delta_loss_tiebreak_group',**min(rows,key=lambda x:(-x['delta_loss'],x['group'])))])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'epoch_summaries.json',epoch_summaries)
    write_json(output/'budget_strata.json',strata)
    write_json(output/'paired_epoch_summaries.json',paired_epochs)
    write_json(output/'paired_budget_strata.json',paired_strata)
    write_json(output/'preselected_train_cases.json',preselected)
    write_json(output/'posthoc_terminal_extrema.json',extrema)
    with (output/'all_matched_train_cases.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(pairs)
    summary=dict(status='completed_saved_online_train_diagnosis',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='POST HOC after overall VALID outcomes; no new frozen-checkpoint reevaluation, no paired TRAIN-versus-VALID causal subtraction',
        source_sha256=sha256(__file__),registration_sha256=sha256(registration_path),input_files_hashed=len(hashes),
        raw_logs=48,epoch_receipts=48,terminal_result_receipts=6,raw_group_rows=12288,matched_group_pairs=len(pairs),
        train_groups_per_epoch=256,target_occurrences_per_epoch=3141,
        first_and_terminal_epoch=[row for row in epoch_summaries if row['epoch'] in (1,8)],
        terminal_epoch_pairs=[row for row in paired_epochs if row['epoch']==8],
        terminal_budget_pairs=[row for row in paired_strata if row['epoch']==8],
        preselected_groups=[263,238],result_receipts=result_checks,
        interpretation_limits=['Online observations occur at changing generator states, group order and per-case random noise.',
            'Mean margin is benign minus fraud; positive is the desired attack direction. These are distributions of group means, not per-node margins.',
            'Group-mean Softplus matches the equally weighted group training objective; occurrence-weighted summaries are descriptive.',
            'Wrong counts alone cannot identify preserved/new per-node successes because TRAIN rows contain aggregate counts, not predictions.',
            'Repeated target memberships are correlated. No IID occurrence inference or causal attribution is claimed.',
            'TRAIN and VALID are different cohorts; differences are not a paired generalization-gap experiment.',
            'Observed weak terminal online TRAIN objective may motivate an independently frozen next study, but does not establish optimizer/temperature/edge-score causation.'],
        model_queries=0,tensor_loads=0,dataset_loads=0,API_calls=0,GPU_hours=0,
        process_cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.monotonic()-started)
    write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',dict(status='completed',source_sha256=sha256(__file__),registration_sha256=sha256(registration_path),
        inputs=hashes,outputs={path.name:sha256(path) for path in output.iterdir() if path.is_file()},
        model_queries=0,tensor_loads=0,dataset_loads=0,API_calls=0,GPU_hours=0,
        process_cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=summary['status'],output=str(output),source_sha256=summary['source_sha256'],
        input_files_hashed=len(hashes),process_cpu_seconds=summary['process_cpu_seconds'],wall_seconds=summary['wall_seconds'])))


if __name__=='__main__':main()
