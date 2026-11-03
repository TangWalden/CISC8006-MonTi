"""Post-hoc numerical cache comparison, without models or tensor deserialization.

Overall development outcomes are already known. Definitions/source are frozen
before opening four immutable NPY caches. All 374 unique selected VALID targets
are included regardless of correctness. This is not a causal experiment.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root=pathlib.Path(args.repo).resolve();stage=pathlib.Path(args.stage).resolve();out=pathlib.Path(args.output).resolve()
    out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    source_sha=sha(pathlib.Path(__file__))
    write(out/'registered_before_cache_reads.json',dict(status='POSTHOC_after_overall_VALID_outcomes_known',
        source_sha256=source_sha,cohort='All374 unique node IDs from all128 saved clean VALID groups; no outcome filtering',
        comparisons='Self token; six relation-token means and group counts under FIT80 versus full original TRAIN masks',
        threshold_for_changed_vector=1e-6,cache_read_limit=4,maximum_selected_nodes=512,maximum_wall_seconds=300,
        model_queries=0,training_steps=0,API_calls=0,PyTorch_tensor_loads=0,graph_loads=0))
    protocol_path=stage/'experiment_protocol_v4.json';protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    teachers=stage/'teachers_attempt1';manifest_path=teachers/'fit_cache_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    clean=stage/'evaluation_attempt1/clean_predictions.jsonl';records=[json.loads(line) for line in clean.read_text().splitlines()]
    ids=sorted({n for row in records for n in row['node_ids']})
    if len(records)!=128 or len(ids)!=374 or sum(len(row['node_ids']) for row in records)!=1091:raise RuntimeError('Fixed development cohort mismatch.')
    split_path=teachers/'train_split_private.json';split=json.loads(split_path.read_text())
    if set(ids)&set(split['original_train_nodes']):raise RuntimeError('Selected VALID nodes overlap TRAIN identities.')
    original={row['path']:row['sha256'] for row in protocol['inputs']}
    directory='Datasets/GossipCop/data_split/'
    mapping={
        'full_sequence':(root/(directory+'gossipcop_0.05_gaga_sequence_0.npy'),original[directory+'gossipcop_0.05_gaga_sequence_0.npy']),
        'full_degree':(root/(directory+'gossipcop_0.05_gaga_degree_0.npy'),original[directory+'gossipcop_0.05_gaga_degree_0.npy']),
        'fit_sequence':(teachers/manifest['files']['sequences']['path'],manifest['files']['sequences']['sha256']),
        'fit_degree':(teachers/manifest['files']['d']['path'],manifest['files']['d']['sha256'])}
    hashes={}
    for key,(path,wanted) in mapping.items():
        actual=sha(path)
        if actual!=wanted:raise RuntimeError('Frozen cache changed: '+key)
        hashes[key]=dict(path=str(path.relative_to(root)),sha256=actual)
    for path in (protocol_path,manifest_path,clean,split_path):hashes[str(path.relative_to(root))]=dict(sha256=sha(path))
    write(out/'input_hashes.json',hashes)
    import numpy as np
    arrays={k:np.load(path,mmap_mode='r',allow_pickle=False) for k,(path,_) in mapping.items()}
    a=arrays['full_sequence'][ids].astype(np.float64);b=arrays['fit_sequence'][ids].astype(np.float64)
    da=arrays['full_degree'][ids].reshape(len(ids),7).astype(np.float64)
    db=arrays['fit_degree'][ids].reshape(len(ids),7).astype(np.float64)
    if a.shape!=b.shape or a.shape[1:]!=(7,768) or len(ids)>512:raise RuntimeError('Unexpected bounded sequence shape.')
    if not all(np.isfinite(x).all() for x in (a,b,da,db)):raise RuntimeError('Nonfinite cached input.')
    names=('self','hop1_train_benign','hop1_train_fraud','hop1_unknown_including_self',
           'hop2_train_benign','hop2_train_fraud','hop2_unknown_including_self')
    tokens=[]
    for token,name in enumerate(names):
        x,y=a[:,token],b[:,token];delta=np.linalg.norm(x-y,axis=1)
        nx,ny=np.linalg.norm(x,axis=1),np.linalg.norm(y,axis=1);valid=(nx>1e-12)&(ny>1e-12)
        cosine=np.clip((x[valid]*y[valid]).sum(1)/(nx[valid]*ny[valid]),-1,1)
        relative=delta[nx>1e-12]/nx[nx>1e-12]
        tokens.append(dict(token=token,name=name,nodes=len(ids),exact_vector_changes=int(np.any(x!=y,axis=1).sum()),
            changes_above_tolerance=int((np.max(np.abs(x-y),axis=1)>1e-6).sum()),max_abs_difference=float(np.abs(x-y).max()),
            median_relative_L2_to_full=float(np.median(relative)),median_cosine=float(np.median(cosine)),cosine_valid_nodes=int(valid.sum()),
            full_mean_group_count=float(da[:,token].mean()),fit_mean_group_count=float(db[:,token].mean()),
            full_empty_groups=int((da[:,token]==0).sum()),fit_empty_groups=int((db[:,token]==0).sum())))
    hops=[]
    for hop in (1,2):
        j=1+(hop-1)*3;full_known=da[:,j:j+2].sum(1);fit_known=db[:,j:j+2].sum(1)
        unknown_increase=db[:,j+2]-da[:,j+2]
        if not np.array_equal(da[:,j:j+3].sum(1),db[:,j:j+3].sum(1)):raise RuntimeError('Total neighborhood counts changed.')
        if not np.array_equal(full_known-fit_known,unknown_increase):raise RuntimeError('Group-count redistribution does not reconcile.')
        hops.append(dict(hop=hop,nodes_with_known_to_unknown_moves=int((unknown_increase>0).sum()),
            full_total_known_incidence=int(full_known.sum()),fit_total_known_incidence=int(fit_known.sum()),
            known_incidence_moved_to_unknown=int(unknown_increase.sum()),
            fraction_full_known_incidence_moved_to_unknown=float(unknown_increase.sum()/full_known.sum()),
            all_total_neighbor_counts_identical=True,all_known_losses_equal_unknown_gains=True))
    result=dict(status='completed_saved_numeric_cache_comparison',source_sha256=source_sha,numpy_version=np.__version__,
        selected_unique_VALID_nodes=len(ids),selected_VALID_occurrences=1091,selected_VALID_groups=128,
        original_TRAIN_nodes=len(split['original_train_nodes']),teacher_FIT_nodes=len(split['fit_nodes']),teacher_TUNE_nodes=len(split['stop_nodes']),
        self_token_exactly_equal=bool(np.array_equal(a[:,0],b[:,0])),
        nodes_with_any_relation_vector_change_above_1e_6=int(np.any(np.abs(a[:,1:]-b[:,1:])>1e-6,axis=(1,2)).sum()),
        tokens=tokens,hops=hops,model_queries=0,optimizer_steps=0,API_calls=0,GPU_hours=0,
        PyTorch_imported='torch' in sys.modules,PyTorch_tensor_loads=0,NPY_caches_read=4,
        cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        limits=['Clean cached inputs, not attacked inputs or model decisions.',
            'Input divergence is not evidence that it caused the negative transfer.',
            'No new GAGA teacher VALID prediction exists in these records; overfit versus teacher-victim mismatch remains unresolved.',
            'No test indices, labels, predictions, article bodies or PyTorch tensors were read.',
            'Cosine/L2 are feature-space descriptors, not calibrated reliability or semantic truth.'])
    if result['PyTorch_imported'] or result['wall_seconds']>300:raise RuntimeError('Declared analysis scope exceeded.')
    write(out/'summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
