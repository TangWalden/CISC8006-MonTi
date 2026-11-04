"""Freeze a post-hoc diagnostic before new own-teacher/capture tensor reads."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT,private_output,sha256,write_json


BASE='runs/feature_polish_20261104_attempt4'
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_PROTOCOL_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit diagnostic source before registration.')
    directory=ROOT/BASE;base_path=directory/'experiment_protocol_v1.json'
    if sha256(base_path)!=BASE_PROTOCOL_SHA:raise RuntimeError('The completed feature experiment changed.')
    base=json.loads(base_path.read_text(encoding='utf-8'))
    complete=json.loads((directory/'queue_complete.json').read_text(encoding='utf-8'))
    if (base['source_commit']!=BASE_SOURCE or complete['status']!='completed' or
            complete['experiment_protocol_sha256']!=BASE_PROTOCOL_SHA or base['execution_device']!='cpu'):
        raise RuntimeError('This diagnostic only compares the complete original CPU study.')
    sources=dict(base['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:raise RuntimeError('The reused scientific implementation changed: '+name)
    sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    inputs={}
    def retain(path,wanted=None):
        path=path.resolve()
        if ROOT not in path.parents:raise RuntimeError('Input escaped the project.')
        measured=sha256(path)
        if wanted is not None and measured!=wanted:raise RuntimeError('Registered input changed: '+str(path))
        name=path.relative_to(ROOT).as_posix();inputs[name]=measured
        return dict(path=name,sha256=measured)
    for row in base['inputs']:retain(ROOT/row['path'],row['sha256'])
    base_ref=retain(base_path,BASE_PROTOCOL_SHA)
    retain(directory/'queue_complete.json');retain(directory/'all3_refinements_complete.json')
    teacher=ROOT/base['teacher_directory']
    for name in ('registered.json','result.json','teacher_artifacts.json','gaga_full_train_teacher.pt'):
        retain(teacher/name)
    teacher_ref=retain(ROOT/base['teacher_ref']['path'],base['teacher_ref']['sha256'])
    runs={}
    for seed in (3,4,5):
        run=directory/'queue_attempt1/refinements'/('seed%d'%seed)
        result=json.loads((run/'result.json').read_text(encoding='utf-8'))
        if (result['status']!='completed' or result['groups']!=128 or result['seed']!=seed or
                result['source_commit']!=BASE_SOURCE or result['protocol_sha256']!=BASE_PROTOCOL_SHA):
            raise RuntimeError('A complete original refinement run is required.')
        retain(run/'registered.json')
        captures=[]
        for row in result['exports']:
            capture=retain(run/row['path'],row['sha256'])
            captures.append(dict(group=row['group'],**capture))
        if [row['group'] for row in captures]!=base['valid_groups']:
            raise RuntimeError('Saved capture coverage/order changed.')
        predictions={arm:retain(directory/'evaluation_attempt1'/('%s_seed%d_predictions.jsonl'%(arm,seed)))
                     for arm in ('baseline','gaga_pgd')}
        runs[str(seed)]=dict(result_ref=retain(run/'result.json'),
            groups_ref=retain(run/'groups.jsonl',result['groups_jsonl_sha256']),
            captures=captures,victim_prediction_refs=predictions)
    out=private_output(args.output,create=True)
    p=dict(schema='monti_feature_parity_diagnostic_v1',status='frozen',post_hoc=True,owner_execution_authorized=True,
        context='Selected after the complete negative five-arm development result; not a preregistered efficacy study.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files=sources,base_experiment_ref=base_ref,teacher_ref=teacher_ref,
        inputs=[dict(path=name,sha256=value) for name,value in inputs.items()],runs=runs,
        seeds=[3,4,5],valid_groups=base['valid_groups'],arms=['baseline','gaga_pgd'],
        deadline_seconds=1800,cpu_threads=8,execution_device='cpu',cuda_visible_devices='',
        sequence_tolerances=dict(atol=2e-5,rtol=2e-5),logit_tolerances=dict(atol=2e-4,rtol=2e-4),
        expected_counts=dict(capture_tensor_loads=384,hard_graph_reconstructions=384,affine_builders=384,
            canonical_sequence_updates=768,fresh_teacher_completed_forwards=768,stored_affine_comparisons=768,
            fresh_teacher_forward_dispatches=768,guard_alias_checks=384,data_load_dispatches=1,data_loads_completed=1,
            teacher_load_dispatches=1,teacher_loads_completed=1,new_affine_model_forwards=0,
            generator_model_constructions=0,generator_forwards=0,GCN_model_constructions=0,GCN_forwards=0,
            official_victim_model_constructions=0,official_victim_forwards=0,optimizer_steps=0,autograd_calls=0,API_calls=0,TEST_queries=0),
        data_scope='Existing original graph/caches, TRAIN masks and frozen own-teacher checkpoint plus384saved captures. Dataset construction may materialize stored labels; only TRAIN labels enter masks. Saved completed VALID predictions are read, not new victim queries.',
        cache_checks='Entire original zero-self adjacency support symmetry; selected374VALIDunknown1/2hop cache denominators and two-hop logical support. Preserve unsupported-domain/mismatch findings without binarizing away original nonunit arcs.',
        interpretation='Retain all768 comparisons/violations. Model-scale differences are not calibration or comparable gradient magnitudes; report finite two-endpoint margin/loss changes, not derivative slopes. No arm/case selection or efficacy claim.',
        new_affine_model_forwards=0,final_TEST_allowed=False,full_test_candidate_gate='closed after both failed development criteria')
    path=out/'protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),inputs=len(inputs),output=str(path))))


if __name__=='__main__':main()
