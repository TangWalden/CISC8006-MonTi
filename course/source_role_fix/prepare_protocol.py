"""Freeze committed code and unchanged scientific inputs before new calls."""
import argparse
import json
import pathlib
import subprocess

from course.source_role_fix.common import ROOT,MASTER,ARMS,private_output,sha256,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--ancestor-protocol',default='runs/dual_surrogate_20261103_attempt1/experiment_protocol_v4.json')
    args=parser.parse_args()
    stage=private_output(args.stage,create=True)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit before freezing new execution.')
    ancestor=ROOT/args.ancestor_protocol
    prior=json.loads(ancestor.read_text(encoding='utf-8'))
    master=json.loads((ROOT/MASTER).read_text(encoding='utf-8'))
    p=dict(prior)
    for key in ('prior_progress_ref','construction_amendment','resource_amendment','timing_amendment',
                'reused_training_runs','reuse_passed_preflight','preflight_ref','scientific_core_source'):
        p.pop(key,None)
    p.update(master)
    if tuple(p['arms']) != ARMS:
        raise RuntimeError('The narrow two-arm design is required.')
    sources=dict(prior['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name) != wanted:
            raise RuntimeError('Frozen original scientific source changed: '+name)
    sources.update({f.relative_to(ROOT).as_posix():sha256(f) for f in pathlib.Path(__file__).parent.glob('*.py')})
    sources[MASTER]=sha256(ROOT/MASTER)
    for name in sources:
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
    for row in prior['inputs']:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('Scientific input bytes changed: '+row['path'])
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=prior['inputs'],stage_root=str(stage),
        retained_inputs=[dict(path=ancestor.relative_to(ROOT).as_posix(),sha256=sha256(ancestor))],
        ancestor_protocol_sha256=sha256(ancestor),ancestor_source=prior['source_commit'],
        accounting='Two aggregate GPU hours include preflight, six fresh TRAIN jobs, development evaluation and failures; charge maximum outer/guardian lifetime.')
    if p['companion_saved_graph_diagnostic']['enabled']:
        evaluation=ROOT/p['companion_saved_graph_diagnostic']['prior_evaluation_directory']
        old_result=evaluation/'result.json'
        result=json.loads(old_result.read_text(encoding='utf-8'))
        if result.get('status')!='completed' or len(result['summaries'])!=12:
            raise RuntimeError('The saved twelve-arm/seed development receipt is incomplete.')
        clean=evaluation/'clean_predictions.jsonl'
        diagnostic=dict(old_protocol_path=ancestor.relative_to(ROOT).as_posix(),old_protocol_sha256=sha256(ancestor),
            evaluation_path=evaluation.relative_to(ROOT).as_posix(),evaluation_result_sha256=sha256(old_result),
            clean_predictions_sha256=sha256(clean),prediction_files={},export_files=[],max_forwards=104,wall_seconds=300)
        retained=[old_result,clean]
        for summary in result['summaries']:
            identity=summary['arm']+'_seed'+str(summary['seed'])
            predictions=evaluation/(identity+'_predictions.jsonl')
            diagnostic['prediction_files'][identity]=dict(path=predictions.relative_to(ROOT).as_posix(),sha256=sha256(predictions))
            retained.append(predictions)
            if len(summary['exports'])!=8:raise RuntimeError('Expected eight predetermined saved exports per old run.')
            for row in summary['exports']:
                export=evaluation/row['path']
                if sha256(export)!=row['sha256']:raise RuntimeError('A saved export differs from its original result manifest.')
                diagnostic['export_files'].append(dict(arm=summary['arm'],seed=summary['seed'],group=row['group'],
                    path=export.relative_to(ROOT).as_posix(),sha256=row['sha256']))
                retained.append(export)
        if len(diagnostic['export_files'])!=96:raise RuntimeError('Expected exactly96saved attack graphs.')
        p['saved_teacher_diagnostic']=diagnostic
        p['retained_inputs'].extend(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)) for f in retained)
    path=stage/'execution_protocol_v1.json'
    write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
