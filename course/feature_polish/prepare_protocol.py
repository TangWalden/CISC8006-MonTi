"""Bind inherited scientific bytes and all retained baseline models, no ML."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT,MASTER,sha256,write_json,private_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit before freezing execution.')
    p=json.loads((ROOT/MASTER).read_text(encoding='utf-8'))
    ancestor=ROOT/p['ancestor_protocol']
    old=json.loads(ancestor.read_text(encoding='utf-8'))
    sources=dict(old['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Inherited scientific source changed: '+name)
    sources.update({f.relative_to(ROOT).as_posix():sha256(f) for f in pathlib.Path(__file__).parent.glob('*.py')})
    sources[MASTER]=sha256(ROOT/MASTER)
    for name in sources:
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
    for row in old['inputs']:
        if sha256(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Scientific input changed: '+row['path'])
    retained=[dict(path=ancestor.relative_to(ROOT).as_posix(),sha256=sha256(ancestor))]
    baselines={}
    for seed in p['seeds']:
        directory=ROOT/p['baseline_directory']/('legacy_seed%d'%seed)
        result_path=directory/'result.json'
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if (result['status']!='completed' or result['seed']!=seed or result['arm']!='legacy' or
                result['source_mode']!='legacy' or result['optimizer_steps']!=2048 or result['epochs']!=8 or
                sha256(directory/'final_model.pt')!=result['final_model_sha256']):
            raise RuntimeError('Retained original-GCN baseline is not the registered terminal model.')
        baselines[str(seed)]=dict(directory=directory.relative_to(ROOT).as_posix(),
            final_model_sha256=result['final_model_sha256'],result_sha256=sha256(result_path))
        for name in ('result.json','registered.json','initial_state.json','final_model.pt'):
            f=directory/name;retained.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)))
    initial_ref=ROOT/p['teacher_initial_reference']
    initial=json.loads(initial_ref.read_text(encoding='utf-8'))
    # The exact old teacher initialization is known before any new fit, whereas
    # its fitted weights are never reused or deserialized here.
    teacher=initial['teachers']['gaga']
    expected=teacher.get('initial_state_sha256',teacher.get('initial_model_state_sha256'))
    if not expected:raise RuntimeError('Missing original independent GAGA initialization hash.')
    if expected!=p['teacher_settings']['reference_initial_state_sha256']:
        raise RuntimeError('Original independent GAGA initialization differs from the committed reference.')
    retained.append(dict(path=initial_ref.relative_to(ROOT).as_posix(),sha256=sha256(initial_ref)))
    stage=private_output(args.stage,create=True)
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=old['inputs'],retained_inputs=retained,baseline_runs=baselines,
        teacher_reference_initial_state_sha256=expected,stage_root=str(stage),
        execution_device=None,cuda_visible_devices='',physical_gpu_uuid=None)
    path=stage/'preparation_protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
