"""Freeze committed source, TRAIN probes and references before new training/API."""
import argparse
import json
import pathlib
import subprocess

from course.alignment_agent.common import ARMS, ROOT, sha256, write_json
from course.alignment_agent.api_bridge import CAPS, PROMPT, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    stage=pathlib.Path(args.stage).resolve()
    if (ROOT/'runs').resolve() not in stage.parents:
        raise RuntimeError('Execution files must remain in the ignored stage.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit intended source before freezing execution.')
    master=json.loads((ROOT/'course/ALIGNMENT_AGENT_V2_PROTOCOL.json').read_text())
    old=json.loads((ROOT/'evidence/new_methods_20261103/alignment_execution_protocol_v1.json').read_text())
    diagnostic=json.loads((stage/'gradient_diagnostic_attempt1/result.json').read_text())
    if diagnostic['status']!='completed' or diagnostic['cases']!=32 or diagnostic['optimizer_steps']!=0:
        raise RuntimeError('The registered diagnostic has not completed.')
    selection=json.loads((stage/'gradient_diagnostic_attempt1/selected_train_groups.json').read_text())
    probes=[]
    for name,count in (('one_edge',3),('one_injection_multi_edge',3),('multi_injection',2)):
        rows=[r for r in selection['groups'] if r['stratum']==name]
        probes.extend(sorted(r['target_group'] for r in rows)[:count])
    if len(probes)!=8:
        raise RuntimeError('The predetermined 3/3/2 TRAIN probe allocation is infeasible.')
    reference=json.loads((stage/'filter_reference_v2.json').read_text())
    if reference['status']!='completed' or reference['model_forwards']!=0:
        raise RuntimeError('Independent filter reference fitting did not complete.')
    sources={row['path']:row['sha256'] for row in old['code']}
    sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    sources['course/llm_fusion/invoke_batch.ps1']=sha256(ROOT/'course/llm_fusion/invoke_batch.ps1')
    sources['course/ALIGNMENT_AGENT_V2_PROTOCOL.json']=sha256(ROOT/'course/ALIGNMENT_AGENT_V2_PROTOCOL.json')
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:
            raise RuntimeError('A registered old source changed: '+name)
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
    protocol=dict(master, status='frozen',owner_execution_authorized=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files=sources,inputs=old['inputs'],base_config_sha256=old['base_config_sha256'],
        train_groups=old['train_groups'],valid_groups=old['valid_groups'],gradient_probe_groups=probes,
        reference=old['reference'],stage_root=str(stage),
        alignment_reference_file=dict(path='runs/owner_methods_20261103_attempt1/alignment-reference-v1.json',sha256=old['reference_sha256']),
        filter_reference_file=dict(path=str((stage/'filter_reference_v2.json').relative_to(ROOT)),sha256=sha256(stage/'filter_reference_v2.json')),
        diagnostic_guardian_sha256=sha256(stage/'gradient_diagnostic_guardian.json'),
        diagnostic_gpu_hours=json.loads((stage/'gradient_diagnostic_guardian.json').read_text())['charged_gpu_wall_hours'],
        api=dict(provider='deepseek',model='deepseek-flash',model_version='DeepSeek-V4.1-Flash-1017 advertised; backend weights not independently certified',
                 caps=CAPS,max_logical_requests=21,prompt_sha256=sha(PROMPT.encode()),
                 pricing_url='https://api-docs.deepseek.com/quick_start/pricing/',
                 registered_peak_input_usd_per_million=.30,registered_peak_output_usd_per_million=1.20))
    if tuple(protocol['arms'])!=ARMS:
        raise RuntimeError('Master arms and implementation differ.')
    output=stage/'execution_protocol_v2.json'
    write_json(output,protocol,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=protocol['source_commit'],protocol_sha256=sha256(output),
                         gradient_probe_groups=probes,output=str(output))),flush=True)


if __name__=='__main__':
    main()
