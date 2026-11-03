"""Freeze committed source and all scientific inputs before the new phase."""
import argparse
import json
import pathlib
import subprocess
from course.dual_surrogate.common import ROOT,private_output,sha256,write_json
from course.dual_surrogate.verify import SEALED_GAGA_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
    stage=private_output(args.stage,create=True)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit before freezing the new phase.')
    master=ROOT/'course/DUAL_SURROGATE_PROTOCOL_V1.json'
    p=json.loads(master.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'evidence/alignment_agent_v2/execution_protocol_v2.json').read_text(encoding='utf-8'))
    sources=dict(old['source_files'])
    sources.update({f.relative_to(ROOT).as_posix():sha256(f) for f in pathlib.Path(__file__).parent.glob('*.py')})
    sources[master.relative_to(ROOT).as_posix()]=sha256(master)
    for name,wanted in sources.items():
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Registered source changed: '+name)
    inputs=list(old['inputs'])+p.pop('additional_input_manifest')
    for row in inputs:
        if sha256(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Input byte mismatch: '+row['path'])
    # This independent file hash does not deserialize or query the victim.
    victim='models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle'
    if sha256(ROOT/victim)!=SEALED_GAGA_SHA256:raise RuntimeError('Sealed victim bytes changed.')
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=inputs,stage_root=str(stage),sealed_victim_sha256=SEALED_GAGA_SHA256,
        sealed_victim=dict(path=victim,sha256=SEALED_GAGA_SHA256),
        accounting='Two aggregate GPU hours include preparation/preflight/train/evaluation/failures; max outer/guardian.')
    path=stage/'preparation_protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
