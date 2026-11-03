"""Fit the separately registered unoptimized TRAIN anomaly diagnostic on CPU."""
import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import time

from course.conditional_alignment.common import ROOT, sha256, validate_protocol, gc_only_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    started=time.monotonic()
    output=pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents or output.exists():
        raise RuntimeError('Use a new exclusive output below ignored runs/.')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU fitting requires empty CUDA visibility.')
    protocol=json.loads(pathlib.Path(args.protocol).read_text())
    if protocol.get('status')!='frozen' or not protocol.get('owner_execution_authorized'):
        raise RuntimeError('The CPU diagnostic protocol is not frozen/authorized.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Tracked source must be committed.')
    for row in protocol['code']:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('Source mismatch: '+row['path'])
    old=protocol['old_execution_protocol']
    if sha256(ROOT/old['path'])!=old['sha256']:
        raise RuntimeError('Original input registration changed.')
    base, provenance=validate_protocol(ROOT/old['path'])
    def expired(signum,frame):
        raise TimeoutError('CPU reference fitting wall-time cap reached.')
    signal.signal(signal.SIGALRM,expired)
    limit=int(protocol.get('max_wall_seconds',120))
    if not 1 <= limit <=120:
        raise RuntimeError('Invalid CPU fitting limit.')
    signal.alarm(limit)
    result=dict(status='failed',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                model_forwards=0,optimizer_steps=0,api_calls=0,gpu_hours=0)
    try:
        import torch
        from modules.data_handler import DataHandler
        from course.alignment_agent.filter import fit_reference
        torch.set_num_threads(8)
        data=DataHandler(gc_only_config(base,3,'cpu'))
        reference=fit_reference(data.graph,data.idx_train)
        reference.update(source_commit=provenance['source_commit'],
            source_sha256={r['path']:r['sha256'] for r in protocol['code']},
            inputs=base['inputs'],protocol_sha256=sha256(args.protocol),
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            purpose='Frozen unoptimized evaluation diagnostic; never attack/controller feedback.')
        result.update(status='completed',reference=reference)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.alarm(0)
        result['elapsed_process_seconds']=time.monotonic()-started
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,result,exclusive=True)
    print(json.dumps(dict(status=result['status'],output=str(output),sha256=sha256(output),
                         model_forwards=0,optimizer_steps=0,gpu_hours=0)),flush=True)


if __name__=='__main__':
    main()
