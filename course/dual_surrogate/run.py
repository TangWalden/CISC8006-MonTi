"""One committed-source TRAIN-only attacker job. No victim evaluation here."""
import argparse
import json
import pathlib
import time
import traceback

import dgl
import torch

from utils.utils import set_seed
from course.dual_surrogate.common import ARMS, ROOT, private_output, sha256, validate_protocol, write_json
from course.dual_surrogate.runtime import config_for,load_data
from course.dual_surrogate.experiment import DualExperiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True,choices=ARMS);parser.add_argument('--seed',type=int,choices=[3,4,5],required=True)
    args=parser.parse_args();output=private_output(args.output,create=True);(output/'raw').mkdir()
    start=time.monotonic();engine=None
    result=dict(status='failed',arm=args.arm,seed=args.seed,test_queries=0,victim_loaded=False)
    try:
        p,provenance=validate_protocol(args.protocol)
        gate=json.loads((ROOT/p['preflight_ref']['path']).read_text(encoding='utf-8'))
        if sha256(ROOT/p['preflight_ref']['path'])!=p['preflight_ref']['sha256'] or gate['status']!='passed':
            raise RuntimeError('Real TRAIN preflight did not pass.')
        torch.set_num_threads(8);set_seed(args.seed);dgl.seed(args.seed);dgl.random.seed(args.seed)
        data=load_data(p,args.seed);config=config_for(args.seed)
        write_json(output/'registered.json',dict(provenance=provenance,arm=args.arm,seed=args.seed,
            teacher_artifacts_ref=p['teacher_artifacts_ref'],h_cache='fixed original h_GCN for every arm',
            counter_caps=dict(generator_forwards=2048,optimizer_steps=2048),epochs=8,
            evaluation_performed=False,sealed_victim_deserialized=False),exclusive=True)
        engine=DualExperiment(config,data,p,args.arm,ROOT/p['teacher_directory'],output,
            start+p['train_job_wall_seconds'])
        write_json(output/'initial_state.json',dict(arm=args.arm,seed=args.seed,
            initial_state_sha256=engine.initial_model_hash,captured_before_training=True,optimizer_steps=0),exclusive=True)
        result.update(engine.run())
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if engine is not None:result.update(generator_forwards=engine.generator_forwards,
            optimizer_steps=engine.optimizer_steps,classifier_forwards=engine.classifier_forwards)
        result.update(elapsed_seconds=time.monotonic()-start)
        write_json(output/'result.json',result,exclusive=True);print('DUAL_TRAIN_END',json.dumps(result),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
