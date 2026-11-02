"""CPU-only reference fitting before any alignment-pilot optimization.

Run with CUDA_VISIBLE_DEVICES empty. This imports no detector or attack model,
loads no model weights, evaluates no model and writes no original data/cache.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import time

import torch

from modules.data_handler import DataHandler
from course.conditional_alignment.common import gc_only_config, sha256, validate_protocol, write_json
from course.conditional_alignment.reference import build_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Reference preparation requires CUDA_VISIBLE_DEVICES empty.')
    output = pathlib.Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing reference: ' + str(output))
    protocol, provenance = validate_protocol(args.protocol)
    torch.set_num_threads(8)
    started = time.monotonic()
    data = DataHandler(gc_only_config(protocol, 3, 'cpu'))
    settings = protocol['reference']
    reference = build_reference(data.graph, data.idx_train,
                                min_count=settings['min_count'],
                                gate_quantile=settings['gate_quantile'],
                                scale_floor=settings.get('scale_floor', 0.05),
                                chunk_size=settings.get('chunk_size', 4096))
    reference.update(provenance=provenance,
                     reference_algorithm_sha256=sha256(pathlib.Path(__file__).with_name('reference.py')),
                     registered_inputs=protocol['inputs'],
                     created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     source='original full TRAIN benign centers; original graph before injection',
                     model_forwards=0, optimizer_steps=0, gpu_hours=0,
                     elapsed_seconds=time.monotonic()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, reference, exclusive=True)
    print(json.dumps(dict(status='completed', reference=str(output),
                          benign_train_nodes=reference['benign_train_nodes'],
                          supported_buckets=sorted(reference['buckets']),
                          model_forwards=0, gpu_hours=0)), flush=True)


if __name__ == '__main__':
    main()
