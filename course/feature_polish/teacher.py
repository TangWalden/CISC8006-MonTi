"""Fit one fresh GAGA on original TRAIN using a fixed, previously chosen horizon.

The original full-TRAIN group-input convention is reused. No validation/test
label indexes, checkpoint selection, official victim loading or API calls occur.
Final TRAIN metrics are explicitly in-sample, not a readiness/generalization gate.
"""
import argparse
import json
import math
import pathlib
import time
import traceback

from course.feature_polish.common import ROOT,private_output,sha256,validate_protocol,write_json


SETTINGS = dict(seed=20261103,epochs=38,batch_size=512,lr=.001,weight_decay=.0001,train_nodes=6595)
SEQUENCE_PATH = 'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy'


def constructor(width):
    from models.victim_model import GAGA
    return GAGA(width,64,2,num_relations=1,n_hops=2,n_heads=4,ff_dim=128,n_layers=3,dropout=.1)


def load_teacher(protocol,data):
    """Load only this phase's fresh full-TRAIN teacher, with byte/state checks."""
    import torch
    from course.dual_surrogate.runtime import model_hash
    reference=protocol['teacher_ref']
    relative=pathlib.Path(reference['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Teacher manifest must remain repository-relative.')
    manifest_path=ROOT/relative
    if sha256(manifest_path)!=reference['sha256']:
        raise RuntimeError('Fresh full-TRAIN teacher manifest changed.')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['provenance']['source_commit']!=protocol.get('teacher_training_source',protocol['source_commit']):
        raise RuntimeError('Fresh teacher training source differs from its registered lineage.')
    if manifest['provenance']['protocol_sha256']!=protocol['teacher_preparation_protocol_sha256']:
        raise RuntimeError('Fresh teacher preparation protocol differs from its registered lineage.')
    if (manifest.get('status')!='completed' or manifest['settings']!=SETTINGS or
            manifest['labels_scope']!='original TRAIN only; full-TRAIN group caches' or
            manifest['official_victim_loaded'] or manifest['original_VALID_used'] or manifest['TEST_used']):
        raise RuntimeError('Fresh teacher scope/settings do not match the fixed phase.')
    checkpoint=pathlib.Path(manifest['checkpoint'])
    if checkpoint.is_absolute() or '..' in checkpoint.parts:
        raise RuntimeError('Teacher checkpoint must remain relative to its manifest.')
    checkpoint=manifest_path.parent/checkpoint
    if sha256(checkpoint)!=manifest['checkpoint_sha256']:
        raise RuntimeError('Fresh full-TRAIN teacher checkpoint changed.')
    model=constructor(data.feat_dim)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu'))
    model.to(data.device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if model_hash(model)!=manifest['final_state_sha256']:
        raise RuntimeError('Restored fresh teacher state differs from its receipt.')
    return model,manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=private_output(args.output,create=True);(output/'raw').mkdir()
    started,cpu_started=time.monotonic(),time.process_time()
    counts=dict(optimizer_steps=0,training_forwards=0,final_train_forwards=0)
    result=dict(status='failed',official_victim_loaded=False,original_VALID_used=False,TEST_used=False,API_calls=0)
    try:
        protocol,provenance=validate_protocol(args.protocol)
        settings=protocol['teacher_settings']
        if any(settings.get(key)!=value for key,value in SETTINGS.items()):
            raise RuntimeError('Frozen full-TRAIN teacher settings differ from implemented constants.')
        deadline=started+protocol['teacher_wall_seconds']
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('Teacher frozen wall deadline reached.')
        write_json(output/'registered.json',dict(provenance=provenance,settings=SETTINGS,
            source_sha256=sha256(__file__),labels_scope='Only original TRAIN labels are indexed',
            checkpoint_rule='Terminal38 only, fixed before this phase; no new held-out selection',
            original_VALID_used=False,TEST_used=False,official_victim_loaded=False),exclusive=True)
        import dgl
        import numpy as np
        import torch
        import torch.nn.functional as F
        from utils.utils import set_seed
        from course.dual_surrogate.runtime import checked_logits,model_hash

        device=torch.device(protocol['teacher_device'])
        if device.type=='cpu' and torch.cuda.is_available():
            raise RuntimeError('The registered CPU teacher must not expose a CUDA device.')
        if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
            raise RuntimeError('The registered CUDA teacher requires exactly one visible device.')
        torch.set_num_threads(8)
        graphs,_=dgl.load_graphs(str(ROOT/protocol['data']['graph']))
        graph=graphs[0].to('cpu')
        raw_indices=np.load(ROOT/protocol['data']['train_indices'],allow_pickle=False).reshape(-1)
        ids_np=raw_indices.astype(np.int64)
        if (not np.array_equal(raw_indices.astype(np.float64),ids_np.astype(np.float64)) or
                len(ids_np)!=SETTINGS['train_nodes'] or len(set(ids_np.tolist()))!=len(ids_np) or
                bool((ids_np<0).any()) or bool((ids_np>=graph.num_nodes()).any())):
            raise RuntimeError('Original TRAIN identities differ from their exact integral cohort.')
        ids=torch.tensor(ids_np,dtype=torch.long)
        # The serialized graph contains all labels: index only original TRAIN,
        # then discard the global field before any classifier input or fitting.
        all_labels=graph.ndata.pop('y')
        labels=all_labels[ids].to(torch.long).clone()
        del all_labels,graphs
        if not bool(((labels==0)|(labels==1)).all()) or len(torch.unique(labels))!=2:
            raise RuntimeError('Full original TRAIN must retain both binary classes.')
        width=int(graph.ndata['x'].shape[1])
        sequences=np.load(ROOT/SEQUENCE_PATH,allow_pickle=False,mmap_mode='r')
        if sequences.shape!=(graph.num_nodes(),7,width) or sequences.dtype!=np.float32:
            raise RuntimeError('Original full-TRAIN sequence cache shape/dtype changed.')
        set_seed(SETTINGS['seed']);dgl.seed(SETTINGS['seed']);dgl.random.seed(SETTINGS['seed'])
        model=constructor(width).to(device)
        initial=model_hash(model)
        expected_initial=settings.get('reference_initial_state_sha256')
        if expected_initial is not None and initial!=expected_initial:
            raise RuntimeError('Fresh full-TRAIN teacher initial state differs from the frozen original initialization.')
        optimizer=torch.optim.Adam(model.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
        class_counts=[int((labels==label).sum()) for label in (0,1)]
        class_weights=torch.tensor([len(ids)/(2*count) for count in class_counts],device=device,dtype=torch.float32)
        def forward(batch_ids):
            check()
            batch=np.array(sequences[batch_ids.numpy()],dtype=np.float32,copy=True)
            return checked_logits(model(torch.from_numpy(batch).to(device)),len(batch_ids))
        history=[]
        for epoch in range(1,SETTINGS['epochs']+1):
            check();model.train();order=torch.randperm(len(ids));losses=[]
            with (output/'raw'/('train_epoch_%03d.jsonl'%epoch)).open('x',encoding='utf-8',newline='\n') as stream:
                for positions in order.split(SETTINGS['batch_size']):
                    optimizer.zero_grad();counts['training_forwards']+=1
                    logits=forward(ids[positions]);truth=labels[positions].to(device)
                    loss=F.cross_entropy(logits,truth,weight=class_weights)
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite fresh teacher training loss.')
                    loss.backward();optimizer.step();counts['optimizer_steps']+=1
                    row=dict(epoch=epoch,batch=len(losses)+1,positions=positions.tolist(),node_ids=ids[positions].tolist(),
                        targets=len(positions),loss=float(loss.detach()),correct=int((logits.detach().argmax(1)==truth).sum()))
                    stream.write(json.dumps(row,allow_nan=False)+'\n');losses.append(row['loss'])
            row=dict(epoch=epoch,optimizer_steps=counts['optimizer_steps'],training_forwards=counts['training_forwards'],
                batches=len(losses),mean_batch_loss=sum(losses)/len(losses),validation_performed=False)
            history.append(row);write_json(output/('epoch_%03d.json'%epoch),row,exclusive=True)
            print('FULL_TRAIN_TEACHER_EPOCH',json.dumps(row),flush=True)
        if counts['optimizer_steps']!=494 or counts['training_forwards']!=494:
            raise RuntimeError('Fixed38epochs/full6595TRAIN/batch512 update count differs.')
        model.eval();correct=0;confusion=[[0,0],[0,0]];loss_sum=0.
        with torch.no_grad(),(output/'raw'/'final_in_sample_train_predictions.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for positions in torch.arange(len(ids)).split(SETTINGS['batch_size']):
                counts['final_train_forwards']+=1
                logits=forward(ids[positions]);truth=labels[positions].to(device);pred=logits.argmax(1)
                loss_sum+=float(F.cross_entropy(logits,truth,reduction='sum'))
                correct+=int((pred==truth).sum())
                for y,predicted in zip(truth.cpu().tolist(),pred.cpu().tolist()):confusion[y][predicted]+=1
                stream.write(json.dumps(dict(node_ids=ids[positions].tolist(),labels=truth.cpu().tolist(),logits=logits.cpu().tolist()),allow_nan=False)+'\n')
        checkpoint=output/'gaga_full_train_teacher.pt'
        with checkpoint.open('xb') as stream:torch.save(model.state_dict(),stream)
        final=model_hash(model)
        if initial==final or sha256(checkpoint)==protocol.get('sealed_victim_sha256'):
            raise RuntimeError('Fresh teacher did not train or unexpectedly equals the sealed victim bytes.')
        identity=__import__('hashlib').sha256(json.dumps(sorted(ids_np.tolist()),separators=(',',':')).encode()).hexdigest()
        manifest=dict(status='completed',schema='monti_full_train_gaga_teacher_v1',settings=SETTINGS,
            execution_device=str(device),reference_initial_state_sha256=expected_initial,initial_state_matches_reference=(initial==expected_initial if expected_initial is not None else None),
            checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint),initial_state_sha256=initial,final_state_sha256=final,
            constructor=dict(feat_dim=width,emb_dim=64,num_classes=2,num_relations=1,n_hops=2,n_heads=4,ff_dim=128,n_layers=3,dropout=.1),
            full_train_nodes=len(ids),train_ids_sha256=identity,train_class_counts=class_counts,
            class_weights=class_weights.cpu().tolist(),labels_scope='original TRAIN only; full-TRAIN group caches',
            checkpoint_selection='Terminal38 only; fixed before new calls from previously selected internal horizon',
            official_victim_loaded=False,original_VALID_used=False,TEST_used=False,
            cache=dict(sequence=dict(path=SEQUENCE_PATH,sha256=sha256(ROOT/SEQUENCE_PATH))),
            final_in_sample_train=dict(accuracy=correct/len(ids),unweighted_cross_entropy=loss_sum/len(ids),confusion_true_by_prediction=confusion,
                held_out=False,calibrated=False),counters=counts,provenance=provenance)
        write_json(output/'teacher_artifacts.json',manifest,exclusive=True)
        result.update(status='completed',teacher_artifacts_sha256=sha256(output/'teacher_artifacts.json'),checkpoint_sha256=sha256(checkpoint),
            initial_state_sha256=initial,final_state_sha256=final,in_sample_train=manifest['final_in_sample_train'])
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(counters=counts,elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu_started)
        write_json(output/'result.json',result,exclusive=True);print('FULL_TRAIN_TEACHER_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
