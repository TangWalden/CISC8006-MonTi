"""Freeze complete original TEST cohorts, immutable method and finite resources."""
import argparse,pathlib,datetime,subprocess
from full_common import read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);rt=s.parent.parent;parent=s.parent/'channel_retrieval_20261116_attempt1';old=read(parent/'protocol.json');assert read(parent/'guardian_receipt.json')['status']=='completed'
 assert old['source_commit']=='6e2982ccf3634a1b924d561f92ff14b7163be075' and sha(parent/'protocol.json')=='d3a7875ff443c96277fd8518825749ba46b5580d313b622c5f8d64f78baad247'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==old['runtime_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
 inputs={str(parent/n):sha(parent/n) for n in ['protocol.json','guardian_receipt.json']}
 for n,h in old['sources'].items():inputs[str(parent/'source'/n)]=h
 datasets={};seeds=[0,1,2]
 for ds in old['datasets']:
  seal=read(parent/ds/'sealed_before_victim.json');pre=[e for e in seal['entries'] if e['seed']==0][:2];assert len(pre)==2
  for e in pre:inputs[str(parent/ds/e['path'])]=e['sha256']
  datasets[ds]=dict(seeds=seeds,preflight_entries=pre)
 gc=s.parent/'portal_full_20261113_attempt1';gp=read(gc/'protocol.json');gs=read(gc/'main/sealed_before_victim.json')
 assert sha(gc/'protocol.json')=='206225b7a23e40e941571fd050630d7eb42f6c2ed7b7d406317f98272885f5a1'
 assert sha(gc/'main/sealed_before_victim.json')=='3fe4e6b3b03e48d05877c68d44d53abd093f37dbcd2c147cc48837b0d40465f3'
 for n in ['protocol.json','main/sealed_before_victim.json']:inputs[str(gc/n)]=sha(gc/n)
 g=datasets['gossipcop'];g.update(test_groups=gp['test_groups'],test_members=gp['test_members'],native_parent=str(gc/'main'),native_cards=[{k:e[k] for k in ['seed','group','path','sha256']} for e in gs['entries']],native_prediction_reference=str(gc/'main/evaluation.jsonl'))
 # Hash the old native prediction file without reading prediction values into selection.
 g['native_prediction_reference_sha256']=sha(g['native_prediction_reference'])
 import numpy as np,scipy.sparse as sp
 d=s.parent/'yelpchi_assets_20261114_attempt1/Datasets/YelpChi_RTR/data_split'
 names=['test_target_mask_0.npy','target_assign_matrix_0.npz','test_idx_surro_0.npy','train_idx_surro_0.npy','valid_idx_surro_0.npy']
 for n in names:inputs[str(d/n)]=sha(d/n)
 groups=sorted(np.load(d/names[0],allow_pickle=False).astype(int).reshape(-1).tolist());assignment=sp.load_npz(d/names[1]).tocsr();members={str(k):assignment[k].indices.tolist() for k in groups}
 ids={k:set(np.load(d/(k+'_idx_surro_0.npy'),allow_pickle=False).astype(int).reshape(-1).tolist()) for k in ['test','train','valid']}
 assert not ids['test']&(ids['train']|ids['valid']);assert all(v and set(v)<=ids['test'] for v in members.values())
 datasets['yelpchi'].update(test_groups=groups,test_members=members)
 for ds,v in datasets.items():
  v['occurrences']=sum(map(len,v['test_members'].values()));v['unique_targets']=len(set(n for vv in v['test_members'].values() for n in vv));assert len(v['test_groups'])==len(set(v['test_groups']))
  assert (len(v['test_groups']),v['occurrences'],v['unique_targets'])==({'gossipcop':(909,10515,1221),'yelpchi':(548,2682,2452)}[ds])
 assert {(e['seed'],e['group']) for e in g['native_cards']}=={(seed,group) for seed in seeds for group in g['test_groups']}
 manifest=read(s/'source_manifest.json')
 p=dict(schema='channel_full_v1',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=manifest['source_commit'],sources=manifest['files'],inputs=inputs,runtime_root=str(rt),runtime_commit=old['runtime_commit'],method_parent=str(parent),method_source_commit=old['source_commit'],method_protocol_sha256=sha(parent/'protocol.json'),settings=old['settings'],datasets=datasets,arms=['native','channel','random_channel','random_joint'],TEST_allowed=True,selection_shards=2,resources=dict(CPU_threads_per_worker=4,max_selection_workers=4,nice=10,GPU_hours=0,API_calls=0,preflight_seconds=1800,selection_seconds=28800,evaluation_seconds=7200,audit_seconds=3600,minimum_available_GiB=48,minimum_disk_GiB=40,automatic_retry=False),caps_per_case=dict(physical_builds=20,teacher_contexts=110,backward_calls=2),primary_candidate='channel',primary='Compare pooled TEST MR against native author checkpoint, same-class equal-dose random and joint-pool equal-dose random separately in each dataset. Success requires higher mean MR than all three controls in BOTH datasets. Report every repetition and contrast without tuning, exclusions or seed expansion.',limitations=['Owner explicitly authorized full TEST on both datasets. No new tuning or training is permitted from these outputs.','GossipCop TEST was used by earlier historical comparisons; this is a frozen benchmark, not an untouched confirmation cohort. YelpChi original TEST has not been evaluated by this downstream method.','Author and independently TRAIN-fitted teacher checkpoints are fixed. Repetitions are conditional on these checkpoints, not three independent teacher/upstream trainings. Repeated targets within groups are correlated.','Native GossipCop captures are reused byte-for-byte. YelpChi native captures use the same original author checkpoint and frozen original budget/candidate/RNG implementation, shared by all four arms.','Method cr_core and gr_core are unchanged. Candidate endpoint labels come only from known TRAIN labels. All TEST targets, features, direct target attachments, injected degrees, nonself edge counts and evaluation denominators remain fixed.','Same-class random matches selected slots, classes and exact edge dose. Joint random matches slots/dose but not class frequency. Equal physical budgets do not mean equal computation.','Both complete dataset action sets must be sealed before any new official victim scoring. No failed or identity case may be removed.','Saved-record/tensor audit is root-authored, not independent or human verification. Final submission, human contribution verification and final tag are separate.'])
 write(s/'protocol.json',p);print(dict(status='frozen',protocol_sha256=sha(s/'protocol.json'),model_calls=0))
if __name__=='__main__':main()
