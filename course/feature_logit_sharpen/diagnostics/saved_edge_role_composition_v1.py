"""Saved geometry JSON only: derive baseline injected-edge endpoint roles.

No labels, logits, outcomes, features, tensors or models are accessed. Whole JSON
rows are parsed, but only the frozen geometry/count fields below are selected.
Earlier public target-incidence summaries were already seen; private physical
degree values are first decoded only after this source/scope freeze.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import time
import traceback


EXPECTED_SHA = '8ada8dd4bcfa7bd0a2ffbb29d181d415792ad8cce182abeb4533553a243123cb'
SEEDS = (3, 4, 5)
EXPECTED_EDGES = 358491
EXPECTED_TARGETS = {3:12337, 4:20507, 5:29348}
ROLES = ('injected_to_target', 'injected_to_original_candidate', 'injected_to_injected')


def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();source=pathlib.Path(args.geometry).resolve();out=pathlib.Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);cpu,wall=time.process_time(),time.monotonic()
    receipt=dict(status='failed',source_sha256=sha(__file__),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input_expected_sha256=EXPECTED_SHA,geometry_files_decoded=0,model_queries=0,tensor_loads=0,
        dataset_loads=0,checkpoint_loads=0,API_calls=0,GPU_hours=0,label_fields_accessed=0,outcome_fields_accessed=0)
    checks=0
    def require(value,message):
        nonlocal checks
        checks+=1
        if not value:raise AssertionError(message)
    try:
        raw=source.read_bytes();actual=hashlib.sha256(raw).hexdigest();receipt['input_actual_sha256']=actual
        require(actual==EXPECTED_SHA,'Saved audited geometry bytes changed before decoding.')
        rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        receipt['geometry_files_decoded']=1;require(len(rows)==384,'Expected all384 saved group-seed cases.')
        cases=[];seen=set();groups={seed:set() for seed in SEEDS}
        for row in rows:
            seed,group=row['seed'],row['group'];geometry=row['geometry'];baseline=row['variants']['baseline']
            require(seed in SEEDS and (seed,group) not in seen,'Unexpected/duplicate seed-group case.');seen.add((seed,group));groups[seed].add(group)
            n,t,e=row['node_budget'],row['targets'],row['edge_budget'];d=geometry['physical_injected_degrees']
            require(all(type(v) is int and v>0 for v in (n,t,e)) and len(d)==n and
                    all(type(v) is int and v>=1 for v in d),'Invalid baseline counts or physical degree entries.')
            require(d==baseline['physical_injected_degrees'] and baseline['actual_injected_nodes']==n and
                    baseline['actual_injected_nonself_undirected_edges']==e,'Baseline physical receipts differ.')
            owned=geometry['original_row_edge_counts'];require(len(owned)==n and sum(owned)==e,'Owned matrix-edge counts do not equal physical undirected budget.')
            target=geometry['total_target_incidence'];before=geometry['target_incoming_injected_degrees_before']
            require(len(before)==t and sum(before)==target and n<=target<=n*t,'Target incidence does not match the complete target prefix.')
            require(sum(plan['original_target_count'] for plan in geometry['row_plan'])==target,'Per-row target counts differ.')
            degree_sum=sum(d);intra=degree_sum-e;original=2*e-degree_sum;candidate=original-target
            roles=dict(injected_to_target=target,injected_to_original_candidate=candidate,injected_to_injected=intra)
            require(all(type(v) is int and v>=0 for v in roles.values()) and sum(roles.values())==e,
                    'Derived edge roles must be nonnegative and sum to the actual physical budget.')
            require(intra<=n*(n-1)//2 and (n!=1 or intra==0),'Inter-injected count violates simple undirected geometry.')
            for arm,variant in row['variants'].items():
                require(variant['physical_injected_degrees']==d and variant['actual_injected_nodes']==n and
                        variant['actual_injected_nonself_undirected_edges']==e and
                        sum(geometry['target_incoming_injected_degrees_after'][arm])==target,
                        'Earlier target-only variants changed fixed edge-role/physical degree totals.')
            fractions={role:count/e for role,count in roles.items()}
            require(all(0<=value<=1 for value in fractions.values()) and math.isclose(sum(fractions.values()),1.,rel_tol=0,abs_tol=1e-12),
                    'Invalid edge-role fractions.')
            cases.append(dict(seed=seed,group=group,injected_nodes=n,targets=t,total_nonself_undirected_edges=e,
                physical_injected_degree_sum=degree_sum,original_endpoint_edges=original,edge_roles=roles,edge_role_fractions=fractions,
                missing_injected_target_pairs=n*t-target,
                candidate_to_target_move_upper_bound=min(candidate,n*t-target),
                intra_to_target_donor_upper_bound=min(intra,n*t-target)))
        require(all(len(values)==128 for values in groups.values()) and groups[3]==groups[4]==groups[5],
                'Preserve the same128 original groups for each seed.')
        totals=[];strata=[]
        for seed in SEEDS:
            selected=[row for row in cases if row['seed']==seed];e=sum(row['total_nonself_undirected_edges'] for row in selected)
            role_counts={role:sum(row['edge_roles'][role] for row in selected) for role in ROLES}
            require(e==EXPECTED_EDGES and role_counts['injected_to_target']==EXPECTED_TARGETS[seed],
                    'Known frozen edge/target incidence checksum totals do not reproduce.')
            totals.append(dict(seed=seed,cases=len(selected),summed_injected_node_instances=sum(row['injected_nodes'] for row in selected),
                total_nonself_undirected_edges=e,edge_roles=role_counts,
                fraction_of_summed_edges={role:value/e for role,value in role_counts.items()},
                mean_per_scenario_edge_fraction={role:math.fsum(row['edge_role_fractions'][role] for row in selected)/len(selected) for role in ROLES},
                cases_with_candidate_donors=sum(row['edge_roles']['injected_to_original_candidate']>0 for row in selected),
                cases_with_intra_donors=sum(row['edge_roles']['injected_to_injected']>0 for row in selected),
                summed_candidate_move_upper_bounds=sum(row['candidate_to_target_move_upper_bound'] for row in selected),
                summed_intra_move_upper_bounds=sum(row['intra_to_target_donor_upper_bound'] for row in selected)))
            for name,condition in (('one_node_one_edge',lambda r:r['injected_nodes']==r['total_nonself_undirected_edges']==1),
                ('one_node_multiple_edges',lambda r:r['injected_nodes']==1 and r['total_nonself_undirected_edges']>1),
                ('multiple_nodes',lambda r:r['injected_nodes']>1)):
                values=[row for row in selected if condition(row)]
                strata.append(dict(seed=seed,regime=name,cases=len(values),edges=sum(row['total_nonself_undirected_edges'] for row in values),
                    edge_roles={role:sum(row['edge_roles'][role] for row in values) for role in ROLES}))
        report=dict(status='completed_saved_count_derivation',input_sha256=actual,full_cases=384,seed_totals=totals,budget_strata=strata,
            identities=dict(intra='sum(nonself physical injected degrees)-E',original='2E-sum(nonself physical injected degrees)',
                            candidate='original-target incidence',role_sum='target+candidate+intra=E'),
            scope='Only baseline role-count geometry already retained by the independently audited target-balancing study. All earlier variants have identical role totals.',
            limitations=['These are counts over128 separate scenarios per seed, not one combined graph or independently sampled edges.',
                'Positive candidate/intra counts establish donor existence, not exact feasible reallocation: per-row candidate incidence, available target pairs and chosen donor endpoints need a later separately frozen adjacency-level feasibility audit.',
                'The movement bounds are coarse count upper bounds, not feasible plans or efficacy estimates.',
                'Replacing an injected-to-original-candidate edge by an injected-to-target edge can preserve that injected node degree. Replacing an inter-injected edge by one target edge generally changes physical injected degrees.',
                'No outcomes, labels, features or model predictions were used. This derivation does not show that either reallocation improves MR, transfer, stealth, defense or course acceptance.',
                'No new experiment is selected by these counts. Earlier public target-incidence values were seen during the prior-summary check before this private physical-degree derivation was frozen.'])
        write(out/'summary.json',report);write(out/'all_case_roles.json',cases)
        receipt.update(status='passed',checks_passed=checks,retained_cases=384,
            output_hashes={name:sha(out/name) for name in ('summary.json','all_case_roles.json')})
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),checks_before_failure=checks)
    finally:
        receipt.update(process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'receipt.json',receipt)
        print(json.dumps({key:value for key,value in receipt.items() if key!='traceback'},allow_nan=False),flush=True)
    if receipt['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
