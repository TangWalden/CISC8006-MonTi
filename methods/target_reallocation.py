"""Pure count-based candidate-edge reallocation; no graph/model/label reads.

Rows are injected nodes; columns are [targets, original candidates, injected].
The strict lower-triangular injected block owns each undirected injected edge
once. Every arm preserves all old target edges, the full injected block, each
row edge count and every physical injected-node degree. Primary target ranking
uses BASELINE injected-attachment counts, never full clean graph node degrees.
"""
import hashlib
import json
import random


ARMS = ('baseline', 'balanced_target_fill', 'random_target_fill', 'candidate_rewire')
NAMESPACE = 'candidate-target-reallocation-v1'


def _validate(adjacency, target_ids, candidate_ids, injected_ids):
    t,c,n = len(target_ids),len(candidate_ids),len(injected_ids)
    ids = list(target_ids)+list(candidate_ids)+list(injected_ids)
    if (not t or not n or any(type(value) is not int or value<0 for value in ids)
            or len(ids)!=len(set(ids))):
        raise ValueError('Disjoint nonnegative Python integer IDs and nonempty target/injection lists required.')
    if len(adjacency)!=n or any(len(row)!=t+c+n for row in adjacency):
        raise ValueError('Adjacency must have B rows and T+C+B columns.')
    for i,row in enumerate(adjacency):
        if any(type(value) not in (int,bool) or value not in (0,1) for value in row):
            raise ValueError('Only exact binary Python int/bool adjacency entries are allowed.')
        if any(row[t+c+i:]) or not any(row[:t]):
            raise ValueError('Strict lower-triangle injection ownership and an old target edge per row are required.')
    return t,c,n


def _key(seed, group_key, source_id, purpose, endpoint=None):
    value = [seed,str(group_key),source_id,purpose,endpoint,NAMESPACE]
    return hashlib.sha256(json.dumps(value,separators=(',', ':'),ensure_ascii=True).encode('ascii')).hexdigest()


def _sample(values,count,key):
    return random.Random(int(key,16)).sample(sorted(values),count)


def _injected_degrees(matrix,t,c):
    return [sum(row)+sum(matrix[j][t+c+i] for j in range(i+1,len(matrix)))
            for i,row in enumerate(matrix)]


def _identity(matrix,baseline):
    changes = [sum(a!=b for a,b in zip(row,baseline[i])) for i,row in enumerate(matrix)]
    return dict(exact_equals_baseline=not any(changes),changed_rows=sum(value>0 for value in changes),
                changed_coordinates=sum(changes),comparison='Exact binary owned-adjacency entries')


def reallocate_candidates(adjacency,target_ids,candidate_ids,injected_ids,*,seed,group_key):
    """Apply one fixed maximal common-feasibility rule and matched controls.

    k_i=min(original candidate ones, original target zeros, original candidate
    zeros). All nonbaseline arms remove the SAME candidate donors. Both target
    arms add target zeros; the strict role control adds ORIGINAL candidate
    zeros instead. No fraction, training, logits, labels or outcome gate enters
    this function. Target IDs are the existing attack list under its known
    fraud-target assumption; this function never verifies/looks up that truth.
    """
    t,c,n = _validate(adjacency,target_ids,candidate_ids,injected_ids)
    if type(seed) is not int or seed<0 or type(group_key) is not int or group_key<0:
        raise ValueError('Seed and original group key must be nonnegative Python integers.')
    baseline = [[int(value) for value in row] for row in adjacency]
    arms = {arm:[row[:] for row in baseline] for arm in ARMS}
    target_column = {node:i for i,node in enumerate(target_ids)}
    candidate_column = {node:t+i for i,node in enumerate(candidate_ids)}
    # Frozen ONCE before any row is processed. No newly added edge changes the
    # priority available to another row, avoiding an iterative feedback rule.
    attachment = [sum(row[j] for row in baseline) for j in range(t)]
    baseline_candidates = sum(sum(row[t:t+c]) for row in baseline)
    row_plan = []
    for i,old in enumerate(baseline):
        donor_pool = [node for node in candidate_ids if old[candidate_column[node]]]
        target_pool = [node for node in target_ids if not old[target_column[node]]]
        candidate_pool = [node for node in candidate_ids if not old[candidate_column[node]]]
        k = min(len(donor_pool),len(target_pool),len(candidate_pool))
        donor_key = _key(seed,group_key,injected_ids[i],'candidate-donors')
        random_key = _key(seed,group_key,injected_ids[i],'random-target-addition')
        control_key = _key(seed,group_key,injected_ids[i],'control-candidate-addition')
        removed = _sample(donor_pool,k,donor_key)
        priority = sorted(target_pool,key=lambda node:(attachment[target_column[node]],
            _key(seed,group_key,injected_ids[i],'primary-target-tie',node),node))
        primary_added = priority[:k]
        random_added = _sample(target_pool,k,random_key)
        control_added = _sample(candidate_pool,k,control_key)
        for arm in ARMS[1:]:
            for node in removed:arms[arm][i][candidate_column[node]]=0
        for node in primary_added:arms['balanced_target_fill'][i][target_column[node]]=1
        for node in random_added:arms['random_target_fill'][i][target_column[node]]=1
        for node in control_added:arms['candidate_rewire'][i][candidate_column[node]]=1
        row_plan.append(dict(row=i,source_id=injected_ids[i],baseline_candidate_count=len(donor_pool),
            missing_target_count=len(target_pool),missing_candidate_count=len(candidate_pool),k=k,
            candidate_donor_fraction=k/len(donor_pool) if donor_pool else None,
            removed_candidate_ids=removed,primary_added_target_ids=primary_added,
            random_added_target_ids=random_added,control_added_candidate_ids=control_added,
            donor_key_sha256=donor_key,random_target_key_sha256=random_key,control_candidate_key_sha256=control_key,
            primary_selected_target_priorities=[dict(target_id=node,baseline_injected_attachment_count=attachment[target_column[node]],
                tie_key_sha256=_key(seed,group_key,injected_ids[i],'primary-target-tie',node)) for node in primary_added]))
    total_k = sum(row['k'] for row in row_plan)
    row_counts = [sum(row) for row in baseline]
    degrees = _injected_degrees(baseline,t,c)
    total_edges = sum(row_counts)
    after = {arm:[sum(row[j] for row in matrix) for j in range(t)] for arm,matrix in arms.items()}
    candidate_after = {arm:sum(sum(row[t:t+c]) for row in matrix) for arm,matrix in arms.items()}
    identities = {arm:_identity(matrix,baseline) for arm,matrix in arms.items()}
    for arm,matrix in arms.items():
        _validate(matrix,target_ids,candidate_ids,injected_ids)
        if ([sum(row) for row in matrix]!=row_counts or _injected_degrees(matrix,t,c)!=degrees
                or sum(map(sum,matrix))!=total_edges
                or any(row[t+c:]!=baseline[i][t+c:] for i,row in enumerate(matrix))
                or any(old and not new for i,row in enumerate(matrix) for old,new in zip(baseline[i][:t],row[:t]))):
            raise AssertionError('An arm changed mandatory target edges, injected block, row/physical degree or edge cost.')
        expected_target_gain = total_k if arm in ('balanced_target_fill','random_target_fill') else 0
        if sum(after[arm])!=sum(attachment)+expected_target_gain or candidate_after[arm]!=baseline_candidates-expected_target_gain:
            raise AssertionError('Candidate/target role incidence does not reconcile.')
        if arm=='baseline':continue
        if identities[arm]['changed_coordinates']!=2*total_k or identities[arm]['changed_rows']!=sum(plan['k']>0 for plan in row_plan):
            raise AssertionError('A nonbaseline arm does not have the matched final Hamming displacement.')
        for i,row in enumerate(matrix):
            removed=[node for node in candidate_ids if baseline[i][candidate_column[node]] and not row[candidate_column[node]]]
            if set(removed)!=set(row_plan[i]['removed_candidate_ids']):
                raise AssertionError('Nonbaseline arms do not share exact original candidate donors.')
            if arm=='candidate_rewire' and row[:t]!=baseline[i][:t]:
                raise AssertionError('The candidate role control changed a target edge.')
    covered_before = {target_ids[j] for j,value in enumerate(attachment) if value}
    covered_after = {arm:{target_ids[j] for j,value in enumerate(values) if value} for arm,values in after.items()}
    if any(not covered_before<=values for values in covered_after.values()):
        raise AssertionError('An old covered target was uncovered.')
    return dict(arms=arms,identity=identities,row_plan=row_plan,total_reallocated_edges=total_k,
        target_incoming_injected_degrees_before=attachment,target_incoming_injected_degrees_after=after,
        total_target_incidence_before=sum(attachment),total_target_incidence_after={arm:sum(value) for arm,value in after.items()},
        candidate_incidence_before=baseline_candidates,candidate_incidence_after=candidate_after,
        original_row_edge_counts=row_counts,physical_injected_degrees=degrees,total_owned_nonself_undirected_edges=total_edges,
        newly_covered_target_ids={arm:sorted(values-covered_before) for arm,values in covered_after.items()},
        previously_covered_target_ids_lost={arm:sorted(covered_before-values) for arm,values in covered_after.items()},
        primary_equals_random=arms['balanced_target_fill']==arms['random_target_fill'],
        priority='Original injected-to-target column sums computed once, then row-specific SHA ties and target ID; not full clean graph target degree',
        common_feasibility='k=min(original candidate ones, original target zeros, original candidate zeros) per injected row',
        key_scheme='compact JSON [seed,str(group_key),physical_injected_id,purpose,endpoint_or_null,namespace]; SHA256 hex -> private Random integer seed',
        namespace=NAMESPACE,all_old_target_edges_preserved=True,injected_block_preserved=True,
        row_and_total_edge_counts_preserved=True,physical_injected_degrees_preserved=True,
        target_attachment_counts_intentionally_increase=True,known_fraud_target_assumption=True,
        uses_full_clean_target_degree=False,labels_read=False,model_outputs_used=False,features_read_or_changed=False,
        maximum_common_feasible_k=True,tunable_fraction=False,
        limitation='A row may lose all its original candidate neighbors; fixed maximal reallocation is not a small perturbation or guaranteed performance gain.')
