"""Private pure-standard-library actual target-incidence balancing prototype.

No datasets, tensors, features, labels, model outputs or network calls are read.
All candidate/inter-injected entries and each row target count remain fixed.
"""
import hashlib
import json
import random


ARMS=('baseline','balanced_target','matched_random')


def _validate(adjacency,target_ids,candidate_ids,injected_ids):
    t,c,n=len(target_ids),len(candidate_ids),len(injected_ids)
    ids=list(target_ids)+list(candidate_ids)+list(injected_ids)
    if not t or not n or len(ids)!=len(set(ids)) or any(type(value) is not int or value<0 for value in ids):
        raise ValueError('Explicit disjoint nonnegative node IDs and nonempty targets/injections required.')
    if len(adjacency)!=n or any(len(row)!=t+c+n for row in adjacency):raise ValueError('Adjacency shape differs.')
    for i,row in enumerate(adjacency):
        if any(type(value) not in (int,bool) or value not in (0,1) for value in row):raise ValueError('Binary adjacency required.')
        if any(row[t+c+i:]) or not any(row[:t]):raise ValueError('Triangle or mandatory-target contract failed.')
    return t,c,n


def _sample(values,count,seed,group_key,source_id,purpose):
    value=json.dumps([seed,str(group_key),source_id,purpose],separators=(',',':')).encode('ascii')
    keyed_seed=int.from_bytes(hashlib.sha256(value).digest(),'big')
    return random.Random(keyed_seed).sample(sorted(values),count)


def _injected_degrees(matrix,t,c):
    return [sum(row)+sum(matrix[j][t+c+i] for j in range(i+1,len(matrix)))
            for i,row in enumerate(matrix)]


def balance_targets(adjacency,target_ids,candidate_ids,injected_ids,*,seed,group_key):
    """Balance column degree, with a final-Hamming-matched random control.

    Global target IDs break degree ties; physical injected IDs select an
    eligible row. Eligibility never uses outcomes. The random control draws
    from ORIGINAL target ones/zeros, with per-role keyed sampling.
    """
    t,c,n=_validate(adjacency,target_ids,candidate_ids,injected_ids)
    if type(seed) is not int or type(group_key) not in (str,int):raise ValueError('Explicit integer seed and string/integer group key required.')
    base=[list(map(int,row)) for row in adjacency]
    balanced=[row[:] for row in base]
    before=[sum(row[j] for row in base) for j in range(t)]
    degrees=before[:];initial_phi=sum(value*value for value in degrees);history=[]
    total=sum(degrees);q,remainder=divmod(total,t)
    optimal_phi=remainder*(q+1)**2+(t-remainder)*q*q
    while max(degrees)-min(degrees)>1:
        high=min(range(t),key=lambda j:(-degrees[j],target_ids[j]))
        low=min(range(t),key=lambda j:(degrees[j],target_ids[j]))
        eligible=[i for i in range(n) if balanced[i][high] and not balanced[i][low]]
        if not eligible:
            raise AssertionError('Higher column degree guarantees a row with high=1 and low=0.')
        row=min(eligible,key=lambda i:injected_ids[i])
        old_high,old_low=degrees[high],degrees[low]
        balanced[row][high]=0;balanced[row][low]=1
        degrees[high]-=1;degrees[low]+=1
        delta=-2*(old_high-old_low-1)
        if delta>=0 or degrees[high]<1:
            raise AssertionError('Potential must fall strictly and no previously covered target may be uncovered.')
        history.append(dict(iteration=len(history)+1,row=row,source_id=injected_ids[row],
            removed_target_id=target_ids[high],added_target_id=target_ids[low],
            previous_high_degree=old_high,previous_low_degree=old_low,delta_sum_squared_target_degree=delta))
    if sum(value*value for value in degrees)!=optimal_phi:
        raise AssertionError('Terminal column degrees do not attain the fixed-total convex optimum.')
    random_control=[row[:] for row in base];row_plan=[]
    by_target={target_ids[j]:j for j in range(t)}
    for i,(old,new) in enumerate(zip(base,balanced)):
        removed=[target_ids[j] for j in range(t) if old[j] and not new[j]]
        added=[target_ids[j] for j in range(t) if not old[j] and new[j]]
        count=len(removed)
        if count!=len(added):raise AssertionError('Final row target count changed.')
        original_ones=[target_ids[j] for j in range(t) if old[j]]
        original_zeros=[target_ids[j] for j in range(t) if not old[j]]
        random_removed=_sample(original_ones,count,seed,group_key,injected_ids[i],'remove-target')
        random_added=_sample(original_zeros,count,seed,group_key,injected_ids[i],'add-target')
        for node in random_removed:random_control[i][by_target[node]]=0
        for node in random_added:random_control[i][by_target[node]]=1
        row_plan.append(dict(row=i,source_id=injected_ids[i],original_target_count=sum(old[:t]),
            final_k=count,balanced_removed_target_ids=sorted(removed),balanced_added_target_ids=sorted(added),
            random_removed_target_ids=random_removed,random_added_target_ids=random_added,
            final_target_hamming_distance=2*count))
    arms=dict(baseline=base,balanced_target=balanced,matched_random=random_control)
    physical_degrees=_injected_degrees(base,t,c);row_counts=[sum(row) for row in base]
    for arm,matrix in arms.items():
        _validate(matrix,target_ids,candidate_ids,injected_ids)
        if ([sum(row) for row in matrix]!=row_counts or
                _injected_degrees(matrix,t,c)!=physical_degrees or
                any(row[t:]!=base[i][t:] for i,row in enumerate(matrix)) or
                any(sum(row[:t])!=sum(base[i][:t]) for i,row in enumerate(matrix))):
            raise AssertionError('Role block, row target/total count or physical injection degree changed.')
        if arm!='baseline':
            for i,row in enumerate(matrix):
                if sum(a!=b for a,b in zip(row[:t],base[i][:t]))!=2*row_plan[i]['final_k']:
                    raise AssertionError('Random/balanced final Hamming turnover differs.')
    after={arm:[sum(row[j] for row in matrix) for j in range(t)] for arm,matrix in arms.items()}
    old_covered={target_ids[j] for j,value in enumerate(before) if value}
    covered={arm:{target_ids[j] for j,value in enumerate(values) if value} for arm,values in after.items()}
    if not old_covered<=covered['balanced_target']:
        raise AssertionError('Balancing uncovered a previously covered target.')
    return dict(arms=arms,row_plan=row_plan,iteration_history=history,
        target_incoming_injected_degrees_before=before,target_incoming_injected_degrees_after=after,
        total_target_incidence=total,sum_squared_target_degree_before=initial_phi,
        sum_squared_target_degree_balanced=optimal_phi,
        balanced_degree_range=max(degrees)-min(degrees),
        balancing_iterations=len(history),final_rewired_edges=sum(row['final_k'] for row in row_plan),
        original_row_edge_counts=row_counts,physical_injected_degrees=physical_degrees,
        newly_covered_target_ids={arm:sorted(values-old_covered) for arm,values in covered.items()},
        previously_covered_target_ids_lost={arm:sorted(old_covered-values) for arm,values in covered.items()},
        balanced_equals_baseline=(balanced==base),balanced_equals_random=(balanced==random_control),
        matched_random_rule='Per-role SHA256-derived seed; Random.sample on sorted ORIGINAL target ones/zeros; freeze Python runtime for replay',
        model_outputs_used=False,features_read_or_changed=False)
