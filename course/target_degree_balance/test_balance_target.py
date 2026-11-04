"""Synthetic counterproof/property checks, without models or real data."""
import itertools
import unittest

try:
    from .balance_target import ARMS,balance_targets
except ImportError:
    from balance_target import ARMS,balance_targets


def blank(t,c,n):return [[0]*(t+c+n) for _ in range(n)]


def edge_set(a,targets,candidates,injected):
    ids=list(targets)+list(candidates)+list(injected)
    edges={tuple(sorted((injected[i],ids[j]))) for i,row in enumerate(a) for j,value in enumerate(row) if value}
    if len(edges)!=sum(sum(row) for row in a):raise AssertionError('Independent physical mapping found duplicate edges.')
    return edges


def degree(edges,node):return sum(node in edge for edge in edges)


class BalanceTests(unittest.TestCase):
    def test_zero_candidate_has_nontrivial_balancing_headroom(self):
        a=blank(3,0,3)
        for row in a:row[0]=1
        r=balance_targets(a,[0,1,2],[],[10,11,12],seed=3,group_key=1)
        self.assertEqual(r['target_incoming_injected_degrees_after']['balanced_target'],[1,1,1])
        self.assertEqual(r['sum_squared_target_degree_before'],9)
        self.assertEqual(r['sum_squared_target_degree_balanced'],3)
        self.assertEqual([row['final_k'] for row in r['row_plan']],[1,1,0])
        self.assertEqual(r['newly_covered_target_ids']['balanced_target'],[1,2])

    def test_full_candidate_block_does_not_block_target_balancing(self):
        a=blank(3,2,3)
        for row in a:row[0]=1;row[3]=row[4]=1
        a[2][5]=1
        r=balance_targets(a,[0,1,2],[3,4],[10,11,12],seed=3,group_key=2)
        self.assertGreater(r['final_rewired_edges'],0)
        for output in r['arms'].values():self.assertEqual([row[3:] for row in output],[row[3:] for row in a])

    def test_high_low_global_target_and_source_id_tie_known_answer(self):
        a=blank(5,0,3)
        for row in a:row[0]=1
        r=balance_targets(a,[50,20,30,40,10],[],[100,101,102],seed=4,group_key=3)
        history=r['iteration_history']
        self.assertEqual((history[0]['source_id'],history[0]['removed_target_id'],history[0]['added_target_id']),(100,50,10))
        self.assertEqual((history[1]['source_id'],history[1]['added_target_id']),(101,20))
        self.assertEqual(r['target_incoming_injected_degrees_after']['balanced_target'],[1,1,0,0,1])
        self.assertEqual(r['previously_covered_target_ids_lost']['balanced_target'],[])

    def test_single_injection_is_identity_even_with_multiple_targets(self):
        a=blank(4,2,1);a[0][0]=a[0][2]=a[0][4]=1
        r=balance_targets(a,[0,1,2,3],[4,5],[10],seed=4,group_key=4)
        self.assertEqual(r['final_rewired_edges'],0)
        for output in r['arms'].values():self.assertEqual(output,a)

    def test_already_balanced_or_single_target_is_identity(self):
        for targets,candidates,injected in [([0,1],[],[10,11]),([0],[],[10,11])]:
            if len(targets)==2:
                a=blank(2,0,2);a[0][0]=a[1][1]=1
            else:
                a=blank(1,0,2);a[0][0]=a[1][0]=1;a[1][1]=1
            r=balance_targets(a,targets,candidates,injected,seed=5,group_key=5)
            for output in r['arms'].values():self.assertEqual(output,a)

    def test_mixed_row_target_counts_and_hamming_match(self):
        a=blank(4,1,3);a[0][0]=1;a[1][:4]=[1,1,1,0];a[2][:4]=[1,1,0,0]
        a[0][4]=1;a[2][5]=1
        r=balance_targets(a,[0,1,2,3],[4],[10,11,12],seed=5,group_key=6)
        self.assertEqual(sorted(r['target_incoming_injected_degrees_after']['balanced_target']),[1,1,2,2])
        self.assertEqual(r['sum_squared_target_degree_balanced'],10)
        for arm in ARMS:
            self.assertEqual([sum(row[:4]) for row in r['arms'][arm]],[1,3,2])
            self.assertEqual([sum(row) for row in r['arms'][arm]],[2,3,3])
        for i,item in enumerate(r['row_plan']):
            for arm in ARMS[1:]:self.assertEqual(sum(x!=y for x,y in zip(a[i],r['arms'][arm][i])),2*item['final_k'])

    def test_no_inputs_mutated_and_random_control_stateless(self):
        a=blank(4,0,4)
        for row in a:row[0]=1
        before=[row[:] for row in a]
        r=balance_targets(a,[0,1,2,3],[],[10,11,12,13],seed=6,group_key=7)
        self.assertEqual(a,before)
        self.assertEqual(r,balance_targets(a,[0,1,2,3],[],[10,11,12,13],seed=6,group_key=7))

    def test_target_column_permutation_equivariance(self):
        a=blank(4,1,4)
        for i,row in enumerate(a):row[0]=1;row[4]=i%2
        targets,candidates,injected=[4,1,3,2],[8],[10,11,12,13]
        r=balance_targets(a,targets,candidates,injected,seed=7,group_key=8)
        permutation=[2,0,3,1]
        changed=[ [row[j] for j in permutation]+row[4:] for row in a]
        other_targets=[targets[j] for j in permutation]
        other=balance_targets(changed,other_targets,candidates,injected,seed=7,group_key=8)
        for arm in ARMS:self.assertEqual(edge_set(r['arms'][arm],targets,candidates,injected),edge_set(other['arms'][arm],other_targets,candidates,injected))

    def test_invalid_triangle_or_missing_target_rejected(self):
        a=blank(2,0,2);a[0][0]=a[1][0]=1
        b=[row[:] for row in a];b[0][2]=1
        with self.assertRaises(ValueError):balance_targets(b,[0,1],[],[10,11],seed=8,group_key=9)
        b=[row[:] for row in a];b[1][:2]=[0,0];b[1][2]=1
        with self.assertRaises(ValueError):balance_targets(b,[0,1],[],[10,11],seed=8,group_key=9)

    def test_exhaustive2744_binary_target_interinjection_patterns(self):
        targets,candidates,injected=[0,1,2],[3],[10,11,12]
        prefixes=[bits for bits in itertools.product((0,1),repeat=3) if any(bits)]
        count=0
        for rows in itertools.product(prefixes,repeat=3):
            for inter in itertools.product((0,1),repeat=3):
                a=blank(3,1,3)
                for i,row in enumerate(a):row[:3]=rows[i];row[3]=i%2
                a[1][4],a[2][4],a[2][5]=inter
                before=[row[:] for row in a];r=balance_targets(a,targets,candidates,injected,seed=9,group_key=10)
                self.assertEqual(a,before)
                original_edges=edge_set(a,targets,candidates,injected)
                d0=r['target_incoming_injected_degrees_before'];d1=r['target_incoming_injected_degrees_after']['balanced_target']
                self.assertLessEqual(max(d1)-min(d1),1)
                self.assertEqual(sum(d0),sum(d1))
                self.assertTrue(all(not d0[j] or d1[j]>0 for j in range(3)))
                total=sum(d0);q,remainder=divmod(total,3)
                self.assertEqual(sum(x*x for x in d1),remainder*(q+1)**2+(3-remainder)*q*q)
                self.assertEqual(r['sum_squared_target_degree_before']+sum(x['delta_sum_squared_target_degree'] for x in r['iteration_history']),r['sum_squared_target_degree_balanced'])
                for arm,out in r['arms'].items():
                    edges=edge_set(out,targets,candidates,injected)
                    self.assertEqual(len(edges),len(original_edges))
                    self.assertEqual([degree(edges,node) for node in injected],[degree(original_edges,node) for node in injected])
                    self.assertEqual(degree(edges,3),degree(original_edges,3))
                    self.assertEqual([row[3:] for row in out],[row[3:] for row in a])
                    self.assertEqual([sum(row[:3]) for row in out],[sum(row[:3]) for row in a])
                    if arm!='baseline':
                        for i,item in enumerate(r['row_plan']):self.assertEqual(sum(x!=y for x,y in zip(a[i][:3],out[i][:3])),2*item['final_k'])
                count+=1
        self.assertEqual(count,2744)


if __name__=='__main__':unittest.main()
