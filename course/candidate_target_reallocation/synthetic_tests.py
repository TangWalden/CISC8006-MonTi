"""Pure synthetic geometry contracts; run only after committed source freeze."""
import itertools
import random
import unittest

from course.candidate_target_reallocation.core import ARMS,reallocate_candidates


class ReallocationTests(unittest.TestCase):
    def call(self,matrix,targets,candidates,injected,**options):
        kwargs=dict(seed=3,group_key=938);kwargs.update(options)
        return reallocate_candidates(matrix,targets,candidates,injected,**kwargs)

    def degrees(self,matrix,t,c):
        return [sum(row)+sum(matrix[j][t+c+i] for j in range(i+1,len(matrix))) for i,row in enumerate(matrix)]

    def assert_invariants(self,original,targets,candidates,injected,result):
        t,c,n=len(targets),len(candidates),len(injected)
        self.assertEqual(tuple(result['arms']),ARMS)
        k=sum(plan['k'] for plan in result['row_plan'])
        for arm,matrix in result['arms'].items():
            self.assertEqual([sum(row) for row in matrix],[sum(row) for row in original],arm)
            self.assertEqual(self.degrees(matrix,t,c),self.degrees(original,t,c),arm)
            for i,row in enumerate(matrix):
                self.assertEqual(row[t+c:],original[i][t+c:],arm)
                self.assertFalse(any(old and not new for old,new in zip(original[i][:t],row[:t])),arm)
                self.assertFalse(any(row[t+c+i:]),arm)
            self.assertEqual(sum(map(sum,matrix)),sum(map(sum,original)),arm)
            if arm!='baseline':
                self.assertEqual(sum(a!=b for row,old in zip(matrix,original) for a,b in zip(row,old)),2*k,arm)
                for i,row in enumerate(matrix):
                    actual=[candidates[j] for j in range(c) if original[i][t+j] and not row[t+j]]
                    self.assertEqual(set(actual),set(result['row_plan'][i]['removed_candidate_ids']),arm)

    def test_closed_single_injection_role_cost_known_answer(self):
        matrix=[[1,0,0,1,1,0,0,0]];result=self.call(matrix,[10,20,30],[40,50,60,70],[100])
        self.assertEqual(result['row_plan'][0]['k'],2)
        self.assertEqual(result['arms']['balanced_target_fill'],[[1,1,1,0,0,0,0,0]])
        self.assertEqual(result['arms']['random_target_fill'],[[1,1,1,0,0,0,0,0]])
        self.assertEqual(result['arms']['candidate_rewire'],[[1,0,0,0,0,1,1,0]])
        self.assertEqual(result['total_target_incidence_before'],1)
        self.assertEqual(result['total_target_incidence_after']['balanced_target_fill'],3)
        self.assertEqual(result['candidate_incidence_after']['balanced_target_fill'],0)
        self.assertEqual(result['physical_injected_degrees'],[3])
        self.assert_invariants(matrix,[10,20,30],[40,50,60,70],[100],result)

    def test_frozen_baseline_attachment_priority_with_prior_injection_edges(self):
        matrix=[[1,0,0,1,0,0,0,0,0],[1,0,0,1,0,0,1,0,0],[1,1,0,1,0,0,1,1,0]]
        result=self.call(matrix,[10,20,30],[40,50,60],[100,101,102])
        self.assertEqual(result['target_incoming_injected_degrees_before'],[3,1,0])
        self.assertEqual([plan['primary_added_target_ids'] for plan in result['row_plan']],[[30],[30],[30]])
        self.assertEqual(result['target_incoming_injected_degrees_after']['balanced_target_fill'],[3,1,3])
        self.assertEqual(result['physical_injected_degrees'],[4,4,5])
        self.assert_invariants(matrix,[10,20,30],[40,50,60],[100,101,102],result)

    def test_each_of_three_common_capacity_limits(self):
        for targets,candidate_bits,expected in (([10,20,30,40],[1,0,0,0],1),
                ([10,20],[1,1,0,0],1),([10,20,30,40],[1,1,1,0],1),([10,20,30],[1,1,0,0],2)):
            matrix=[[1]+[0]*(len(targets)-1)+candidate_bits+[0]]
            result=self.call(matrix,targets,[50,60,70,80],[100])
            self.assertEqual(result['row_plan'][0]['k'],expected)
            self.assert_invariants(matrix,targets,[50,60,70,80],[100],result)

    def test_no_candidate_no_zero_candidate_and_no_missing_target_identity(self):
        examples=[([[1,0,0]],[10,20],[],[100]),
                  ([[1,0,1,1,0]],[10,20],[30,40],[100]),
                  ([[1,1,1,0,0]],[10,20],[30,40],[100]),
                  ([[1,1,0,0]],[10],[30,40],[100])]
        for matrix,targets,candidates,injected in examples:
            with self.subTest(matrix=matrix):
                result=self.call(matrix,targets,candidates,injected)
                self.assertEqual(result['total_reallocated_edges'],0)
                for arm in ARMS:self.assertTrue(result['identity'][arm]['exact_equals_baseline'])

    def test_exact_shared_donors_and_control_uses_only_original_zeros(self):
        matrix=[[1,0,0,0,1,1,1,0,0,0]]
        result=self.call(matrix,[10,20,30,40],[50,60,70,80,90],[100])
        plan=result['row_plan'][0]
        self.assertEqual(plan['k'],2)
        self.assertEqual(set(plan['control_added_candidate_ids']),{80,90})
        self.assertTrue(set(plan['removed_candidate_ids'])<={50,60,70})
        self.assertTrue(set(plan['primary_added_target_ids'])<={20,30,40})
        self.assertTrue(set(plan['random_added_target_ids'])<={20,30,40})
        self.assert_invariants(matrix,[10,20,30,40],[50,60,70,80,90],[100],result)

    def test_deterministic_keys_do_not_change_global_rng(self):
        matrix=[[1,0,0,1,1,0,0,0]];state=random.getstate()
        first=self.call(matrix,[10,20,30],[40,50,60,70],[100])
        second=self.call(matrix,[10,20,30],[40,50,60,70],[100])
        self.assertEqual(first,second);self.assertEqual(random.getstate(),state)
        self.assertNotEqual(first['row_plan'][0]['donor_key_sha256'],self.call(matrix,[10,20,30],[40,50,60,70],[100],group_key=939)['row_plan'][0]['donor_key_sha256'])
        self.assertNotEqual(first['row_plan'][0]['donor_key_sha256'],first['row_plan'][0]['random_target_key_sha256'])

    def test_original_inputs_and_returned_arm_storage_are_independent(self):
        matrix=[[1,0,0,1,1,0,0,0]];original=[row[:] for row in matrix]
        targets,candidates,injected=[10,20,30],[40,50,60,70],[100]
        result=self.call(matrix,targets,candidates,injected)
        self.assertEqual(matrix,original)
        result['arms']['baseline'][0][0]=0
        self.assertEqual(matrix,original);self.assertEqual(result['arms']['balanced_target_fill'][0][0],1)

    def test_target_and_candidate_column_permutation_equivariance(self):
        targets,candidates,injected=[10,20,30],[40,50,60,70],[100]
        matrix=[[1,0,0,1,1,0,0,0]];a=self.call(matrix,targets,candidates,injected)
        tp,cp=[2,0,1],[3,1,0,2]
        permuted=[[row[j] for j in tp]+[row[len(targets)+j] for j in cp]+row[len(targets)+len(candidates):] for row in matrix]
        b=self.call(permuted,[targets[j] for j in tp],[candidates[j] for j in cp],injected)
        for arm in ARMS:
            restored=[[row[tp.index(j)] for j in range(len(targets))]+[row[len(targets)+cp.index(j)] for j in range(len(candidates))]+row[len(targets)+len(candidates):] for row in b['arms'][arm]]
            self.assertEqual(restored,a['arms'][arm],arm)

    def test_exhaustive_small_legal_geometry_and_exact_roles(self):
        cases=0
        nonempty_targets=((1,0),(0,1),(1,1));candidate_bits=tuple(itertools.product((0,1),repeat=2))
        for first,second,c0,c1,intra in itertools.product(nonempty_targets,nonempty_targets,candidate_bits,candidate_bits,(0,1)):
            matrix=[list(first)+list(c0)+[0,0],list(second)+list(c1)+[intra,0]]
            result=self.call(matrix,[10,20],[30,40],[100,101])
            self.assert_invariants(matrix,[10,20],[30,40],[100,101],result)
            expected=sum(min(sum(row[2:4]),2-sum(row[:2]),2-sum(row[2:4])) for row in matrix)
            self.assertEqual(result['total_reallocated_edges'],expected)
            self.assertEqual(result['total_target_incidence_after']['balanced_target_fill'],result['total_target_incidence_before']+expected)
            self.assertEqual(result['candidate_incidence_after']['candidate_rewire'],result['candidate_incidence_before'])
            cases+=1
        self.assertEqual(cases,288)

    def test_invalid_shape_binary_triangle_and_mandatory_edges(self):
        for matrix in ([],[[1,0]],[[1,0,1,0,0.]],[[1,0,1,0,1]],[[0,0,1,0,0]],[[1,0,2,0,0]]):
            with self.subTest(matrix=matrix),self.assertRaises(ValueError):self.call(matrix,[10,20],[30,40],[100])

    def test_invalid_ids_seed_and_group_keys(self):
        for targets,candidates,injected in (([10,10],[30,40],[100]),([10,20],[20,40],[100]),
                ([10,20],[30,40],[40]),([10,20],[30,40],[-1]),([10.,20],[30,40],[100])):
            with self.subTest(targets=targets,candidates=candidates,injected=injected),self.assertRaises(ValueError):
                self.call([[1,0,1,0,0]],targets,candidates,injected)
        for options in (dict(seed=True),dict(seed=-1),dict(group_key=1.),dict(group_key=-1)):
            with self.subTest(options=options),self.assertRaises(ValueError):self.call([[1,0,1,0,0]],[10,20],[30,40],[100],**options)


if __name__=='__main__':unittest.main()
