"""Synthetic numerical contracts; optional NumPy/SciPy are explicit.

Without optional dependencies this module reports skips, keeping the original
standard-library method checks usable. The registered server numeric check must
execute every test with zero skips. No data, checkpoint or neural model is read.
"""
import itertools
import importlib.util
import random
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('numpy', 'scipy'))
if AVAILABLE:
    import numpy as np
    import scipy
    from methods.feature_assignment import (ACCEPTANCE_TOLERANCE, accept_if_improved,
        apply_permutation, cycle_lengths, cycle_matched_random, propose_assignment)
else:
    np = None


CHECK_COUNTERS = dict(bruteforce_permutations=0, brute_force_cases=0, conjugacy_cases=0)


@unittest.skipUnless(AVAILABLE, 'Optional NumPy/SciPy unavailable; assignment checks not executed.')
class FeatureAssignmentTests(unittest.TestCase):
    def test_known_orientation_position_receives_source_feature(self):
        x = np.array([[0.], [1.]], dtype=np.float32)
        g = np.array([[-1.], [1.]], dtype=np.float32)
        result = propose_assignment(x, g)
        self.assertTrue(np.array_equal(result['permutation'], [1, 0]))
        self.assertTrue(np.array_equal(apply_permutation(x, result['permutation']), [[1.], [0.]]))
        self.assertEqual(result['linear_delta'], -2.)
        self.assertTrue(result['linear_improvement_certified'])

    def test_bruteforce_optimum_every_size_one_through_six(self):
        rng = np.random.RandomState(9127)
        for b in range(1, 7):
            for repeat in range(3):
                x = rng.randint(-4, 5, size=(b, 4)).astype(np.float64)
                g = rng.randint(-3, 4, size=(b, 4)).astype(np.float64)
                result = propose_assignment(x, g)
                chosen = result['permutation']
                actual = float(np.sum(g*(x[chosen]-x), dtype=np.float64))
                for permutation in itertools.permutations(range(b)):
                    other = float(np.sum(g*(x[list(permutation)]-x), dtype=np.float64))
                    self.assertLessEqual(actual, other+result['roundoff_bound']+1e-12)
                    CHECK_COUNTERS['bruteforce_permutations'] += 1
                CHECK_COUNTERS['brute_force_cases'] += 1

    def test_exact_bijection_multiset_dtype_and_input_immutability(self):
        x = np.array([[.2, -0.], [.5, 2.], [.9, -1.]], dtype=np.float32)
        g = np.array([[-2., 1.], [1., -2.], [1., 1.]], dtype=np.float32)
        before, before_g = x.copy(), g.copy()
        result = propose_assignment(x, g);p = result['permutation'];value = apply_permutation(x, p)
        self.assertTrue(np.array_equal(np.sort(p), np.arange(len(x))))
        self.assertEqual(value.dtype, x.dtype)
        self.assertEqual(sorted(row.tobytes() for row in value), sorted(row.tobytes() for row in x))
        self.assertEqual(x.tobytes(), before.tobytes());self.assertEqual(g.tobytes(), before_g.tobytes())

    def test_one_row_identical_features_and_zero_gradient_identity(self):
        for x,g,reason in ((np.array([[1.,2.]]),np.array([[2.,3.]]),'one_row'),
                (np.array([[1.,2.],[1.,2.]]),np.array([[2.,3.],[-4.,1.]]),'identical_feature_rows'),
                (np.array([[1.,2.],[3.,4.]]),np.zeros((2,2)),'zero_gradient')):
            result = propose_assignment(x,g)
            self.assertTrue(np.array_equal(result['permutation'],np.arange(len(x))))
            self.assertEqual(result['reason'],reason)

    def test_roundoff_gate_rejects_uncertifiable_cancellation(self):
        x = np.array([[1.], [1.+2.**-30]], dtype=np.float64)
        g = np.array([[1.+2.**-30], [1.]], dtype=np.float64)
        result = propose_assignment(x,g)
        self.assertFalse(result['linear_improvement_certified'])
        self.assertGreater(result['roundoff_bound'],0.)
        self.assertTrue(np.array_equal(result['permutation'],[0,1]))
        self.assertGreaterEqual(result['solver_linear_delta'],-result['roundoff_bound'])

    def test_equal_gradient_rows_have_zero_multiset_linear_gain(self):
        x=np.array([[0.,1.],[4.,2.],[8.,-3.]])
        g=np.tile([2.,-1.],(3,1))
        result=propose_assignment(x,g)
        self.assertTrue(np.array_equal(result['permutation'],np.arange(3)))
        self.assertFalse(result['linear_improvement_certified'])

    def test_raw_cycle_conjugation_matches_lengths_and_movement(self):
        for primary in ([0], [1,0], [1,2,0,4,3], [1,0,2,4,5,3]):
            for group in range(8):
                result=cycle_matched_random(primary,seed=3,group_key=group)
                p=np.asarray(primary);q=result['permutation'];r=result['relabeling']
                self.assertTrue(np.array_equal(q[r],r[p]))
                self.assertEqual(cycle_lengths(q),cycle_lengths(p))
                self.assertEqual(result['moved_positions'],int(np.count_nonzero(p!=np.arange(len(p)))))
                self.assertTrue(result['before_common_acceptance'])
                CHECK_COUNTERS['conjugacy_cases'] += 1

    def test_random_key_determinism_and_global_rng_unchanged(self):
        python_state=random.getstate();numpy_state=np.random.get_state()
        one=cycle_matched_random([1,2,0,4,3],seed=3,group_key=938)
        two=cycle_matched_random([1,2,0,4,3],seed=3,group_key=938)
        self.assertTrue(np.array_equal(one['permutation'],two['permutation']))
        self.assertEqual(one['key_sha256'],two['key_sha256'])
        self.assertEqual(random.getstate(),python_state)
        after=np.random.get_state()
        self.assertEqual(after[0],numpy_state[0]);self.assertTrue(np.array_equal(after[1],numpy_state[1]))
        self.assertEqual(after[2:],numpy_state[2:])

    def test_common_acceptance_requires_strict_teacher_improvement(self):
        base=np.array([[0.,1.],[0.,.5]])
        gcn=np.array([[1.,0.],[0.,1.]])
        good=accept_if_improved(base,base-np.array([0.,.1]),gcn,gcn)
        self.assertTrue(good['accepted']);self.assertGreater(good['teacher_improvement'],ACCEPTANCE_TOLERANCE)
        self.assertFalse(accept_if_improved(base,base,gcn,gcn)['accepted'])
        self.assertFalse(accept_if_improved(base,base-np.array([0.,1e-9]),gcn,gcn)['accepted'])
        self.assertFalse(accept_if_improved(base,base+np.array([0.,.1]),gcn,gcn)['accepted'])

    def test_pointwise_GCN_class0_guard_and_first_class_tie(self):
        base=np.array([[0.,1.],[0.,.5]]);good=base-np.array([0.,.2])
        gcn_base=np.array([[1.,1.],[0.,1.]])
        self.assertTrue(accept_if_improved(base,good,gcn_base,gcn_base)['accepted'])
        changed=np.array([[0.,1.],[1.,0.]])
        receipt=accept_if_improved(base,good,gcn_base,changed)
        self.assertFalse(receipt['accepted']);self.assertEqual(receipt['lost_protected_GCN_count'],1)
        self.assertEqual(receipt['protected_GCN_count'],1)
        empty=np.array([[0.,1.],[0.,1.]])
        self.assertTrue(accept_if_improved(base,good,empty,empty)['accepted'])

    def test_acceptance_nonfinite_rejected_and_shapes_invalid(self):
        x=np.array([[0.,1.]])
        for value in (np.array([[np.nan,1.]]),np.array([[0.,np.inf]])):
            result=accept_if_improved(x,value,x,x)
            self.assertFalse(result['accepted']);self.assertFalse(result['finite'])
        with self.assertRaises(ValueError):accept_if_improved(x,np.ones((1,3)),x,x)
        with self.assertRaises(ValueError):accept_if_improved(x,np.ones((2,2)),x,x)

    def test_invalid_feature_gradient_permutation_or_keys_rejected(self):
        x=np.array([[0.,1.],[2.,3.]])
        for value in (np.ones(2),np.empty((0,2)),np.array([[np.nan,0.]]),np.array([[np.inf,0.]])):
            with self.subTest(value=value),self.assertRaises(ValueError):propose_assignment(value,value)
        with self.assertRaises(ValueError):propose_assignment(x,np.ones((2,1)))
        for p in ([0,0],[1,2],[0.,1.],[],[True,False]):
            with self.subTest(p=p),self.assertRaises(ValueError):apply_permutation(x,p)
        for options in (dict(seed=True,group_key=1),dict(seed=1,group_key=-1)):
            with self.subTest(options=options),self.assertRaises(ValueError):cycle_matched_random([1,0],**options)


if __name__=='__main__':unittest.main()
