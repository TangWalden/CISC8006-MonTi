"""Synthetic CPU checks; no data/model/credential/API access or work on import."""
import math
import unittest

from course.dual_surrogate.prepare_teachers import (
    binary_metrics, build_sequence_arrays, fit_group_masks, improves_checkpoint,
    passes_quality, stratified_train_split)


class TeacherPreparationChecks(unittest.TestCase):
    def test_split_is_stratified_deterministic_and_covers_train_only(self):
        ids = list(range(20)); labels = [0]*10+[1]*10
        fit, stop = stratified_train_split(ids, labels)
        self.assertEqual((len(fit),len(stop)),(16,4))
        self.assertEqual(sum(labels[n] for n in fit),8)
        self.assertEqual(sum(labels[n] for n in stop),2)
        self.assertFalse(set(fit)&set(stop)); self.assertEqual(set(fit)|set(stop),set(ids))
        self.assertEqual((fit,stop),stratified_train_split(list(reversed(ids)),list(reversed(labels))))

    def test_split_rejects_duplicate_or_missing_class(self):
        for ids,labels in (([0,0],[0,1]),([0,1],[0,0]),([0,1],[0,2])):
            with self.assertRaises(ValueError): stratified_train_split(ids,labels)

    def test_only_fit_labels_enter_group_masks(self):
        masks=fit_group_masks(5,[0,2],[0,1])
        self.assertEqual(masks['train_neg'],[1,0,0,0,0])
        self.assertEqual(masks['train_pos'],[0,0,1,0,0])
        self.assertEqual(masks['train_masked'],[0,1,0,1,1])
        self.assertTrue(all(sum(masks[k][i] for k in masks)==1 for i in range(5)))

    def test_mask_rejects_invalid_id_and_label(self):
        for ids,labels in (([5],[0]),([0],[2]),([0,0],[0,1])):
            with self.assertRaises(ValueError): fit_group_masks(5,ids,labels)

    def test_earliest_maximum_checkpoint_ties_are_retained(self):
        self.assertTrue(improves_checkpoint(.6,None))
        self.assertFalse(improves_checkpoint(.6,.6))
        self.assertFalse(improves_checkpoint(.5,.6))
        self.assertTrue(improves_checkpoint(.7,.6))
        with self.assertRaises(ValueError): improves_checkpoint(float('nan'),.6)

    def test_metrics_known_perfect_and_tied_auc(self):
        result=binary_metrics([0,1,0,1],[.1,.9,.2,.8])
        self.assertEqual((result['accuracy'],result['macro_f1'],result['auc']),(1,1,1))
        ties=binary_metrics([0,1],[.5,.5])
        self.assertEqual(ties['auc'],.5)
        self.assertEqual((ties['tn'],ties['fn']),(1,1))

    def test_actual_argmax_can_be_supplied_for_rounded_probability_tie(self):
        result=binary_metrics([0,1],[.5,.5],[0,1])
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['auc'],.5)

    def test_quality_gate_rejects_majority_only_teacher(self):
        self.assertFalse(passes_quality(binary_metrics([0,0,0,1],[.1,.1,.1,.1])))
        self.assertTrue(passes_quality(binary_metrics([0,0,1,1],[.1,.2,.8,.9])))

    def test_sparse_sequences_mask_center_and_keep_nonfit_unknown(self):
        import numpy as np
        import scipy.sparse as sp
        # Path0--1--2. FIT nodes0 benign,2fraud; node1 internalSTOP unknown.
        adjacency=sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.float32))
        two=(adjacency@adjacency).astype(bool).astype(np.int64);two.setdiag(0);two.eliminate_zeros()
        x=np.array([[2.],[5.],[11.]],dtype=np.float32)
        seq,sums,degrees=build_sequence_arrays(x,adjacency,two,fit_group_masks(3,[0,2],[0,1]),chunk_size=1)
        # Center1 receives class groups(2,11) and its OWN5 as unknown.
        np.testing.assert_array_equal(seq[1,:,0],[5,2,11,5,0,0,5])
        # Center0 own benign label must not leak through length-two return walk.
        np.testing.assert_array_equal(seq[0,:,0],[2,0,0,3.5,0,11,2])
        self.assertEqual(float(degrees[0,3,0]),2)
        self.assertEqual(float(sums[0,3,0]),7)


if __name__ == '__main__':
    unittest.main()
