"""Meaningful CPU-only reference/mixing contracts; no real data or model calls.

Execute only after committed source and the synthetic registration are frozen.
"""
import unittest

import torch

from course.train_prototype_mix.core import ARMS, selectreference, mixfeatures


class PrototypeTests(unittest.TestCase):
    def source(self):
        features = torch.tensor([[0.,2.,7.], [2.,4.,7.], [10.,12.,7.], [-100.,100.,7.], [100.,-100.,7.]])
        # Held-out label values are intentionally invalid. A reference function
        # that examines them rather than TRAIN-only labels will fail this test.
        labels = torch.tensor([0.,0.,1.,float('nan'),-19.])
        return features, labels, torch.tensor([0,1,2]), torch.tensor([3]), torch.tensor([4])

    def test_known_train_only_means_and_global_unlabeled_box(self):
        reference = selectreference(*self.source())
        self.assertTrue(torch.equal(reference['benign_mean'], torch.tensor([1.,3.,7.])))
        self.assertTrue(torch.equal(reference['all_train_mean'], torch.tensor([4.,6.,7.])))
        self.assertTrue(torch.equal(reference['fraud_mean'], torch.tensor([10.,12.,7.])))
        self.assertTrue(torch.equal(reference['minimum'], torch.tensor([-100.,-100.,7.])))
        self.assertTrue(torch.equal(reference['maximum'], torch.tensor([100.,100.,7.])))
        self.assertEqual(reference['counts']['label_values_read'], 3)
        self.assertEqual(reference['counts']['benign_train_nodes'], 2)
        self.assertEqual(reference['counts']['fraud_train_nodes'], 1)

    def test_heldout_label_poison_does_not_change_any_reference(self):
        values = self.source()
        first = selectreference(*values)
        labels = values[1].clone();labels[3]=float('inf');labels[4]=2345
        second = selectreference(values[0], labels, *values[2:])
        for key in ('minimum','maximum','benign_mean','all_train_mean','fraud_mean','train_ids','train_labels','benign_ids','fraud_ids'):
            self.assertTrue(torch.equal(first[key], second[key]), key)

    def test_exact_integral_float_ids_are_accepted(self):
        values = self.source()
        reference = selectreference(values[0], values[1], torch.tensor([0.,1.,2.]), values[3], values[4])
        self.assertEqual(reference['train_ids'].dtype, torch.int64)
        self.assertTrue(torch.equal(reference['train_ids'], values[2]))

    def test_duplicate_fractional_and_out_of_range_train_ids_rejected(self):
        values = self.source()
        for ids in ([0,0,2], [0.,1.5,2.], [-1,1,2], [0,1,5], [0.,float('nan'),2.]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                selectreference(values[0], values[1], ids, values[3], values[4])

    def test_partition_overlap_and_heldout_duplicates_rejected(self):
        values = self.source()
        for valid, test in (([2,3],[4]), ([3],[3,4]), ([3,3],[4])):
            with self.subTest(valid=valid, test=test), self.assertRaises(ValueError):
                selectreference(values[0], values[1], values[2], valid, test)

    def test_both_train_classes_required_and_train_labels_binary(self):
        values = self.source()
        with self.assertRaises(ValueError):selectreference(values[0],values[1],[0,1],[2,3],[4])
        for bad in (2.,float('nan')):
            labels=values[1].clone();labels[0]=bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                selectreference(values[0],labels,*values[2:])

    def test_fixed_half_mean_arms_and_half_row_headroom(self):
        reference = selectreference(*self.source())
        baseline = torch.tensor([[1.,3.,7.],[4.,6.,7.]])
        result = mixfeatures(baseline,reference)
        self.assertEqual(tuple(result['arms']), ARMS)
        expected = dict(baseline=baseline,
            benign_mix=torch.tensor([[1.,3.,7.],[2.5,4.5,7.]]),
            all_train_mix=torch.tensor([[2.5,4.5,7.],[4.,6.,7.]]),
            fraud_mix=torch.tensor([[5.5,7.5,7.],[7.,9.,7.]]))
        for arm,value in expected.items():self.assertTrue(torch.equal(result['arms'][arm],value),arm)
        self.assertEqual(result['alpha'], .5)
        self.assertEqual(result['identity']['baseline']['changed_rows'], 0)
        self.assertEqual(result['identity']['benign_mix']['changed_rows'], 1)
        self.assertEqual(result['identity']['all_train_mix']['changed_rows'], 1)
        self.assertEqual(result['identity']['fraud_mix']['changed_rows'], 2)

    def test_zero_width_dimensions_and_box_boundaries_are_frozen(self):
        reference=selectreference(*self.source())
        baseline=torch.tensor([[-100.,100.,7.],[100.,-100.,7.]])
        result=mixfeatures(baseline,reference)
        self.assertEqual(result['zero_width_dimensions'],1)
        for value in result['arms'].values():
            self.assertTrue(torch.equal(value[:,2],baseline[:,2]))
            self.assertTrue(bool(((value>=reference['minimum'])&(value<=reference['maximum'])).all()))
        self.assertTrue(torch.equal(result['arms']['baseline'].view(torch.uint8),baseline.view(torch.uint8)))

    def test_float32_signed_zero_identity_uses_bytes(self):
        features=torch.tensor([[-1.],[1.],[0.]])
        reference=selectreference(features,torch.tensor([0,0,1]),[0,1,2],[],[])
        baseline=torch.tensor([[-0.]])
        result=mixfeatures(baseline,reference)
        self.assertTrue(torch.equal(result['arms']['benign_mix'],baseline))
        self.assertFalse(result['identity']['benign_mix']['exact_equals_baseline'])
        self.assertEqual(result['identity']['benign_mix']['changed_coordinates'],1)
        self.assertTrue(result['identity']['baseline']['exact_equals_baseline'])

    def test_original_features_labels_ids_and_baseline_are_not_modified(self):
        values=self.source();reference=selectreference(*values)
        snapshots=[value.clone() for value in values]
        baseline=torch.tensor([[1.,3.,7.]]);old=baseline.clone()
        mixfeatures(baseline,reference)
        for before,after in zip(snapshots,values):
            self.assertTrue(torch.equal(before.contiguous().view(torch.uint8),after.contiguous().view(torch.uint8)))
        self.assertTrue(torch.equal(baseline.view(torch.uint8),old.view(torch.uint8)))

    def test_baseline_roundoff_tolerance_does_not_clip_baseline(self):
        features=torch.tensor([[0.,7.],[1.,7.],[.5,7.]])
        reference=selectreference(features,torch.tensor([0,0,1]),[0,1,2],[],[])
        baseline=torch.tensor([[-5e-7,7.]])
        result=mixfeatures(baseline,reference)
        self.assertFalse(result['baseline_exact_box'])
        self.assertLessEqual(result['baseline_box_max_violation'],1e-6)
        self.assertTrue(torch.equal(result['arms']['baseline'].view(torch.uint8),baseline.view(torch.uint8)))
        with self.assertRaises(ValueError):mixfeatures(torch.tensor([[-2e-6,7.]]),reference)

    def test_shapes_nonfinite_features_and_wrong_fixed_coordinate_rejected(self):
        values=self.source();reference=selectreference(*values)
        damaged=values[0].clone();damaged[4,0]=float('nan')
        with self.assertRaises(ValueError):selectreference(damaged,*values[1:])
        with self.assertRaises(ValueError):selectreference(values[0],values[1].reshape(-1,1),*values[2:])
        for baseline in (torch.zeros(1,2),torch.tensor([[float('nan'),0.,7.]]),torch.tensor([[0.,0.,7.0000005]])):
            with self.subTest(shape=baseline.shape),self.assertRaises(ValueError):mixfeatures(baseline,reference)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
