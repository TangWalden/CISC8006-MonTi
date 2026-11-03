"""Known-answer regressions for two concrete pre-freeze integration defects."""
import random
import unittest
import numpy as np
import torch
from course.dual_surrogate.runtime import checked_logits,fork_rng


class RuntimeChecks(unittest.TestCase):
    def test_singleton_gaga_logits_keep_target_axis(self):
        x=torch.tensor([.2,.8]);self.assertEqual(tuple(checked_logits(x,1).shape),(1,2))
        with self.assertRaises(RuntimeError):checked_logits(x,2)
        with self.assertRaises(RuntimeError):checked_logits(torch.tensor([[float('nan'),0.]]),1)

    def test_extra_teacher_construction_preserves_later_training_order(self):
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
        py=random.getstate();ns=np.random.get_state();ts=torch.random.get_rng_state().clone()
        expected=torch.randperm(256);torch.random.set_rng_state(ts)
        with fork_rng(20261103):
            torch.nn.Linear(768,64);torch.nn.Linear(64,2)
            random.random();np.random.rand()
        self.assertTrue(torch.equal(expected,torch.randperm(256)))
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(ns[1],np.random.get_state()[1]))


if __name__=='__main__':unittest.main()
