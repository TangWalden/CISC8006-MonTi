"""CPU synthetic contracts; execute only after committed source is frozen."""
import random
import unittest

import torch

from course.feature_logit_sharpen.core import ARMS, transform_features


class SharpenTests(unittest.TestCase):
    def call(self, x, lo=None, hi=None, **kwargs):
        lo = torch.zeros(x.shape[1]) if lo is None else lo
        hi = torch.ones(x.shape[1]) if hi is None else hi
        options = dict(seed=3, group=938, original_count=100)
        options.update(kwargs)
        return transform_features(x, lo, hi, **options)

    def test_closed_known_answer_and_fixed_arm_order(self):
        x = torch.tensor([[0., .25, .5, .75, 1.]])
        result = self.call(x)
        self.assertEqual(tuple(result['arms']), ARMS)
        self.assertTrue(torch.equal(result['arms']['logit_sharp'], torch.tensor([[0., .1, .5, .9, 1.]])))
        self.assertTrue(torch.equal(result['arms']['linear_contrast'], torch.tensor([[0., .125, .5, .875, 1.]])))
        self.assertEqual(result['beta'], 2.)
        self.assertEqual(result['linear_slope'], 1.5)

    def test_monotone_and_preserves_half_sides_endpoints(self):
        x = torch.linspace(0, 1, 101).reshape(1, -1)
        result = self.call(x)
        for arm in ('logit_sharp', 'linear_contrast'):
            y = result['arms'][arm]
            self.assertTrue(bool((y[:, 1:] >= y[:, :-1]).all()))
            self.assertTrue(bool((y[x < .5] <= x[x < .5]).all()))
            self.assertTrue(bool((y[x > .5] >= x[x > .5]).all()))
            self.assertEqual(float(y[0, 0]), 0.)
            self.assertEqual(float(y[0, -1]), 1.)

    def test_nonunit_raw_box_and_exact_flat_signed_zero(self):
        x = torch.tensor([[-1., 4., -0.], [1., 12., 0.]])
        lo, hi = torch.tensor([-2., 0., 0.]), torch.tensor([2., 16., 0.])
        result = self.call(x, lo, hi)
        self.assertTrue(torch.equal(result['arms']['logit_sharp'][:, :2], torch.tensor([[-1.6, 1.6], [1.6, 14.4]])))
        for arm, y in result['arms'].items():
            self.assertTrue(torch.equal(y[:, 2].contiguous().view(torch.uint8), x[:, 2].contiguous().view(torch.uint8)), arm)
            self.assertTrue(bool(((y >= lo)&(y <= hi)).all()))
        self.assertEqual(result['zero_width_dimensions'], 1)

    def test_baseline_bytes_immutability_and_no_aliases(self):
        x = torch.tensor([[.2, .8, .4], [.3, .6, .9]])
        lo, hi = torch.zeros(3), torch.ones(3)
        old = [value.clone() for value in (x, lo, hi)]
        result = self.call(x, lo, hi)
        self.assertTrue(torch.equal(result['arms']['baseline'].view(torch.uint8), x.view(torch.uint8)))
        self.assertTrue(result['identity']['baseline']['exact_equals_baseline'])
        for value, before in zip((x, lo, hi), old):self.assertTrue(torch.equal(value, before))
        result['arms']['baseline'][0, 0] = .7
        self.assertTrue(torch.equal(x, old[0]))

    def test_roundoff_baseline_preserved_other_arms_inside_box(self):
        x = torch.tensor([[-5e-7, 1.+5e-7]])
        result = self.call(x)
        self.assertTrue(torch.equal(result['arms']['baseline'], x))
        self.assertFalse(result['baseline_exact_box'])
        for arm in ARMS[1:]:self.assertTrue(bool(((result['arms'][arm] >= 0)&(result['arms'][arm] <= 1)).all()))
        with self.assertRaises(ValueError):self.call(torch.tensor([[-2e-6, .5]]))

    def test_all_flat_and_all_fixed_point_identity(self):
        flat = self.call(torch.tensor([[7., -0.]]), torch.tensor([7., 0.]), torch.tensor([7., 0.]))
        fixed = self.call(torch.tensor([[0., .5, 1.]]))
        for result in (flat, fixed):
            for arm in ARMS:self.assertTrue(result['identity'][arm]['exact_equals_baseline'], arm)
            self.assertEqual(result['maximum_recorded_preclip_unit_l2_gap'], 0.)

    def test_private_shuffle_deterministic_and_no_global_rng_change(self):
        x = torch.tensor([[.02, .15, .3, .65, .9, .97], [.1, .2, .4, .6, .7, .9]])
        python_before, torch_before = random.getstate(), torch.random.get_rng_state().clone()
        first, second = self.call(x), self.call(x)
        self.assertEqual(random.getstate(), python_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertEqual(first['shuffle_receipts'], second['shuffle_receipts'])
        self.assertTrue(torch.equal(first['arms']['shuffled_delta'], second['arms']['shuffled_delta']))
        self.assertNotEqual(first['shuffle_receipts'][0]['key_sha256'], self.call(x, group=939)['shuffle_receipts'][0]['key_sha256'])
        self.assertNotEqual(first['shuffle_receipts'][0]['key_sha256'], first['shuffle_receipts'][1]['key_sha256'])

    def test_shuffle_matches_preclip_unit_norm_not_terminal_norm(self):
        x = torch.tensor([[0., .01, .1, .2, .8, .9, .99, 1.]])
        # Fixed, synthetic keys exercise clipping without selecting real cases.
        clipped = []
        for group in range(12):
            result = self.call(x, group=group)
            primary, shuffled = (result['stats'][arm]['rows'][0] for arm in ('logit_sharp', 'shuffled_delta'))
            self.assertAlmostEqual(primary['preclip_unit_l2'], shuffled['preclip_unit_l2'], places=12)
            if shuffled['preclip_below_zero_coordinates']+shuffled['preclip_above_one_coordinates']:
                clipped.append(abs(shuffled['postclip_unit_l2']-shuffled['preclip_unit_l2']))
        self.assertTrue(any(value > 1e-5 for value in clipped))

    def test_raw_dose_is_independently_reported_for_anisotropic_box(self):
        lo, hi = torch.tensor([0., -100., 3.]), torch.tensor([1., 100., 3.])
        x = torch.tensor([[.25, 50., 3.]])
        result = self.call(x, lo, hi)
        for arm, y in result['arms'].items():
            actual = y.double()-x.double()
            self.assertAlmostEqual(result['stats'][arm]['raw_delta_frobenius'], float(actual.square().sum().sqrt()), places=12)
            self.assertAlmostEqual(result['stats'][arm]['raw_delta_linf'], float(actual.abs().max()), places=12)
        self.assertEqual(result['identity']['logit_sharp']['changed_coordinates'], 2)

    def test_invalid_feature_shapes_dtypes_and_nonfinite_values(self):
        for x in (torch.ones(3), torch.empty(0, 3), torch.ones(2, 3, dtype=torch.float64),
                  torch.tensor([[float('nan')]]), torch.tensor([[float('inf')]])):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                transform_features(x, torch.zeros(3), torch.ones(3), seed=3, group=1, original_count=10)

    def test_invalid_bounds_and_flat_value_rejected(self):
        x = torch.tensor([[.2, .8]])
        for lo, hi in ((torch.zeros(1), torch.ones(2)),
                       (torch.zeros(2, dtype=torch.float64), torch.ones(2)),
                       (torch.tensor([1., 0.]), torch.tensor([0., 1.])),
                       (torch.tensor([float('nan'), 0.]), torch.ones(2)),
                       (torch.zeros(2), torch.tensor([float('inf'), 1.]))):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):self.call(x, lo, hi)
        with self.assertRaises(ValueError):self.call(torch.tensor([[.2, 7.1]]), torch.tensor([0., 7.]), torch.tensor([1., 7.]))

    def test_invalid_identity_keys_rejected(self):
        for options in (dict(seed=True), dict(seed=-1), dict(group=1.5), dict(group=-1),
                        dict(original_count=0), dict(original_count=2.)):
            with self.subTest(options=options), self.assertRaises(ValueError):self.call(torch.tensor([[.2]]), **options)


if __name__ == '__main__':
    unittest.main()
