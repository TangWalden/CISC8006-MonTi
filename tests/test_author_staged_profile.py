"""Standard-library profile/lock guards only; no ML imports or asset reads."""
import importlib.util
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('author_profile_common',ROOT/'experiments/final_staged/common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


class AuthorProfileChecks(unittest.TestCase):
    def test_formal_profile_preserves_existing_default_lock(self):
        self.assertEqual(common.execution_profile(dict(schema='monti_final_staged_v1')),
            ('formal_reproduction','final_staged_exclusive.lock'))
    def test_author_profile_requires_explicit_separate_lock(self):
        valid=dict(schema='monti_author_staged_v1',execution_profile='author_upstream',guardian_lock='author_staged_exclusive.lock')
        self.assertEqual(common.execution_profile(valid),('author_upstream','author_staged_exclusive.lock'))
        for key in ('execution_profile','guardian_lock'):
            bad=dict(valid);bad.pop(key)
            with self.assertRaises(RuntimeError):common.execution_profile(bad)
    def test_unknown_or_cross_profile_lock_rejected(self):
        for value in (dict(schema='unknown'),dict(schema='monti_final_staged_v1',execution_profile='author_upstream'),
                dict(schema='monti_author_staged_v1',execution_profile='author_upstream',guardian_lock='final_staged_exclusive.lock'),
                dict(schema='monti_author_staged_v1',execution_profile='author_upstream',guardian_lock='../escape')):
            with self.assertRaises(RuntimeError):common.execution_profile(value)


if __name__=='__main__':unittest.main()
