"""File-handshake checks; no SSH, credentials, API, dataset or model calls."""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from course.alignment_agent.api_bridge import resolve_decision
from course.alignment_agent.controller import build_policy


def fixture_state(completed=1):
    return dict(completed_epoch=completed,train_groups=256,target_occurrences=500,
        margin_mean=1.,margin_median=1.,margin_positive_fraction=.8,hinge_mean=.2,
        softplus_mean=.3,distance_mean=.5,outside_fraction=.2,reference_quantile=.9,
        gradient_probe_groups=8,gradient_cosine_median=-.2,gradient_norm_ratio_median=2.,
        gradient_missing_count=0,prior_actions=[False]*completed)


class ControllerFileChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.stage=pathlib.Path(self.temporary.name)
        self.evidence=self.stage/'job';self.evidence.mkdir()
        self.protocol=dict(stage_root=str(self.stage),_protocol_sha256='a'*64)

    def tearDown(self):
        self.temporary.cleanup()

    def test_common_first_epoch_never_requests_agent(self):
        for arm in ('legacy_hinge_off','smooth_off','smooth_threshold','smooth_rule','smooth_agent'):
            p=build_policy(arm,3,self.protocol,self.evidence)
            self.assertFalse(p(1,None)['align'])
        self.assertEqual(list((self.stage/'requests').glob('*.json')),[])

    def test_fixed_controls_and_only_lagged_epoch(self):
        for arm,expected in (('legacy_hinge_off',False),('smooth_off',False),('smooth_threshold',True),('smooth_rule',True)):
            self.assertEqual(build_policy(arm,3,self.protocol,self.evidence)(2,fixture_state())['align'],expected)
        with self.assertRaises(RuntimeError):
            build_policy('smooth_agent',3,self.protocol,self.evidence)(3,fixture_state())

    def test_timeout_applies_retained_fallback_without_transport(self):
        p=build_policy('smooth_agent',3,self.protocol,self.evidence)
        with patch('course.alignment_agent.controller.time.monotonic',side_effect=[0.,.1,21.,21.1]):
            with patch('course.alignment_agent.controller.time.sleep'):
                result=p(2,fixture_state())
        self.assertTrue(result['fallback']);self.assertTrue(result['align'])
        self.assertEqual(result['decision_source'],'registered_fallback')
        self.assertEqual(len(list((self.stage/'requests').glob('*.json'))),1)
        self.assertTrue((self.evidence/'applied_controller_epoch_02.json').exists())

    def publish(self,raw,wrong_hash=False):
        request=next((self.stage/'requests').glob('*.json'))
        envelope=json.loads(request.read_text())
        decision=resolve_decision(envelope,raw,'a'*64)
        if wrong_hash:decision['state_sha256']='b'*64
        (self.stage/'responses'/request.name).write_text(json.dumps(decision))

    def test_valid_response_is_applied_with_exact_context(self):
        p=build_policy('smooth_agent',3,self.protocol,self.evidence)
        with patch('course.alignment_agent.controller.time.sleep',side_effect=lambda _:self.publish('{"align":false,"reason":"Measured training state supports delay."}')):
            result=p(2,fixture_state())
        self.assertFalse(result['align']);self.assertFalse(result['fallback'])
        self.assertEqual(result['decision_source'],'external_agent')

    def test_mismatched_context_is_never_applied(self):
        p=build_policy('smooth_agent',3,self.protocol,self.evidence)
        with patch('course.alignment_agent.controller.time.sleep',side_effect=lambda _:self.publish('{"align":false,"reason":"Wrong context must not be applied."}',True)):
            result=p(2,fixture_state())
        self.assertTrue(result['fallback']);self.assertTrue(result['align'])

    def test_malformed_response_keeps_epoch_and_falls_back(self):
        def invalid(_):
            request=next((self.stage/'requests').glob('*.json'))
            (self.stage/'responses'/request.name).write_text('{broken')
        with patch('course.alignment_agent.controller.time.sleep',side_effect=invalid):
            result=build_policy('smooth_agent',3,self.protocol,self.evidence)(2,fixture_state())
        self.assertTrue(result['fallback'])

    def test_nonnumeric_hash_field_cannot_crash_invalid_response_fallback(self):
        def invalid(_):
            request=next((self.stage/'requests').glob('*.json'))
            envelope=json.loads(request.read_text())
            decision=resolve_decision(envelope,'{"align":false,"reason":"Malformed hash fixture."}','a'*64)
            decision['response_sha256']=123
            (self.stage/'responses'/request.name).write_text(json.dumps(decision))
        with patch('course.alignment_agent.controller.time.sleep',side_effect=invalid):
            result=build_policy('smooth_agent',3,self.protocol,self.evidence)(2,fixture_state())
        self.assertTrue(result['fallback'])

    def test_matched_random_schedule_uses_actual_boolean_actions(self):
        schedule=self.stage/'schedule.json'
        actions=[False,True,False,True,False,False,True,False]
        schedule.write_text(json.dumps(dict(seed=3,protocol_sha256='a'*64,actions=actions,source_enabled_epochs=3)))
        p=build_policy('smooth_random',3,self.protocol,self.evidence,matched_schedule_path=schedule)
        observed=[p(1,None)['align']]+[p(e,fixture_state(e-1))['align'] for e in range(2,9)]
        self.assertEqual(observed,actions)
        self.assertEqual(sum(observed),3)

    def test_random_control_rejects_wrong_count_or_protocol(self):
        path=self.stage/'bad.json'
        path.write_text(json.dumps(dict(seed=3,protocol_sha256='a'*64,actions=[False]*8,source_enabled_epochs=2)))
        with self.assertRaises(RuntimeError):
            build_policy('smooth_random',3,self.protocol,self.evidence,matched_schedule_path=path)


if __name__=='__main__':
    unittest.main()
