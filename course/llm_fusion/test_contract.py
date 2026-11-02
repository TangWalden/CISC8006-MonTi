"""Synthetic offline checks only. No APIs, datasets, checkpoints or ML imports."""

import copy
from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

from course.llm_fusion.contract import (
    BranchEvidence, ContractError, EvidenceCard, FusionConfig, GraphSummary,
    LocalCase, cache_key, card_from_mapping, fact_index, public_payload,
    resolve_batch, resolve_decision, strict_json, validate_decision,
    validate_description,
)


def fixture(p_self=0.8, p_graph=0.2):
    return EvidenceCard(BranchEvidence(p_self, 'uncalibrated'),
                        BranchEvidence(p_graph, 'not_assessed'), GraphSummary(3))


def description_payload(card):
    return {
        'self': {'summary': 'The self branch supplies numeric feature evidence.',
                 'refs': ['self.p_fraud', 'self.calibration'],
                 'echoes': [{'fact_id': 'self.p_fraud', 'value': card.self_evidence.p_fraud}]},
        'graph_context': {'summary': 'The graph branch includes center and network evidence.',
                          'refs': ['graph_context.p_fraud', 'current_graph.neighbor_similarity_mean'],
                          'echoes': [{'fact_id': 'graph_context.p_fraud', 'value': card.graph_context.p_fraud},
                                     {'fact_id': 'current_graph.neighbor_similarity_mean', 'value': None}]},
        'conflict': {'summary': 'The branch estimates can disagree; reliability remains unverified.',
                     'refs': ['self.p_fraud', 'graph_context.p_fraud'], 'echoes': []},
    }


class ContractTests(unittest.TestCase):
    def test_known_fusion_and_equal_threshold(self):
        result = resolve_decision(fixture(), '{"alpha_self":0.5}')
        self.assertEqual((result.p_fraud, result.predicted_fraud, result.used_fallback), (0.5, True, False))
        result = resolve_decision(fixture(), '{"alpha_self":1}')
        self.assertEqual(result.p_fraud, 0.8)
        self.assertEqual(resolve_decision(fixture(), '{"alpha_self":0}').p_fraud, 0.2)

    def test_both_wrong_has_no_convex_fusion_headroom(self):
        # Truth remains local test data; it never enters either evidence card.
        for card, truth in ((fixture(0.1, 0.4), True), (fixture(0.7, 0.9), False)):
            for alpha in (0, 0.5, 1):
                result = resolve_decision(card, json.dumps({'alpha_self': alpha}))
                self.assertNotEqual(result.predicted_fraud, truth)

    def test_exact_boundary_and_adjacent_float(self):
        below = 0.5 - 2 ** -54
        self.assertFalse(resolve_decision(fixture(below, below), '{"alpha_self":1}').predicted_fraud)
        self.assertTrue(resolve_decision(fixture(0.5, 0.5), '{"alpha_self":1}').predicted_fraud)

    def test_card_round_trip_and_no_local_identity_in_payload(self):
        card = fixture()
        local = LocalCase('local-sensitive-identity', 'a' * 64, card)
        payload = public_payload(local.card)
        self.assertEqual(card_from_mapping(payload), card)
        serialized = json.dumps(payload)
        self.assertNotIn(local.case_id, serialized)
        self.assertNotIn(local.context_sha256, serialized)
        self.assertIsNone(payload['current_graph']['neighbor_similarity_mean'])

    def test_top_level_card_leakage_is_rejected(self):
        for field, value in (('node_id', 99), ('label', 1), ('attack_status', 'injected'),
                             ('clean_difference', 0.1), ('article_text', 'invented text')):
            payload = public_payload(fixture())
            payload[field] = value
            with self.assertRaises(ContractError):
                card_from_mapping(payload)

    def test_nested_card_leakage_and_missing_branch_rejected(self):
        for section in ('self', 'graph_context', 'current_graph'):
            payload = public_payload(fixture())
            payload[section]['node_id'] = 44
            with self.assertRaises(ContractError):
                card_from_mapping(payload)
        payload = public_payload(fixture())
        del payload['self']
        with self.assertRaises(ContractError):
            card_from_mapping(payload)

    def test_numeric_types_ranges_and_source_marker(self):
        for value in (True, False, '0.5', None, -0.01, 1.01, float('nan'), float('inf'), 10 ** 400):
            with self.assertRaises(ContractError):
                BranchEvidence(value)
        with self.assertRaises(ContractError):
            BranchEvidence(0.5, 'certainly_reliable')
        with self.assertRaises(ContractError):
            replace(fixture(), source='real_data')

    def test_registered_pilot_marker_is_structural_not_provenance_verification(self):
        # No data/protocol is opened here. The live caller must verify its envelope.
        pilot = replace(fixture(), source='registered_numeric_pilot')
        self.assertEqual(card_from_mapping(public_payload(pilot)), pilot)
        self.assertEqual(public_payload(pilot)['source'], 'registered_numeric_pilot')
        self.assertEqual(fixture().source, 'synthetic_fixture')
        with self.assertRaises(ContractError):
            replace(pilot, source='unregistered_numeric_data')

    def test_graph_statistics_missingness_and_bounds(self):
        self.assertEqual(GraphSummary(0).degree_nonself, 0)
        self.assertEqual(GraphSummary(1, -1, 0).neighbor_similarity_mean, -1)
        for args in ((True,), (-1,), (3.0,), (0, 0, 0), (2, None, 0),
                     (2, 1.1, 0), (2, 0, -0.1), (2, 0, 1.1), (1, 0, 0.2)):
            with self.assertRaises(ContractError):
                GraphSummary(*args)

    def test_card_and_fact_index_are_immutable(self):
        card = fixture()
        with self.assertRaises(FrozenInstanceError):
            card.self_evidence.p_fraud = 0.9
        with self.assertRaises(TypeError):
            fact_index(card)['self.p_fraud'] = 0.9
        public_payload(card)['self']['p_fraud'] = 0.9
        self.assertEqual(card.self_evidence.p_fraud, 0.8)

    def test_valid_grounded_description_and_immutable_echoes(self):
        card = fixture()
        description = validate_description(card, json.dumps(description_payload(card)))
        self.assertIsInstance(description.sections, tuple)
        self.assertIsInstance(description.sections[0][1].echoes, tuple)
        with self.assertRaises(FrozenInstanceError):
            description.sections[0][1].summary = 'changed'
        result = resolve_decision(card, '{"alpha_self":0.5}', FusionConfig(require_description=True), description=description)
        self.assertFalse(result.used_fallback)

    def test_description_hallucinated_reference_rejected(self):
        card = fixture()
        payload = description_payload(card)
        payload['self']['refs'].append('article.truth')
        with self.assertRaises(ContractError):
            validate_description(card, json.dumps(payload))

    def test_description_mismatched_numeric_echo_rejected(self):
        card = fixture()
        for value in (0.9, True, '0.8'):
            payload = description_payload(card)
            payload['self']['echoes'][0]['value'] = value
            with self.assertRaises(ContractError):
                validate_description(card, json.dumps(payload))

    def test_description_missing_fact_cannot_be_numeric(self):
        card = fixture()
        payload = description_payload(card)
        payload['graph_context']['echoes'][1]['value'] = 0
        with self.assertRaises(ContractError):
            validate_description(card, json.dumps(payload))

    def test_description_role_grounding(self):
        card = fixture()
        for name, refs in (('self', ['graph_context.p_fraud']),
                           ('graph_context', ['self.p_fraud']), ('conflict', ['self.p_fraud'])):
            payload = description_payload(card)
            payload[name]['refs'], payload[name]['echoes'] = refs, []
            with self.assertRaises(ContractError):
                validate_description(card, json.dumps(payload))

    def test_prose_numeric_literals_banned_but_prose_not_truth_verified(self):
        card = fixture()
        payload = description_payload(card)
        payload['self']['summary'] = 'The chance is 0.9.'
        with self.assertRaises(ContractError):
            validate_description(card, json.dumps(payload))
        # This acceptance deliberately demonstrates the semantic-validation limit.
        payload['self']['summary'] = 'An unverified qualitative interpretation.'
        self.assertIsNotNone(validate_description(card, json.dumps(payload)))

    def test_description_cannot_be_reused_for_changed_card(self):
        card = fixture()
        description = validate_description(card, json.dumps(description_payload(card)))
        result = resolve_decision(fixture(0.7, 0.2), '{"alpha_self":1}', description=description)
        self.assertEqual(result.status, 'fallback_invalid_description')

    def test_decision_duplicates_nonfinite_bool_extra_and_non_grid(self):
        invalid = ('{"alpha_self":0,"alpha_self":1}', '{"alpha_self":NaN}',
                   '{"alpha_self":Infinity}', '{"alpha_self":1e999}', '{"alpha_self":true}',
                   '{"alpha_self":"0.5"}', '{"alpha_self":0.3}', '{"alpha_self":-1}',
                   '{"alpha_self":0.5,"verdict":"fraud"}', '[]', 'null')
        for raw in invalid:
            with self.assertRaises(ContractError):
                validate_decision(raw)

    def test_nested_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ContractError):
            strict_json('{"outer":{"x":1,"x":2}}', FusionConfig())

    def test_response_byte_limit_utf8_and_malformed_json(self):
        config = FusionConfig(max_response_bytes=24)
        for raw in ('{"alpha_self":0.5}' + ' ' * 30, b'\xff', '{not-json', '{"alpha_self":0.5} trailing'):
            self.assertEqual(resolve_decision(fixture(), raw, config).status, 'fallback_invalid_response')
        self.assertFalse(resolve_decision(fixture(), b'{"alpha_self":0.5}', config).used_fallback)

    def test_fixed_fallback_missing_timeout_and_invalid_description(self):
        config = FusionConfig(fallback_alpha_self=1)
        for raw, kwargs, status in ((None, {}, 'fallback_missing_response'),
                                    ('not json', {}, 'fallback_invalid_response'),
                                    ('{"alpha_self":0}', {'timed_out': True}, 'fallback_timeout'),
                                    ('{"alpha_self":0}', {'description': '{}'}, 'fallback_invalid_description')):
            result = resolve_decision(fixture(), raw, config, **kwargs)
            self.assertEqual((result.p_fraud, result.alpha_self, result.used_fallback), (0.8, 1.0, True))
            self.assertEqual(result.status, status)
        self.assertEqual(resolve_decision(fixture(), '{"alpha_self":0}', FusionConfig(require_description=True)).status,
                         'fallback_missing_description')

    def test_all_invalid_responses_preserve_every_case(self):
        cases = tuple(LocalCase('case-' + str(i), format(i, '064x'), fixture()) for i in range(5))
        results = resolve_batch(cases, (None, '', 'null', '{"alpha_self":true}', '{"alpha_self":NaN}'))
        self.assertEqual(len(results), len(cases))
        self.assertTrue(all(x.used_fallback and x.p_fraud == 0.5 for x in results))

    def test_malformed_input_batch_rejected_before_response_parsing(self):
        case = LocalCase('local', 'a' * 64, fixture())
        with patch('course.llm_fusion.contract.validate_decision', side_effect=AssertionError('must not parse')):
            with self.assertRaises(ContractError):
                resolve_batch((case, object()), ('{}', '{}'))
            with self.assertRaises(ContractError):
                resolve_batch((case,), ())
            with self.assertRaises(ContractError):
                resolve_decision(object(), '{}')

    def test_config_rejects_mutability_bool_and_invalid_grid(self):
        for kwargs in ({'alpha_grid': [0, 1]}, {'alpha_grid': (0, True)},
                       {'alpha_grid': (1, 0)}, {'alpha_grid': (0, 0.5, 0.5)},
                       {'fallback_alpha_self': 0.3}, {'threshold': float('nan')},
                       {'max_response_bytes': True}, {'require_description': 1}):
            with self.assertRaises(ContractError):
                FusionConfig(**kwargs)

    def test_cache_key_changes_for_context_identity_prompts_examples_and_settings(self):
        card = fixture()
        args = dict(model_id='synthetic-model', model_version='fixture-v1', prompt_sha256='b' * 64,
                    description_prompt_sha256='c' * 64, examples_sha256='d' * 64,
                    settings={'temperature': 0, 'top_p': 1, 'max_output_tokens': 200})
        original = cache_key(card, 'a' * 64, **args)
        self.assertEqual(original, cache_key(card, 'a' * 64, **copy.deepcopy(args)))
        self.assertNotEqual(original, cache_key(card, 'e' * 64, **args))
        for field, value in (('model_id', 'other-model'), ('model_version', 'fixture-v2'),
                             ('prompt_sha256', 'f' * 64), ('description_prompt_sha256', '0' * 64),
                             ('examples_sha256', '1' * 64), ('settings', {'temperature': 0.5})):
            changed = dict(args)
            changed[field] = value
            self.assertNotEqual(original, cache_key(card, 'a' * 64, **changed))
        self.assertNotEqual(original, cache_key(card, 'a' * 64, **args, stage='description'))

    def test_cache_key_covers_full_card_config_and_description(self):
        card = fixture()
        args = dict(model_id='synthetic-model', model_version='fixture-v1', prompt_sha256='b' * 64)
        original = cache_key(card, 'a' * 64, **args)
        variants = (fixture(0.7, 0.2), fixture(0.8, 0.3),
                    replace(card, self_evidence=BranchEvidence(0.8, 'not_assessed')),
                    replace(card, current_graph=GraphSummary(4)),
                    replace(card, current_graph=GraphSummary(3, 0.1, 0.2)))
        for changed in variants:
            self.assertNotEqual(original, cache_key(changed, 'a' * 64, **args))
        for config in (FusionConfig(threshold=0.6), FusionConfig(fallback_alpha_self=1),
                       FusionConfig(max_response_bytes=9000), FusionConfig(alpha_grid=(0, 0.25, 0.5, 1))):
            self.assertNotEqual(original, cache_key(card, 'a' * 64, **args, config=config))
        desc = validate_description(card, json.dumps(description_payload(card)))
        described = cache_key(card, 'a' * 64, **args, description=desc)
        self.assertNotEqual(original, described)
        payload = description_payload(card)
        payload['self']['summary'] = 'A different unverified interpretation of identical evidence.'
        changed = validate_description(card, json.dumps(payload))
        self.assertNotEqual(described, cache_key(card, 'a' * 64, **args, description=changed))

    def test_cache_rejects_ambiguous_context_and_missing_required_description(self):
        args = dict(model_id='synthetic-model', model_version='fixture-v1', prompt_sha256='b' * 64)
        with self.assertRaises(ContractError):
            cache_key(fixture(), 'node-only-id', **args)
        with self.assertRaises(ContractError):
            cache_key(fixture(), 'a' * 64, **args, config=FusionConfig(require_description=True))


if __name__ == '__main__':
    unittest.main()
