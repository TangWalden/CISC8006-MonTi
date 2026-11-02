"""Preregistered optional in-context extension sharing the primary API ledger.

Four train-meta examples receive label-blind descriptions. Only afterward do
local training labels determine their supervised grid weights. Twenty fixed,
class-balanced validation cases receive few-shot decisions. No LLM weights are
updated. All API calls, including retries, share the original 120-attempt cap.
This module imports no model/data runtime and never changes primary results.
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys
import unittest
from unittest.mock import patch

from course.llm_fusion import api_pilot as primary
from course.llm_fusion.contract import (
    BranchEvidence, ContractError, EvidenceCard, GraphSummary, LocalCase,
    cache_key, card_from_mapping, fact_index, public_payload, resolve_decision,
    strict_json, validate_description,
)


SELECTION_RULE = 'sha256_label_stratified_case_id_v1'
SEED = 20261103
EXTENSION_RULES = dict(enabled=True, mode='in_context', selection_rule=SELECTION_RULE,
                       selection_seed=SEED, train_per_class=2, probe_per_class=10,
                       max_extra_logical_requests=24, global_physical_request_cap=120,
                       invalid_train_description='deterministic_template')
FEWSHOT_PROMPT = primary.GENERATED_PROMPT + '''
The training_examples are four registered train-meta cases, not validation
answers. target_alpha_self is a supervised training target chosen locally from
the same finite grid by minimum binary cross-entropy using that training case's
label. Treat examples as demonstrations, not hidden validation labels or new
query facts. Infer a decision for query only. Return exactly {"alpha_self": NUMBER}.
This is in-context adaptation; you are not updating your model weights.'''


def select_ids(labels, population, per_class):
    """Hash only seed/population/class/local ID; probabilities never enter selection."""
    if population not in ('train_meta', 'validation_probe') or type(per_class) is not int or per_class < 1:
        raise ContractError('Invalid fixed selection population/count')
    if type(labels) is not dict or any(type(k) is not str or type(v) is not int or v not in (0, 1) for k, v in labels.items()):
        raise ContractError('Selection requires local binary supervision')
    chosen = []
    for label in (0, 1):
        candidates = [case_id for case_id, truth in labels.items() if truth == label]
        if len(candidates) < per_class:
            raise ContractError('Insufficient registered examples in one class')
        def order(case_id):
            return primary.sha(('%d|%s|%d|%s' % (SEED, population, label, case_id)).encode()), case_id
        chosen.extend(sorted(candidates, key=order)[:per_class])
    return tuple(chosen)


def frozen_selection(meta_labels, probe_labels):
    """Can be called before the first primary API request to freeze exact IDs."""
    return dict(selected_train_case_ids=list(select_ids(meta_labels, 'train_meta', 2)),
                selected_probe_case_ids=list(select_ids(probe_labels, 'validation_probe', 10)))


def load_population(cards_path, labels_path):
    cases, labels = {}, {}
    for row in primary.read_jsonl(cards_path):
        if type(row) is not dict or set(row) != {'case_id', 'context_sha256', 'card'}:
            raise ContractError('Invalid local evidence envelope')
        card = card_from_mapping(row['card'])
        if card.source != 'registered_numeric_pilot':
            raise ContractError('Registered numeric evidence source required')
        case = LocalCase(row['case_id'], row['context_sha256'], card)
        if case.case_id in cases:
            raise ContractError('Duplicate local evidence case')
        cases[case.case_id] = case
    for row in primary.read_jsonl(labels_path):
        if type(row) is not dict or set(row) != {'case_id', 'label'} or type(row['case_id']) is not str or type(row['label']) is not int or row['label'] not in (0, 1):
            raise ContractError('Invalid private supervision envelope')
        if row['case_id'] in labels:
            raise ContractError('Duplicate local supervision')
        labels[row['case_id']] = row['label']
    if set(cases) != set(labels):
        raise ContractError('Supervision and evidence coverage differ')
    return cases, labels


def training_target(card, label):
    """TRAIN ONLY label-supervised oracle target, never a deployed query oracle.

    Equal BCE uses nearest-to-half, then lower-alpha tie breaking. Clipping only
    makes endpoint probabilities finite for local training-target calculation.
    """
    if type(label) is not int or label not in (0, 1):
        raise ContractError('Invalid local training label')
    def objective(alpha):
        probability = alpha * card.self_evidence.p_fraud + (1-alpha) * card.graph_context.p_fraud
        probability = min(1-1e-12, max(1e-12, probability))
        loss = -math.log(probability if label else 1-probability)
        return loss, abs(alpha-0.5), alpha
    return min(primary.GRID, key=objective)


def template_description(card):
    """Frozen neutral fallback for invalid/missing TRAIN descriptions only."""
    facts = fact_index(card)
    roles = dict(self=['self.p_fraud', 'self.calibration'],
                 graph_context=['graph_context.p_fraud', 'current_graph.degree_nonself',
                                'current_graph.neighbor_similarity_mean', 'current_graph.neighbor_similarity_std'],
                 conflict=['self.p_fraud', 'graph_context.p_fraud'])
    summaries = dict(self='The self branch supplies own-feature evidence; its reliability is not verified.',
                     graph_context='The graph branch includes center features and network context; missing values remain missing.',
                     conflict='The streams overlap; neither estimate is verified as correct by this description.')
    payload = {name: dict(summary=summaries[name], refs=refs,
                          echoes=[dict(fact_id=key, value=facts[key]) for key in refs])
               for name, refs in roles.items()}
    return validate_description(card, primary.canonical(payload), primary.CONFIG)


def example_payload(card, description, target_alpha):
    # This explicit allowlist intentionally excludes local case/context IDs and
    # the training label. The label-derived target alpha is permitted supervision.
    if type(target_alpha) not in (int, float) or target_alpha not in primary.GRID:
        raise ContractError('Training target outside the registered grid')
    return dict(evidence=public_payload(card), descriptions=primary.descriptions_payload(description),
                target_alpha_self=target_alpha)


def extension_request(case, protocol, protocol_sha, kind, description=None, examples=None):
    if kind not in ('train_description', 'fewshot'):
        raise ContractError('Unknown extension request type')
    describing = kind == 'train_description'
    prompt = primary.DESCRIPTION_PROMPT if describing else FEWSHOT_PROMPT
    max_tokens = 512 if describing else 64
    config = primary.CONFIG if describing else primary.DESCRIBED_CONFIG
    context = primary.sha(primary.canonical(dict(context=case.context_sha256, protocol=protocol_sha,
                                                 branches=protocol['branch_identity_sha256'])).encode())
    version = primary.sha(primary.canonical(dict(model_version=protocol['model_version'],
                                                 branches=protocol['branch_identity_sha256'])).encode())
    examples = [] if examples is None else examples
    key = cache_key(case.card, context, model_id=protocol['model'], model_version=version,
                    prompt_sha256=primary.sha(prompt.encode()),
                    description_prompt_sha256=primary.sha(primary.DESCRIPTION_PROMPT.encode()),
                    examples_sha256=primary.sha(primary.canonical(examples).encode()),
                    settings=dict(temperature=0, max_output_tokens=max_tokens), config=config,
                    description=description, stage='description' if describing else 'decision')
    if describing:
        payload = public_payload(case.card)
    else:
        if description is None or len(examples) != 4:
            raise ContractError('Few-shot input requires a grounded query and four fixed demonstrations')
        payload = dict(training_examples=examples,
                       query=dict(evidence=public_payload(case.card), descriptions=primary.descriptions_payload(description)))
    return dict(request_id=kind + '_' + key, max_tokens=max_tokens,
                messages=[dict(role='system', content=prompt), dict(role='user', content=primary.canonical(payload))])


def validate_extension(protocol, protocol_sha, primary_output, meta_cards, meta_labels, selection):
    extension = protocol.get('text_learning')
    if type(extension) is not dict or any(type(extension.get(k)) is not type(v) or extension.get(k) != v for k, v in EXTENSION_RULES.items()):
        raise ContractError('Text-learning rules were not frozen as this bounded extension')
    for name, path in (('meta_cards', meta_cards), ('meta_labels', meta_labels)):
        if extension.get(name + '_sha256') != primary.sha(path.read_bytes()):
            raise ContractError('Frozen train-meta input hash mismatch')
    if 'course/llm_fusion/text_learning.py' not in protocol['source_files']:
        raise ContractError('Extension source must be committed and frozen before primary API execution')
    for name, selected in selection.items():
        if name in extension and extension[name] != selected:
            raise ContractError('Exact frozen selection differs from the hash rule')
    registration = json.loads((primary_output / 'registered.json').read_text(encoding='utf-8'))
    if registration.get('protocol_sha256') != protocol_sha or registration.get('protocol', {}).get('text_learning') != extension:
        raise ContractError('Primary registration must already contain the identical extension rules')
    summary = json.loads((primary_output / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('protocol_sha256') != protocol_sha or summary.get('cases') != 32:
        raise ContractError('Complete primary case coverage required before this extension')


def checked_ledger(attempts, protocol_sha):
    ledger, responses = primary.completed_responses(attempts)
    if any(row.get('protocol_sha256') != protocol_sha for row in ledger):
        raise ContractError('Cross-protocol API ledger refused')
    if sum(row.get('attempted') is True for row in ledger) > 120:
        raise ContractError('Global request cap was already exceeded')
    return ledger, responses


def bounded_batch(requests, output, stem, attempts, key_file, protocol_sha):
    ledger, _ = checked_ledger(attempts, protocol_sha)
    if sum(row.get('attempted') is True for row in ledger) >= 120:
        return dict(returncode=0, stdout='Global physical-call cap reached; all affected cases use fixed fallback.', skipped_logical_requests=len(requests))
    # Reuse the EXACT primary ledger and PS transport, including its retry/token/
    # dollar reservations. Never reset the budget or open an extension ledger.
    return primary.run_batch(requests, output, stem, attempts, key_file, protocol_sha)


def run(args):
    for path in (args.meta_cards, args.meta_labels, args.cards, args.labels,
                 args.primary_output, args.output, args.key_file):
        primary.local_private(path)
    protocol_bytes = args.protocol.read_bytes()
    protocol = strict_json(protocol_bytes, primary.FusionConfig(max_response_bytes=1048576))
    primary.validate_protocol(protocol, args.cards, args.labels)
    meta, train_labels = load_population(args.meta_cards, args.meta_labels)
    probes, probe_labels = load_population(args.cards, args.labels)
    if len(probes) != 32 or set(meta) & set(probes):
        raise ContractError('Train and validation evidence must be distinct registered populations')
    selection = frozen_selection(train_labels, probe_labels)
    protocol_sha = primary.sha(protocol_bytes)
    validate_extension(protocol, protocol_sha, args.primary_output, args.meta_cards, args.meta_labels, selection)
    attempts = args.primary_output / 'api_attempts.jsonl'
    if not attempts.is_file():
        raise ContractError('Primary API ledger is missing')
    ledger_before, primary_responses = checked_ledger(attempts, protocol_sha)
    original_outcomes = primary.read_jsonl(args.primary_output / 'case_outcomes.jsonl')
    original = {row['case_id']: row for row in original_outcomes}
    if len(original_outcomes) != 32 or set(original) != set(probes):
        raise ContractError('Primary outcome cohort differs')
    args.output.mkdir(parents=True, exist_ok=False)
    primary.write_json(args.output / 'extension_registered.json', dict(
        schema='monti_text_learning_extension_execution_v1', protocol_sha256=protocol_sha,
        rules=EXTENSION_RULES, selection=selection,
        selection_timing='Rule/source/input hashes fixed in the primary registration; deterministic IDs recomputed here without API outcomes.',
        target_alpha_rule='Train-label minimum clipped BCE; nearest to half then smaller alpha on ties.',
        graph_meta_status='Original graph branch training evidence is not out-of-fold.',
        prior_physical_attempts=sum(row.get('attempted') is True for row in ledger_before),
        global_physical_request_cap=120, llm_weight_updates=0, new_branch_inference_calls=0))
    train_ids, probe_ids = selection['selected_train_case_ids'], selection['selected_probe_case_ids']
    train_requests = {key: extension_request(meta[key], protocol, protocol_sha, 'train_description') for key in train_ids}
    train_batch = bounded_batch(list(train_requests.values()), args.output, 'train_description', attempts, args.key_file, protocol_sha)
    _, responses = checked_ledger(attempts, protocol_sha)
    examples, train_records = [], []
    for key in train_ids:
        raw = responses.get(train_requests[key]['request_id'])
        try:
            description = validate_description(meta[key].card, raw, primary.CONFIG)
            status = 'generated_description_syntactically_valid'
        except ContractError:
            description = template_description(meta[key].card)
            status = 'frozen_template_fallback'
        # This is the FIRST use of the training label in constructing a worked
        # example. It was absent from the preceding description request.
        alpha = training_target(meta[key].card, train_labels[key])
        examples.append(example_payload(meta[key].card, description, alpha))
        train_records.append(dict(case_id=key, local_training_label=train_labels[key],
                                  target_alpha_self=alpha, description_status=status))
    primary.write_json(args.output / 'training_examples_private.json', dict(records=train_records, payloads=examples))
    query_descriptions, query_status, requests = {}, {}, {}
    for key in probe_ids:
        request = primary.request_for(probes[key], 'description', protocol, protocol_sha)
        raw = primary_responses.get(request['request_id'])
        try:
            description = validate_description(probes[key].card, raw, primary.CONFIG)
            query_descriptions[key] = description
            query_status[key] = 'reused_primary_grounded_description'
            requests[key] = extension_request(probes[key], protocol, protocol_sha, 'fewshot', description, examples)
        except ContractError:
            query_status[key] = 'primary_description_invalid_or_missing_static_fallback'
    decision_batch = bounded_batch(list(requests.values()), args.output, 'fewshot_decision', attempts, args.key_file, protocol_sha)
    ledger_after, responses = checked_ledger(attempts, protocol_sha)
    outcomes = []
    for key in probe_ids:
        raw = responses.get(requests.get(key, {}).get('request_id'))
        decision = resolve_decision(probes[key].card, raw, primary.DESCRIBED_CONFIG,
                                    description=query_descriptions.get(key))
        if original[key]['label'] != probe_labels[key]:
            raise ContractError('Local primary/extension labels differ')
        outcomes.append(dict(case_id=key, label=probe_labels[key],
                             alpha_self=decision.alpha_self, p_fraud=decision.p_fraud,
                             predicted_fraud=decision.predicted_fraud, status=decision.status,
                             used_fallback=decision.used_fallback, query_description_status=query_status[key],
                             primary_template_prediction=original[key]['predictions']['template'],
                             primary_generated_prediction=original[key]['predictions']['generated'],
                             self_only_prediction=original[key]['predictions']['self_only'],
                             graph_only_prediction=original[key]['predictions']['graph_only'],
                             static_half_prediction=original[key]['predictions']['static_half']))
    with (args.output / 'extension_outcomes.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in outcomes:
            stream.write(primary.canonical(row) + '\n')
    truth = [row['label'] for row in outcomes]
    comparisons = {name: primary.metrics([row[name] for row in outcomes], truth) for name in
                   ('predicted_fraud', 'primary_template_prediction', 'primary_generated_prediction',
                    'self_only_prediction', 'graph_only_prediction', 'static_half_prediction')}
    charged_p = sum(row.get('charged_prompt_tokens', 0) for row in ledger_after if row.get('attempted'))
    charged_c = sum(row.get('charged_completion_tokens', 0) for row in ledger_after if row.get('attempted'))
    summary = dict(schema='monti_text_learning_extension_summary_v1', status='completed_with_full_twenty_case_coverage',
                   protocol_sha256=protocol_sha, selection=selection, training_examples=4,
                   validation_cases=20, class_counts=[10, 10], fallback_cases=sum(row['used_fallback'] for row in outcomes),
                   training_template_fallbacks=sum(row['description_status'] == 'frozen_template_fallback' for row in train_records),
                   comparisons_on_identical_twenty_cases=comparisons, numeric_gate_comparison='Pending separate same-case join; not claimed here.',
                   cumulative_physical_attempts=sum(row.get('attempted') is True for row in ledger_after),
                   additional_physical_attempts=sum(row.get('attempted') is True for row in ledger_after)-sum(row.get('attempted') is True for row in ledger_before),
                   charged_prompt_tokens_cumulative=charged_p, charged_completion_tokens_cumulative=charged_c,
                   charged_usd_upper_bound_cumulative=charged_p*0.30/1000000+charged_c*1.20/1000000,
                   transport_batches=[train_batch, decision_batch], primary_results_unchanged=True,
                   adaptation='Four supervised train-meta demonstrations in context; no LLM weight fine-tuning.',
                   limitations=['Selected twenty-case subcohort, not the primary thirty-two-case denominator or a final test.',
                                'Original graph branch train-meta evidence is in-sample, not out-of-fold.',
                                'Descriptions have syntactic checks, not semantic truth verification.',
                                'Training target weights are supervised; validation labels never choose query weights.'],
                   new_branch_inference_calls=0, llm_weight_updates=0)
    primary.write_json(args.output / 'extension_summary.json', summary)
    print(primary.canonical(dict(status=summary['status'], cases=20,
                                 cumulative_attempts=summary['cumulative_physical_attempts'],
                                 fallback_cases=summary['fallback_cases'])))


class SyntheticTextLearningTests(unittest.TestCase):
    def test_balanced_deterministic_selection(self):
        labels = {'local-%02d' % i: i % 2 for i in range(32)}
        a = select_ids(labels, 'validation_probe', 10)
        b = select_ids(dict(reversed(list(labels.items()))), 'validation_probe', 10)
        self.assertEqual(a, b)
        self.assertEqual((len(a), sum(labels[k] for k in a)), (20, 10))

    def test_selection_has_no_prediction_argument(self):
        labels = {'local-%02d' % i: i % 2 for i in range(32)}
        self.assertEqual(len(select_ids(labels, 'train_meta', 2)), 4)
        with self.assertRaises(ContractError):
            select_ids({'only-benign': 0}, 'train_meta', 2)

    def test_training_oracle_and_tie_are_local(self):
        card = EvidenceCard(BranchEvidence(.2), BranchEvidence(.8), GraphSummary(0))
        self.assertEqual(training_target(card, 1), 0)
        self.assertEqual(training_target(card, 0), 1)
        tied = EvidenceCard(BranchEvidence(.8), BranchEvidence(.8), GraphSummary(0))
        self.assertEqual(training_target(tied, 1), .5)

    def test_template_is_grounded_and_missingness_preserved(self):
        card = EvidenceCard(BranchEvidence(.2), BranchEvidence(.8), GraphSummary(0))
        desc = template_description(card)
        encoded = primary.canonical(primary.descriptions_payload(desc))
        self.assertEqual(validate_description(card, encoded).card_sha256, desc.card_sha256)
        self.assertIn('null', encoded)

    def test_query_payload_has_no_private_identity_or_label(self):
        card = EvidenceCard(BranchEvidence(.2), BranchEvidence(.8), GraphSummary(0))
        case = LocalCase('sensitive-private-validation-id', 'a'*64, card)
        desc = template_description(card)
        examples = [example_payload(card, desc, .5) for _ in range(4)]
        protocol = dict(branch_identity_sha256='b'*64, model_version='fixture-version', model='deepseek-flash')
        request = extension_request(case, protocol, 'c'*64, 'fewshot', desc, examples)
        content = request['messages'][1]['content']
        self.assertNotIn(case.case_id, content)
        self.assertNotIn(case.context_sha256, content)
        self.assertNotIn('"label"', content)
        self.assertEqual(set(json.loads(content)), {'training_examples', 'query'})

    def test_invalid_query_preserves_static_fallback(self):
        card = EvidenceCard(BranchEvidence(.2), BranchEvidence(.8), GraphSummary(0))
        result = resolve_decision(card, None, primary.DESCRIBED_CONFIG, description=None)
        self.assertTrue(result.used_fallback)
        self.assertEqual((result.p_fraud, result.alpha_self), (.5, .5))
        self.assertEqual(EXTENSION_RULES['global_physical_request_cap'], primary.CAPS['max_requests'])

    def test_exhausted_shared_cap_never_invokes_transport(self):
        ledger = [dict(attempted=True) for _ in range(120)]
        with patch(__name__ + '.checked_ledger', return_value=(ledger, {})):
            with patch.object(primary, 'run_batch', side_effect=AssertionError('must not call transport')):
                result = bounded_batch([{}, {}], None, 'fixture', None, None, 'a'*64)
        self.assertEqual(result['skipped_logical_requests'], 2)


def main():
    if sys.argv[1:] == ['--self-test']:
        unittest.main(argv=[sys.argv[0]])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('meta-cards', 'meta-labels', 'cards', 'labels', 'protocol', 'primary-output', 'output', 'key-file'):
        parser.add_argument('--' + name, required=True, type=pathlib.Path)
    run(parser.parse_args())


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(primary.canonical(dict(status='text_learning_extension_failed', error_type=type(error).__name__)))
        sys.exit(1)
