"""Strict, transport-free numeric evidence fusion.

Defaults are SYNTHETIC CONVENTIONS, not an experimental freeze. Admitted source
markers are synthetic_fixture and registered_numeric_pilot. These are structural
declarations, not proof of provenance. A live pilot caller must verify its local
protocol envelope and branch/checkpoint/calibrator identities before admission;
protocol hashes stay local and belong in the complete context/cache identity.
Real evidence must never be relabeled synthetic. No raw article text is accepted.
Descriptions are syntactically grounded, not verified explanations: spelled-out
numbers and qualitative interpretations are not semantically checked.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType


SCHEMA = 'monti_numeric_fusion_evidence_v1'
ADMITTED_SOURCES = frozenset(('synthetic_fixture', 'registered_numeric_pilot'))
CALIBRATION = frozenset(('uncalibrated', 'not_assessed', 'held_out_calibration_declared'))
ZERO_DIGEST = '0' * 64


class ContractError(ValueError):
    """An input violates the declared interface; never silently omit its case."""


def _number(value, name, low=None, high=None):
    if type(value) not in (int, float):
        raise ContractError(name + ': expected a number, not bool/string')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ContractError(name + ': nonfinite number')
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ContractError(name + ': out of range or nonfinite')
    return value


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ContractError(name + ': unexpected or missing properties')


def _digest(value, name):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError(name + ': expected lowercase SHA-256')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)


@dataclass(frozen=True)
class BranchEvidence:
    p_fraud: float
    calibration: str = 'not_assessed'

    def __post_init__(self):
        object.__setattr__(self, 'p_fraud', _number(self.p_fraud, 'p_fraud', 0, 1))
        if type(self.calibration) is not str or self.calibration not in CALIBRATION:
            raise ContractError('Unknown calibration declaration')


@dataclass(frozen=True)
class GraphSummary:
    degree_nonself: int
    neighbor_similarity_mean: object = None
    neighbor_similarity_std: object = None

    def __post_init__(self):
        if type(self.degree_nonself) is not int or not 0 <= self.degree_nonself <= 2 ** 53 - 1:
            raise ContractError('degree_nonself: expected a nonnegative safe integer')
        mean, std = self.neighbor_similarity_mean, self.neighbor_similarity_std
        if (mean is None) != (std is None):
            raise ContractError('Similarity mean and population std must both be missing or supplied')
        if mean is not None:
            if self.degree_nonself == 0:
                raise ContractError('No neighbor similarities are available for an isolated node')
            mean = _number(mean, 'neighbor_similarity_mean', -1, 1)
            std = _number(std, 'neighbor_similarity_std', 0, 1)
            if self.degree_nonself == 1 and std != 0:
                raise ContractError('One neighbor has population std zero')
            object.__setattr__(self, 'neighbor_similarity_mean', mean)
            object.__setattr__(self, 'neighbor_similarity_std', std)


@dataclass(frozen=True)
class EvidenceCard:
    self_evidence: BranchEvidence
    graph_context: BranchEvidence
    current_graph: GraphSummary
    source: str = 'synthetic_fixture'

    def __post_init__(self):
        if type(self.self_evidence) is not BranchEvidence or type(self.graph_context) is not BranchEvidence:
            raise ContractError('Both validated branches are required')
        if type(self.current_graph) is not GraphSummary or type(self.source) is not str or self.source not in ADMITTED_SOURCES:
            raise ContractError('Unsupported numeric evidence source or graph summary')


def card_from_mapping(value):
    """Allowlist every level; identities, labels, text and attack metadata fail."""
    _keys(value, ('schema', 'source', 'self', 'graph_context', 'current_graph'), 'card')
    if value['schema'] != SCHEMA or type(value['source']) is not str or value['source'] not in ADMITTED_SOURCES:
        raise ContractError('Unsupported evidence schema/source')
    branches = []
    for name in ('self', 'graph_context'):
        _keys(value[name], ('p_fraud', 'calibration'), name)
        branches.append(BranchEvidence(**value[name]))
    _keys(value['current_graph'], ('degree_nonself', 'neighbor_similarity_mean', 'neighbor_similarity_std'), 'current_graph')
    return EvidenceCard(branches[0], branches[1], GraphSummary(**value['current_graph']), value['source'])


def public_payload(card):
    """Fresh deterministic copy. Local case IDs and graph hashes are never sent."""
    if type(card) is not EvidenceCard:
        raise ContractError('Expected a validated evidence card')
    return dict(schema=SCHEMA, source=card.source,
                self=dict(p_fraud=card.self_evidence.p_fraud, calibration=card.self_evidence.calibration),
                graph_context=dict(p_fraud=card.graph_context.p_fraud, calibration=card.graph_context.calibration),
                current_graph=dict(degree_nonself=card.current_graph.degree_nonself,
                                   neighbor_similarity_mean=card.current_graph.neighbor_similarity_mean,
                                   neighbor_similarity_std=card.current_graph.neighbor_similarity_std))


def fact_index(card):
    payload = public_payload(card)
    return MappingProxyType({section + '.' + key: value
                             for section in ('self', 'graph_context', 'current_graph')
                             for key, value in payload[section].items()})


@dataclass(frozen=True)
class FusionConfig:
    alpha_grid: tuple = (0.0, 0.5, 1.0)
    threshold: float = 0.5
    fallback_alpha_self: float = 0.5
    max_response_bytes: int = 8192
    require_description: bool = False

    def __post_init__(self):
        if type(self.alpha_grid) is not tuple or not 1 <= len(self.alpha_grid) <= 33:
            raise ContractError('alpha_grid must be a bounded immutable tuple')
        grid = tuple(_number(x, 'alpha_grid', 0, 1) for x in self.alpha_grid)
        if len(set(grid)) != len(grid) or tuple(sorted(grid)) != grid:
            raise ContractError('alpha_grid must be increasing without duplicates')
        object.__setattr__(self, 'alpha_grid', grid)
        object.__setattr__(self, 'threshold', _number(self.threshold, 'threshold', 0, 1))
        fallback = _number(self.fallback_alpha_self, 'fallback_alpha_self', 0, 1)
        if fallback not in grid:
            raise ContractError('Fallback must belong to alpha_grid')
        object.__setattr__(self, 'fallback_alpha_self', fallback)
        if type(self.max_response_bytes) is not int or not 1 <= self.max_response_bytes <= 1048576:
            raise ContractError('Invalid response byte limit')
        if type(self.require_description) is not bool:
            raise ContractError('require_description must be bool')


def _config_payload(config):
    if type(config) is not FusionConfig:
        raise ContractError('Expected a validated fusion configuration')
    return dict(alpha_grid=list(config.alpha_grid), threshold=config.threshold,
                fallback_alpha_self=config.fallback_alpha_self,
                max_response_bytes=config.max_response_bytes, require_description=config.require_description)


def strict_json(raw, config):
    """Reject duplicate keys at every depth, invalid constants and oversized input."""
    _config_payload(config)
    if type(raw) not in (str, bytes):
        raise ContractError('Response must be JSON text or UTF-8 bytes')
    try:
        encoded = raw.encode('utf-8') if type(raw) is str else raw
        if len(encoded) > config.max_response_bytes:
            raise ContractError('Response exceeds byte limit')
        text = encoded.decode('utf-8')
        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ContractError('Duplicate JSON key')
                result[key] = value
            return result
        def invalid_constant(value):
            raise ContractError('Nonfinite JSON constant')
        result = json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
        def check(value, depth=0):
            if depth > 16:
                raise ContractError('JSON nesting exceeds limit')
            if type(value) is float and not math.isfinite(value):
                raise ContractError('Nonfinite JSON number')
            if type(value) is dict:
                for item in value.values():
                    check(item, depth + 1)
            elif type(value) is list:
                for item in value:
                    check(item, depth + 1)
        check(result)
        return result
    except (UnicodeError, ValueError, RecursionError, OverflowError) as error:
        raise ContractError('Malformed or unsupported JSON') from error


@dataclass(frozen=True)
class DescriptionSection:
    summary: str
    refs: tuple
    echoes: tuple


@dataclass(frozen=True)
class EvidenceDescription:
    card_sha256: str
    sections: tuple


def _description_payload(description):
    return {name: dict(summary=section.summary, refs=list(section.refs),
                       echoes=[dict(fact_id=key, value=value) for key, value in section.echoes])
            for name, section in description.sections}


def validate_description(card, raw, config=FusionConfig()):
    """Check references/explicit echoes, not semantic truth or written-out numbers."""
    facts = fact_index(card)
    value = strict_json(raw, config)
    _keys(value, ('self', 'graph_context', 'conflict'), 'description')
    sections = []
    for name in ('self', 'graph_context', 'conflict'):
        section = value[name]
        _keys(section, ('summary', 'refs', 'echoes'), 'description.' + name)
        summary, refs, echoes = section['summary'], section['refs'], section['echoes']
        if type(summary) is not str or not summary.strip() or len(summary) > 500 or re.search(r'\d', summary):
            raise ContractError('Prose must be bounded and have no numeric literals; use checked echoes')
        if type(refs) is not list or not 1 <= len(refs) <= 16 or any(type(x) is not str for x in refs):
            raise ContractError('Invalid fact references')
        if len(set(refs)) != len(refs) or any(x not in facts for x in refs):
            raise ContractError('Unknown or duplicate fact reference')
        if name == 'self' and any(not x.startswith('self.') for x in refs):
            raise ContractError('Self description must cite self facts')
        if name == 'graph_context' and any(not x.startswith(('graph_context.', 'current_graph.')) for x in refs):
            raise ContractError('Graph description must cite graph facts')
        if name == 'conflict' and not {'self.p_fraud', 'graph_context.p_fraud'}.issubset(refs):
            raise ContractError('Conflict description must cite both branch probabilities')
        if type(echoes) is not list or len(echoes) > 16:
            raise ContractError('Invalid fact echoes')
        checked, seen = [], set()
        for echo in echoes:
            _keys(echo, ('fact_id', 'value'), 'echo')
            key, number = echo['fact_id'], echo['value']
            if type(key) is not str or key not in refs or key in seen:
                raise ContractError('Echo must cite a unique referenced fact')
            expected = facts[key]
            if type(expected) in (int, float):
                number = _number(number, 'echo value')
                if number != expected:
                    raise ContractError('Numeric echo differs from evidence')
            elif type(number) is not type(expected) or number != expected:
                raise ContractError('Echo differs from enum or missingness')
            checked.append((key, number))
            seen.add(key)
        sections.append((name, DescriptionSection(summary, tuple(refs), tuple(checked))))
    digest = hashlib.sha256(_canonical(public_payload(card)).encode()).hexdigest()
    return EvidenceDescription(digest, tuple(sections))


def _checked_description(card, description, config):
    if type(description) is EvidenceDescription:
        # Revalidation also prevents a manually constructed object from bypassing checks.
        try:
            checked = validate_description(card, _canonical(_description_payload(description)), config)
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError('Invalid immutable description') from error
        if checked.card_sha256 != description.card_sha256:
            raise ContractError('Description belongs to another evidence card')
        return checked
    return validate_description(card, description, config)


def validate_decision(raw, config=FusionConfig()):
    value = strict_json(raw, config)
    _keys(value, ('alpha_self',), 'decision')
    alpha = _number(value['alpha_self'], 'alpha_self', 0, 1)
    if alpha not in config.alpha_grid:
        raise ContractError('alpha_self is outside the declared grid')
    return alpha


@dataclass(frozen=True)
class DecisionResult:
    p_fraud: float
    predicted_fraud: bool
    alpha_self: float
    used_fallback: bool
    status: str
    description: object = None


def resolve_decision(card, raw_response, config=FusionConfig(), *, description=None, timed_out=False):
    """One result per valid card, including timeout/missing/invalid responses.

    Invalid INPUTS raise before response handling. No API attempt occurs here.
    Equal-to-threshold means fraud. The fallback is the fixed configured mixture.
    """
    public_payload(card)
    _config_payload(config)
    if type(timed_out) is not bool:
        raise ContractError('timed_out must be bool')
    alpha, status, checked = config.fallback_alpha_self, 'valid', None
    if timed_out:
        status = 'fallback_timeout'
    elif config.require_description and description is None:
        status = 'fallback_missing_description'
    else:
        if description is not None:
            try:
                checked = _checked_description(card, description, config)
            except ContractError:
                status = 'fallback_invalid_description'
        if status == 'valid':
            if raw_response is None:
                status = 'fallback_missing_response'
            else:
                try:
                    alpha = validate_decision(raw_response, config)
                except ContractError:
                    status = 'fallback_invalid_response'
    probability = alpha * card.self_evidence.p_fraud + (1 - alpha) * card.graph_context.p_fraud
    return DecisionResult(probability, probability >= config.threshold, alpha, status != 'valid', status, checked)


@dataclass(frozen=True)
class LocalCase:
    case_id: str
    context_sha256: str
    card: EvidenceCard

    def __post_init__(self):
        if type(self.case_id) is not str or not self.case_id or len(self.case_id) > 128:
            raise ContractError('Invalid local case ID')
        _digest(self.context_sha256, 'context_sha256')
        public_payload(self.card)


def resolve_batch(cases, responses, config=FusionConfig(), *, descriptions=None, timed_out=None):
    """Validate the WHOLE input batch first; return every valid case in order."""
    _config_payload(config)
    if type(cases) is not tuple or type(responses) is not tuple or len(cases) != len(responses):
        raise ContractError('Cases/responses must be equally sized tuples')
    if any(type(case) is not LocalCase for case in cases) or len({case.case_id for case in cases}) != len(cases):
        raise ContractError('Cases must be validated and have unique local IDs')
    for case in cases:
        public_payload(case.card)
        _digest(case.context_sha256, 'context_sha256')
    descriptions = (None,) * len(cases) if descriptions is None else descriptions
    timed_out = (False,) * len(cases) if timed_out is None else timed_out
    if type(descriptions) is not tuple or type(timed_out) is not tuple or len(descriptions) != len(cases) or len(timed_out) != len(cases):
        raise ContractError('Description/timeout tuples must preserve batch length')
    if any(type(flag) is not bool for flag in timed_out):
        raise ContractError('Timeout flags must be bool')
    return tuple(resolve_decision(case.card, response, config, description=description, timed_out=timeout)
                 for case, response, description, timeout in zip(cases, responses, descriptions, timed_out))


def cache_key(card, context_sha256, *, model_id, model_version, prompt_sha256,
              config=FusionConfig(), description=None, description_prompt_sha256=ZERO_DIGEST,
              examples_sha256=ZERO_DIGEST, settings=None, stage='decision'):
    """Hash all declared inputs. Context hashes stay local, never in public_payload.

    The caller must digest the full graph/case context, not just a node ID. This
    interface cannot verify that a caller's opaque digest covers those inputs.
    Settings must describe the actual call; no provider capabilities are assumed.
    """
    payload = public_payload(card)
    _config_payload(config)
    _digest(context_sha256, 'context_sha256')
    for name, value in (('prompt', prompt_sha256), ('description_prompt', description_prompt_sha256), ('examples', examples_sha256)):
        _digest(value, name)
    for value in (model_id, model_version):
        if type(value) is not str or not value.strip() or len(value) > 200:
            raise ContractError('Explicit model identity/version required')
    if stage not in ('description', 'decision'):
        raise ContractError('Unknown cache stage')
    if config.require_description and stage == 'decision' and description is None:
        raise ContractError('Decision cache requires the actual description input')
    settings = {} if settings is None else settings
    if type(settings) is not dict or not set(settings).issubset({'temperature', 'top_p', 'max_output_tokens', 'seed'}):
        raise ContractError('Unknown LLM setting')
    normalized = {}
    for key, value in settings.items():
        if key in ('temperature', 'top_p'):
            normalized[key] = _number(value, key, 0, 2 if key == 'temperature' else 1)
        elif type(value) is not int or not 0 <= value <= 2 ** 53 - 1 or (key == 'max_output_tokens' and value == 0):
            raise ContractError('Invalid integer LLM setting')
        else:
            normalized[key] = value
    desc = None if description is None else _description_payload(_checked_description(card, description, config))
    key = dict(schema=SCHEMA, stage=stage, card=payload, context_sha256=context_sha256,
               model_id=model_id, model_version=model_version, prompt_sha256=prompt_sha256,
               description_prompt_sha256=description_prompt_sha256, examples_sha256=examples_sha256,
               llm_settings=normalized, fusion_config=_config_payload(config), description=desc)
    return hashlib.sha256(_canonical(key).encode()).hexdigest()


cachekey = cache_key
