"""Standard-library provenance checks for the isolated dual-surrogate pilot.

These helpers never import the project/ML libraries, deserialize tensors, load
models, inspect predictions or call a network service. A receipt's declarations
are checked for consistency; they do not independently prove its author's
information-boundary statements or scientific correctness.
"""
import hashlib
import json
import pathlib


ORIGINAL_GCN_SHA256 = 'cbd561057e37d2a90b4797b16b8c0fff1229cf47145c4dfb860fc12b8bd8fe5e'
SEALED_GAGA_SHA256 = '729b8624baac7429bd60e7bc2d3741a9d9ee7c4c7df921d618b53ceb0435c383'
TEACHER_FAMILIES = {'gcn_base': 'GCN', 'gcn_new': 'GCN', 'gaga_new': 'GAGA'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_string(value, field):
    _require(type(value) is str and len(value) == 64 and
             all(char in '0123456789abcdef' for char in value),
             'Invalid SHA256: ' + field)
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read_json(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=_unique_pairs,
                         parse_constant=_invalid_constant)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _file_under(root, relative):
    root = pathlib.Path(root).resolve()
    _require(type(relative) is str and relative, 'An explicit relative path is required.')
    pure = pathlib.PurePosixPath(relative.replace('\\', '/'))
    _require(not pure.is_absolute() and not pathlib.PureWindowsPath(relative).is_absolute()
             and '..' not in pure.parts, 'Only repository-relative artifact paths are allowed.')
    path = root.joinpath(*pure.parts).resolve()
    _require(root in path.parents and path.is_file(), 'Missing/outside-root artifact: ' + relative)
    return path


def verify_manifest(root, entries):
    """Rehash available bytes; missing artifacts fail instead of becoming passes.

    ``entries`` is a list of {path, sha256, optional bytes}. The returned rows
    retain only artifact names, lengths and digests, never tensor contents.
    """
    _require(type(entries) is list and entries, 'A nonempty explicit manifest is required.')
    seen, verified = set(), []
    for row in entries:
        _require(type(row) is dict and {'path', 'sha256'} <= set(row), 'Malformed manifest row.')
        relative = row['path']
        _require(relative not in seen, 'Duplicate artifact in manifest: ' + str(relative))
        seen.add(relative)
        wanted = _sha_string(row['sha256'], relative)
        path = _file_under(root, relative)
        size = path.stat().st_size
        if 'bytes' in row:
            _require(type(row['bytes']) is int and row['bytes'] == size, 'Artifact byte length differs: ' + relative)
        actual = sha256(path)
        _require(actual == wanted, 'Artifact byte hash differs: ' + relative)
        verified.append(dict(path=relative, bytes=size, sha256=actual))
    return verified


def _node_ids(values, field):
    _require(isinstance(values, (list, tuple)) and values, 'Nonempty node list required: ' + field)
    _require(all(type(value) is int and value >= 0 for value in values), 'Invalid node identity: ' + field)
    _require(len(values) == len(set(values)), 'Duplicate node identities: ' + field)
    return set(values)


def node_ids_sha256(values):
    """Identity-set digest; ordering does not redefine teacher membership."""
    return canonical_sha256(sorted(_node_ids(values, 'node IDs')))


def verify_teacher_partition(original_train, fit, tune, *, expected_fit_count=None):
    """Check a disjoint exact cover of original TRAIN, without reading labels.

    The caller freezes how the 80/20 partition is formed. A stratified split
    may round differently from a single global floor, so any exact fit count
    is supplied explicitly rather than silently inferred here.
    """
    train_set = _node_ids(original_train, 'original TRAIN')
    fit_set, tune_set = _node_ids(fit, 'internal FIT'), _node_ids(tune, 'internal TUNE')
    _require(not fit_set & tune_set, 'Internal FIT/TUNE overlap.')
    _require(fit_set | tune_set == train_set, 'Teacher partition is not an exact original TRAIN cover.')
    if expected_fit_count is not None:
        _require(type(expected_fit_count) is int and len(fit_set) == expected_fit_count,
                 'Teacher FIT count differs from its frozen rule.')
    return dict(original_train_nodes=len(train_set), fit_nodes=len(fit_set), tune_nodes=len(tune_set),
                observed_fit_fraction=len(fit_set)/len(train_set),
                original_train_ids_sha256=node_ids_sha256(original_train),
                fit_ids_sha256=node_ids_sha256(fit), tune_ids_sha256=node_ids_sha256(tune),
                disjoint_exact_cover=True, labels_read=False)


def verify_gaga_label_masks(fit, tune, benign_label_ids, fraud_label_ids):
    """Prove membership boundaries of supplied masks, not correctness of labels.

    Internal TUNE nodes must be masked even though they belong to the broader
    original TRAIN partition. Only FIT labels may enter GAGAnew group means.
    """
    fit_set, tune_set = _node_ids(fit, 'internal FIT'), _node_ids(tune, 'internal TUNE')
    def mask(values, field):
        _require(isinstance(values, (list, tuple)) and
                 all(type(value) is int and value >= 0 for value in values), 'Invalid mask IDs: ' + field)
        _require(len(values) == len(set(values)), 'Duplicate mask IDs: ' + field)
        return set(values)
    negative = mask(benign_label_ids, 'benign')
    positive = mask(fraud_label_ids, 'fraud')
    _require(not fit_set & tune_set and not negative & positive, 'GAGA mask/partition overlap.')
    _require(negative | positive == fit_set, 'GAGA group labels must cover only internal FIT.')
    _require(not (negative | positive) & tune_set, 'GAGA internal TUNE labels leaked into aggregation.')
    return dict(fit_nodes=len(fit_set), masked_internal_tune_nodes=len(tune_set),
                fit_ids_sha256=node_ids_sha256(fit), tune_ids_sha256=node_ids_sha256(tune),
                label_scope='internal_fit_only', label_truth_independently_checked=False)


def verify_teacher_training_receipt(receipt, partition, family):
    """Validate an explicitly normalized receipt, with no hidden defaults.

    Required receipt fields are deliberately independent of an engine's output
    schema; its caller supplies this small normalized view. This is a consistency
    check on saved declarations, to accompany source and execution auditing.
    """
    _require(family in ('GCN', 'GAGA') and type(receipt) is dict, 'Invalid teacher family/receipt.')
    for field in ('original_valid_used', 'test_used', 'sealed_victim_loaded'):
        _require(receipt.get(field) is False, 'Teacher information boundary not explicitly false: ' + field)
    _require(receipt.get('checkpoint_selection') == 'internal_tune_only', 'Teacher selection must use only internal TUNE.')
    for field in ('fit_ids_sha256', 'tune_ids_sha256'):
        _require(_sha_string(receipt.get(field), field) == partition[field], 'Teacher partition receipt differs: ' + field)
    _require(type(receipt.get('optimizer_steps')) is int and receipt['optimizer_steps'] > 0,
             'A new teacher requires positive recorded optimizer steps.')
    _require(type(receipt.get('seed')) is int, 'Teacher seed must be explicit.')
    _sha_string(receipt.get('initial_state_sha256'), 'teacher initial state')
    _sha_string(receipt.get('final_state_sha256'), 'teacher final state')
    _require(receipt['initial_state_sha256'] != receipt['final_state_sha256'], 'Recorded new teacher state never changed.')
    if family == 'GAGA':
        _require(receipt.get('aggregation_label_scope') == 'internal_fit_only' and
                 receipt.get('aggregation_label_ids_sha256') == partition['fit_ids_sha256'],
                 'GAGAnew aggregation must retain the internal-FIT-only label convention.')
    return dict(family=family, seed=receipt['seed'], optimizer_steps=receipt['optimizer_steps'],
                fit_ids_sha256=partition['fit_ids_sha256'], tune_ids_sha256=partition['tune_ids_sha256'],
                declared_boundary_consistent=True,
                limitation='Receipt consistency and changed hashes do not independently prove training or information exclusion.')


def verify_teacher_roles(root, records, sealed_victim, partition):
    """Check three named checkpoint roles and the sealed victim's byte identity.

    ``records`` keys: gcn_base, gcn_new, gaga_new. Every row has path, sha256,
    family; new rows additionally have a normalized training_receipt as above.
    Victim bytes are hashed only. Different byte/content hashes are necessary
    bookkeeping checks, not proof of statistical independence or provenance.
    """
    _require(type(records) is dict and set(records) == set(TEACHER_FAMILIES), 'Three explicit teacher roles are required.')
    victim = verify_manifest(root, [sealed_victim])[0]
    _require(victim['sha256'] == SEALED_GAGA_SHA256, 'Original sealed GAGA identity changed.')
    result = {}
    for role, family in TEACHER_FAMILIES.items():
        row = records[role]
        _require(row.get('family') == family, 'Teacher family mismatch: ' + role)
        checkpoint = verify_manifest(root, [dict(path=row['path'], sha256=row['sha256'])])[0]
        _require(checkpoint['sha256'] != victim['sha256'], 'A teacher is the sealed GAGA victim: ' + role)
        if role == 'gcn_base':
            _require(checkpoint['sha256'] == ORIGINAL_GCN_SHA256, 'Original GCN teacher identity changed.')
        else:
            _require(checkpoint['sha256'] != ORIGINAL_GCN_SHA256, 'New teacher reuses original GCN bytes.')
            checkpoint['training'] = verify_teacher_training_receipt(row['training_receipt'], partition, family)
        result[role] = checkpoint
    _require(records['gcn_new']['path'] != records['gcn_base']['path'], 'New and original GCN paths alias.')
    return dict(teachers=result, sealed_victim=victim, victim_weights_deserialized=False,
                limitation='Byte identity checks cannot establish statistical independence of trained teachers.')


def verify_shared_settings(arm_records, keys):
    """Check explicitly listed nuisance settings across all supplied arm records."""
    _require(type(arm_records) is dict and len(arm_records) >= 2 and keys, 'Explicit arms/settings required.')
    arms = sorted(arm_records)
    first = arm_records[arms[0]]
    selected = {key: first[key] for key in keys}
    for arm in arms[1:]:
        _require(all(key in arm_records[arm] and arm_records[arm][key] == value
                     for key, value in selected.items()), 'Shared generator/input setting differs: ' + arm)
    return dict(arms=arms, checked_keys=list(keys), shared_settings_sha256=canonical_sha256(selected))


def verify_initial_pairing(rows, expected_arms, seeds=(3, 4, 5)):
    """Compare captured initial attacker content hashes, without loading tensors."""
    _require(type(rows) is list and rows and len(set(expected_arms)) == len(expected_arms), 'Explicit unique pairing layout required.')
    seen, by_seed = set(), {seed: set() for seed in seeds}
    for row in rows:
        identity = (row['arm'], row['seed'])
        _require(identity not in seen and row['arm'] in expected_arms and row['seed'] in by_seed,
                 'Duplicate/unregistered initial pairing row.')
        seen.add(identity)
        _require(row.get('captured_before_training') is True and row.get('optimizer_steps') == 0,
                 'Initial attacker state was not captured before training.')
        by_seed[row['seed']].add(_sha_string(row['initial_state_sha256'], 'attacker initial state'))
    _require(seen == {(arm, seed) for arm in expected_arms for seed in seeds}, 'Initial pairing matrix incomplete.')
    _require(all(len(values) == 1 for values in by_seed.values()), 'Initial attacker weights differ within a paired seed.')
    return dict(arms=list(expected_arms), seeds=list(seeds), complete=True,
                initial_state_sha256_by_seed={str(seed): next(iter(values)) for seed, values in by_seed.items()},
                tensors_independently_rehashed=False)
