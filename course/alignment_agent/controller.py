"""TRAIN-only outer-epoch controller; never exposes credentials to the server."""
import hashlib
import json
import pathlib
import re
import time

from course.alignment_agent.api_bridge import (
    SCHEMA, ContractError, canonical, fallback_rule, resolve_decision, state_sha256,
    strict_json, validate_request, validate_state)
from course.conditional_alignment.common import sha256, write_json


def _atomic(path,value):
    path=pathlib.Path(path)
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(canonical(value)+'\n')
    if path.exists():
        raise RuntimeError('A controller request already exists; do not duplicate it.')
    temporary.rename(path)


def _read_schedule(path,seed,protocol):
    row=json.loads(pathlib.Path(path).read_text())
    actions=row['actions']
    if (row['seed']!=seed or row['protocol_sha256']!=protocol['_protocol_sha256'] or
            len(actions)!=8 or any(type(v) is not bool for v in actions) or actions[0]):
        raise RuntimeError('Invalid matched-random applied-action schedule.')
    if sum(actions)!=row['source_enabled_epochs']:
        raise RuntimeError('Matched schedule does not preserve the agent ON count.')
    return actions


def build_policy(arm,seed,protocol,evidence,matched_schedule_path=None):
    evidence=pathlib.Path(evidence)
    stage=pathlib.Path(protocol['stage_root'])
    requests=stage/'requests';responses=stage/'responses'
    requests.mkdir(parents=True,exist_ok=True);responses.mkdir(parents=True,exist_ok=True)
    schedule=None
    if arm=='smooth_random':
        if matched_schedule_path is None:
            raise RuntimeError('Random control needs the same-seed frozen derivation.')
        schedule=_read_schedule(matched_schedule_path,seed,protocol)
    def policy(epoch,previous_train_state):
        if not 1<=epoch<=8:
            raise RuntimeError('Invalid outer-epoch decision slot.')
        if epoch==1:
            return dict(align=False,fallback=False,reason='Common first-epoch no-alignment warm-up.',decision_source='registered_warmup')
        validate_state(previous_train_state)
        if previous_train_state['completed_epoch']!=epoch-1:
            raise RuntimeError('Only the preceding completed TRAIN epoch can select an action.')
        if arm in ('legacy_hinge_off','smooth_off'):
            return dict(align=False,fallback=False,reason='Registered no-alignment control.',decision_source='fixed')
        if arm in ('smooth_threshold','smooth_rule'):
            return dict(align=True,fallback=False,reason='Registered ON schedule after common warm-up.',decision_source='fixed')
        if arm=='smooth_random':
            return dict(align=schedule[epoch-1],fallback=False,reason='Outcome-independent permutation of same-seed agent applied actions.',decision_source='matched_random')
        if arm!='smooth_agent':
            raise RuntimeError('Unknown controller arm.')
        state_hash=state_sha256(previous_train_state)
        identity=hashlib.sha256(('%s|%s|%s|%s'%(protocol['_protocol_sha256'],seed,epoch,state_hash)).encode()).hexdigest()
        request_id='s%d_e%02d_%s'%(seed,epoch,identity[:32])
        envelope=dict(schema=SCHEMA,request_id=request_id,protocol_sha256=protocol['_protocol_sha256'],
                      seed=int(seed),epoch=int(epoch),state=previous_train_state)
        validate_request(envelope,protocol['_protocol_sha256'])
        _atomic(requests/(request_id+'.json'),envelope)
        response=responses/(request_id+'.json')
        start=time.monotonic();deadline=start+20.0
        result=None
        while time.monotonic()<deadline:
            if response.exists():
                try:
                    candidate=strict_json(response.read_text(encoding='utf-8'))
                except (OSError,ValueError,UnicodeError,ContractError):
                    break
                required={'request_id','protocol_sha256','state_sha256','align','reason','fallback','response_sha256'}
                if (type(candidate) is dict and set(candidate)==required and candidate['request_id']==request_id and
                    candidate['protocol_sha256']==protocol['_protocol_sha256'] and candidate['state_sha256']==state_hash and
                    type(candidate['align']) is bool and type(candidate['fallback']) is bool and
                    type(candidate['reason']) is str and 0<len(candidate['reason'])<=1024 and
                    type(candidate['response_sha256']) is str and
                    re.fullmatch(r'[0-9a-f]{64}',candidate['response_sha256'])):
                    result=candidate;break
                # A malformed response cannot stall or remove this epoch.
                break
            time.sleep(.2)
        if result is None:
            result=resolve_decision(envelope,None,protocol['_protocol_sha256'])
            result.update(reason='No valid response before the registered twenty-second deadline; TRAIN fallback applied.',timeout_or_invalid_response=True)
        result.update(decision_source='external_agent' if not result['fallback'] else 'registered_fallback',
                      wait_seconds=time.monotonic()-start,epoch=epoch)
        write_json(evidence/('applied_controller_epoch_%02d.json'%epoch),result,exclusive=True)
        return result
    return policy
