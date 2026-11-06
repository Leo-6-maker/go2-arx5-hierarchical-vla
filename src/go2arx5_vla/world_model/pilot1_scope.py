"""PILOT1 admission through the existing owner, registry, Journal and worker locks.

No accumulated engineering caps; each worker retains a finite, declared endpoint.
Historical scopes and their restrictions are not modified.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import time

from .storage import digest

PLAN_SHA = 'fec841e74d406b3b556c84b733d2de1615581bf7'
C08 = '29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2'
SCOPE = 'WM02_PILOT1_SIM_ONLY'
GPU_UUIDS = {'GPU-845348f1-1511-dc19-0877-cc97658e3edd', 'GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'}


def binding_contract(value, now):
    expected = dict(schema='WM02_PILOT1_SCOPE_V1', plan_sha=PLAN_SHA, scope=SCOPE,
                    owner='EXISTING_LOCAL_PC', host='corelab-4080s', uid=2009,
                    engineering_budget_caps=None, max_native_slots=2, C08_native_binary_sha256=C08)
    if (any(value.get(k) != v for k, v in expected.items())
            or not value['activation_epoch'] <= now < value['deadline_epoch']
            or value['deadline_epoch'] > value['parent_campaign_deadline_epoch']):
        raise PermissionError('LIVE_PILOT1_OWNER_SITE_AND_FINITE_ENDPOINT_REQUIRED')


def validate(spec, registry=None, now=None):
    now = time.time() if now is None else now
    path = Path(spec['pilot1_binding'])
    binding = json.loads(path.read_text())
    binding_contract(binding, now)
    lease = json.loads((path.parent / 'LEASE.json').read_text())
    parent = json.loads((Path(spec['campaign_root']) / 'control/LEASE.json').read_text())
    if (digest(path) != spec['pilot1_binding_sha256'] or binding['campaign_root'] != spec['campaign_root']
            or lease.get('status') != 'ACTIVE' or lease.get('binding_sha256') != digest(path)
            or lease.get('owner') != binding['owner'] or lease.get('scope') != SCOPE
            or parent.get('status') != 'ACTIVE' or parent.get('owner') != binding['owner']
            or parent.get('scope') != 'WM02_AUTO1_SIM_ONLY' or now >= parent['deadline_epoch']
            or spec['deadline_epoch'] > min(binding['deadline_epoch'], parent['deadline_epoch'])
            or any(spec.get(k) for k in ('next2_binding', 'next1_binding', 'night1_binding'))
            or (path.parent / 'STOP.json').exists() or (Path(spec['campaign_root']) / 'STOP.json').exists()):
        raise PermissionError('FRESH_PILOT1_LEASE_REQUIRED_NO_HISTORICAL_SCOPE_REUSE')
    registry = json.loads(Path(spec['registry']).read_text()) if registry is None else registry
    if not any(e.get('scope') == SCOPE and e.get('path') == str(path.parent.parent)
               and e.get('owner') == binding['owner'] and e.get('local_lease') == str(path.parent / 'LEASE.json')
               for e in registry['entries']):
        raise PermissionError('PILOT1_SCOPE_NOT_IN_ACTUAL_REGISTRY')
    if not any(e.get('path') == spec['registered_root'] and e.get('source_sha') == spec['source_sha']
               and e.get('pilot1_binding') == str(path) for e in registry['entries']):
        raise PermissionError('PILOT1_WORKER_NOT_IN_ACTUAL_REGISTRY')
    return binding


def validate_gpu(spec, actual):
    if (spec.get('profile') != 'wm02-pilot1-neural' or spec.get('physics') is not False
            or spec.get('GPU_uuid') not in GPU_UUIDS or actual['uuid'] != spec['GPU_uuid']
            or spec.get('kind') not in ('ACTOR_DISTILLATION', 'ACTOR_STUDY', 'ARX_RESIDUAL', 'ARX_EQUIVALENCE','CONTACT_WM','CONTACT_POLICY')
            or type(spec.get('endpoint')) is not int or spec['endpoint'] < 1
            or not 0 < spec.get('allocator_limit_bytes', 0) < actual['free_bytes'] - 1024**3
            or os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']):
        raise PermissionError('PILOT1_FINITE_SINGLE_LOCAL_GPU_AND_REAL_HEADROOM_REQUIRED')
    validate(spec)


def register(path):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.night10.native_relay import snapshot, once
    path = Path(path)
    binding = json.loads(path.read_text())
    binding_contract(binding, time.time())
    campaign, window = Path(binding['campaign_root']), path.parent.parent
    if os.uname().nodename != binding['host'] or os.getuid() != binding['uid']:
        raise PermissionError('PILOT1_ACTUAL_HOST_UID_REQUIRED')
    with closing(OwnerLock(campaign / 'OWNER.lock')), closing(OwnerLock(binding['physical_owner_lock'])):
        if (path.parent / 'LEASE.json').exists():
            raise FileExistsError('RECONCILE_EXISTING_PILOT1_REGISTRATION')
        parent = json.loads((campaign / 'control/LEASE.json').read_text())
        if (parent.get('status') != 'ACTIVE' or parent.get('owner') != binding['owner']
                or parent.get('scope') != 'WM02_AUTO1_SIM_ONLY'
                or parent['deadline_epoch'] != binding['parent_campaign_deadline_epoch']):
            raise PermissionError('PILOT1_ORIGINAL_OWNER_LEASE_NOT_CURRENT')
        registry, before = snapshot(binding['registry'])
        for entry in registry['entries']:
            if entry.get('scope') in ('WM02_NEXT1_SIM_ONLY', 'WM02_NEXT2_SIM_ONLY', SCOPE):
                lease = Path(entry['local_lease'])
                if not lease.exists() or json.loads(lease.read_text()).get('status') != 'CLOSED':
                    raise PermissionError('EXISTING_RESEARCH_OWNER_NOT_TERMINAL')
        profiles = json.loads(Path(binding['environments']).read_text())['profiles']
        if not any(p['name'] == '4080-simulation' and binding['host'] in p['hostnames']
                   and p['python'] == binding['runtime_python'] for p in profiles):
            raise PermissionError('PILOT1_EXISTING_SITE_PROFILE_REQUIRED')
        for name, pin in binding['evidence_files'].items():
            if digest(name) != pin:
                raise PermissionError('PILOT1_PINNED_INPUT_CHANGED:' + name)
        registry['entries'].append(dict(role='WM02_PILOT1_WINDOW', path=str(window), run_uuid=window.name,
                                       executor_node=binding['host'], owner=binding['owner'], scope=SCOPE,
                                       source_sha=binding['source_sha'], plan_sha=PLAN_SHA,
                                       local_lease=str(path.parent / 'LEASE.json')))
        once(Path(binding['registry']), registry, window.name + '-REGISTER', expected_before_sha256=before)
        write(path.parent / 'LEASE.json', dict(status='ACTIVE', owner=binding['owner'], scope=SCOPE,
              run_uuid=window.name, binding_sha256=digest(path), activated_epoch=binding['activation_epoch'],
              deadline_epoch=binding['deadline_epoch']))
        state, before = snapshot(campaign / 'AUTO1_STATE.json')
        if state['owner'] != binding['owner']:
            raise PermissionError('CANONICAL_OWNER_CHANGED')
        state.update(current_phase='PILOT1_REGISTERED', updated_epoch=time.time())
        state['pilot1'] = dict(status='REGISTERED_NOT_NATIVE_EVALUATED', binding=str(path), plan_sha=PLAN_SHA,
                               Actor_updates=0, WM_updates=0, next='A1 frozen ARX and A2 real local Actor')
        once(campaign / 'AUTO1_STATE.json', state, window.name + '-INDEX', expected_before_sha256=before)
        result = dict(status='PILOT1_REGISTERED', binding_sha256=digest(path), old_closed_leases_reopened=False,
                      engineering_budget_caps=None, new_integrations=0, Actor_updates=0)
        write(window / 'REGISTERED.private.json', result)
        return result
