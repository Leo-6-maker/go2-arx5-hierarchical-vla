"""A narrow FUSION supplement in the existing registry; no new scheduler/owner."""
from pathlib import Path
from contextlib import closing
import json, os, time, subprocess
from .fusion_contract import PLAN
from .storage import digest

SCOPE = 'WM02_FUSION_REFERENCE_SIM_ONLY'


def binding_contract(binding, now):
    expected = dict(schema='WM02_FUSION_NIGHT1_SCOPE_V1', plan_sha=PLAN, scope=SCOPE,
        owner='EXISTING_LOCAL_PC', host='corelab-4080s', uid=2009, max_native_slots=2,
        GR00T_updates=0, WBC_changes=False, physics_dt_s=.002,
        native_reference_domain='SINGLE_DECISION_LOCAL_XY4_RAMP8_H16_OPEN_GRIPPER')
    if (any(binding.get(k) != v for k, v in expected.items())
            or not binding['activation_epoch'] <= now < binding['deadline_epoch']
            or binding['deadline_epoch'] > binding['parent_deadline_epoch']-1200):
        raise PermissionError('FINITE_FUSION_SCOPE_OR_FROZEN_DOMAIN_INVALID')


def validate(spec, registry=None, now=None):
    from .pilot1_scope import validate as validate_parent
    now = time.time() if now is None else now
    validate_parent(spec, registry, now)
    path = Path(spec['fusion_binding']); binding = json.loads(path.read_text())
    binding_contract(binding, now)
    lease = json.loads((path.parent/'LEASE.json').read_text())
    registry = json.loads(Path(spec['registry']).read_text()) if registry is None else registry
    if (digest(path) != spec['fusion_binding_sha256']
            or binding['campaign_root'] != spec['campaign_root']
            or binding['pilot1_binding'] != spec['pilot1_binding']
            or binding['pilot1_binding_sha256'] != spec['pilot1_binding_sha256']
            or lease.get('status') != 'ACTIVE' or lease.get('binding_sha256') != digest(path)
            or lease.get('owner') != spec['owner'] or lease.get('scope') != SCOPE
            or spec['deadline_epoch'] > binding['deadline_epoch']
            or (path.parent/'STOP.json').exists()):
        raise PermissionError('ACTUAL_FUSION_SUPPLEMENT_LEASE_REQUIRED')
    if not any(e.get('path') == str(path.parent.parent) and e.get('scope') == SCOPE
               and e.get('owner') == spec['owner'] and e.get('local_lease') == str(path.parent/'LEASE.json')
               for e in registry['entries']):
        raise PermissionError('FUSION_SUPPLEMENT_NOT_REGISTERED')
    if not any(e.get('path') == spec['registered_root'] and e.get('source_sha') == spec['source_sha']
               and e.get('fusion_binding') == str(path) and e.get('scope') == SCOPE
               for e in registry['entries']):
        raise PermissionError('FRESH_FUSION_WORKER_NOT_REGISTERED')
    return binding


def register(path):
    """Register once under the same canonical locks, only after the sealed batch.

    Caller supplies actual immutable binding bytes. Missing or active predecessor
    is a wait, never a reason to interrupt it or mutate its capsule.
    """
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.night10.native_relay import snapshot, once
    path = Path(path); binding = json.loads(path.read_text()); binding_contract(binding, time.time())
    campaign = Path(binding['campaign_root']); window = path.parent.parent
    if os.uname().nodename != binding['host'] or os.getuid() != binding['uid']:
        raise PermissionError('ACTUAL_EXISTING_FUSION_HOST_UID_REQUIRED')
    with closing(OwnerLock(campaign/'OWNER.lock')), closing(OwnerLock(binding['physical_owner_lock'])):
        if (path.parent/'LEASE.json').exists(): raise FileExistsError('RECONCILE_EXISTING_FUSION_SCOPE')
        parent = json.loads((campaign/'control/LEASE.json').read_text())
        p1 = json.loads((Path(binding['pilot1_binding']).parent/'LEASE.json').read_text())
        prior = Path(binding['predecessor_result']); result = json.loads(prior.read_text())
        previous_lease = json.loads((prior.parent/'control/LEASE.json').read_text())
        process = subprocess.check_output(['systemctl', '--user', 'show', binding['predecessor_unit'],
                                           '-p', 'MainPID'], text=True).strip()
        if (parent.get('status') != 'ACTIVE' or parent.get('owner') != binding['owner']
                or parent['deadline_epoch'] != binding['parent_deadline_epoch']
                or p1.get('status') != 'ACTIVE'
                or digest(binding['pilot1_binding']) != binding['pilot1_binding_sha256']
                or p1.get('binding_sha256') != binding['pilot1_binding_sha256']
                or digest(prior) != binding['predecessor_result_sha256']
                or previous_lease.get('status') != 'CLOSED'
                or process != 'MainPID=0' or result.get('exit_code') != 0):
            raise PermissionError('CURRENT_OWNER_AND_ACTUALLY_SEALED_PREDECESSOR_REQUIRED')
        reg, before = snapshot(binding['registry'])
        reg['entries'].append(dict(role='WM02_FUSION_NIGHT1_SUPPLEMENT', path=str(window),
            run_uuid=window.name, owner=binding['owner'], executor_node=binding['host'], scope=SCOPE,
            source_sha=binding['source_sha'], plan_sha=PLAN, local_lease=str(path.parent/'LEASE.json')))
        once(Path(binding['registry']), reg, window.name+'-REGISTER', expected_before_sha256=before)
        value = dict(status='ACTIVE', scope=SCOPE, owner=binding['owner'], run_uuid=window.name,
            binding_sha256=digest(path), deadline_epoch=binding['deadline_epoch'],
            new_campaign=False, predecessor_closed_lease_reopened=False)
        write(path.parent/'LEASE.json', value)
        return dict(status='FUSION_SUPPLEMENT_REGISTERED_NOT_NATIVE_QUALIFIED',
                    binding_sha256=digest(path), native_integrations=0, VLA_inferences=0)
