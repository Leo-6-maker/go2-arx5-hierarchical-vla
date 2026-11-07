"""Renew an explicitly authorized finite window using the original owner/Journal.

Metadata only: close expired supplement windows, renew an ACTIVE campaign's
deadline, then call the existing PILOT1 and FUSION registration functions.
Never reopens a CLOSED lease or starts a worker.
"""
import argparse
from contextlib import closing, ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def need(ok, reason):
    if not ok:
        raise PermissionError('FUSION_RENEWAL_' + reason)


def request_contract(request, parent, pilot, fusion, *, now):
    need(request.get('schema') == 'WM02_FUSION_EXPLICIT_FINITE_RENEWAL_V1'
         and request.get('owner') == 'EXISTING_LOCAL_PC'
         and request.get('host') == 'corelab-4080s' and request.get('uid') == 2009
         and request.get('user_authorization')
         and request.get('WBC_changes') is False
         and request.get('physics_dt_s') == .002
         and request.get('GR00T_updates') == request.get('Actor_updates') == 0,
         'EXPLICIT_USER_AND_FROZEN_SIM_ONLY_REQUEST')
    end = request['user_authorized_deadline_epoch']
    need(type(end) in (int, float) and math.isfinite(end) and now < end
         and request['parent_deadline_epoch'] == end
         and request['fusion_deadline_epoch'] == end - 1200
         and now + 1260 < request['fusion_deadline_epoch'],
         'FINITE_USER_DEADLINE_AND_ORIGINAL_CLOSEOUT_RESERVE')
    need(parent.get('status') == 'ACTIVE'
         and parent.get('owner') == request['owner']
         and parent.get('scope') == 'WM02_AUTO1_SIM_ONLY'
         and parent['deadline_epoch'] <= now < end,
         'ONLY_EXPIRED_ACTIVE_PARENT_NEVER_CLOSED_PARENT')
    for lease, scope in ((pilot, 'WM02_PILOT1_SIM_ONLY'),
                         (fusion, 'WM02_FUSION_REFERENCE_SIM_ONLY')):
        need(lease.get('status') == 'ACTIVE' and lease.get('owner') == request['owner']
             and lease.get('scope') == scope and lease['deadline_epoch'] <= now,
             'ONLY_EXPIRED_ACTIVE_SUPPLEMENT_TO_CLOSE')


def load(path, pin=None):
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink() and path.resolve() == path,
         'CANONICAL_METADATA_PATH')
    raw = path.read_bytes()
    need(len(raw) <= 16 * 1024**2, 'BOUNDED_METADATA')
    digest = hashlib.sha256(raw).hexdigest()
    need(pin is None or digest == pin, 'PINNED_METADATA_BYTES')
    return json.loads(raw), digest


def preflight(request):
    from go2arx5_vla.world_model.pilot1_scope import binding_contract as pilot_contract
    from go2arx5_vla.world_model.fusion_scope import binding_contract as fusion_contract
    now = time.time()
    need(os.uname().nodename == request['host'] and os.getuid() == request['uid'],
         'ACTUAL_EXISTING_HOST_UID')
    campaign = Path(request['campaign_root'])
    parent, _ = load(campaign/'control/LEASE.json', request['old_parent_lease_sha256'])
    old_pilot, _ = load(request['old_pilot_binding'], request['old_pilot_binding_sha256'])
    old_fusion, _ = load(request['old_fusion_binding'], request['old_fusion_binding_sha256'])
    ppath = Path(request['old_pilot_binding']).parent/'LEASE.json'
    fpath = Path(request['old_fusion_binding']).parent/'LEASE.json'
    pilot, _ = load(ppath, request['old_pilot_lease_sha256'])
    fusion, _ = load(fpath, request['old_fusion_lease_sha256'])
    request_contract(request, parent, pilot, fusion, now=now)
    need(not (campaign/'STOP.json').exists(), 'ORIGINAL_STOP_RETAINED')
    registry, _ = load(request['registry'])
    masters = {str(ppath), str(fpath), str(campaign/'control/LEASE.json')}
    for entry in registry['entries']:
        if entry.get('scope') not in ('WM02_AUTO1_SIM_ONLY', 'WM02_PILOT1_SIM_ONLY',
                                      'WM02_FUSION_REFERENCE_SIM_ONLY'):
            continue
        lease_path = entry.get('local_lease')
        need(lease_path, 'REGISTERED_LEASE_PATH_REQUIRED')
        if lease_path not in masters:
            lease, _ = load(lease_path)
            need(lease.get('status') == 'CLOSED', 'OTHER_RESEARCH_ROLE_NOT_CLOSED')
    for row in request['latest_terminal_runs']:
        root = Path(row['root'])
        spec, _ = load(root/'EXECUTION_SPEC.json', row['spec_sha256'])
        lease, _ = load(root/'control/LEASE.json', row['lease_sha256'])
        worker, _ = load(root/'WORKER_EXIT.private.json', row['worker_sha256'])
        need(lease.get('status') == 'CLOSED' and worker.get('exit_code') == 0
             and lease.get('spec_sha256') == row['spec_sha256']
             and spec['campaign_root'] == str(campaign), 'LATEST_PAIR_CLOSED_AND_BOUND')
        unit = subprocess.run(['systemctl', '--user', 'show', 'wm02-'+root.name,
                               '--property=MainPID', '--value'], capture_output=True,
                              text=True, timeout=10)
        need(unit.returncode == 0 and unit.stdout.strip() == '0', 'LATEST_PAIR_PID_ZERO')
    for root in (Path(request['new_pilot_root']), Path(request['new_fusion_root'])):
        need(not root.exists() and root.parent.is_dir()
             and root.parent.stat().st_dev == request['qualified_storage_device'],
             'FRESH_WINDOWS_ON_QUALIFIED_DATA_VOLUME')
    new_pilot = copy.deepcopy(old_pilot)
    new_pilot.update(activation_epoch=now, deadline_epoch=request['parent_deadline_epoch'],
                     parent_campaign_deadline_epoch=request['parent_deadline_epoch'],
                     source_root=request['runtime_source_root'],
                     source_sha=request['runtime_source_sha'],
                     evidence_files=request['evidence_files'],
                     user_authorization=request['user_authorization'])
    new_fusion = copy.deepcopy(old_fusion)
    new_fusion.update(activation_epoch=now, deadline_epoch=request['fusion_deadline_epoch'],
                      parent_deadline_epoch=request['parent_deadline_epoch'],
                      pilot1_binding=str(Path(request['new_pilot_root'])/'control/BINDING.private.json'),
                      source_root=request['runtime_source_root'],
                      source_sha=request['runtime_source_sha'],
                      user_authorization=request['user_authorization'])
    pilot_contract(new_pilot, now)
    fusion_contract(new_fusion, now)
    return parent, pilot, fusion, new_pilot, new_fusion


def renew(path, *, execute=False):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import once
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.pilot1_scope import register as register_pilot
    from go2arx5_vla.world_model.fusion_scope import register as register_fusion
    request, request_sha = load(path)
    values = preflight(request)
    if not execute:
        return dict(status='RENEWAL_READONLY_PREFLIGHT_PASSED', effects=0,
                    jobs_started=0, request_sha256=request_sha,
                    parent_deadline_epoch=request['parent_deadline_epoch'],
                    fusion_deadline_epoch=request['fusion_deadline_epoch'])
    campaign = Path(request['campaign_root']); key = request['request_id']
    journal = Journal(campaign/'control/journal', campaign.name)
    with ExitStack() as locks:
        for lock in (campaign/'OWNER.lock', Path(request['physical_owner_lock']),
                     Path(request['neural_owner_lock'])):
            locks.enter_context(closing(OwnerLock(lock)))
        parent, pilot, fusion, new_pilot, new_fusion = preflight(request)
        for label, binding, lease, pin in (
                ('OLD-FUSION-CLOSE', request['old_fusion_binding'], fusion, request['old_fusion_lease_sha256']),
                ('OLD-PILOT-CLOSE', request['old_pilot_binding'], pilot, request['old_pilot_lease_sha256'])):
            target = Path(binding).parent/'LEASE.json'
            value = dict(lease, status='CLOSED', closed_epoch=time.time(),
                         closeout_reason='EXPIRED_WINDOW_ALL_CHILDREN_CLOSED_BEFORE_AUTHORIZED_SUCCESSOR',
                         renewal_request_sha256=request_sha)
            journal.once(key+'-'+label, dict(request_sha256=request_sha, before_sha256=pin),
                         lambda target=target, value=value, pin=pin, label=label:
                         once(target, value, key+'-'+label, expected_before_sha256=pin))
        updated = dict(parent, deadline_epoch=request['parent_deadline_epoch'],
                       previous_deadline_epoch=parent['deadline_epoch'],
                       finite_renewal_request=str(path), finite_renewal_request_sha256=request_sha,
                       closed_lease_reopened=False)
        journal.once(key+'-PARENT-DEADLINE', dict(request_sha256=request_sha),
                     lambda: once(campaign/'control/LEASE.json', updated, key+'-PARENT-DEADLINE',
                                  expected_before_sha256=request['old_parent_lease_sha256']))
        def prepare():
            p = Path(request['new_pilot_root']); f = Path(request['new_fusion_root'])
            for root in (p, f):
                root.mkdir(); (root/'control').mkdir()
            write(p/'control/BINDING.private.json', new_pilot)
            new_fusion['pilot1_binding_sha256'] = digest(p/'control/BINDING.private.json')
            write(f/'control/BINDING.private.json', new_fusion)
            return dict(pilot_binding=str(p/'control/BINDING.private.json'),
                        fusion_binding=str(f/'control/BINDING.private.json'))
        paths = journal.once(key+'-PREPARE', dict(request_sha256=request_sha), prepare)
    # Original register functions acquire their own canonical locks.
    pilot_result = journal.once(key+'-PILOT-REGISTER', dict(request_sha256=request_sha),
                                lambda: register_pilot(paths['pilot_binding']))
    fusion_result = journal.once(key+'-FUSION-REGISTER', dict(request_sha256=request_sha),
                                 lambda: register_fusion(paths['fusion_binding']))
    return dict(status='NEW_FINITE_FUSION_WINDOW_REGISTERED_NO_WORKERS',
                request_sha256=request_sha, pilot=pilot_result, fusion=fusion_result,
                bindings=paths, native_integrations=0, queries=0, updates=0,
                new_campaign=False, closed_lease_reopened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(renew(args.request, execute=args.execute)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
