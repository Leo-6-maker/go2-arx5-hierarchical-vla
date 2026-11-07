"""Renewal admission boundaries, independent of actual site qualification."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.wm02.renew_fusion_scope import preflight, request_contract
from go2arx5_vla.world_model.pilot1_scope import C08, PLAN_SHA
from go2arx5_vla.world_model.fusion_contract import PLAN


def fixtures(tmp_path):
    def put(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(body).encode(); path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    campaign = tmp_path/'campaign'
    parent = dict(status='ACTIVE', owner='EXISTING_LOCAL_PC',
                  scope='WM02_AUTO1_SIM_ONLY', deadline_epoch=99)
    pilot = dict(status='ACTIVE', owner=parent['owner'], scope='WM02_PILOT1_SIM_ONLY', deadline_epoch=99)
    fusion = dict(status='ACTIVE', owner=parent['owner'], scope='WM02_FUSION_REFERENCE_SIM_ONLY', deadline_epoch=99)
    ppath = tmp_path/'old-pilot/control/BINDING.private.json'
    fpath = tmp_path/'old-fusion/control/BINDING.private.json'
    pb = dict(schema='WM02_PILOT1_SCOPE_V1', plan_sha=PLAN_SHA, scope=pilot['scope'],
              owner=parent['owner'], host='corelab-4080s', uid=2009, engineering_budget_caps=None,
              max_native_slots=2, C08_native_binary_sha256=C08)
    fb = dict(schema='WM02_FUSION_NIGHT1_SCOPE_V1', plan_sha=PLAN, scope=fusion['scope'],
              owner=parent['owner'], host='corelab-4080s', uid=2009, max_native_slots=2,
              GR00T_updates=0, WBC_changes=False, physics_dt_s=.002,
              native_reference_domain='SINGLE_DECISION_LOCAL_XY4_RAMP8_H16_OPEN_GRIPPER')
    request = dict(schema='WM02_FUSION_EXPLICIT_FINITE_RENEWAL_V1', owner=parent['owner'],
                   host='corelab-4080s', uid=2009, user_authorization='explicit finite successor request',
                   WBC_changes=False, physics_dt_s=.002, GR00T_updates=0, Actor_updates=0,
                   user_authorized_deadline_epoch=10000, parent_deadline_epoch=10000,
                   fusion_deadline_epoch=8800, campaign_root=str(campaign),
                   old_parent_lease_sha256=put(campaign/'control/LEASE.json', parent),
                   old_pilot_binding=str(ppath), old_pilot_binding_sha256=put(ppath, pb),
                   old_pilot_lease_sha256=put(ppath.parent/'LEASE.json', pilot),
                   old_fusion_binding=str(fpath), old_fusion_binding_sha256=put(fpath, fb),
                   old_fusion_lease_sha256=put(fpath.parent/'LEASE.json', fusion),
                   new_pilot_root=str(tmp_path/'new-pilot'), new_fusion_root=str(tmp_path/'new-fusion'),
                   qualified_storage_device=tmp_path.stat().st_dev,
                   runtime_source_root=str(tmp_path/'source'), runtime_source_sha='d'*40,
                   evidence_files={}, latest_terminal_runs=[])
    registry = tmp_path/'REGISTRY.json'
    put(registry, dict(entries=[dict(scope=pilot['scope'], local_lease=str(ppath.parent/'LEASE.json')),
                               dict(scope=fusion['scope'], local_lease=str(fpath.parent/'LEASE.json'))]))
    request['registry'] = str(registry)
    return request, parent, pilot, fusion


def actual_context():
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch('scripts.wm02.renew_fusion_scope.time.time', return_value=100))
    stack.enter_context(patch('scripts.wm02.renew_fusion_scope.os.uname', return_value=SimpleNamespace(nodename='corelab-4080s')))
    stack.enter_context(patch('scripts.wm02.renew_fusion_scope.os.getuid', return_value=2009))
    return stack


def test_explicit_renewal_closes_old_windows_without_changing_binding_bytes(tmp_path):
    request, *leases = fixtures(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with actual_context():
        values = preflight(request)
    assert values[3]['deadline_epoch'] == 10000
    assert values[4]['deadline_epoch'] == 8800
    assert {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('change', [
    dict(user_authorization=''), dict(owner='different'), dict(WBC_changes=True),
    dict(physics_dt_s=.001), dict(GR00T_updates=1), dict(Actor_updates=1),
    dict(parent_deadline_epoch=11000), dict(fusion_deadline_epoch=9999),
    dict(user_authorized_deadline_epoch=float('inf'), parent_deadline_epoch=float('inf'), fusion_deadline_epoch=float('inf')),
    dict(user_authorized_deadline_epoch=90, parent_deadline_epoch=90, fusion_deadline_epoch=-1110),
])
def test_request_cannot_expand_authority_or_frozen_domain(tmp_path, change):
    request, parent, pilot, fusion = fixtures(tmp_path)
    with pytest.raises(PermissionError):
        request_contract(dict(request, **change), parent, pilot, fusion, now=100)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_closed_parent_or_supplement_is_never_reopened(tmp_path, index):
    request, *leases = fixtures(tmp_path)
    leases[index]['status'] = 'CLOSED'
    with pytest.raises(PermissionError):
        request_contract(request, *leases, now=100)


def test_unexpired_parent_cannot_be_silently_replaced(tmp_path):
    request, parent, pilot, fusion = fixtures(tmp_path)
    parent['deadline_epoch'] = 101
    with pytest.raises(PermissionError):
        request_contract(request, parent, pilot, fusion, now=100)


def test_other_registered_active_role_blocks_before_mutation(tmp_path):
    request, *_ = fixtures(tmp_path)
    child = tmp_path/'other/control/LEASE.json'; child.parent.mkdir(parents=True)
    child.write_text(json.dumps(dict(status='ACTIVE')))
    registry = Path(request['registry']); value = json.loads(registry.read_text())
    value['entries'].append(dict(scope='WM02_PILOT1_SIM_ONLY', local_lease=str(child)))
    registry.write_text(json.dumps(value))
    with actual_context(), pytest.raises(PermissionError, match='OTHER_RESEARCH_ROLE_NOT_CLOSED'):
        preflight(request)


def test_stop_and_existing_successor_remain_rejections(tmp_path):
    request, *_ = fixtures(tmp_path)
    stop = Path(request['campaign_root'])/'STOP.json'; stop.write_text('{}')
    with actual_context(), pytest.raises(PermissionError, match='ORIGINAL_STOP_RETAINED'):
        preflight(request)
    stop.unlink(); Path(request['new_fusion_root']).mkdir()
    with actual_context(), pytest.raises(PermissionError, match='FRESH_WINDOWS'):
        preflight(request)


def test_changed_parent_bytes_fail_the_pinned_preflight(tmp_path):
    request, *_ = fixtures(tmp_path)
    Path(request['campaign_root'], 'control/LEASE.json').write_text('{}')
    with actual_context(), pytest.raises(PermissionError, match='PINNED_METADATA_BYTES'):
        preflight(request)
