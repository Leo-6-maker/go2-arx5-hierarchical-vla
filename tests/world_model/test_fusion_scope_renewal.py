"""Renewal admission boundaries, independent of actual site qualification."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.wm02.renew_fusion_scope import preflight, request_contract, close_data_volume_window
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


def test_qualified_volume_close_preserves_expired_deadline_and_refuses_replay(tmp_path):
    request, parent, pilot, fusion = fixtures(tmp_path)
    path = Path(request['old_fusion_binding']).parent/'LEASE.json'
    value = dict(fusion, status='CLOSED')
    with actual_context():
        receipt = close_data_volume_window(path, value, request['old_fusion_lease_sha256'], request)
        assert json.loads(path.read_bytes()) == value
        assert receipt['before_sha256'] == request['old_fusion_lease_sha256']
        with pytest.raises(PermissionError, match='PINNED_METADATA_BYTES'):
            close_data_volume_window(path, value, request['old_fusion_lease_sha256'], request)


def test_data_volume_writer_cannot_write_other_lease_or_device(tmp_path):
    request, parent, pilot, fusion = fixtures(tmp_path)
    path = Path(request['old_fusion_binding']).parent/'LEASE.json'
    before = path.read_bytes()
    with actual_context(), pytest.raises(PermissionError, match='EXACT_QUALIFIED'):
        close_data_volume_window(path, dict(fusion, status='CLOSED'), request['old_fusion_lease_sha256'],
                                 dict(request, qualified_storage_device=-1))
    other=tmp_path/'unregistered/LEASE.json';other.parent.mkdir();other.write_text('{}')
    with actual_context(), pytest.raises(PermissionError, match='EXACT_QUALIFIED'):
        close_data_volume_window(other,dict(pilot,status='CLOSED'),request['old_pilot_lease_sha256'],request)
    assert path.read_bytes() == before


def test_expired_pilot_on_same_qualified_volume_uses_original_writer(tmp_path):
    request,parent,pilot,fusion=fixtures(tmp_path)
    path=Path(request['old_pilot_binding']).parent/'LEASE.json'
    with actual_context():
        close_data_volume_window(path,dict(pilot,status='CLOSED'),request['old_pilot_lease_sha256'],request)
    assert json.loads(path.read_bytes())==dict(pilot,status='CLOSED')


@pytest.mark.parametrize('exit_code',[0,1])
def test_terminal_failure_is_not_relabelled_as_success_or_a_renewal_block(tmp_path,exit_code):
    request,*_=fixtures(tmp_path);root=tmp_path/'last-run';(root/'control').mkdir(parents=True)
    def put(path,value):
        path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()
    spec=put(root/'EXECUTION_SPEC.json',dict(campaign_root=request['campaign_root']))
    lease=put(root/'control/LEASE.json',dict(status='CLOSED',spec_sha256=spec,worker_exit_code=exit_code))
    worker=put(root/'WORKER_EXIT.private.json',dict(exit_code=exit_code))
    request['latest_terminal_runs']=[dict(root=str(root),spec_sha256=spec,lease_sha256=lease,worker_sha256=worker)]
    terminal=SimpleNamespace(returncode=0,stdout='0\n')
    with actual_context(),patch('scripts.wm02.renew_fusion_scope.subprocess.run',return_value=terminal):
        preflight(request)
    assert json.loads((root/'WORKER_EXIT.private.json').read_bytes())['exit_code']==exit_code
