"""Pure target identity fixtures. No native/runtime imports or dispatch."""
import hashlib,json
from copy import deepcopy
import pytest
from scripts.pi0294.dec1 import target_readonly_identity as unit


def raw(v):return (json.dumps(v,sort_keys=True)+'\n').encode()
def h(v):return hashlib.sha256(v).hexdigest()


@pytest.fixture
def packet():
    pin='a'*64;source='b'*40
    seal=dict(source_sha=source,files={'scripts/pi0294/task_native_worker.py':pin})
    adoption=dict(authorizes_execution=True,counter_reset=False,old_lease_reopened=False)
    out=dict(run_uuid='original-target',role='ORIGINAL_REGISTERED_ROLE',run_root='/private/registered/original-target')
    docs=dict(manifest=dict(scope=unit.SCOPE,run_uuid='original-target',owner='EXISTING_LOCAL_PC',
        authority=dict(adoption_sha256=h(raw(adoption))),
        code=dict(vla_commit=source,source_seal_sha256=h(raw(seal))),
        execution=dict(profile=unit.PROFILE,profile_python='/private/python',entry_path='/private/entry',entry_sha256=pin,
            output_binding_sha256=h(raw(out)))),
        input_binding=dict(asset_root='/private/assets',runtime_python='/private/python'),output_binding=out,
        registry=dict(entries=[dict(role=out['role'],path=out['run_root'])]),
        lease=dict(status='CLOSED',run_uuid='original-target',deadline_epoch=100,adoption_sha256=h(raw(adoption))),
        source_seal=seal,window_adoption=adoption,
        startup_config=dict(initialization=dict(compiled_model='/private/frozen.mjb',
            pinned_files={'/private/frozen.mjb':pin,'/private/frozen-leg.pt':pin,'/private/task_floating.info':pin})),
        clock_build=dict(source_pins={'original.cpp':pin},binary_sha256={'go2_x5_ocs2_node':pin}),
        pause_build=dict(binary_sha256={'rl_sim_mujoco':pin},original_clock_binaries={'go2_x5_ocs2_node':pin}),
        adapter_build=dict(binaries={'task_actuation':pin,'task_observer':pin}),
        task_acceptance=dict(version=unit.ASSET,physical_qualification=False,WBC_changes=0,safety_threshold_changes=0),
        pilot_plan=dict(cards=[dict(id='TASK-PILOT-'+f+'-'+t,task=t) for f in ('W0','W1') for t in ('T1','T2')]))
    files={'/private/entry':pin,'/private/frozen.mjb':pin,'/private/frozen-leg.pt':pin,
        '/private/task_floating.info':pin,'native/original.cpp':pin,'clock/go2_x5_ocs2_node':pin,
        'pause/rl_sim_mujoco':pin,'adapter/task_actuation.so':pin,'adapter/task_observer.so':pin,
        'source/scripts/pi0294/task_native_worker.py':pin}
    def call():
        encoded={k:raw(v) for k,v in docs.items()}
        return unit.inspect(encoded,{k:h(v) for k,v in encoded.items()},files,now_epoch=200)
    return docs,files,call


def test_complete_historical_identity_never_reopens_expired_native_entry(packet):
    docs,files,call=packet;before=deepcopy(docs);r=call()
    assert not r['missing'] and not r['mismatch']
    assert r['status']=='IDENTITY_BOUND_CURRENT_NATIVE_ADMISSION_UNCHECKED'
    assert r['historical_window']=='HISTORICAL_OR_CLOSED'
    assert r['native_dispatch_admitted'] is r['target_task_native_qualified'] is False
    assert not r['executor_manifest_created'] and not r['controller_or_role_changed']
    assert docs==before and not r['source_training_waits_for_target']


def test_fd2_source_profile_and_assets_do_not_become_target_wbc(packet):
    docs,files,call=packet;docs['manifest']['scope']='MP1_SINGLE_SOURCE_CONFIRMATION_NO_TRAINING'
    r=call();assert r['status']=='SUPPLIED_METADATA_IS_NOT_FROZEN_TARGET_TASK_ENTRY'
    assert 'field:manifest.scope' in r['mismatch'] and not r['target_task_native_qualified']


def test_empty_inventory_names_exact_existing_delta_without_claim_interface_missing():
    r=unit.inspect({}, {}, {}, now_epoch=200)
    assert 'document:clock_build' in r['missing'] and 'document:startup_config' in r['missing']
    assert r['existing_callable']=='scripts.pi0294.task_native_worker.run'
    assert r['interface']['wire']=='WORLD_EE_JSON_41_POINTS_NO_PREVIEW9'
    assert not r['hold_RMS_repair_required']


def test_private_paths_not_exported_and_ack_stop_sources_not_fabricated(packet):
    r=packet[2]();assert '/private/' not in json.dumps(r)
    assert 'native_receiver' in r['interface']['ACK'] and 'process terminal' in r['interface']['stop']
    assert r['interface']['tcp_site']=='x5_ee'
    assert r['interface']['cameras']==['body_rgb','wrist_rgb']


@pytest.mark.parametrize('fault',['entry','controller','policy_weights','plant','adapter','pause_base',
    'task_overlay','adoption_reset','registry','profile','source_member','pilot_duplicate'])
def test_wrong_existing_identity_fails_readiness_without_touching_controller(packet,fault):
    docs,files,call=packet
    if fault=='entry':files['/private/entry']='c'*64
    elif fault=='controller':files['clock/go2_x5_ocs2_node']='c'*64
    elif fault=='policy_weights':files.pop('/private/frozen-leg.pt')
    elif fault=='plant':files['/private/frozen.mjb']='c'*64
    elif fault=='adapter':files['adapter/task_actuation.so']='c'*64
    elif fault=='pause_base':docs['pause_build']['original_clock_binaries']={}
    elif fault=='task_overlay':docs['task_acceptance']['version']='SOURCE_FIXED_BASE_NOT_TARGET'
    elif fault=='adoption_reset':docs['window_adoption']['counter_reset']=True
    elif fault=='registry':docs['registry']['entries']=[]
    elif fault=='profile':docs['manifest']['execution']['profile']='HOPPER_TRAINING'
    elif fault=='source_member':files['source/scripts/pi0294/task_native_worker.py']='c'*64
    elif fault=='pilot_duplicate':docs['pilot_plan']['cards'][1]=deepcopy(docs['pilot_plan']['cards'][0])
    r=call();assert r['missing'] or r['mismatch']
    assert not r['native_dispatch_admitted'] and not r['target_task_native_qualified']


def test_document_pin_mismatch_and_duplicate_json_never_ready(packet):
    docs,files,_=packet;data={k:raw(v) for k,v in docs.items()};pins={k:h(v) for k,v in data.items()}
    data['manifest']+=b' '
    r=unit.inspect(data,pins,files,now_epoch=200)
    assert 'document:manifest' in r['mismatch']
    data['manifest']=b'{"scope":"A","scope":"B"}';pins['manifest']=h(data['manifest'])
    assert 'document:manifest' in unit.inspect(data,pins,files,now_epoch=200)['mismatch']
