"""Artifact/role/partial-response software fixtures; never site qualifications."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_source_records_v1 as sources
from test_fusion_live_program_replay_v1 import artifacts


def fixture(tmp_path,monkeypatch,role='TRAIN'):
    monkeypatch.setattr(sources,'local_path',lambda spec,path:Path(path))
    checked=[];monkeypatch.setattr(sources,'_pid0',lambda root:checked.append(root.name))
    nroot=tmp_path/'native';groot=tmp_path/'gpu'
    for root in [nroot,groot]:
        (root/'control').mkdir(parents=True)
    nr=nroot/'native-01';gr=groot/'fit-01';nr.mkdir();gr.mkdir()
    def write(path,value):
        path.write_text(json.dumps(value));return descriptor(path)
    def descriptor(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    raw,prefix,decision,segment=artifacts();registry=tmp_path/'REGISTRY.json';panel=tmp_path/'PANEL.json'
    p=write(panel,dict(assignments=[dict(native_run=nroot.name,GPU_run=groot.name,parent='NEW_P0',split=role)]))
    raw.update(registered_root=str(nroot),source_sha='a'*40,role='NATIVE',owner='TEST_OWNER',registry=str(registry),
        response_learning_assignment=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',split=role,parent='NEW_P0',panel_sha256=p['sha256']),
        response_learning_panel=str(panel),response_learning_panel_sha256=p['sha256'])
    ns=write(nroot/'EXECUTION_SPEC.json',raw)
    g=dict(registered_root=str(groot),source_sha='a'*40,role='GPU',owner='TEST_OWNER',registry=str(registry),
        producer_native_spec=ns['path'],producer_native_spec_sha256=ns['sha256'],kind='FROZEN_VLA_LIVE_PAUSED_QUERY',max_queries=1,
        checkpoint_manifest_sha256=sources.CHECKPOINT,numeric_mode='ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST',Actor_updates=0,WM_updates=0,GR00T_updates=0)
    gs=write(groot/'EXECUTION_SPEC.json',g)
    entries=[dict(path=str(root),role=role0,source_sha='a'*40,owner='TEST_OWNER',local_lease=str(root/'control/LEASE.json'))for root,role0 in [(nroot,'NATIVE'),(groot,'GPU')]]
    registry.write_text(json.dumps(dict(entries=entries)))
    bundle=dict(source_sha='a'*40,native_spec=ns,gpu_spec=gs,learning_panel=p)
    for side,root,spec,code in [('native',nroot,ns,1),('gpu',groot,gs,0)]:
        bundle[side+'_lease']=write(root/'control/LEASE.json',dict(status='CLOSED',run_uuid=root.name,source_sha='a'*40,spec_sha256=spec['sha256'],worker_exit_code=code))
        bundle[side+'_worker']=write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=None))
    bundle['native_result']=write(nr/'RESULT.json',dict(source_sha='a'*40,phase=sources.replay.PHASE,native_binary_sha256=sources.NATIVE_SHA,observer_sha256=sources.OBSERVER_SHA,status='NATIVE_STAGE_FAILED_PRESERVED'))
    bundle['initialization']=write(nr/'LIVE_SKILL_INITIALIZATION_STAGE.private.json',prefix)
    bundle['decision']=write(nr/'LIVE_SKILL_SEGMENT_DECISION.private.json',decision)
    bundle['segment']=write(nr/'LIVE_SKILL_SEGMENT.private.json',segment)
    action=gr/'REAL_VLA_ACTION.npz';np.savez(action,physical_H40=np.asarray(decision['raw_H40']))
    bundle['action']=descriptor(action)
    bundle['gpu_result']=write(gr/'RESULT.private.json',dict(actual_VLA_inferences=1,action_sha256=bundle['action']['sha256'],producer_native_spec_sha256=ns['sha256'],checkpoint_manifest_sha256=sources.CHECKPOINT,source_sha='a'*40,execution_spec_sha256=gs['sha256']))
    manifest=tmp_path/'SOURCES.json';mp=write(manifest,dict(schema=sources.MANIFEST_SCHEMA,records=[bundle]))
    spec=dict(owner='TEST_OWNER',registry=str(registry),source_manifest=mp['path'],source_manifest_sha256=mp['sha256'])
    def refresh():
        manifest.write_text(json.dumps(dict(schema=sources.MANIFEST_SCHEMA,records=[bundle])))
        spec['source_manifest_sha256']=descriptor(manifest)['sha256']
    return spec,bundle,refresh,checked


def test_closed_partial_native_failure_keeps_actual_labels_and_original_group(tmp_path,monkeypatch):
    spec,bundle,refresh,checked=fixture(tmp_path,monkeypatch)
    data,meta=sources.load_sources(spec)
    assert checked==['native','gpu'] and data['X'].shape==(3,1614)
    assert data['valid'].sum()==27 and set(data['split'])=={'TRAIN'}
    assert meta['attempts'][0]['native_exit_code']==1 and meta['summary']['physical_runs']==1


@pytest.mark.parametrize('role',['DEV','TEST','DIAGNOSTIC'])
def test_other_roles_never_become_train(tmp_path,monkeypatch,role):
    spec,_,_,_=fixture(tmp_path,monkeypatch,role)
    with pytest.raises(ValueError,match='CANNOT_BECOME_TRAIN'):sources.load_sources(spec)


def test_active_lease_and_changed_artifact_are_rejected_before_fitting(tmp_path,monkeypatch):
    spec,bundle,refresh,_=fixture(tmp_path,monkeypatch)
    path=Path(bundle['native_lease']['path']);value=json.loads(path.read_text());value['status']='ACTIVE';path.write_text(json.dumps(value))
    bundle['native_lease']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();refresh()
    with pytest.raises(ValueError,match='CLOSED_WORKER'):sources.load_sources(spec)


def test_tampered_decision_is_not_another_valid_source(tmp_path,monkeypatch):
    spec,bundle,_,_=fixture(tmp_path,monkeypatch)
    Path(bundle['decision']['path']).write_text('{}')
    with pytest.raises(ValueError,match='ARTIFACT_SHA256'):sources.load_sources(spec)
