"""Synthetic proof fixtures, not trained models or native successes."""
import json
from pathlib import Path
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_evaluation_v1 as evaluation
from go2arx5_vla.world_model import stable_task_scene_v1 as scene
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_stable_task_data_v1 import fixture,put


def configuration(tmp_path,monkeypatch):
    spec,pin=fixture(tmp_path,monkeypatch,np.eye(4))
    static=Path(spec['stable_task_scene_reference']['static_result']['path'])
    proof=json.loads(static.read_text());proof.update(layout='DEV01',parent='STABLE_TASK_LAYOUT_DEV01',split='DEV')
    proof['compiled']['layout']='DEV01';proof['compiled']['changes']['offset_xy_m']=[.015,.03]
    spec['stable_task_scene_reference']['static_result']=put(static,proof)
    spec.update(phase=evaluation.PHASE,kind=evaluation.KIND,data_role='DEV',data_parent=proof['parent'],scene_id='STABLE_TASK_DEV01',
                snapshot_binding_mode='PRE_REGISTERED_STABLE_TASK_ADAPTER_EVAL_V1')
    row=dict(native_run=Path(spec['registered_root']).name,parent=proof['parent'],split='DEV',
        model_sha256=pin,static_sha256=digest(static),command=spec['task_command'])
    panel=put(tmp_path/'EVAL_PANEL.json',dict(schema='WM02_STABLE_TASK_ADAPTER_NATIVE_PANEL_V1',assignments=[row]))
    spec.update(stable_task_panel=panel['path'],stable_task_panel_sha256=panel['sha256'])
    job=tmp_path/'TRAIN_JOB';output=job/'fit-01';output.mkdir(parents=True);(job/'control').mkdir()
    put(job/'EXECUTION_SPEC.json',dict(kind='TASK_ADAPTER',profile='wm02-pilot1-neural',TASK_ADAPTER_updates=2000))
    put(job/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,spec_sha256=digest(job/'EXECUTION_SPEC.json')))
    put(job/'WORKER_EXIT.private.json',dict(exit_code=0))
    from go2arx5_vla.world_model import stable_task_adapter_v1 as candidate
    export=output/'TASK_ADAPTER_EXPORT.npz'
    np.savez(export,schema=np.array(candidate.SCHEMA),mean=np.zeros(116,np.float32),
        scale=np.ones(116,np.float32),**{'hidden.weight':np.zeros((32,116),np.float32),
        'hidden.bias':np.zeros(32,np.float32),'output.weight':np.zeros((3,32),np.float32),
        'output.bias':np.zeros(3,np.float32)})
    trained=dict(schema='WM02_STABLE_TASK_ADAPTER_TRAIN_ACTUAL_V1',status='CANDIDATE_TRAINED_RELOADED_NOT_NATIVE_EVALUATED',
        TASK_ADAPTER_updates=2000,GR00T_updates=0,Actor_updates=0,WM_updates=0,optimizer_restored=True,sampler_restored=True,
        RNG_restored=True,export_sha256=digest(export))
    result=put(output/'RESULT.private.json',trained)
    spec['stable_task_evaluation']=dict(method='LEARNED_TASK_ADAPTER',step_m=.004,suffix_atoms=176,
        minimum_course_clearance_m=.14,producer='INDEPENDENT_TASK_ADAPTER_NOT_TEACHER',training_result=result,
        export=dict(path=str(export),sha256=digest(export)))
    return spec,trained,output


def test_candidate_has_real_dev_identity_and_separate_training_contract(tmp_path,monkeypatch):
    spec,_,_=configuration(tmp_path,monkeypatch)
    assert evaluation.validate(spec)['method']=='LEARNED_TASK_ADAPTER'
    assert scene.episode(spec)['actual_kind']=='STABLE_TASK_EVAL_V1'
    with pytest.raises(ValueError,match='TYPED_PHASE_AND_ROLE'):
        evaluation.validate(dict(spec,data_role='TRAIN'))
    with pytest.raises(ValueError,match='EXPLICIT_REAL_DEV'):
        evaluation.validate(dict(spec,paused_query_pair='/old/pair'))


@pytest.mark.parametrize('change',[dict(TASK_ADAPTER_updates=0),dict(GR00T_updates=1),dict(optimizer_restored=False)])
def test_untrained_or_incomplete_candidate_is_not_native_admitted(tmp_path,monkeypatch,change):
    spec,trained,output=configuration(tmp_path,monkeypatch);trained.update(change)
    spec['stable_task_evaluation']['training_result']=put(output/'RESULT.private.json',trained)
    with pytest.raises(ValueError,match='REAL_SEALED_NONZERO'):
        evaluation.validate(spec)


def test_candidate_dispatch_does_not_enter_teacher_route(monkeypatch):
    from go2arx5_vla.world_model.auto1_native import phase_limit
    from go2arx5_vla.world_model import stable_task_data_v1 as data
    seen=[]
    monkeypatch.setattr(evaluation,'episode_limit',lambda s,p:seen.append(s)or evaluation.E)
    monkeypatch.setattr(data,'episode_limit',lambda s,p:pytest.fail('teacher route used'))
    spec=dict(phase=evaluation.PHASE,kind=evaluation.KIND)
    assert phase_limit(evaluation.PHASE,None,spec)==evaluation.E
    assert seen==[spec]


def test_bad_export_fails_before_native_prefix(tmp_path,monkeypatch):
    spec,_,output=configuration(tmp_path,monkeypatch)
    (output/'TASK_ADAPTER_EXPORT.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='PINNED_CANONICAL_SMALL_EXPORT'):
        evaluation.validate(spec)
