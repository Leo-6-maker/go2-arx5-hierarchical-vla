from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from go2arx5_vla.world_model import expert_plan_capture_v61 as capture


def recorder(tmp_path,monkeypatch):
    monkeypatch.setattr(capture,'configuration',lambda spec:{'transport_statistics':{'sha256':'s'*64}})
    adapter=SimpleNamespace(output=tmp_path,spec={'source_root':str(Path(__file__).resolve().parents[1]),'registered_root':str(tmp_path)},observe=lambda:{'step':5703})
    r=capture.Recorder(adapter)
    raw=np.repeat(np.array([[0.,0.,.6,1.,0.,0.,0.,1.,0.,.5]]),40,axis=0)
    p=SimpleNamespace(decoded_H40=raw,reference_path=raw)
    r.begin(p,index=0,proposal={'preplanner':True},reference=np.eye(4))
    r.active['transported_reference_H40']=raw.tolist()
    return r,raw


def certify(r,raw):
    r.certify(dict(native_step=5703,action_EEF10=raw.tolist(),native_reference_certificate={
        'status':'PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF','physics_integrations':0}))


def row(raw,k):
    return dict(actual_installed_consumed_EEF10_BASE7=raw[k].tolist()+[0.]*7,
        native_start_step=5703+25*k,native_end_step=5703+25*(k+1),actual_ticks=25,
        solver_policies_with_exact_installed_target=[{'original_join':True}],gripper={'command_consumed':True})


def test_before_install_async_record_then_real_interval_mask_and_partial_failure(tmp_path,monkeypatch):
    r,raw=recorder(tmp_path,monkeypatch);pin=r.active['plan_sha256'];ids=r.active['command_ids'].copy()
    assert not any(r.active['supervision_mask'])
    certify(r,raw)
    for k in range(6):r.confirm(k,row(raw,k))
    r.close();s=json.loads((tmp_path/'FINAL_PLAN_CAPTURE_ACTUAL.private.json').read_bytes())
    assert (s['supervised_points'],s['consumed_points'])==(40,6)
    assert s['plans'][0]['plan_sha256']==pin and s['plans'][0]['command_ids']==ids
    assert not s['expert_task_success_observed'] and not s['training_dataset_admitted']
    events=[json.loads(line)for line in(tmp_path/'FINAL_PLAN_EVENTS.private.jsonl').read_text().splitlines()]
    assert events[0]['kind']=='FINAL_PLAN_BEFORE_INSTALL' and not any(events[0]['plan']['supervision_mask'])
    assert events[1]['kind']=='ORIGINAL_NATIVE_CERTIFICATE_AND_INSTALL_ACK'
    assert sum(x['kind']=='CONSUMPTION'for x in events)==6 and s['writer']['committed']


def test_wrong_reference_or_clock_is_retained_as_error_not_consumption(tmp_path,monkeypatch):
    r,raw=recorder(tmp_path,monkeypatch);certify(r,raw)
    bad=row(raw,0);bad['actual_installed_consumed_EEF10_BASE7'][0]+=.01
    with pytest.raises(ValueError,match='EXACT_ORIGIN'):r.confirm(0,bad)
    bad=row(raw,0);bad['native_start_step']+=25
    with pytest.raises(ValueError,match='ORIGINAL_QUERY'):r.confirm(0,bad)
    r.close()
