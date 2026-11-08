"""CPU scope/identity fixtures; not actual Core GPU qualification."""
import json
import pytest
from go2arx5_vla.world_model import target_gr00t_inference_v1 as inference
from go2arx5_vla.world_model.target_gr00t_profile_v1 import PARENT
from go2arx5_vla.world_model.storage import digest
from scripts.pi0294.numerics import MODE


def fixture(tmp_path):
    root=tmp_path/'data';root.mkdir()
    m=dict(bundle_kind='FIT4_DIAGNOSTIC',episodes=[dict(split='TRAIN')for _ in range(4)],
        TRAIN_only_statistics=True,TEST_results_read=False,WM_or_HOLD_or_diagnostic_relabelled=False)
    (root/'DATASET_MANIFEST.json').write_text(json.dumps(m));(root/'statistics.json').write_text('{}')
    a=dict(status='PASS_ACTUAL_OFFICIAL_PROCESSOR_ROUNDTRIP',
        manifest_sha256=digest(root/'DATASET_MANIFEST.json'),statistics_sha256=digest(root/'statistics.json'))
    (root/'PROCESSOR_TRAINING_ADMISSION.json').write_text(json.dumps(a))
    s=dict(profile=inference.PROFILE,kind=inference.KIND,execution_site='CORE_FUSION_NEURAL',
        physics=False,device='cuda',max_queries=4,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        numeric_mode=MODE,baseline_changed=False,checkpoint_manifest_sha256=PARENT,
        target_inference_mode=inference.PARENT_SHADOW,target_dataset=str(root),
        target_dataset_sha256=a['manifest_sha256'],target_statistics_sha256=a['statistics_sha256'],
        candidate_package=None)
    return s


def test_target_processor_shadow_has_separate_identity_and_no_candidate_weights(tmp_path):
    s=fixture(tmp_path);assert inference.validate_spec(s)['bundle_kind']=='FIT4_DIAGNOSTIC'
    s['candidate_package']={'path':'unqualified','sha256':'a'*64}
    with pytest.raises(PermissionError,match='PARENT_TARGET_PROCESSOR'):inference.validate_spec(s)


@pytest.mark.parametrize('key,value',[
    ('max_queries',8),('GR00T_updates',1),('WM_updates',200),('physics',True),
    ('baseline_changed',True),('kind','FROZEN_VLA_SHADOW'),('checkpoint_manifest_sha256','f'*64),
    ('numeric_mode','UNQUALIFIED_DEFAULT')])
def test_new_shadow_does_not_alias_baseline_or_enable_training(tmp_path,key,value):
    s=fixture(tmp_path);s[key]=value
    with pytest.raises(PermissionError):inference.validate_spec(s)


def test_target_processor_must_match_actual_admission_and_no_test_data(tmp_path):
    s=fixture(tmp_path);root=tmp_path/'data'
    m=json.loads((root/'DATASET_MANIFEST.json').read_bytes());m['TEST_results_read']=True
    (root/'DATASET_MANIFEST.json').write_text(json.dumps(m));s['target_dataset_sha256']=digest(root/'DATASET_MANIFEST.json')
    a=json.loads((root/'PROCESSOR_TRAINING_ADMISSION.json').read_bytes());a['manifest_sha256']=s['target_dataset_sha256']
    (root/'PROCESSOR_TRAINING_ADMISSION.json').write_text(json.dumps(a))
    with pytest.raises(PermissionError,match='TRAIN_ONLY_PROCESSOR'):inference.validate_spec(s)


@pytest.mark.parametrize('kind,module',[
    (inference.KIND,'scripts.wm02.infer_target_gr00t_v1'),
    ('FROZEN_VLA_SHADOW','scripts.wm02.infer_fusion_vla')])
def test_actual_neural_wrapper_routes_and_closes_only_its_owned_lease(tmp_path,monkeypatch,kind,module):
    import time,subprocess
    from types import SimpleNamespace
    from scripts.wm02 import pilot1_neural_worker as worker
    root=tmp_path/'job';root.mkdir();(root/'control').mkdir();spec=root/'EXECUTION_SPEC.json'
    lease=root/'control/LEASE.json';spec.write_text(json.dumps(dict(registered_root=str(root),
        lease=str(lease),kind=kind,source_sha='f'*40,wall_limit_s=60,deadline_epoch=time.time()+60)))
    lease.write_text(json.dumps(dict(status='ACTIVE',run_uuid=root.name,spec_sha256=digest(spec))))
    calls=[]
    def child(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',child)
    assert worker.main(['--spec',str(spec)])==0
    assert calls[0][3]==module
    assert json.loads(lease.read_bytes())['status']=='CLOSED'
    assert json.loads((root/'WORKER_EXIT.private.json').read_bytes())['exit_code']==0


def test_cross_hardware_gate_keeps_predeclared_tolerances_and_model_identity(tmp_path):
    import numpy as np
    rows=[dict(episode=str(i),split='TRAIN',anchor=0,seed=i,noise={'seed':i},model_global_step=0,
        model_identity=dict(dataset_sha256='a'*64,statistics_sha256='b'*64))for i in range(4)]
    a,b=tmp_path/'a',tmp_path/'b'
    for root,delta in ((a,0.),(b,5e-5)):
        root.mkdir();(root/'RESULT.json').write_text(json.dumps(dict(records=rows)))
        for i in range(4):np.savez(root/(str(i)+'.npz'),physical_H40=np.full((40,10),delta),normalized=np.zeros((1,40,132)))
    proof=inference.compare_cross_site(a,b)
    assert proof['absolute_component_tolerances']['XYZ_m']==1e-4
    np.savez(b/'3.npz',physical_H40=np.full((40,10),2e-4),normalized=np.zeros((1,40,132)))
    with pytest.raises(PermissionError,match='NUMERICAL_GATE_FAILED'):inference.compare_cross_site(a,b)
    rows[0]['model_identity']['statistics_sha256']='c'*64
    (b/'RESULT.json').write_text(json.dumps(dict(records=rows)))
    with pytest.raises(PermissionError,match='MODEL_DATA_STATISTICS'):inference.compare_cross_site(a,b)
