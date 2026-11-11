"""Dataset shape/role fixtures do not grant native or Processor qualification."""
import copy
import json
import numpy as np
import pytest
from PIL import Image
from test_task_learning_dataset_v1 import packet
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_dataset_v1 import exposure_schedule
from go2arx5_vla.world_model.task_learning_native_dataset_v1 import TaskLearningNativeDataset,sample_rows
from go2arx5_vla.world_model.task_learning_export_v1 import SCHEMA,DOMAIN,PROFILE
from go2arx5_vla.world_model.target_gr00t_dataset_v1 import TargetNativeDataset
from go2arx5_vla.world_model.storage import digest


def manifest(root):
    root.mkdir(exist_ok=True);rows=[];p=packet()
    for restart in range(4):
        for layout in range(25):
            valid=np.ones(15,bool)
            if restart>=2:valid[8]=False
            q=dict(p,valid_window=valid)
            rows.append(dict(id=f'fixture-{layout:02d}-{restart}',parent=f'TRAIN{layout:02d}',split='TRAIN',
                command=task.COMMAND,file=f'episodes/{layout:02d}-{restart}.npz',sha256='missing-native-data',
                supervised_anchors=np.flatnonzero(valid).tolist(),optimization_anchors=exposure_schedule(q),
                real_policy_decisions=15,supervised_policy_decisions=int(valid.sum()),
                expert_stage_audit_only=p['expert_stage_audit_only'].tolist(),genuine_recovery_success=restart>=2))
    (root/'statistics.json').write_text('{}')
    m=dict(schema=SCHEMA,domain=DOMAIN,profile=PROFILE,bundle_kind='CONTINUOUS_TRAIN_BASE',
        statistics_sha256=digest(root/'statistics.json'),episodes=rows,population=dict(
            complete=True,passes=True,requested=100,attempted=100,task_successes=100,
            successful_layouts=25,genuine_recovery_successes=50,recovery_layouts=25,
            hard_safety_stop_conditions=0,technical_failure_conditions=0),
        TRAIN_only_statistics=True,statistics_fit_before_oversampling=True,interventions_masked=True,
        construction_or_qualification_relabelled=False,TEST_results_read=False)
    save_manifest(root,m);return m


def save_manifest(root,m):
    (root/'DATASET_MANIFEST.json').write_text(json.dumps(m))


def test_unique_observations_are_separate_from_balanced_optimization_visits(tmp_path):
    m=manifest(tmp_path)
    observed=sample_rows(m['episodes'],'TRAIN',False);visited=sample_rows(m['episodes'],'TRAIN',True)
    assert len(observed)==1450 and len(visited)==3200
    assert all(a<15 for _,a in visited)
    assert all(a!=8 for r,a in visited if r['genuine_recovery_success'])
    bad=copy.deepcopy(m['episodes']);bad[50]['optimization_anchors'][0]=8
    with pytest.raises(ValueError):sample_rows(bad,'TRAIN',True)
    with pytest.raises(ValueError):sample_rows(m['episodes'],'DEV',True)


def test_old_short_dataset_cannot_be_used_as_continuous_task_and_reverse(tmp_path):
    m=manifest(tmp_path)
    with pytest.raises(ValueError,match='TARGET_PROFILE_NOT_OLD_SOURCE'):
        TargetNativeDataset(tmp_path,'unused','unused')
    m.update(schema=TargetNativeDataset.MANIFEST_SCHEMA,domain=TargetNativeDataset.MANIFEST_DOMAIN,
        profile=TargetNativeDataset.MANIFEST_PROFILE);save_manifest(tmp_path,m)
    with pytest.raises(ValueError,match='TARGET_PROFILE_NOT_OLD_SOURCE'):
        TaskLearningNativeDataset(tmp_path,'unused','unused')


def test_processor_family_route_preserves_both_domains_and_rejects_guessing(tmp_path):
    from scripts.wm02.qualify_target_gr00t_processor_v1 import dataset_type
    manifest(tmp_path)
    assert dataset_type('TARGET_SHORT_V1')is TargetNativeDataset
    assert dataset_type('TASK_LEARNING_CONTINUOUS_V1')is TaskLearningNativeDataset
    assert len(dataset_type('TASK_LEARNING_CONTINUOUS_V1')(tmp_path,'unused','unused'))==1450
    with pytest.raises(ValueError,match='TARGET_PROFILE_NOT_OLD_SOURCE'):
        dataset_type('TARGET_SHORT_V1')(tmp_path,'unused','unused')
    with pytest.raises(ValueError,match='EXPLICIT_ORIGINAL'):
        dataset_type('infer-from-filename')


def test_train_base_or_synthetic_success_metadata_cannot_authorize_optimization(tmp_path):
    m=manifest(tmp_path)
    with pytest.raises(ValueError,match='FIXED_NEW_DEV_REFERENCES'):
        TaskLearningNativeDataset(tmp_path,'unused','unused',for_optimization=True)
    m['bundle_kind']='FORMAL_TASK_LEARNING_V1'
    for i in range(4):
        row=copy.deepcopy(m['episodes'][i]);row.update(id=f'dev-{i}',parent=f'DEV{i}',split='DEV',purpose='DEV_REFERENCE_NOT_BC')
        m['episodes'].append(row)
    save_manifest(tmp_path,m)
    with pytest.raises(FileNotFoundError,match='PROCESSOR_TRAINING_ADMISSION'):
        TaskLearningNativeDataset(tmp_path,'unused','unused',for_optimization=True)


def test_raw_uses_real_decision_anchor_to_select_the_later_committed_h40(tmp_path):
    m=manifest(tmp_path);p=packet();p['valid_window'][:]=True
    p['instruction']=np.asarray(task.LANGUAGE)
    image=tmp_path/'rgb.png';Image.new('RGB',(256,256),(20,40,60)).save(image)
    p.update(rgb_paths=np.full((15,2),'rgb.png'),rgb_sha256=np.full((15,2),digest(image)))
    path=tmp_path/m['episodes'][0]['file'];path.parent.mkdir();np.savez_compressed(path,**p)
    m['episodes'][0]['sha256']=digest(path);save_manifest(tmp_path,m)
    data=TaskLearningNativeDataset(tmp_path,'unused','unused')
    value=data.raw(9);assert len(value)==4
    state,actions,images,text=value
    assert np.array_equal(state,p['actual_state10'][9])
    assert np.array_equal(actions,p['committed_absolute10'][360:400])
    assert not np.array_equal(actions,p['installed_consumed_50ms_audit_only'][360:400])
    assert len(images)==2 and text==task.LANGUAGE
