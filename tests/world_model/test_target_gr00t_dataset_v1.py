"""Synthetic supervision/split/time-boundary tests; no native demonstrations."""
import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from go2arx5_vla.world_model import target_gr00t_dataset_v1 as ds
from go2arx5_vla.world_model import target_gr00t_collection_v1 as collect
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_stable_task_dataset_training_v1 import qualification


def test_only_consumed_commands_enter_target_action_labels():
    q=qualification()
    a=ds.actual_commands(q)
    for row in q['generated_segment']['windows']:row['actual_TCP'][0][3]=999.
    np.testing.assert_array_equal(ds.actual_commands(q),a)
    q['generated_segment']['windows'][-1]['solver_policies_with_exact_installed_target']=[]
    with pytest.raises(ValueError,match='SOLVER_GRIPPER'):ds.actual_commands(q)


def synthetic():
    state=np.tile([0,0,.3,1,0,0,0,1,0,1.],(27,1))
    command=np.tile(state[0],(40,1));command[:,0]=np.linspace(0,.004,40)
    return dict(packet=dict(actual_state10=state,committed_absolute10=command,
        native_command_anchor=np.asarray(collect.CAPTURE_ATOMS),
        valid_window=np.arange(27)==0),row=dict(split='TRAIN',purpose='DEMONSTRATION'),
        qualification_only=False)


def test_statistics_use_training_only_and_executed_window_grid():
    e=synthetic();s=ds.fit_statistics([e])
    assert s['new_embodiment']['state']['eef_9d']['mean'][2]==pytest.approx(.3)
    for patch in (dict(row=dict(split='DEV',purpose='DEMONSTRATION')),
                  dict(row=dict(split='TRAIN',purpose='QUALIFICATION')),
                  dict(qualification_only=True)):
        with pytest.raises(ValueError,match='TRAIN_ONLY'):ds.fit_statistics([dict(e,**patch)])


def dataset_fixture(tmp_path):
    root=tmp_path/'real-looking-software-fixture';root.mkdir()
    p=synthetic()['packet'];pic=root/'image.png'
    Image.fromarray(np.zeros((256,256,3),np.uint8)).save(pic)
    p.update(instruction=np.asarray(collect.LANGUAGES[collect.COMMANDS[0]]),
        rgb_paths=np.full((27,2),'image.png'),rgb_sha256=np.full((27,2),digest(pic)),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
    np.savez(root/'episode.npz',**p)
    (root/'statistics.json').write_text(json.dumps(ds.fit_statistics([synthetic()])))
    row=dict(id='SYNTHETIC_NOT_NATIVE',split='TRAIN',windows=1,file='episode.npz',sha256=digest(root/'episode.npz'))
    (root/'DATASET_MANIFEST.json').write_text(json.dumps(dict(schema=ds.SCHEMA,domain=ds.DOMAIN,profile='TARGET_GR00T_V1',
        statistics_sha256=digest(root/'statistics.json'),episodes=[row])))
    return root


def test_target_loader_preserves_50ms_actions_and_sparse_same_clock_rgb(tmp_path):
    root=dataset_fixture(tmp_path);d=ds.TargetNativeDataset(root,'unused','unused')
    assert len(d)==1
    _,commands,images,text=d.raw(0)
    assert commands.shape==(40,10) and commands[0,0]==0
    assert len(images)==2 and text==collect.LANGUAGES[collect.COMMANDS[0]]
    assert commands[-1,0]==pytest.approx(.004)
    with pytest.raises(IndexError):d.raw(1)
    with pytest.raises(FileNotFoundError):ds.TargetNativeDataset(root,'unused','unused',for_optimization=True)
    with pytest.raises(ValueError,match='NO_DEV'):ds.TargetNativeDataset(root,'unused','unused','DEV',for_optimization=True)


def test_old_source_or_wm_manifest_is_not_target_vla_data(tmp_path):
    root=dataset_fixture(tmp_path);path=root/'DATASET_MANIFEST.json';v=json.loads(path.read_text())
    for key,value in [('domain','FIXED_BASE_ARX5_TASK_SIM'),('schema','WM02_REAL_STABLE_TASK_SUPERVISION_V1')]:
        path.write_text(json.dumps(dict(v,**{key:value})))
        with pytest.raises(ValueError,match='NOT_OLD_SOURCE'):ds.TargetNativeDataset(root,'unused','unused')


def test_teacher_policy_boundary_undoes_no_grip_or_time_mapping_twice(tmp_path):
    from go2arx5_vla.world_model.stable_start_short_task_v1 import build_for_config
    actual=np.eye(4);cube=np.eye(4);cube[0,3]=.2;base=np.array([0,0,0,1,0,0,0.])
    program,proposal=build_for_config(actual,actual,base,cube,collect.config(collect.COMMANDS[0],'TRAIN'))
    n=tmp_path/'native-01';n.mkdir()
    def put(name,value):
        def plain(x):
            if isinstance(x,dict):return{k:plain(v)for k,v in x.items()}
            if isinstance(x,np.ndarray):return x.tolist()
            return x
        p=n/name;p.write_text(json.dumps(plain(value)));return digest(p)
    pin=put('STABLE_TASK_EXPERT_PROPOSAL.private.json',proposal)
    put('LIVE_SKILL_SEGMENT_DECISION.private.json',dict(raw_H40=program.decoded_H40,
        desired_path=program.desired_path,nominal_reference_path=program.reference_path))
    q=qualification();q['executed_reference_proposal_sha256']=pin;q['adaptation']=program.evidence()
    q['generated_segment']['original_raw_H40']=program.decoded_H40.tolist()
    for i,row in enumerate(q['generated_segment']['windows']):
        row['actual_installed_consumed_EEF10_BASE7']=np.r_[program.preview(i)[0],base].tolist()
    policy,proof=ds.policy_commands(tmp_path,q)
    assert len(policy)==40 and np.all(policy[:,9]==1)
    assert ds.actual_commands(q)[-1,9]==pytest.approx(.042/.044)
    assert proof['policy_decisions']==1 and proof['renewed_previews_as_additional_expert_decisions']==0
    assert proof['raw_policy_grip_preserved_in_label']is True
    # Altering achieved TCP has no effect on the expert policy command.
    for row in q['generated_segment']['windows']:row['actual_TCP'][0][3]=999
    np.testing.assert_array_equal(ds.policy_commands(tmp_path,q)[0],policy)
    i=program.knot_indices[10]
    q['generated_segment']['windows'][i]['actual_installed_consumed_EEF10_BASE7'][0]+=.000001
    with pytest.raises(ValueError,match='ACTUALLY_COMPILED'):ds.policy_commands(tmp_path,q)
