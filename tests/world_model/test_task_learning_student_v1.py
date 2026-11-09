"""Pure student proposals and panel checks; no native execution or model load."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import task_learning_student_v1 as student
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_dev_v1 import DEV_OFFSETS
from go2arx5_vla.world_model.task_learning_scene_v1 import scene_id
from go2arx5_vla.world_model.target_gr00t_profile_v1 import PARENT
from go2arx5_vla.world_model.storage import digest
from scripts.pi0291_arx5_data.native_dataset import pose10


def panel():
    policies={m:dict(arm=m,parent_manifest_sha256=PARENT,package=None,
        Core_qualification=None)for m in('B0','B1')}
    policies['B1'].update(package=dict(path='/fixture',sha256='f'*64),
        Core_qualification=dict(path='/fixture/proof',sha256='e'*64))
    rows=[]
    for i,layout in enumerate(DEV_OFFSETS):
        for restart in(1,2):
            for method in(('B0','B1')if restart==1 else('B1','B0')):
                rows.append(dict(native_run=f'{method}-{layout}-r{restart}',method=method,
                    layout=layout,parent=scene_id(layout),restart=restart,split='DEV',
                    purpose=student.PURPOSE,command=task.COMMAND,model_sha256='a'*64,
                    query_seeds=list(range(1000*i+100*restart,1000*i+100*restart+15))))
    return dict(schema=student.SCHEMA,task_learning_role=task.STUDENT_DEV_ROLE,
        protocol=task.protocol(task.STUDENT_DEV_ROLE),source_sha='b'*40,policies=policies,assignments=rows,
        B1_global_step=2000)


def test_panel_retains_forty_conditions_for_each_frozen_method_and_matched_noise():
    p=panel();assert len(student.validate(p))==80
    for change in('missing','seed','teacher','policy'):
        q=copy.deepcopy(p)
        if change=='missing':q['assignments'].pop()
        elif change=='seed':q['assignments'][0]['query_seeds'][0]+=40
        elif change=='teacher':q['assignments'][0]['Teacher_mode']='NORMAL_TEACHER_V7'
        else:q['policies']['B0']['package']=q['policies']['B1']['package']
        with pytest.raises(ValueError):student.validate(q)


def action_file(tmp_path,grip=.5):
    actual=np.eye(4);actual[:3,3]=[.26,-.02,.70]
    frame=np.eye(4);frame[:3,3]=[.1,.2,.3]
    poses=np.repeat(actual[None],40,axis=0)
    # A real first-knot error must survive the frame map and reference transport.
    poses[:,0,3]+=np.linspace(.0001,.002,40)
    raw=pose10(frame@poses,np.full(40,grip))
    action=tmp_path/'ACTION.npz';np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,132)))
    reference=actual.copy();reference[:3,3]+=[.002,.003,.004]
    bindings=dict(source_from_target=frame,query_state10=pose10(frame@actual,.5),
        actual_TCP=actual,consumed_reference_TCP=reference,base_hold=[0,0,.3,1,0,0,0])
    return action,poses,bindings


def test_one_frame_map_keeps_first_knot_error_and_original_gripper(tmp_path):
    path,poses,bindings=action_file(tmp_path)
    program,proof=student.compile_proposal(path,digest(path),**bindings)
    assert np.allclose(program.decoded_H40[:,:3],poses[:,:3,3],rtol=0,atol=1e-15)
    assert program.decoded_H40[0,0]-bindings['actual_TCP'][0,3]==pytest.approx(.0001)
    assert np.all(program.decoded_H40[:,9]==.5)
    assert np.allclose(program.reference_path[:,:3],bindings['consumed_reference_TCP'][:3,3]
        +program.decoded_H40[:,:3]-bindings['actual_TCP'][:3,3])
    assert proof['command_clips']==0 and proof['policy_frame_mapping_count']==1
    assert proof['expert_training_eligible']is False


def test_ood_gripper_and_stale_query_pose_are_rejected_without_clipping(tmp_path):
    path,_,bindings=action_file(tmp_path,grip=1.01)
    with pytest.raises(ValueError,match='WITHOUT_CLIPPING'):
        student.compile_proposal(path,digest(path),**bindings)
    path,_,bindings=action_file(tmp_path)
    bindings['query_state10'][0]+=.001
    with pytest.raises(ValueError,match='SAME_CAPTURED_POLICY_STATE'):
        student.compile_proposal(path,digest(path),**bindings)


def test_partial_dev_retains_all_requested_failures_and_unrun_denominators():
    p=panel();r=dict(p['assignments'][0],attempted=True,reset_ready=True,
        task_success=False,technical_valid=True,hard_safety_stop=False,actual_GR00T_queries=1)
    summary=student.summarize(p,[r])
    assert summary['complete']is False
    assert summary['methods']['B0']['requested']==40
    assert summary['methods']['B0']['attempted']==1 and summary['methods']['B0']['unrun']==39
    assert summary['methods']['B0']['P_success_per_requested_episode']==0
    assert summary['methods']['B1']['unrun']==40
    with pytest.raises(ValueError):student.summarize(p,[r,r])
    r['method']='B1'
    with pytest.raises(ValueError):student.summarize(p,[r])
