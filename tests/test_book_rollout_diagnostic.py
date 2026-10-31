import hashlib
from copy import deepcopy
import numpy as np
import pytest
from scripts.pi0294.multitask.book_policy import Student
from scripts.pi0294.multitask import book_rollout_diagnostic as d
from scripts.pi0294.multitask.capture_core import capture
from scripts.pi0294.multitask.libero_book_teacher import route_card,seed_card,RELEASE_REVISION


def test_train_cards_reproduce_seen_tasks_and_reject_dev():
    for condition in d.conditions():
        actual=d.card(condition);p=condition['TRAIN_parent']
        expected=route_card(RELEASE_REVISION)if p==0 else seed_card(p)
        for key in('initial_tcp','initial_book_native_world','book','book_route','instruction','task_duration_s'):
            assert actual[key]==expected[key]
        assert actual['study_split']=='TRAIN_AUTONOMOUS_DIAGNOSTIC'
    with pytest.raises(ValueError):d.card(dict(id='BOOK_DEV000',TRAIN_parent=0,xy_offset_m=[.006,0]))
    m=dict(kind=d.KIND,rollout_diagnostic=d.plan(),delta_checkpoint_manifest_sha256=d.B512,
        limits=dict(model_loads=1,optimizer_updates=0,physics_integrations=55500,
            generation_queries=240,RGB_pairs=1926,source_constructors=3))
    d.validate_manifest(m)
    bad=deepcopy(m);bad['rollout_diagnostic']['expert_takeover']=True
    with pytest.raises(PermissionError):d.validate_manifest(bad)
    bad=deepcopy(m);bad['rollout_diagnostic']=d.plan(4)
    with pytest.raises(PermissionError):d.validate_manifest(bad)


def test_paired_feedback_seeds_follow_observation_time_and_default_seed_is_unchanged():
    a=d.student_factory(Student,8)(None,'BOOK_TRAIN000',None,None)
    b=d.student_factory(Student,4)(None,'BOOK_TRAIN000',None,None)
    for index in range(0,640,8):
        a.index=b.index=index
        assert a.generation_seed(index//8)==b.generation_seed(index//4)
    old=Student(None,'BOOK_DEV000',None,None)
    expected=int.from_bytes(hashlib.sha256(b'BOOK5_DEV10/BOOK_DEV000/5').digest()[:8],'big')%(2**63)
    assert old.generation_seed(5)==expected


@pytest.mark.parametrize('prefix,plan_indices',[ (8,[0]),(4,[0,4]) ])
def test_capture_replans_from_fresh_observation_and_installs_only_declared_prefix(prefix,plan_indices):
    rows=[]
    class Engine:
        steps=2500
        def snapshot(self):return dict(time_s=self.steps*.002,tcp=np.eye(4),gripper_fraction=.9,safety_valid=True)
        def render(self,state):
            image=np.zeros((256,256,3),np.uint8)
            calibration=dict(timestamp_s=state['time_s'],T_world_optical=np.eye(4),K=np.eye(3))
            return dict(body_rgb=image,wrist_rgb=image),dict(body_rgb=calibration,wrist_rgb=calibration)
        def step(self,target,opening):self.steps+=1
    class Policy:
        index=-1;stage='STUDENT';candidate_complete=False
        def observe(self,state):self.index+=1
        def plan(self):
            assert self.index in plan_indices
            poses=np.repeat(np.eye(4)[None],40,axis=0);poses[:,0,3]=self.index+np.arange(40)*.001
            return dict(poses=poses,opening=np.full(40,.9))
    result=capture(Engine(),Policy(),instruction='Put the book away',guard=lambda:None,
        emit=lambda k,v:rows.append((k,v)),duration_s=.4,execution_prefix=prefix)
    assert result['physics_integrations']==200 and result['completed_intervals']==8
    assert [v['observation_index']for k,v in rows if k=='planned_H40']==plan_indices
    completed=[v for k,v in rows if k=='completed_interval']
    for i,c in enumerate(completed):
        assert c['prefix_index']==i%prefix
        assert c['action_EEF10'][0]==pytest.approx(i-i%prefix+(i%prefix)*.001)
    assert result['H40_plans']==len(plan_indices)
