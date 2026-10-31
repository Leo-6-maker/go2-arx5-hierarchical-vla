from copy import deepcopy
import numpy as np
import pytest

from scripts.pi0294.multitask import book_recovery as r
from scripts.pi0294.multitask.book_policy import dev_card
from scripts.pi0294.multitask.libero_book_teacher import route_card, RELEASE_REVISION


def test_recovery_keeps_train_layout_home_goal_and_original_duration():
    card=r.recovery_card(r.CONDITION)
    original=route_card(RELEASE_REVISION)
    for name in ('initial_tcp','initial_book_native_world','book','warmup_s','task_duration_s'):
        assert card[name]==original[name]
    assert card['study_split']=='TRAIN_RECOVERY_DIAGNOSTIC'
    with pytest.raises(ValueError):r.recovery_card(dict(id='BOOK_DEV000',xy_offset_m=[.006,0]))
    assert dev_card(dict(id='BOOK_DEV000',xy_offset_m=[.006,0]))['study_split']=='DEV'


def test_exact_handover_uses_real_state_and_never_queries_student_afterward():
    records=[];experts=[];states=[]
    class Base:
        def __init__(self,model,identifier,record,guard,*,seed_namespace):
            assert seed_namespace=='BOOK_RECOVERY_TRAIN32'
            self.index=-1;self.queries=self.completed_generations=0
            self.guard=guard;self.state=None
        def observe(self,state):self.index+=1;self.state=np.asarray(state['state10'])
        def plan(self):self.queries+=1;self.completed_generations+=1;return 'STUDENT'
    class Expert:
        stage='approach';candidate_complete=False
        def __init__(self,card):experts.append(self);self.observed=[]
        def observe(self,state):self.observed.append(deepcopy(state))
        def plan(self):return 'EXPERT'
    student=r.make_student(Base,None,r.CONDITION['id'],lambda *a:records.append(a),lambda:None,teacher_type=Expert)
    plans=[]
    for i in range(41):
        state=dict(state10=[i]*10,safety_valid=True,native_grasp=False,time_s=5+i*.05)
        states.append(state);student.observe(state)
        if i%8==0:plans.append(student.plan())
    assert plans==['STUDENT']*4+['EXPERT']*2
    assert student.queries==student.completed_generations==4 and len(experts)==1
    assert experts[0].observed[0]==states[32] and len(experts[0].observed)==9
    assert records[0][0]=='expert_takeover'and records[0][1]['index']==32
    assert records[0][1]['student_prefix_is_expert']is False


def test_only_complete_safe_success_expert_suffix_can_be_eligible():
    indices=list(range(640))
    eligible=r.eligible_recovery_anchors(indices,full_task_success=True,native_safe=True)
    assert eligible==tuple(range(32,601))and len(eligible)==569
    assert r.eligible_recovery_anchors(indices,full_task_success=False,native_safe=True)==()
    assert r.eligible_recovery_anchors(indices,full_task_success=True,native_safe=False)==()
    with pytest.raises(ValueError):r.eligible_recovery_anchors(indices[:-1],full_task_success=True,native_safe=True)
    with pytest.raises(ValueError):r.eligible_recovery_anchors([0]+indices[1:-1]+[638],full_task_success=True,native_safe=True)


def test_default_factory_is_the_actual_qualified_routed_release_teacher():
    from scripts.pi0294.multitask.book_policy import Student
    from scripts.pi0294.multitask.libero_book_teacher import RoutedLiberoBookTeacher
    card=route_card(RELEASE_REVISION)
    student=r.make_student(Student,None,r.CONDITION['id'],lambda *a:None,lambda:None)
    student.index=31;student.queries=student.completed_generations=4
    state=dict(tcp=np.asarray(card['initial_tcp']),object_pose=np.asarray(card['initial_book_native_world']),
        safety_valid=True,native_grasp=False,gripper_fraction=.95,time_s=6.6)
    student.observe(state)
    assert isinstance(student.expert,RoutedLiberoBookTeacher)
    assert student.expert.card==card
    plan=student.plan()
    assert plan['poses'].shape==(40,4,4)
    np.testing.assert_array_equal(plan['poses'][0],state['tcp'])


def test_diagnostic_does_not_silently_expand_to_dev_or_more_model_queries():
    manifest=dict(kind='V7_BOOK_TRAIN_RECOVERY_DIAGNOSTIC',recovery_plan=r.plan(),
        delta_checkpoint_manifest_sha256=r.CHECKPOINT,
        limits=dict(optimizer_updates=0,physics_integrations=18500,generation_queries=4,
            RGB_pairs=642,source_constructors=1,model_loads=1))
    r.validate_manifest(manifest)
    changed=deepcopy(manifest);changed['limits']['generation_queries']=80
    with pytest.raises(PermissionError):r.validate_manifest(changed)
    changed=deepcopy(manifest);changed['recovery_plan']['condition']['id']='BOOK_DEV000'
    with pytest.raises(PermissionError):r.validate_manifest(changed)
