from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.pi0294.multitask import book_late_recovery as late
from scripts.pi0294.multitask.book_held_teacher import from_held_state
from scripts.pi0294.multitask.feedback_teacher import CLOSED
from scripts.pi0294.multitask.late_recovery_export import suffix_packet
from go2arx5_vla.task_data.codec import pose10


def state(index=80,rise=.08):
    card=late.original_card(late.conditions()[0]);book=np.eye(4);book[:3,3]=[.36,-.1,.55+rise]
    tcp=book@np.asarray(card['book']['object_to_tcp_grasp'])
    return dict(time_s=5+index*.05,tcp=tcp,object_pose=book,gripper_fraction=.045,
        safety_valid=True,native_abort=False,native_grasp=True,left_object_contact=True,
        right_object_contact=True,opposing_contact=True,relative_motion_valid=True,
        gripper_state_valid=True,object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),
        native_libero_goal=False,object_supported_by_shelf=False,robot_object_contact=True,
        qpos=np.zeros(15),qvel=np.zeros(14),finger_q=np.array([.002,.002]),finger_qd=np.zeros(2))


def test_no_approach_or_clock_restart_when_entering_actual_held_carry():
    condition=late.conditions()[2];card=late.original_card(condition);s=state()
    original=deepcopy(s);expert=from_held_state(card,s,source_index=80,grasp_since=s['time_s']-.4,initial_object_z=.55)
    assert expert.stage=='preinsert'and expert.initial_time==5 and expert.index==79
    expert.observe(s);p=expert.plan()
    assert expert.index==80 and p['stage']=='preinsert'and p['persistent_path_start_observation']==80
    np.testing.assert_allclose(p['poses'][0],s['tcp'],rtol=0,atol=1e-12);np.testing.assert_array_equal(p['opening'],np.full(40,CLOSED))
    for key in('qpos','qvel','tcp','object_pose','finger_q'):np.testing.assert_array_equal(s[key],original[key])
    with pytest.raises(ValueError):expert.observe(dict(s,time_s=s['time_s']))


def test_just_established_grasp_enters_lift_with_closed_first_prefix():
    s=state(96,.005);card=late.original_card(late.conditions()[0])
    expert=from_held_state(card,s,source_index=96,grasp_since=s['time_s']-.4,initial_object_z=.55)
    expert.observe(s);assert expert.stage=='lift'
    np.testing.assert_array_equal(expert.plan()['opening'][:8],np.full(8,CLOSED))


@pytest.mark.parametrize('key,value',[('native_grasp',False),('left_object_contact',False),
    ('safety_valid',False),('native_abort',True),('relative_motion_valid',False)])
def test_unsafe_or_no_actual_bilateral_grasp_never_creates_held_teacher(key,value):
    s=state();s[key]=value
    with pytest.raises(PermissionError):from_held_state(late.original_card(late.conditions()[0]),s,source_index=80,grasp_since=s['time_s']-.4,initial_object_z=.55)


def test_events_are_actual_held_height_and_support_not_time_or_local_suffix_index():
    s=state();c=late.conditions()[2]
    assert late.eligible_event(c,s,source_index=80,grasp_since=s['time_s']-.4,initial_object_z=.55)
    assert not late.eligible_event(c,s,source_index=81,grasp_since=s['time_s']-.4,initial_object_z=.55)
    assert not late.eligible_event(c,s,source_index=80,grasp_since=s['time_s']-.1,initial_object_z=.55)
    assert not late.eligible_event(c,s,source_index=80,grasp_since=s['time_s']-.4,initial_object_z=.629)
    near=late.conditions()[6];s['object_pose']=np.asarray(late.original_card(near)['book']['target_book_world']);s['native_libero_goal']=True;s['object_supported_by_shelf']=True
    assert not late.eligible_event(near,s,source_index=80,grasp_since=s['time_s']-.4,initial_object_z=s['object_pose'][2,3]-.08)


def test_factory_keeps_student_prefix_and_stops_student_queries_at_real_event(monkeypatch):
    records=[]
    class Base:
        def __init__(self,instance,identifier,record,guard,*,seed_namespace):
            self.index=79;self.queries=self.completed_generations=10;self.guard=guard
            assert seed_namespace==late.VERSION
        def observe(self,s):self.index+=1;self.state=pose10(np.asarray(s['tcp']),s['gripper_fraction'])
        def plan(self):raise AssertionError('NO_STUDENT_QUERY_AFTER_TAKEOVER')
    import scripts.pi0294.multitask.recovery_state as snapshot
    monkeypatch.setattr(snapshot,'capture_handover',lambda e,s:dict(sha256='1'*64,snapshot=dict(test_only=True)))
    c=late.conditions()[0];student=late.make_student(Base,None,c['id'],lambda *v:records.append(v),lambda:None)
    s=state(80,.005);student.bind_engine(SimpleNamespace());student.grasp_since=s['time_s']-.4;student.initial_object_z=.55
    student.observe(s);p=student.plan();status=student.recovery_status()
    assert student.expert.stage=='lift'and student.expert.index==80
    assert student.queries==student.completed_generations==10 and status['takeover_index']==80
    assert records[0][0]=='expert_takeover'and records[0][1]['student_prefix_is_expert']is False
    assert records[0][1]['full_state_unchanged_by_initialization']is True
    np.testing.assert_allclose(p['poses'][0],s['tcp'],rtol=0,atol=1e-12)


def test_suffix_drops_all_student_commands_and_keeps_legal_later_open_labels():
    t=160;commands=np.arange(6400,dtype=float).reshape(640,10);commands[:,9]=CLOSED;commands[400:,9]=.042/.044
    full=dict(actual_state10=np.arange(6410).reshape(641,10),committed_absolute10=commands,
        valid_window=np.arange(640)<601,rgb_paths=np.zeros((641,2),dtype='U1'),rgb_sha256=np.zeros((641,2),dtype='U1'),
        offset_s=np.arange(40)/20,instruction=np.asarray('book'),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))
    suffix=suffix_packet(full,t)
    np.testing.assert_array_equal(suffix['committed_absolute10'],commands[t:]);np.testing.assert_array_equal(suffix['actual_state10'],full['actual_state10'][t:])
    assert suffix['valid_window'].sum()==601-t and suffix['source_observation_index'][0]==t
    assert suffix['committed_absolute10'][400-t,9]==commands[400,9]
    assert late.eligible_recovery_anchors(late.conditions()[0],None,range(640),full_task_success=True,native_safe=True)==()
    assert late.eligible_recovery_anchors(late.conditions()[0],t,range(640),full_task_success=False,native_safe=True)==()
    with pytest.raises(ValueError):suffix_packet(full,161)


def test_pilots_are_different_events_and_science_cannot_expand_to_dev_or_different_policy():
    pilots=late.conditions('pilot');assert [c['event']for c in pilots]==['GRASP_CONFIRMED','LIFT_CARRY']
    assert len(late.conditions('remainder'))==6 and not set(c['id']for c in pilots)&set(c['id']for c in late.conditions('remainder'))
    p=late.plan('1'*64,'pilot');m=dict(kind=late.KIND,recovery_plan=p,delta_checkpoint_manifest_sha256='1'*64,
        limits=dict(optimizer_updates=0,physics_integrations=37000,generation_queries=160,RGB_pairs=1284,source_constructors=2,model_loads=1))
    late.validate_manifest(m);m['recovery_plan']['conditions'][0]['split']='DEV'
    with pytest.raises(PermissionError):late.validate_manifest(m)
