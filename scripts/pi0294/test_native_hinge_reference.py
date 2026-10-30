"""Native saved grasp boundary and original future range, no simulation."""
import base64,json,os
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask.native_hinge_reference import from_native_boundary
from scripts.pi0294.multitask.object_relations import door_turn
from scripts.pi0294.multitask import cabinet_port04 as c,cabinet_camera_port03 as old,night_scope as s,source_stage_bundle as b
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher


def args(q=-.0012749542955516303,target=1.4915,limits=(0.,1.57)):
    tool=np.eye(4);tool[:3,3]=[.2,0,0]
    return dict(hinge_world=np.eye(4),axis_hinge=[0,0,1],actual_tcp_world=tool,
        actual_handle_world=tool.copy(),actual_angle_rad=q,target_angle_rad=target,
        actual_grasp_valid=True,native_safety_valid=True,angle_limits_rad=limits)


@pytest.mark.parametrize('direction',[1,-1])
def test_measured_anchor_is_not_clamped_and_every_future_angle_obeys_original_range(direction):
    a=args(q=-direction*.0012749542955516303,target=direction*1.4915,
        limits=(0.,1.57)if direction==1 else(-1.57,0.))
    value=from_native_boundary(**a);low,high=a['angle_limits_rad']
    np.testing.assert_allclose(value.tcp_world[0],a['actual_tcp_world'],atol=1e-12)
    assert value.hinge_angles_rad[0]==a['actual_angle_rad']
    assert np.all(value.hinge_angles_rad[1:]>=low)and np.all(value.hinge_angles_rad[1:]<=high)
    assert np.max(np.abs(np.diff(value.hinge_angles_rad)))/.05<=.3+1e-12
    assert np.max(np.linalg.norm(np.diff(value.tcp_world[:,:3,3],axis=0),axis=1))/.05<=.06+1e-12


def test_existing_in_range_reference_is_numerically_unchanged():
    a=args(q=.12);expected=door_turn(**{k:v for k,v in a.items()if k!='native_safety_valid'})
    actual=from_native_boundary(**a)
    np.testing.assert_array_equal(actual.tcp_world,expected.tcp_world)
    np.testing.assert_array_equal(actual.hinge_angles_rad,expected.hinge_angles_rad)


@pytest.mark.parametrize('fault',['unsafe','lost_grasp','target_outside','too_far_outside','nonfinite'])
def test_no_safety_bypass_joint_range_expansion_or_unbounded_boundary_rescue(fault):
    a=args()
    if fault=='unsafe':a['native_safety_valid']=False
    elif fault=='lost_grasp':a['actual_grasp_valid']=False
    elif fault=='target_outside':a['target_angle_rad']=1.6
    elif fault=='too_far_outside':a['actual_angle_rad']=-.02
    else:a['actual_angle_rad']=float('nan')
    with pytest.raises(ValueError):from_native_boundary(**a)


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Actual native PORT03 saved state required')
    p=Path(directory);v=json.loads((p/'CABINET_PORT03_GRASP_HINGE_REJECTION_ACTUAL.private.json').read_bytes())
    bodies={k:base64.b64decode(x['base64'],validate=True)for k,x in v['chain_and_lease_records'].items()}
    authority=(p.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    return v,bodies,authority


def test_actual_saved_boundary_reproduces_old_rejection_and_new_exact_anchored_plan(actual):
    v,_,_=actual;state=v['final_native_state'];spec=v['card']['door']
    a=dict(hinge_world=spec['hinge_world'],axis_hinge=[0,0,1],actual_tcp_world=state['tcp'],
        actual_handle_world=state['object_pose'],actual_angle_rad=state['hinge_angle_rad'],
        target_angle_rad=spec['target_angle_rad'],actual_grasp_valid=state['native_grasp'],angle_limits_rad=spec['angle_limits_rad'])
    with pytest.raises(ValueError,match='DECLARED_HINGE_LIMITS'):door_turn(**a)
    ref=from_native_boundary(**a,native_safety_valid=state['safety_valid'])
    assert ref.hinge_angles_rad[0]==state['hinge_angle_rad']and ref.hinge_angles_rad[1]>0
    np.testing.assert_allclose(ref.tcp_world[0],state['tcp'],atol=1e-12)
    teacher=SourceSkillTeacher(c.cards()['right']);teacher.stage='turn';teacher.observe(state);plan=teacher.plan()
    np.testing.assert_allclose(plan['poses'][0],state['tcp'],atol=1e-12)
    for side,row in c.cards().items():
        prior=old.cards()[side]
        for key in('id','port_revision'):row.pop(key);prior.pop(key)
        row.pop('hinge_boundary_reference');assert row==prior
    assert b.hashlib.sha256(b.raw(dict(BOOK=books(),ROBOCASA_CABINET=c.cards()))).hexdigest()==b.PORT04_CARD_SHA


def test_actual_four_attempt_account_is_carried_once_with_every_cost_and_clock(actual):
    _,bodies,authority=actual;dest='/var/cache/source/20261030-cabinet-port04-night-source-four01'
    v=c.account(bodies,authority,destination=dest,now=s.SUCCESSOR_STOP-1);before=json.loads(bodies['account'])
    assert v['charged_upper']==before['charged_upper']and v['operations']==before['operations']
    assert v['charged_upper']['attempts']==4 and v['charged_upper']['physics_integrations']==200000
    closed=c.transfer(bodies['account'],b.raw(v),destination=dest)
    assert closed['operations']==before['operations']and closed['charged_upper']==before['charged_upper']
    ns={};exec(compile(s.__loader__.get_source(s.__name__),'<inline-scope>','exec'),ns);assert ns['validate_account'](v)==v
