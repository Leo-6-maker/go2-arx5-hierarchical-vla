"""Cold PORT02 contracts and real closed-cost capture; zero simulation."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask import cabinet_repair02 as r,night_scope as s,source_stage_bundle as bundle
from scripts.pi0294.multitask.robocasa_cabinet_cards import cards as old_cards
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher
from scripts.pi0294.multitask.night_book_chain import checked_plan


def test_versioned_pose_change_keeps_original_cards_fixture_ranges_full_duration_and_scores():
    before=old_cards();new=r.cards()
    for side,c in new.items():
        old=before[side]
        assert r.checked_card(old,side)is False and r.checked_card(c,side)is True
        assert c['source_port']==old['source_port']and c['independent_demo_conditions']==old['independent_demo_conditions']
        assert c['warmup_s']==old['warmup_s']==5. and c['task_duration_s']==old['task_duration_s']==32.
        assert c['cabinet']['size_m']==old['cabinet']['size_m']
        for k in('initial_angle_rad','target_angle_rad','angle_limits_rad','handle_radius_m','handle_length_m'):
            assert c['door'][k]==old['door'][k]
        assert r.attempt_operation(c,side)!=r.attempt_operation(old,side)
    assert old_cards()==before and new['left']['initial_tcp']==new['right']['initial_tcp']
    mixed=dict(BOOK=books(),ROBOCASA_CABINET=new)
    assert bundle.hashlib.sha256(bundle.raw(mixed)).hexdigest()==bundle.REPAIR02_CARD_SHA


@pytest.mark.parametrize('held,opening,expected',[ (True,1.,'release'),(False,.5,'release'),(False,1.,'withdraw')])
def test_wrist_return_only_after_measured_release_and_open_gripper(held,opening,expected):
    c=r.cards()['right'];teacher=SourceSkillTeacher(c);teacher.stage='release'
    tcp=np.eye(4);tcp[:3,3]=[.3,0.,.59];obj=np.eye(4)
    teacher.observe(dict(tcp=tcp,object_pose=obj,time_s=5.,gripper_fraction=opening,
        native_grasp=held,safety_valid=True,hinge_angle_rad=c['door']['target_angle_rad'],hinge_velocity_rad_s=0.))
    assert teacher.stage==expected
    if expected=='withdraw':
        np.testing.assert_allclose(teacher.withdraw[:3,:3],np.asarray(c['initial_tcp'])[:3,:3])
        assert np.linalg.norm(teacher.withdraw[:3,3]-tcp[:3,3])==pytest.approx(.13)
    else:assert teacher.withdraw is None


def test_caller_cannot_change_pose_after_freeze_or_claim_mixed_stage(tmp_path):
    c=r.cards()['right'];bad=deepcopy(c);bad['withdraw_tcp_rotation_world']=np.eye(3).tolist()
    with pytest.raises(PermissionError):SourceSkillTeacher(bad)
    fixed=dict(BOOK=books(),ROBOCASA_CABINET=r.cards())
    (tmp_path/'FIXED_TASK_CARDS.json').write_bytes(bundle.raw(fixed))
    (tmp_path/'ASSET_BINDINGS.private.json').write_text(json.dumps(dict(
        robocasa_definition_members={k:k for k in bundle.CABINET_XML},
        robocasa_texture_members=dict(wood='wood',metal='metal'),
        files={k:{}for k in [*bundle.CABINET_XML,'wood','metal']})))
    assert checked_plan(tmp_path,'ROBOCASA_CABINET')==(('ROBOCASA_CABINET','right'),('ROBOCASA_CABINET','left'))
    fixed['ROBOCASA_CABINET']['left']['task_duration_s']=31.
    (tmp_path/'FIXED_TASK_CARDS.json').write_bytes(bundle.raw(fixed))
    with pytest.raises(PermissionError):checked_plan(tmp_path,'ROBOCASA_CABINET')


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Exact native closed cabinet capture required')
    p=Path(directory);v=json.loads((p/'CABINET_REPAIR02_CLOSED_PREDECESSOR_ACTUAL.private.json').read_bytes())
    bodies={k:base64.b64decode(x['base64'],validate=True)for k,x in v['records'].items()}
    auth=(p.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    return bodies,auth,'/var/cache/source/20261030-cabinet-repair02-night-source-four01'


def test_exact_native_two_attempt_pool_and_all_unknown_uppers_carry_without_refund(actual):
    bodies,auth,dest=actual;v=r.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-3600)
    old=json.loads(bodies['account']);assert v['charged_upper']==old['charged_upper']and v['operations']==old['operations']
    assert v['charged_upper']['attempts']==2 and v['charged_upper']['physics_integrations']==100000
    closed=r.transfer(bodies['account'],bundle.raw(v),destination=dest)
    assert closed['custody']=='CLOSED_CARRIED_TO_ONE_EVIDENCED_CABINET_REPAIR02'
    assert closed['charged_upper']==old['charged_upper']and closed['operations']==old['operations']
    ns={};exec(compile(s.__loader__.get_source(s.__name__),'<actual-inline-prestage-scope>','exec'),ns)
    assert ns['validate_account'](v)==v


@pytest.mark.parametrize('fault',['refund','drop_old','old_attempt','book_attempt','third_repair','deadline','different_stop'])
def test_cannot_reset_pool_replay_old_attempt_or_bypass_closed_evidence(actual,fault):
    bodies,auth,dest=actual
    if fault=='different_stop':
        with pytest.raises(PermissionError):r.account(dict(bodies,stop=b'{}'),auth,destination=dest,now=s.SUCCESSOR_STOP-1)
        return
    v=r.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-1)
    if fault=='refund':v['charged_upper']['physics_integrations']-=1
    elif fault=='drop_old':v['operations']=v['operations'][1:]
    elif fault=='deadline':v['deadline_epoch']+=1
    else:
        op={'old_attempt':'ROBOCASA_RIGHT_FIRST01','book_attempt':'LIBERO_RIGHT_FIRST01','third_repair':'ROBOCASA_RIGHT_REPAIR03'}[fault]
        v['operations'].append(dict(kind='ATTEMPT',task='ROBOCASA_OPEN_CABINET',operation=op))
    with pytest.raises(PermissionError):s.validate_account(v)


def test_real_complete_asset_payload_selects_only_exact_new_card_sha(actual):
    directory=Path(os.environ['NIGHT_SOURCE_PRIVATE_CAPTURE_DIR'])
    p=deepcopy(json.loads((directory/'CABINET_UNRUN_PREPARED.private.json').read_bytes())['payload'])
    p['cards']=dict(BOOK=books(),ROBOCASA_CABINET=r.cards());p['cabinet_port_revision']=r.VERSION
    p['qualification'].update(fixed_task_cards_sha256=bundle.REPAIR02_CARD_SHA,cabinet_port_revision=r.VERSION)
    assert len(bundle.unpack(p)['plan'])==2
    for field in('cabinet_port_revision',):
        bad=deepcopy(p);del bad[field]
        with pytest.raises(PermissionError):bundle.unpack(bad)
    p['cards']['ROBOCASA_CABINET']['right']['door']['target_angle_rad']-=.1
    p['qualification']['fixed_task_cards_sha256']=bundle.hashlib.sha256(bundle.raw(p['cards'])).hexdigest()
    with pytest.raises(PermissionError):bundle.unpack(p)
