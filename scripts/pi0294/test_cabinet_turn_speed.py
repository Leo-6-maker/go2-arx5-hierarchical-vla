"""Slower command only; original grasp rejection and full32s score retained."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask import cabinet_port05 as c,cabinet_port04 as old,night_scope as s,source_stage_bundle as b
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher
from scripts.pi0294.multitask.cabinet_repair02 import checked_card,attempt_operation


def test_only_explicit_turn_speed_changes_and_old_score_home_fixture_and_grip_are_preserved():
    for side,card in c.cards().items():
        before=old.cards()[side];assert checked_card(card,side)
        assert attempt_operation(card,side)=='ROBOCASA_'+side.upper()+'_PORT05_SLOW_TURN'
        assert card['warmup_s']==5. and card['task_duration_s']==32.
        assert card.pop('turn_reference_limits')==dict(max_linear_m_s=.03,max_angular_rad_s=.15)
        for k in('id','port_revision'):card.pop(k);before.pop(k)
        assert card==before
    assert b.hashlib.sha256(b.raw(dict(BOOK=books(),ROBOCASA_CABINET=c.cards()))).hexdigest()==b.PORT05_CARD_SHA


def test_caller_cannot_posthoc_change_reference_or_grasp_fields():
    card=c.cards()['right'];card['turn_reference_limits']['max_linear_m_s']=.04
    with pytest.raises(PermissionError):SourceSkillTeacher(card)


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Actual PORT03/04 closed native captures required')
    p=Path(directory);prior=json.loads((p/'CABINET_PORT03_GRASP_HINGE_REJECTION_ACTUAL.private.json').read_bytes())
    loss=json.loads((p/'CABINET_PORT04_CONTACT_SPEED_ACTUAL.private.json').read_bytes())
    v=json.loads((p/'CABINET_PORT05_CLOSED_PREDECESSOR_ACTUAL.private.json').read_bytes());bodies={k:base64.b64decode(v['base64'],validate=True)for k,v in v['records'].items()}
    auth=(p.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    return p,prior,loss,bodies,auth


def test_actual_pre_loss_grasp_has_half_speed_and_exact_anchor_but_lost_state_still_stops(actual):
    _,prior,loss,_,_=actual;card=c.cards()['right'];teacher=SourceSkillTeacher(card);teacher.stage='turn'
    state=prior['final_native_state'];teacher.observe(state);teacher.plan();path=teacher.path
    assert np.max(np.linalg.norm(np.diff(path[:,:3,3],axis=0),axis=1))/.05<=.03+1e-12
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    assert max(geodesic_angle(a[:3,:3],b[:3,:3])for a,b in zip(path[:-1],path[1:]))/.05<=.15+1e-12
    np.testing.assert_allclose(path[0],state['tcp'],atol=1e-12)
    assert (len(path)-1)*.05<32.
    failed=SourceSkillTeacher(card);failed.stage='turn'
    with pytest.raises(PermissionError,match='ACTUAL_GRASP_LOST'):
        failed.observe(loss['final_actual_state'])


def test_exact_five_attempt_pool_and_native_safety_audit_are_retained_once(actual):
    _,_,_,bodies,auth=actual;dest='/var/cache/source/20261030-cabinet-port05-night-source-four01'
    v=c.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-1);before=json.loads(bodies['account'])
    assert v['charged_upper']==before['charged_upper']and v['operations']==before['operations']
    assert before['charged_upper']['attempts']==5 and before['charged_upper']['physics_integrations']==250000
    closed=c.transfer(bodies['account'],b.raw(v),destination=dest)
    assert closed['charged_upper']==before['charged_upper']and closed['operations']==before['operations']
    ns={};exec(compile(s.__loader__.get_source(s.__name__),'<inline-scope>','exec'),ns);assert ns['validate_account'](v)==v
    bad=dict(bodies,audit=b'{}')
    with pytest.raises(PermissionError):c.account(bad,auth,destination=dest,now=s.SUCCESSOR_STOP-1)


def test_same_failure_cannot_become_another_unbounded_slow_turn_retry(actual):
    _,_,_,bodies,auth=actual;dest='/var/cache/source/20261030-cabinet-port05-night-source-four01'
    v=c.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-1)
    v['operations'].append(dict(kind='ATTEMPT',task='ROBOCASA_OPEN_CABINET',operation='ROBOCASA_RIGHT_PORT06_SLOW_TURN'))
    with pytest.raises(PermissionError):s.validate_account(v)


def test_full_original_asset_bundle_still_required_for_the_slow_version(actual):
    p,_,_,_,_=actual;payload=deepcopy(json.loads((p/'CABINET_REPAIR02_PREPARED.private.json').read_bytes())['payload'])
    payload.update(cards=dict(BOOK=books(),ROBOCASA_CABINET=c.cards()),cabinet_port_revision=c.VERSION)
    payload['qualification'].update(cabinet_port_revision=c.VERSION,fixed_task_cards_sha256=b.PORT05_CARD_SHA)
    assert len(b.unpack(payload)['plan'])==2
