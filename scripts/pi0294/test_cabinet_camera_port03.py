"""One cold fixed-home camera correction, actual old charges and no physics."""
import base64,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.multitask import cabinet_camera_port03 as c,cabinet_repair02 as old,night_scope as s,source_stage_bundle as b
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher


def test_fixed_declared_home_only_camera_FOV_grasp_and_withdrawal_unchanged():
    for side,row in c.cards().items():
        prior=old.cards()[side];assert old.checked_card(row,side)
        assert old.attempt_operation(row,side)=='ROBOCASA_'+side.upper()+'_PORT03_CAMERA'
        teacher=SourceSkillTeacher(row)
        assert teacher.withdraw_rotation.tolist()==prior['withdraw_tcp_rotation_world']
        for key in('id','port_revision','initial_tcp'):row.pop(key);prior.pop(key)
        assert row==prior
    assert b.hashlib.sha256(b.raw(dict(BOOK=books(),ROBOCASA_CABINET=c.cards()))).hexdigest()==b.PORT03_CARD_SHA


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Exact closed PORT02 native capture required')
    p=Path(directory);v=json.loads((p/'CABINET_REPAIR02_PORT03_CLOSED_PREDECESSOR_ACTUAL.private.json').read_bytes())
    assert v['safety_STOP_absent']
    bodies={k:base64.b64decode(x['base64'],validate=True)for k,x in v['records'].items()}
    auth=(p.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    return p,bodies,auth,'/var/cache/source/20261030-cabinet-port03-night-source-four01'


def test_actual_closed_camera_failure_pool_all_three_attempts_retained(actual):
    p,bodies,auth,dest=actual;v=c.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-1);prior=json.loads(bodies['account'])
    assert v['charged_upper']==prior['charged_upper']and v['operations']==prior['operations']
    assert v['charged_upper']['attempts']==3 and v['charged_upper']['physics_integrations']==150000
    closed=c.transfer(bodies['account'],b.raw(v),destination=dest)
    assert closed['operations']==prior['operations']and closed['charged_upper']==prior['charged_upper']
    ns={};exec(compile(s.__loader__.get_source(s.__name__),'<same-inline-prestage>','exec'),ns)
    assert ns['validate_account'](v)==v
    payload=deepcopy(json.loads((p/'CABINET_REPAIR02_PREPARED.private.json').read_bytes())['payload'])
    payload.update(cards=dict(BOOK=books(),ROBOCASA_CABINET=c.cards()),cabinet_port_revision=c.VERSION)
    payload['qualification'].update(fixed_task_cards_sha256=b.PORT03_CARD_SHA,cabinet_port_revision=c.VERSION)
    assert len(b.unpack(payload)['plan'])==2


@pytest.mark.parametrize('fault',['refund','old_attempt','another_camera_retry','clock','old_carry'])
def test_changed_clock_replenished_pool_and_repeat_camera_retry_rejected(actual,fault):
    _,bodies,auth,dest=actual;v=c.account(bodies,auth,destination=dest,now=s.SUCCESSOR_STOP-1)
    if fault=='refund':v['charged_upper']['RGB_pairs']-=1
    elif fault=='clock':v['deadline_epoch']+=1
    elif fault=='old_carry':v[old.FIELD]=json.loads(bodies['account'])[old.FIELD]
    else:v['operations'].append(dict(kind='ATTEMPT',task='ROBOCASA_OPEN_CABINET',operation='ROBOCASA_RIGHT_REPAIR02'if fault=='old_attempt'else'ROBOCASA_RIGHT_PORT04_CAMERA'))
    with pytest.raises(PermissionError):s.validate_account(v)
