"""Bind the measured static repair to one new source condition, never a retry."""
from copy import deepcopy
import base64,json,os
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.multitask import cabinet_layout_port06 as candidate,cabinet_port05 as old
from scripts.pi0294.multitask import source_stage_bundle as bundle
from scripts.pi0294.multitask.cabinet_repair02 import checked_card,attempt_operation
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher
from scripts.pi0294.multitask.libero_book_cards import cards as books


def test_card_matches_actual_full_body_static_candidate_and_retains_task():
    r=Path(__file__).resolve().parents[1]
    proof=json.loads((r/'reports/PI0294/design/science-goal-v7-20261031/BOOK5_ALL3005_LEARNING_ACTIVE_E1_DECODE_AND_CABINET_LAYOUT_STATIC_ACTUAL_20261031.json').read_bytes())['cabinet']
    before=old.cards();after=candidate.cards();card=after['right']
    assert after['left']==before['left']and old.cards()==before
    assert checked_card(card,'right')and attempt_operation(card,'right')=='ROBOCASA_RIGHT_PORT06_LAYOUT40_RETRACT8'
    assert card['door']['object_to_tcp_grasp']==proof['candidate_object_to_tcp_grasp']
    for key in('initial_tcp','warmup_s','task_duration_s','independent_demo_conditions','turn_reference_limits','withdraw_tcp_rotation_world'):
        assert card[key]==before['right'][key]
    np.testing.assert_allclose(np.asarray(card['cabinet']['cabinet_world'])[:3,3]-np.asarray(before['right']['cabinet']['cabinet_world'])[:3,3],np.array(proof['candidate_translation_world_m']),rtol=0.,atol=1e-15)
    for key in('size_m','orientation'):assert card['cabinet'][key]==before['right']['cabinet'][key]
    for key in('target_angle_rad','angle_limits_rad','handle_radius_m','handle_length_m'):assert card['door'][key]==before['right']['door'][key]
    assert bundle.hashlib.sha256(bundle.raw(dict(BOOK=books(),ROBOCASA_CABINET=after))).hexdigest()==bundle.PORT06_CARD_SHA
    bad=deepcopy(card);bad['turn_reference_limits']['max_linear_m_s']*=2
    with pytest.raises(PermissionError):SourceSkillTeacher(bad)
    with pytest.raises(PermissionError):checked_card(card,'left')


def test_single_right_layout_cannot_expand_to_unqualified_left_or_old_run():
    tasks=[dict(task_kind='ROBOCASA_CABINET',side='right')]
    m=dict(task_kind='ROBOCASA_CABINET',condition_side='right',cabinet_port_revision=candidate.VERSION,
        predeclared_tasks=tasks,attempts_this_chain_max=1)
    assert bundle.checked_manifest(m)==('ROBOCASA_CABINET',tasks)
    for changes in(dict(condition_side=None),dict(condition_side='left'),dict(cabinet_port_revision=old.VERSION),dict(attempts_this_chain_max=2)):
        with pytest.raises(PermissionError):bundle.checked_manifest(dict(m,**changes))


def test_actual_minimum_asset_bundle_and_saved_grasp_boundary():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit private pinned asset and saved native state capture')
    p=Path(directory);payload=deepcopy(json.loads((p/'CABINET_PORT05_PREPARED.private.json').read_bytes())['payload'])
    # Native staging reuses these book bytes from the predecessor asset tree;
    # the pure bundle checker takes the original fully captured bytes.
    complete=json.loads((p/'NIGHT_SOURCE_NATIVE_STATIC_PREPARED.private.json').read_bytes())['payload']
    payload['libero_binary_files']=complete['libero_binary_files']
    for name,filename in(('wood','official-wood-tiles.png'),('metal','official-bright_metal.png')):
        payload['robocasa_texture_files'][name]['base64']=base64.b64encode((p/'robocasa-pinned-minimum-source30'/filename).read_bytes()).decode()
    payload.update(cards=dict(BOOK=books(),ROBOCASA_CABINET=candidate.cards()),cabinet_port_revision=candidate.VERSION,condition_side='right')
    payload['qualification'].update(cabinet_port_revision=candidate.VERSION,condition_side='right',fixed_task_cards_sha256=bundle.PORT06_CARD_SHA)
    assert bundle.unpack(payload)['plan']==[dict(task_kind='ROBOCASA_CABINET',side='right')]
    for changes in(dict(condition_side=None),dict(condition_side='left')):
        with pytest.raises(PermissionError):bundle.unpack(dict(payload,**changes))
    state=deepcopy(json.loads((p/'CABINET_PORT05_FULL_Q_ANCHORS.private.json').read_bytes())['states'][1]['actual_state'])
    card=candidate.cards()['right'];obj=np.array(state['object_pose']);obj[0,3]+=.04;tcp=obj@np.array(card['door']['object_to_tcp_grasp'])
    state.update(object_pose=obj,tcp=tcp);teacher=SourceSkillTeacher(card);teacher.stage='turn';teacher.observe(state);teacher.plan()
    np.testing.assert_allclose(teacher.path[0],tcp,atol=1e-12)
    assert np.max(np.linalg.norm(np.diff(teacher.path[:,:3,3],axis=0),axis=1))/.05<=.03+1e-12
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    assert max(geodesic_angle(a[:3,:3],b[:3,:3])for a,b in zip(teacher.path[:-1],teacher.path[1:]))/.05<=.15+1e-12
    failed=SourceSkillTeacher(card);failed.stage='turn'
    with pytest.raises(PermissionError,match='ACTUAL_GRASP_LOST'):failed.observe(dict(state,native_grasp=False))
