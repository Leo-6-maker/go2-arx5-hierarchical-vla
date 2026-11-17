import itertools,struct
from pathlib import Path
import numpy as np
import pytest
from scripts.pi0294.task_state import parse_state as legacy
from scripts.pi0294.task_contacts import allowed,TASK_PAIRS
from go2arx5_vla.world_model.multitask_state_v6 import parse_state
from go2arx5_vla.world_model.multitask_contact_v6 import allowed_pair,observer_permitted,preview_permitted,observer_source


def packet(abi,*,step=100,time_s=.2,contact_nan=False):
    values=np.zeros(sum(abi),dtype='<f8');raw=struct.pack('<8sQdIII',b'GRTSTATE',step,time_s,*abi)+values.tobytes()
    raw+=struct.pack('<I',int(contact_nan))
    if contact_nan:
        rest=np.zeros(18,dtype='<f8');rest[0]=np.nan;raw+=struct.pack('<IId',0,1,0.)+rest.tobytes()
    return raw


def test_two_native_packet_contracts_do_not_silently_accept_each_others_abi():
    assert legacy(packet((34,32,20)))['qpos'].shape==(34,)
    assert parse_state(packet((41,38,20)))['qpos'].shape==(41,)
    with pytest.raises(ValueError,match='ABI'):legacy(packet((41,38,20)))
    with pytest.raises(ValueError,match='ABI'):parse_state(packet((34,32,20)))
    with pytest.raises(ValueError,match='2MS'):parse_state(packet((41,38,20),time_s=.21))
    with pytest.raises(ValueError,match='CONTACT'):parse_state(packet((41,38,20),contact_nan=True))


def test_selected_red_contact_predicate_is_exact_for_every_legacy_pair_and_depth():
    names=['floor','FL','FR','RL','RR','task_cube_geom','task_table','task_left_pad','task_right_pad','robot_forearm']
    for a,b,depth in itertools.product(names,names,[-.01,-.004,0.,.001]):
        original=bool(allowed(a,b)and not(frozenset((a,b))in TASK_PAIRS and depth<-.004))
        assert observer_permitted(a,b,depth,'red')==original


def test_two_object_binding_adds_support_without_waiving_wrong_object_or_table_hand_contacts():
    assert observer_permitted('task_blue_cube_geom','task_table',0.,'red')
    assert not observer_permitted('task_blue_cube_geom','task_table',-.00401,'red')
    assert not allowed_pair('task_left_pad','task_blue_cube_geom','red')
    assert allowed_pair('task_left_pad','task_blue_cube_geom','blue')
    assert not allowed_pair('task_left_pad','task_cube_geom','blue')
    assert not allowed_pair('task_left_pad','task_table','blue')
    assert not allowed_pair('task_cube_geom','task_blue_cube_geom','blue')
    assert not preview_permitted('task_left_pad','task_blue_cube_geom',0.,'approach','blue')
    assert preview_permitted('task_left_pad','task_blue_cube_geom',0.,'close','blue')


def test_observer_candidate_changes_only_role_binding_and_keeps_original_stop_latch():
    import os
    supplied=os.environ.get('SLA_V6_FROZEN_OBSERVER_SOURCE')
    if not supplied:pytest.skip('actual archived Core observer source is supplied during execution qualification')
    original=Path(supplied).read_bytes()
    candidate=observer_source(original)
    assert b'if(safety_stopped) {wm02_init::emit(wm02_init::REFUSED,d->time,count);raise(SIGINT);return;}'in candidate
    assert b'd->contact[i].dist < -.004'in candidate
    assert b'MFREE_V6_SELECTED_COLOR'in candidate
    site=b'    bool task_pair=pair("task_table","task_cube_geom") || pair("task_left_pad","task_cube_geom") || pair("task_right_pad","task_cube_geom");'
    import re
    restored=re.sub(rb'// V6_TASK_ROLE_HELPER_BEGIN.*?// V6_TASK_ROLE_HELPER_END\n\n',b'',candidate,flags=re.S)
    restored=re.sub(rb'    // V6_TASK_ROLE_SELECTION_BEGIN.*?    // V6_TASK_ROLE_SELECTION_END',site,restored,flags=re.S)
    assert restored==original
    with pytest.raises(ValueError):observer_source(original+b'changed')


def test_unbound_repository_observer_cannot_replace_actual_core_frozen_trace_source():
    raw=(Path(__file__).parents[1]/'src/go2arx5_vla/target_mujoco/observer.cpp').read_bytes()
    with pytest.raises(ValueError,match='FROZEN_SERIALIZED'):observer_source(raw)
