"""Mechanical change is explicit; collision, task and old cards stay intact."""
from copy import deepcopy
import base64,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.pi0294.multitask import cabinet_edge_hinge as edge,cabinet_layout_port06 as old
from scripts.pi0294.multitask import robocasa_cabinet_assets as assets,source_stage_bundle as bundle
from scripts.pi0294.multitask.robocasa_cabinet_scene import overlay
from scripts.pi0294.multitask.cabinet_repair02 import checked_card,attempt_operation
from scripts.pi0294.multitask.libero_book_cards import cards as book_cards
from scripts.pi0294.test_robocasa_cabinet_scene import inputs,SOURCE


def test_separate_right_variant_keeps_original_conditions_and_history():
    before=old.cards();after=edge.cards();card=after['right']
    assert old.cards()==before and after['left']==before['left']
    assert checked_card(card,'right')and attempt_operation(card,'right')=='ROBOCASA_RIGHT_PORT07_EDGE_HINGE_LAYOUT60'
    for key in('initial_tcp','warmup_s','task_duration_s','independent_demo_conditions','turn_reference_limits','withdraw_tcp_rotation_world'):
        assert card[key]==before['right'][key]
    for key in('target_angle_rad','angle_limits_rad','handle_radius_m','handle_length_m','object_to_tcp_grasp'):
        assert card['door'][key]==before['right']['door'][key]
    assert card['cabinet']['mechanical_variant']==edge.MECHANICS
    delta=np.asarray(card['door']['hinge_world'])[:3,3]-np.asarray(before['right']['door']['hinge_world'])[:3,3]
    np.testing.assert_allclose(delta,[.02,-.0135,0.],atol=1e-15,rtol=0)
    assert bundle.hashlib.sha256(bundle.raw(dict(BOOK=book_cards(),ROBOCASA_CABINET=after))).hexdigest()==bundle.PORT07_CARD_SHA
    m=dict(task_kind='ROBOCASA_CABINET',condition_side='right',cabinet_port_revision=edge.VERSION,
        predeclared_tasks=[dict(task_kind='ROBOCASA_CABINET',side='right')],attempts_this_chain_max=1)
    assert bundle.checked_manifest(m)[1]==m['predeclared_tasks']
    for side in('left',None):
        with pytest.raises(PermissionError):bundle.checked_manifest(dict(m,condition_side=side))
    bad=deepcopy(card);bad['turn_reference_limits']['max_linear_m_s']*=2
    with pytest.raises(PermissionError):checked_card(bad,'right')


@pytest.mark.parametrize('orientation',['right','left'])
def test_pinned_geometry_changes_only_pivot_and_submicron_quarter_turn(inputs,orientation):
    xml,_=assets.single_cabinet_geometry(inputs['definitions'],size=[.24,.22,.30],orientation=orientation)
    changed,meta=edge.geometry(xml,size=[.24,.22,.30],orientation=orientation)
    before,after=map(ET.fromstring,(xml,changed));diff=[]
    for a,b in zip(before.iter(),after.iter()):
        assert a.tag==b.tag and a.text==b.text
        diff.extend((a.get('name'),k)for k in set(a.attrib)|set(b.attrib)if a.get(k)!=b.get(k))
    assert diff==[('doorhinge','pos'),('door','euler')]
    assert meta['exact_upstream_mechanics']is meta['collision_exclusions_or_mask_changes']is False
    # Independent box-corner rotation: old pivot drives the back edge into
    # the front housing plane; the proposed outer edge opens away from it.
    side=1 if orientation=='right'else-1
    half=np.fromstring(before.find(".//geom[@name='door']").get('size'),sep=' ')
    corners=np.array([[x,y,z]for x in(-half[0],half[0])for y in(-half[1],half[1])for z in(-half[2],half[2])])
    angle=side*.15413973057791916;front=-.11+.03
    def maximum_y(tree,theta):
        joint=tree.find(".//joint[@name='doorhinge']");geom=tree.find(".//geom[@name='door']")
        pivot=np.fromstring(joint.get('pos'),sep=' ')
        local=corners@Rotation.from_euler('x',float(geom.get('euler').split()[0])).as_matrix().T+np.array([0.,-.095,0.])
        return ((local-pivot)@Rotation.from_euler('z',theta).as_matrix().T+pivot)[:,1].max()
    assert .001716<maximum_y(before,angle)-front<.001719
    assert max(maximum_y(after,a)-front for a in np.linspace(0,side*.95*1.57,301))<1e-14
    if orientation=='right':
        changed_scene,binding=overlay(SOURCE,edge.cards()['right'],**inputs)
        source,new=map(ET.fromstring,(SOURCE,changed_scene))
        for key in('actuator','contact','equality','option','compiler',"worldbody/body[@name='base_link']"):
            assert ET.tostring(source.find(key))==ET.tostring(new.find(key))
        assert binding['mechanical_variant']==edge.MECHANICS and not binding['exact_upstream_mechanics']
        bad=deepcopy(edge.cards()['right']);bad['cabinet']['mechanical_variant']='IGNORE_SELF_CONTACTS'
        with pytest.raises(ValueError):overlay(SOURCE,bad,**inputs)


def test_card_matches_saved_complete_route_and_native_contact_screen():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit private saved full-route/native evidence')
    p=Path(directory);route=json.loads((p/'CABINET_EDGE_HINGE_LAYOUT60_FULL_ROUTE.private.json').read_bytes())
    c=edge.cards()['right']
    np.testing.assert_allclose(c['cabinet']['cabinet_world'],route['candidate_card']['cabinet']['cabinet_world'],atol=1e-15,rtol=0)
    np.testing.assert_allclose(c['door']['hinge_world'],route['candidate_card']['door']['hinge_world'],atol=1e-15,rtol=0)
    np.testing.assert_allclose(c['door']['object_to_tcp_grasp'],route['candidate_card']['door']['object_to_tcp_grasp'],atol=1e-15,rtol=0)
    assert route['all_Knot_IK_and_joint_speed_pass']and route['all_full_robot_fixture_knots_clear']
    native=json.loads((p/'CABINET_EDGE_HINGE_LAYOUT60_NATIVE_STATIC.private.json').read_bytes())['result']
    assert native['native_forward_calls']==11676 and native['physics_integrations']==0
    assert native['strict_unexpected_robot_contact_samples']==0 and not native['native_penetration_or_joint_or_arm_table_violation']
    assert native['task_fixture_internal_max_penetration_m']<1e-12 and not native['qualification']


def test_new_bundle_uses_all_actual_minimum_assets_and_only_right_condition():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit private minimum-asset capture')
    p=Path(directory);payload=deepcopy(json.loads((p/'CABINET_PORT05_PREPARED.private.json').read_bytes())['payload'])
    payload['libero_binary_files']=json.loads((p/'NIGHT_SOURCE_NATIVE_STATIC_PREPARED.private.json').read_bytes())['payload']['libero_binary_files']
    for name,filename in(('wood','official-wood-tiles.png'),('metal','official-bright_metal.png')):
        payload['robocasa_texture_files'][name]['base64']=base64.b64encode((p/'robocasa-pinned-minimum-source30'/filename).read_bytes()).decode()
    payload.update(cards=dict(BOOK=book_cards(),ROBOCASA_CABINET=edge.cards()),cabinet_port_revision=edge.VERSION,condition_side='right')
    payload['qualification'].update(cabinet_port_revision=edge.VERSION,condition_side='right',fixed_task_cards_sha256=bundle.PORT07_CARD_SHA)
    assert bundle.unpack(payload)['plan']==[dict(task_kind='ROBOCASA_CABINET',side='right')]
    for side in('left',None):
        with pytest.raises(PermissionError):bundle.unpack(dict(payload,condition_side=side))
