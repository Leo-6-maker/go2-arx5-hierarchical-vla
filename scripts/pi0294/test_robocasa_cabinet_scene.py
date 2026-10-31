"""XML contract and actual pinned assets; no native model construction."""
from copy import deepcopy
import builtins
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts.pi0294.multitask import robocasa_cabinet_assets as a
from scripts.pi0294.multitask.robocasa_cabinet_cards import cards
from scripts.pi0294.multitask.robocasa_cabinet_scene import overlay
from scripts.pi0294.multitask.source_engine import make_engine

SOURCE='''<mujoco model="PI0294_FIXED_BASE_ARX5_MULTITASK"><compiler angle="radian"/>
<option timestep=".002"/><asset/><default><geom friction=".8 .02 .002"/></default>
<worldbody><camera name="body_rgb" pos=".6 .1 .8"/><body name="base_link"><body name="x5_link1"><joint name="arm"/>
<geom type="sphere" size=".01" mass=".1"/><camera name="wrist_rgb" pos="0 0 .1"/></body></body>
<body name="cube"><freejoint/><geom name="cube_geom" type="box" size=".02 .02 .02"/></body>
<geom name="goal_marker_0" type="box" size=".01 .01 .001" contype="0" conaffinity="0"/>
</worldbody><actuator><motor joint="arm"/></actuator><contact/><equality/></mujoco>'''


@pytest.fixture(scope='module')
def inputs():
    directory=os.environ.get('ROBOCASA_PINNED_SOURCE_DIR')
    if not directory:pytest.skip('Explicit privately saved fixed XML and official PNG members required')
    root=Path(directory)
    return dict(definitions={n:(root/'robocasa/models/assets'/p).read_bytes()for n,p in a.XML_PATHS.items()},
        texture_paths=dict(wood=str(root/'official-wood-tiles.png'),metal=str(root/'official-bright_metal.png')))


@pytest.mark.parametrize('orientation',['left','right'])
def test_source_xml_protects_robot_control_cameras_defaults_and_real_handle_center(inputs,orientation):
    card=cards()[orientation];xml,b=overlay(SOURCE,card,**inputs)
    old,new=map(ET.fromstring,(SOURCE,xml))
    for query in("worldbody/body[@name='base_link']","worldbody/camera[@name='body_rgb']",'actuator','contact','equality','option','compiler'):
        assert ET.tostring(old.find(query))==ET.tostring(new.find(query))
    assert ET.tostring(old.find('default/geom'))==ET.tostring(new.find('default/geom'))
    assert new.find(".//body[@name='cube']")is None and new.find(".//geom[@name='goal_marker_0']")is None
    assert len(b['obstacle_geoms'])==8 and b['goal_kind']=='ROBOCASA_NATIVE_DOOR'
    handle=new.find(".//body[@name='rc_door_handle']")
    center=handle.find("geom[@name='rc_handle']").get('pos')
    assert handle.find("body[@name='rc_handle_grasp_frame']").get('pos')==center
    assert b['object_body']!=b['native_handle_body']
    joint=new.find(".//joint[@name='rc_doorhinge']")
    np.testing.assert_array_equal(np.fromstring(joint.get('range'),sep=' '),card['door']['angle_limits_rad'])
    assert joint.get('damping')=='2'and not new.findall('worldbody/body/body/freejoint')
    assert b['new_actuators_equalities_mocap']==0 and b['qualification']=='UNCOMPILED_UNEXECUTED_SOURCE_PORT'


@pytest.mark.parametrize('fault',['floating','degrees','target','active_marker','hinge_frame','range','handle','target_angle'])
def test_rejects_source_or_passive_task_identity_changes(inputs,fault):
    source=SOURCE;card=deepcopy(cards()['right'])
    if fault=='floating':source=source.replace('name="base_link">','name="base_link"><freejoint/>')
    elif fault=='degrees':source=source.replace('angle="radian"','angle="degree"')
    elif fault=='target':source=source.replace('PI0294_FIXED_BASE_ARX5_MULTITASK','TARGET_WBC')
    elif fault=='active_marker':source=source.replace('contype="0"','contype="1"')
    elif fault=='hinge_frame':card['door']['hinge_world'][0][3]+=.001
    elif fault=='range':card['door']['angle_limits_rad'][1]=1.6
    elif fault=='handle':card['door']['handle_radius_m']=.01
    else:card['door']['target_angle_rad']=.5
    with pytest.raises(ValueError):overlay(source,card,**inputs)


def test_two_port_selectors_reject_before_any_native_import(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kw):
        if name=='mujoco':raise AssertionError('Native import must not be reached')
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='ONE_EXPLICIT_PINNED_SOURCE_TASK_PORT'):
        make_engine({},'/unused',pinned_libero={},pinned_robocasa={})


def test_custom_seed_scene_moves_whole_fixture_preserving_relative_hinge_and_robot(inputs):
    from scripts.pi0294.multitask.cabinet_braked_turn import cards as braked_cards
    from scripts.pi0294.multitask.cabinet_seed_cards import seed_card,OFFSETS_M
    original_xml,original_meta=overlay(SOURCE,braked_cards()['right'],**inputs)
    original=ET.fromstring(original_xml)
    for index,offset in enumerate(OFFSETS_M,1):
        xml,meta=overlay(SOURCE,seed_card(index),**inputs);new=ET.fromstring(xml)
        before=original.find(".//body[@name='rc_cabinet_frame']")
        after=new.find(".//body[@name='rc_cabinet_frame']")
        np.testing.assert_allclose(np.fromstring(after.get('pos'),sep=' ')-np.fromstring(before.get('pos'),sep=' '),
                                  [*offset,0],rtol=0,atol=1e-15)
        after.set('pos',before.get('pos'))
        assert ET.tostring(new)==ET.tostring(original)
        np.testing.assert_allclose(np.array(meta['hinge_world'])[:3,3]-np.array(original_meta['hinge_world'])[:3,3],
                                  [*offset,0],rtol=0,atol=1e-15)
        assert meta['qualification']=='UNCOMPILED_UNEXECUTED_SOURCE_PORT'
