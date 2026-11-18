import copy
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml


@pytest.fixture
def original(monkeypatch):
    from go2arx5_vla.world_model import pilot1_curriculum
    q=' '.join(map(str,np.r_[np.zeros(27),.48,-.055,.46,1,0,0,0]));v=' '.join(['0']*32)
    xml=f'''<mujoco model="fixture"><option timestep=".002"/><worldbody><body name="base_link"><freejoint/><geom name="robot" size=".1"/></body>
    <geom name="task_table" size=".17 .24 .025" pos=".515 0 .415"/>
    <geom name="task_goal_0" contype="0" conaffinity="0"/><geom name="task_goal_1" contype="0" conaffinity="0"/>
    <geom name="task_goal_2" contype="0" conaffinity="0"/><geom name="task_goal_3" contype="0" conaffinity="0"/>
    <body name="task_cube" pos=".48 -.055 .46"><freejoint name="task_cube_free"/><geom name="task_cube_geom" size=".02 .02 .02" mass=".05" friction=".8 .02 .002"/></body>
    </worldbody><actuator>''' + ''.join('<motor name="a%d"/>'%i for i in range(20))+f'''</actuator><keyframe><key qpos="{q}" qvel="{v}" ctrl="{' '.join(['0']*20)}"/></keyframe></mujoco>'''
    monkeypatch.setattr(pilot1_curriculum,'scene_xml',lambda spec,m:xml)
    return SimpleNamespace(geom=lambda name:SimpleNamespace(id=0),geom_pos=np.array([[.515,0,.415]]),geom_size=np.array([[.17,.24,.025]]))


def card():return dict(object_positions_world=dict(red=[.48,-.055,.46],blue=[.48,.055,.46]),
    zones={z:dict(center_world_XY=[.58,y],half_size_XY=[.04,.03])for z,y in(('left',.07),('right',-.07))},direction_world_XY=[1,0])


def test_scene_keeps_both_free_objects_and_same_visible_scene_for_every_instruction(original):
    one,binding=overlay_xml({},original,card());root=ET.fromstring(one)
    two,_=overlay_xml({},original,dict(card(),instruction='PLACE-blue-right'))
    assert one==two and binding['scene_depends_on_instruction']is False
    assert binding['expected_ABI']==dict(nq=41,nv=38,nu=20)
    assert len(root.findall('./actuator/motor'))==20
    assert len(np.fromstring(root.find('./keyframe/key').get('qpos'),sep=' '))==41
    assert len(np.fromstring(root.find('./keyframe/key').get('qvel'),sep=' '))==38
    assert root.find("./worldbody/body[@name='task_blue_cube']/geom").get('mass')=='.05'
    assert binding['native_dispatch_qualified']is False


@pytest.mark.parametrize('fault',['object_overlap','outside_table','region_overlap'])
def test_static_unsupported_geometry_is_rejected_before_any_native_request(original,fault):
    c=copy.deepcopy(card())
    if fault=='object_overlap':c['object_positions_world']['blue']=c['object_positions_world']['red']
    elif fault=='outside_table':c['object_positions_world']['blue'][0]=.99
    else:c['zones']['right']=c['zones']['left']
    with pytest.raises(ValueError):overlay_xml({},original,c)


def test_existing_contact_contract_does_not_silently_admit_new_object_names():
    from scripts.pi0294.task_contacts import allowed
    assert allowed('task_cube_geom','task_table')
    assert not allowed('task_blue_cube_geom','task_table')
    assert not allowed('task_blue_cube_geom','task_left_pad')


def test_missing_convenience_asset_root_is_resolved_from_frozen_members_only(tmp_path):
    import hashlib
    from scripts.wm02.multitask_scene_audit_v6 import asset_binding
    files={}
    for name in('go2_x5.xml','scene.xml'):
        path=tmp_path/name;path.write_text('fixed XML '+name);files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    root,pins=asset_binding(dict(frozen_files=files))
    assert root==str(tmp_path) and len(pins)==2
    (tmp_path/'scene.xml').write_text('changed')
    with pytest.raises(ValueError,match='FROZEN_NATIVE_ASSET_XML_CHANGED'):asset_binding(dict(frozen_files=files))
