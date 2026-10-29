"""XML port and synthetic control checks; no simulator construction or ticks."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scripts.pi0294.multitask.libero_book_scene import overlay, VERSION
from scripts.pi0294.multitask.libero_book_teacher import LiberoBookTeacher
from scripts.pi0294.test_multitask_contact import SOURCE_FIXTURE, card, state, translation


def test_upstream_geometry_preserved_and_native_origin_not_replaced_by_grasp_frame():
    path=os.environ.get('NIGHT_LIBERO_ASSET_CAPTURE_DIR')
    if not path:pytest.skip('actual pinned private source XMLs required')
    path=Path(path)/'libero/libero/assets/turbosquid_objects'
    definitions={n:(path/n/(n+'.xml')).read_bytes()for n in('black_book','desk_caddy')}
    assets={n+'/'+a.get('file'):str(path/n/a.get('file'))for n,b in definitions.items()
            for a in ET.fromstring(b).find('asset')if a.get('file')}
    c=dict(source_port=VERSION,goal_side='left',initial_book_native_world=np.eye(4),caddy_world=np.eye(4))
    source=SOURCE_FIXTURE.replace('<worldbody>','<asset/><worldbody>')
    xml,binding=overlay(source,c,definitions=definitions,asset_paths=assets);root=ET.fromstring(xml)
    original=ET.fromstring(source)
    for query in("worldbody/body[@name='base_link']",'actuator','option','contact','equality'):
        assert ET.tostring(root.find(query))==ET.tostring(original.find(query))
    for n,body in definitions.items():
        before=ET.fromstring(body).find("worldbody/body/body[@name='object']")
        after=root.find("worldbody/body[@name='libero_"+n+"']")
        bg=list(before.findall('geom'));ag=list(after.findall('geom'))
        assert len(bg)==len(ag)
        for b,a in zip(bg,ag):
            assert {k:v for k,v in b.attrib.items()if k not in('name','mesh','material')}=={
                k:v for k,v in a.attrib.items()if k not in('name','mesh','material')}
    assert binding['object_body']!=binding['native_object_body']
    assert len(binding['obstacle_geoms'])==17
    assert len(root.findall(".//joint[@type='free']"))==1
    native=root.find("worldbody/body[@name='libero_black_book']")
    assert native.find('joint').get('damping')=='.0005'
    assert float(native.find('inertial').get('mass'))==pytest.approx(100*8*.01434*.05519*.06720)
    assert root.find("worldbody/body[@name='libero_black_book']/body[@name='skill_book']").get('pos')=='-0.00047 -0.01215 0.06937'
    with pytest.raises(ValueError):overlay(source,dict(c,goal_side='middle'),definitions=definitions,asset_paths=assets)


def test_vertical_task_uses_actual_grasp_relation_and_requires_extra_support():
    c=dict(card('BOOK'),source_port=VERSION)
    teacher=LiberoBookTeacher(c)
    teacher.state=state('BOOK',tcp=translation(.45,-.1,.65),native_grasp=True)
    above,insert=teacher._book_targets()
    # Actual relative grasp, not the nominal grasp offset, determines target.
    actual=teacher.state
    np.testing.assert_allclose(np.linalg.inv(insert)@np.asarray(c['book']['target_book_world']),
        np.linalg.inv(actual['tcp'])@actual['object_pose'])
    np.testing.assert_allclose(above[:3,3]-insert[:3,3],[0,0,.12])
    teacher.state.update(native_libero_goal=True,object_supported_by_shelf=False)
    assert not teacher._book_supported_inside()
    teacher.state['object_supported_by_shelf']=True
    assert teacher._book_supported_inside()
    np.testing.assert_array_equal(teacher._withdraw_direction(),[0,0,1])


def test_port_teacher_keeps_native_safety_and_actual_clock_gates():
    teacher=LiberoBookTeacher(dict(card('BOOK'),source_port=VERSION))
    teacher.observe(state('BOOK'))
    with pytest.raises(ValueError):teacher.observe(state('BOOK',.1))
    with pytest.raises(PermissionError):teacher.observe(state('BOOK',.05,safety_valid=False))
