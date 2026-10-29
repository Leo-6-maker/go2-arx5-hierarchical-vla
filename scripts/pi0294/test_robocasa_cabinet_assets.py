"""Compare actual pinned upstream functions, without third-party simulators."""
import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace,MethodType
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scripts.pi0294.multitask import robocasa_cabinet_assets as c

PINS={'cabinets.py':'268fee24027bafea14b2d531c884f892e3e3b80c',
    'cabinet_panels.py':'c1c76b0fc3fba1b6cf96d58af9de014d7db58121',
    'handles.py':'6ecec8a9cc1fb23e33dd8833dd7eb9ac35dbab6f',
    'fixture.py':'75b1dc813e87f738fd37637b8ee7ba203c9e5afc'}


@pytest.fixture(scope='module')
def original():
    directory=os.environ.get('ROBOCASA_PINNED_SOURCE_DIR')
    if not directory:pytest.skip('Explicit privately saved fixed-revision upstream bytes required')
    root=Path(directory);sources={};definitions={}
    for name,pin in PINS.items():
        b=(root/'robocasa/models/fixtures'/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==pin
        sources[name]=ast.parse(b)
    b=(root/'robocasa/utils/object_utils.py').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()=='81e905247ca2f170a65ace0ae10a8037951f4e2a'
    sources['object_utils.py']=ast.parse(b)
    for name,path in c.XML_PATHS.items():definitions[name]=(root/'robocasa/models/assets'/path).read_bytes()
    return sources,definitions


def method(sources,file,cls,name,namespace):
    tree=sources[file]
    if cls is not None:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        '<pinned-upstream-method-only>','exec'),namespace)
    return namespace[name]


def test_native_boundary_and_negative_direction_match_actual_upstream_functions(original):
    sources,_=original;ns={}
    norm=method(sources,'object_utils.py',None,'normalize_joint_value',ns)
    ns=dict(OU=SimpleNamespace(normalize_joint_value=norm))
    state=method(sources,'fixture.py','Fixture','get_joint_state',ns)
    opened=method(sources,'fixture.py','Fixture','is_open',ns)
    closed=method(sources,'fixture.py','Fixture','is_closed',ns)
    ranges=dict(right=(0,1.57),left=(-1.57,0))
    positions={};model=SimpleNamespace(get_joint_qpos_addr=lambda name:name)
    fixture=SimpleNamespace(_joint_infos={k:dict(range=v)for k,v in ranges.items()},door_joint_names=list(ranges))
    fixture.get_joint_state=MethodType(state,fixture)
    env=SimpleNamespace(sim=SimpleNamespace(model=model,data=SimpleNamespace(qpos=positions)))
    points=[0,.005,.1,.899999999,.90,.900000001,1,1.001,-.001]
    pairs=0
    for a in points:
        for b in points:
            positions.update(right=a*1.57,left=-b*1.57)
            result=c.native_door_status(positions,ranges)
            assert result['normalized_openness']==state(fixture,env,list(ranges))
            assert result['native_is_open']==opened(fixture,env)
            assert result['native_is_closed']==closed(fixture,env)
            pairs+=1
    assert pairs==81


def upstream_geometry(sources,definitions,size,orientation):
    """Small XML container only; execute real upstream procedural methods."""
    ns=dict(np=np,a2s=lambda x:' '.join(str(v)for v in x),s2a=lambda x:np.fromstring(x,sep=' '),
        deepcopy=__import__('copy').deepcopy,ET=ET)
    ns['array_to_string']=ns['a2s']
    ns['set_geom_dimensions']=method(sources,'object_utils.py',None,'set_geom_dimensions',ns)
    create_handle=method(sources,'handles.py','BarHandle','_create_handle',ns)
    create_panel=method(sources,'cabinet_panels.py','SlabCabinetPanel','_create_panel',ns)
    add_handle=method(sources,'cabinet_panels.py','CabinetPanel','_add_handle',ns)
    add_door=method(sources,'cabinets.py','Cabinet','_add_door',ns)
    levels=method(sources,'cabinets.py','Cabinet','_add_levels',ns)
    create=method(sources,'cabinets.py','SingleCabinet','_create_cab',ns)
    ns['find_elements']=lambda *args,**kw:None  # this slab has no reg_main
    def components(obj):return {g.get('name'):[g]for g in obj.findall('geom')}
    class Handle:
        def __init__(self,name,panel_w,panel_h,**kw):
            self.obj=c.pinned_definition('bar_handle',definitions['bar_handle']).find("worldbody/body/body[@name='object']")
            self.obj.set('name','door_handle');self.panel_w=panel_w;self.panel_h=panel_h
            self.length=.24;self.handle_pad=.04;self.orientation='vertical'
            self._get_components=lambda:(components(self.obj),{},{});create_handle(self)
        def get_obj(self):return self.obj
        def _get_depth_ofs(self):return None
    ns['BarHandle']=Handle
    class Panel:
        def __init__(self,size,name,texture,handle_type,handle_config,handle_hpos,handle_vpos):
            self.name=name
            self.root=c.pinned_definition('slab',definitions['slab'])
            self.obj=self.root.find("worldbody/body/body[@name='object']");self.obj.set('name','door_panel')
            self.size=size;self.handle_type=handle_type;self.handle_config=handle_config
            self.handle_hpos=handle_hpos;self.handle_vpos=handle_vpos
            self.handle_hpercent=.825;self.handle_vpercent=.80;self.naming_prefix=''
            self._get_components=lambda:components(self.obj);self.merge_assets=lambda obj:None
            create_panel(self);add_handle(self)
        def get_obj(self):return self.obj
    ns['SlabCabinetPanel']=Panel
    root=c.pinned_definition('cabinet_single',definitions['cabinet_single']);obj=root.find("worldbody/body/body[@name='object']")
    fixture=SimpleNamespace(size=np.array(size),thickness=.03,door_gap=.003,orientation=orientation,
        panel_type='slab',panel_config={},handle_type='bar',handle_config={},texture=None,name='source_cabinet',num_levels=1)
    fixture._get_cab_components=lambda:(components(obj),{'hingedoor':obj.find('body')},{'doorhinge':obj.find('body/joint')})
    fixture.merge_assets=lambda item:None;fixture.get_obj=lambda:obj
    regions={};fixture.update_regions=lambda value,**kw:regions.update(value)
    fixture._add_door=MethodType(add_door,fixture);fixture._add_levels=MethodType(levels,fixture)
    create(fixture)
    return root,regions


@pytest.mark.parametrize('orientation',['left','right'])
@pytest.mark.parametrize('size',[(.24,.22,.30),(.36,.28,.36),(.60,.40,.60)])
def test_all_primitive_geometry_matches_actual_upstream_functions(original,size,orientation):
    sources,definitions=original
    actual,meta=c.single_cabinet_geometry(definitions,size=size,orientation=orientation)
    actual=ET.fromstring(actual);expected,regions=upstream_geometry(sources,definitions,size,orientation)
    for name in ('top','bottom','back','left','right','door','handle','handle_connector_top','handle_connector_bottom','doorhinge'):
        a=actual.find(".//*[@name='"+name+"']");b=expected.find(".//*[@name='"+name+"']")
        for field in('pos','size','euler','quat','range','damping','axis','mass','type','class'):
            if field not in b.attrib:assert field not in a.attrib
            elif field in('type','class'):assert a.get(field)==b.get(field)
            else:assert np.array_equal(np.fromstring(a.get(field),sep=' '),np.fromstring(b.get(field),sep=' '))
    for name in('door_panel','door_handle'):
        assert np.array_equal(np.fromstring(actual.find(".//*[@name='"+name+"']").get('pos'),sep=' '),
            np.fromstring(expected.find(".//*[@name='"+name+"']").get('pos'),sep=' '))
    assert np.array_equal(np.fromstring(actual.find(".//*[@name='reg_level0']").get('size'),sep=' '),regions['level0']['halfsize'])
    assert meta['textures_resolved']is meta['native_compilation_physics_or_rendering_qualified']is False
    assert len(meta['raw_texture_dependencies'])==3


def test_invalid_data_is_not_native_success_and_no_clipping_hides_limits():
    for q in (math.nan,math.inf,-math.inf,True):
        with pytest.raises(ValueError):c.native_door_status({'door':q},{'door':(0,1.57)})
    with pytest.raises(ValueError):c.native_door_status({},{})
    with pytest.raises(ValueError):c.native_door_status({'door':0},{'door':(0,0)})
    result=c.native_door_status({'door':1.58},{'door':(0,1.57)})
    assert result['native_is_open']and not result['joint_positions_within_declared_ranges']
    assert not result['release_withdrawal_support_or_hold_in_native_predicate']
    with pytest.raises(ValueError):c.pinned_definition('cabinet_single',b'<mujoco/>')


def test_two_real_official_png_members_bind_without_geometry_or_model_changes(original,tmp_path):
    _,definitions=original;root=Path(os.environ['ROBOCASA_PINNED_SOURCE_DIR'])
    paths=dict(wood=str(root/'official-wood-tiles.png'),metal=str(root/'official-bright_metal.png'))
    if not all(Path(p).exists()for p in paths.values()):pytest.skip('Explicit minimal official PNG members required')
    xml,_=c.single_cabinet_geometry(definitions,size=(.24,.22,.30),orientation='right')
    bound,meta=c.bind_minimum_textures(xml,paths)
    old,new=map(ET.fromstring,(xml,bound))
    assert ET.tostring(old.find('worldbody'))==ET.tostring(new.find('worldbody'))
    assert {x.get('file')for x in new.findall('asset/texture')}==set(paths.values())
    assert meta['textures_resolved']and not meta['native_compilation_physics_or_rendering_qualified']
    bad=tmp_path/'changed.png';bad.write_bytes(b'changed')
    with pytest.raises(ValueError,match='EXACT_FROZEN_TEXTURE_MEMBER_SIZE'):
        c.bind_minimum_textures(xml,dict(paths,wood=str(bad)))
    with pytest.raises(ValueError):c.bind_minimum_textures(xml,dict(paths,other=paths['wood']))
