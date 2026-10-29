"""Pinned RoboCasa single-cabinet geometry and original door predicate.

Procedural formulas adapt RoboCasa dc73a8ce (MIT, copyright 2026 RoboCasa
Team); see docs/third_party/robocasa/LICENSE. No simulator, renderer, network,
robot or data admission. A source ARX5 port is not the original kitchen test.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path,PurePosixPath
import posixpath
import xml.etree.ElementTree as ET

COMMIT='dc73a8ce18e4b8947e8df2ae2958245ca823f3b4'
XML_BLOBS=dict(cabinet_single='4359060211bb42e341c23479249eb3fe25fb4918',
    slab='696e08782235be3d726985b3f77a58a834c5951f',
    bar_handle='6120c2eb0187e48f44ae2560aa5c2829888e187c')
XML_PATHS=dict(cabinet_single='fixtures/cabinets/cabinet_single.xml',
    slab='fixtures/cabinets/cabinet_panels/slab.xml',
    bar_handle='fixtures/handles/bar_handle.xml')
VERSION='ROBOCASA_PINNED_SINGLE_SLAB_BAR_SOURCE_GEOMETRY_V1'
TEXTURE_COMMIT='1b92c3d02ca4354984fec961357db0bff7b32166'
TEXTURE_FILES=dict(wood=dict(path='textures/prev/wood-tiles.png',
    bytes=1460776,sha256='18d87069529422d832559ce969858576dd46f83efc3dfd9e5ffbf26d740be123'),
    metal=dict(path='textures/metals/bright_metal.png',bytes=1998226,
    sha256='e67c23a272f98890b485c5e2ae2ef1088080067954c676cb70170252ab931bf4'))


def numbers(values):return ' '.join(format(float(x),'.17g')for x in values)


def pinned_definition(name,body):
    if (name not in XML_BLOBS or type(body)is not bytes or not 0<len(body)<=32768
            or b'<!DOCTYPE'in body or b'<!ENTITY'in body):
        raise ValueError('ONLY_THREE_PINNED_CABINET_XML_MEMBERS')
    blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if blob!=XML_BLOBS[name]:raise ValueError('ORIGINAL_ROBOCASA_GIT_BLOB_REQUIRED')
    root=ET.fromstring(body)
    if root.tag!='mujoco'or root.find("worldbody/body/body[@name='object']")is None:
        raise ValueError('ORIGINAL_ROBOCASA_OBJECT_FRAME')
    return root


def native_door_status(joint_positions,joint_ranges):
    """Fixture.get_joint_state/is_open/is_closed, for real finite door joints.

    The original normalization is not clipped or replaced by abs(angle).
    All selected doors must be >=90% open. Release, withdrawal, contact and
    hold are additional project checks, never implied by this predicate.
    """
    if not joint_positions or set(joint_positions)!=set(joint_ranges):
        raise ValueError('NONEMPTY_EXACT_NATIVE_DOOR_JOINTS')
    openness={};within=True
    for name,q in joint_positions.items():
        low,high=joint_ranges[name]
        if not all(type(x)in(int,float)and math.isfinite(x)for x in(q,low,high))or low>=high:
            raise ValueError('FINITE_NATIVE_JOINT_POSITION_AND_NONZERO_RANGE')
        value=(q-low)/(high-low)
        if low<0:value=1-value
        openness[name]=value;within=within and low<=q<=high
    return dict(normalized_openness=openness,native_is_open=all(x>=.90 for x in openness.values()),
        native_is_closed=all(x<=.005 for x in openness.values()),
        joint_positions_within_declared_ranges=within,
        release_withdrawal_support_or_hold_in_native_predicate=False)


def single_cabinet_geometry(definitions,*,size,orientation):
    """Build upstream primitive geometry, without resolving textures or physics.

    This selects one SingleCabinet, slab panel, vertical bar, one level and
    default thickness/gap/handle settings. Its size/placement is a declared
    port condition; original kitchen layouts, distractors and robot differ.
    Native compilation, inertia, collision and rendering remain unqualified.
    """
    if set(definitions)!=set(XML_BLOBS)or orientation not in('left','right'):
        raise ValueError('EXACT_SINGLE_SLAB_VERTICAL_BAR_PORT')
    if (type(size)not in(list,tuple)or len(size)!=3
            or not all(type(x)in(int,float)and math.isfinite(x)and x>.08 for x in size)
            or size[2]>.60):
        raise ValueError('FINITE_POSITIVE_ORIGINAL_ONE_LEVEL_CABINET_DIMENSIONS')
    source={n:pinned_definition(n,definitions[n])for n in XML_BLOBS}
    root=deepcopy(source['cabinet_single']);obj=root.find("worldbody/body/body[@name='object']")
    x,y,z=(v/2 for v in size);th=.03/2;gap=.003
    sizes=dict(top=[x,y-th,th],bottom=[x,y-th,th],back=[x-2*th,th,z-2*th],
        left=[th,y-th,z-2*th],right=[th,y-th,z-2*th])
    positions=dict(top=[0,th,z-th],bottom=[0,th,-z+th],back=[0,y-th,0],
        left=[-x+th,th,0],right=[x-th,th,0])
    for name in sizes:
        geom=obj.find("geom[@name='"+name+"']")
        geom.set('size',numbers(sizes[name]));geom.set('pos',numbers(positions[name]))
    door=obj.find("body[@name='hingedoor']");door.set('pos','0 0 0')
    joint=door.find('joint');right=orientation=='right';limits=(0,1.57)if right else(-1.57,0)
    joint.set('pos',numbers([x-th if right else-x+th,-y,0]));joint.set('range',numbers(limits))
    panel=deepcopy(source['slab'].find("worldbody/body/body[@name='object']"))
    panel.set('name','door_panel');panel.set('pos',numbers([0,-y+th,0]))
    panel_w,panel_h=2*x-gap,2*z-gap
    panel_geom=panel.find('geom');panel_geom.set('size',numbers([panel_w/2,panel_h/2,th]));panel_geom.set('pos','0 0 0')
    panel_geom.set('material','panel_mat')
    handle=deepcopy(source['bar_handle'].find("worldbody/body/body[@name='object']"))
    handle.set('name','door_handle');length=min(.24,panel_h-2*.04)
    if length<=0:raise ValueError('ORIGINAL_BAR_HANDLE_LENGTH_AND_PADDING')
    hx=(-panel_w/2 if right else panel_w/2)*.825
    hz=(-panel_h/2+length/2)*.80;handle.set('pos',numbers([hx,0,hz]))
    for geom in handle.findall('geom'):
        name=geom.get('name');geom.set('material','handle_mat')
        sign=1 if name=='handle_connector_top'else-1
        geom.set('pos',numbers([0,-.05 if name=='handle'else-.025,0 if name=='handle'else sign*length/2*.60]))
        geom.set('size',numbers([.013,length/2]if name=='handle'else[.008,.025]))
    panel.append(handle);door.append(panel)
    obj.find("geom[@name='reg_main']").set('size',numbers([x,y,z]))
    ET.SubElement(obj,'geom',name='reg_level0',type='box',group='1',contype='0',conaffinity='0',
        rgba='0 1 0 0',pos='0 0 0',size=numbers([x-2*th,y-2*th,(2*(z-th))/2-th]))
    assets=root.find('asset');unresolved={}
    for part in XML_BLOBS:
        prefix=''if part=='cabinet_single'else'panel_'if part=='slab'else'handle_'
        for item in source[part].findall('asset/*'):
            copied=deepcopy(item)
            for key in('name','texture'):
                if key in copied.attrib:copied.set(key,prefix+copied.attrib[key])
            if 'file'in copied.attrib:
                rel=posixpath.normpath(str(PurePosixPath(XML_PATHS[part]).parent/copied.get('file')))
                if not rel.startswith('fixtures/'):raise ValueError('ASSET_REFERENCE_ESCAPE')
                unresolved[copied.get('name')]=dict(raw_XML_reference=rel,
                    handle_constructor_default_override='textures/metals/bright_metal.png'if part=='bar_handle'else None)
            if part!='cabinet_single':assets.append(copied)
    root.set('model',VERSION)
    return ET.tostring(root,encoding='unicode'),dict(version=VERSION,commit=COMMIT,
        orientation=orientation,size=list(size),joint_name='doorhinge',joint_range=list(limits),
        open_threshold_radians=.90*1.57,closed_initial_joint_position=0,
        handle_body='door_handle',raw_texture_dependencies=unresolved,
        textures_resolved=False,native_compilation_physics_or_rendering_qualified=False,
        native_integrations=0,training_admitted=False,original_kitchen_benchmark=False)


def bind_minimum_textures(xml,paths):
    """Bind two frozen official PNG members; do not compile or render.

    The wood style is an explicit port condition selected before behavior.
    The metal is the original Handle constructor's default. Images originate
    from robocasa/robocasa-assets@TEXTURE_COMMIT (CC BY 4.0). The retrieval
    receipt distinguishes member CRC/SHA checks from an uncomputed full ZIP
    hash. Runtime admission must recheck node-local paths and all asset bytes.
    """
    if set(paths)!=set(TEXTURE_FILES):raise ValueError('EXACT_TWO_FROZEN_OFFICIAL_TEXTURE_MEMBERS')
    root=ET.fromstring(xml)
    if root.get('model')!=VERSION:raise ValueError('EXPLICIT_UNCOMPILED_CABINET_PORT')
    original_geometry=ET.tostring(root.find('worldbody'));bindings={}
    for name,pin in TEXTURE_FILES.items():
        p=Path(paths[name])
        if not p.is_absolute()or any(x.is_symlink()for x in(p,*p.parents)):
            raise ValueError('ABSOLUTE_UNALIASED_PRIVATE_TEXTURE_PATH')
        info=p.stat()
        if not p.is_file()or info.st_size!=pin['bytes']:
            raise ValueError('EXACT_FROZEN_TEXTURE_MEMBER_SIZE')
        b=p.read_bytes()
        if not b.startswith(b'\x89PNG\r\n\x1a\n')or hashlib.sha256(b).hexdigest()!=pin['sha256']:
            raise ValueError('EXACT_FROZEN_OFFICIAL_PNG_BYTES')
        bindings[name]=str(p)
    textures=root.findall('asset/texture')
    if len(textures)!=3 or {x.get('name')for x in textures}!={'tex','panel_tex','handle_tex'}:
        raise ValueError('EXACT_ORIGINAL_THREE_TEXTURE_ROLES')
    for texture in textures:texture.set('file',bindings['metal'if texture.get('name')=='handle_tex'else'wood'])
    if ET.tostring(root.find('worldbody'))!=original_geometry:
        raise ValueError('TEXTURE_BINDING_CHANGED_PHYSICAL_GEOMETRY')
    return ET.tostring(root,encoding='unicode'),dict(status='TWO_OFFICIAL_PNG_MEMBERS_BOUND_GEOMETRY_UNCHANGED',
        repository='robocasa/robocasa-assets',commit=TEXTURE_COMMIT,
        files={k:dict(v)for k,v in TEXTURE_FILES.items()},
        constructor_default_metal_preserved=True,wood_is_declared_port_style=True,
        textures_resolved=True,native_compilation_physics_or_rendering_qualified=False,
        native_integrations=0,training_admitted=False)
