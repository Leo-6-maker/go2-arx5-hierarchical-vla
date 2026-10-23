"""Real temporary mesh bytes and URI parsing; no SDK, ROS or physics."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from . import target_sdk_package_assets as unit

@pytest.fixture
def packet(tmp_path):
    root=tmp_path/'packages/robot';(root/'meshes').mkdir(parents=True)
    mesh=root/'meshes/link.glb';mesh.write_bytes(b'CPU_TEMP_BYTES_NOT_NATIVE_GEOMETRY')
    urdf=b'<robot name="CPU"><link name="base"><collision><geometry><mesh filename="package://robot/meshes/link.glb"/></geometry></collision></link></robot>'
    return urdf,{'robot':str(root)},{str(mesh):hashlib.sha256(mesh.read_bytes()).hexdigest()}

def test_all_original_uris_and_hashes_resolve_without_mutation(packet):
    xml,roots,pins=packet;before=deepcopy(packet)
    result=unit.environment(*packet)
    assert result==dict(ROS_PACKAGE_PATH=str(Path(roots['robot']).parent),AMENT_PREFIX_PATH='')
    assert packet==before and len(pins)==1

@pytest.mark.parametrize('fault',['missing_pin','changed_bytes','extra_mesh','extra_package','wrong_root','escape',
    'absolute','other_scheme','duplicate_separator','colon_root','missing_file'])
def test_incomplete_changed_or_escaping_bindings_refuse_before_SDK(packet,fault):
    xml,roots,pins=deepcopy(packet);mesh=Path(next(iter(pins)))
    if fault=='missing_pin':pins={}
    elif fault=='changed_bytes':mesh.write_bytes(b'CHANGED')
    elif fault=='extra_mesh':pins[str(mesh.parent/'extra.glb')]='a'*64
    elif fault=='extra_package':roots['extra']=roots['robot']
    elif fault=='wrong_root':roots['robot']=str(mesh.parent)
    elif fault=='escape':xml=xml.replace(b'meshes/link.glb',b'../escape.glb')
    elif fault=='absolute':xml=xml.replace(b'meshes/link.glb',b'/absolute.glb')
    elif fault=='other_scheme':xml=xml.replace(b'package://',b'file://')
    elif fault=='duplicate_separator':xml=xml.replace(b'meshes/link.glb',b'meshes//link.glb')
    elif fault=='colon_root':roots['robot']=roots['robot']+':other'
    else:mesh.unlink()
    with pytest.raises((PermissionError,KeyError)):unit.environment(xml,roots,pins)
