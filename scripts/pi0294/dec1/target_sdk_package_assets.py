"""Read-only package-URI binding for the existing frozen SDK's mesh loader.

No URDF/mesh rewrite, ROS node, SDK import, constructor or resource admission.
"""
import hashlib
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET


def need(ok,why):
    if not ok:raise PermissionError('DEC1_SDK_PACKAGE_'+why)


def environment(urdf_bytes,package_roots,mesh_sha256):
    need(type(urdf_bytes) is bytes and len(urdf_bytes)<=2*1024**2,'BOUNDED_ORIGINAL_URDF')
    need(type(package_roots) is dict and type(mesh_sha256) is dict,'EXPLICIT_SEALED_PACKAGE_AND_MESH_BINDINGS')
    uris={node.attrib['filename'] for node in ET.fromstring(urdf_bytes).iter('mesh')}
    need(0<len(uris)<=128,'COMPLETE_BOUNDED_ORIGINAL_MESH_SET')
    packages=set();paths=set();parents=set()
    for uri in uris:
        need(type(uri) is str and uri.startswith('package://'),'ORIGINAL_PACKAGE_URI_ONLY')
        name,sep,relative=uri.removeprefix('package://').partition('/')
        part=PurePosixPath(relative)
        need(sep and re.fullmatch(r'[A-Za-z0-9_-]+',name) is not None
             and relative and not part.is_absolute() and str(part)==relative
             and not any(x in ('.','..') for x in part.parts),'NO_URI_ESCAPE_OR_AMBIGUITY')
        root=Path(package_roots[name]);need(root.is_absolute() and root.is_dir()
             and root.name==name and not any(c in str(root) for c in (':','\x00','\n')),
             'EXACT_INSTALLED_PACKAGE_ROOT')
        path=root/part
        need(path.resolve().is_relative_to(root.resolve()) and path.is_file(),'MESH_INSIDE_ORIGINAL_PACKAGE')
        pin=mesh_sha256[str(path)];need(type(pin) is str and len(pin)==64,'EXACT_MESH_PIN')
        h=hashlib.sha256();size=0
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):
                size+=len(chunk);need(size<=64*1024**2,'BOUNDED_ORIGINAL_MESH');h.update(chunk)
        need(h.hexdigest()==pin,'ORIGINAL_MESH_BYTES_CHANGED')
        packages.add(name);paths.add(str(path));parents.add(str(root.parent))
    need(packages==set(package_roots) and paths==set(mesh_sha256),'NO_OMITTED_OR_EXTRA_PACKAGE_ASSET')
    # The actual installed Pinocchio rosPaths() reads these two variables.
    # A ROS_PACKAGE_PATH member is the directory containing the named package.
    return dict(ROS_PACKAGE_PATH=':'.join(sorted(parents)),AMENT_PREFIX_PATH='')
