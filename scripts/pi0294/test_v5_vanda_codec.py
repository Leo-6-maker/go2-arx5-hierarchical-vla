"""Codec metadata/wrapper fixtures only; actual ABI needs the server CPU test."""
from copy import deepcopy
from pathlib import Path
import os

import pytest
from scripts.pi0294.pv1 import mix_vanda_codec as c,mix_accounting as a


def fixture(tmp_path,monkeypatch):
    tools=tmp_path/'audit-codec';files={}
    elf=b'\x7fELF\x02\x01'+bytes(12)+(62).to_bytes(2,'little')+b'explicit-cpu-fixture'
    values={'bin/ffmpeg':c.wrapper(tools,'ffmpeg'),'bin/ffprobe':c.wrapper(tools,'ffprobe'),
        'libexec/ffmpeg':elf,'libexec/ffprobe':elf,'lib/libavcodec.so.60':elf,
        'licenses/ffmpeg-copyright.txt':b'licence fixture, not actual package evidence'}
    for n,data in values.items():
        p=tools/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o500 if n.startswith(('bin/','libexec/')) else 0o400)
        files[n]=dict(bytes=len(data),sha256=a.sha(data))
    value=dict(schema=c.SCHEMA,source_kind='READONLY_EXISTING_INSTALLED_UBUNTU_PACKAGES',source_ffmpeg_version='ffmpeg version 6.1.1-3ubuntu5 CPU fixture',
        physics_SIF_sha256='2cae97d18c6a41450bb102e3bd54819d0c30552605d222d73ffc387c6ec14c8c',parent_LD_LIBRARY_PATH_modified=False,
        source_license_files_retained=True,files=files)
    p=tmp_path/c.MANIFEST;p.parent.mkdir();p.write_bytes(a.raw(value));p.chmod(0o400)
    m=dict(execution=dict(sealed_inputs={c.MANIFEST:a.sha(p.read_bytes())}))
    monkeypatch.setenv('PATH',str(tools/'bin')+':'+os.environ['PATH'])
    return value,m


def test_only_codec_child_loader_gets_its_library_path(tmp_path,monkeypatch):
    value,m=fixture(tmp_path,monkeypatch);before=os.environ.get('LD_LIBRARY_PATH')
    assert c.validate(tmp_path,m)==value
    assert os.environ.get('LD_LIBRARY_PATH')==before
    script=(tmp_path/'audit-codec/bin/ffmpeg').read_text()
    assert '--library-path' in script and 'export' not in script and 'LD_LIBRARY_PATH=' not in script
    assert '"$@"' in script


@pytest.mark.parametrize('change',['changed_binary','writable','another_path','parent_environment','missing_licence','fake_ELF','source_link','different_SIF'])
def test_unsealed_or_intrusive_codec_cannot_enter_source_runtime(tmp_path,monkeypatch,change):
    value,m=fixture(tmp_path,monkeypatch);path=tmp_path/c.MANIFEST
    if change=='changed_binary':
        p=tmp_path/'audit-codec/libexec/ffmpeg';p.chmod(0o600);p.write_bytes(b'changed');p.chmod(0o400)
    elif change=='writable':(tmp_path/'audit-codec/bin/ffmpeg').chmod(0o700)
    elif change=='another_path':monkeypatch.setenv('PATH','/other/bin:'+os.environ['PATH'])
    elif change=='source_link':
        p=tmp_path/'audit-codec/lib/libavcodec.so.60';original=p.read_bytes();p.unlink();target=tmp_path/'else';target.write_bytes(original);p.symlink_to(target)
    else:
        if change=='parent_environment':value['parent_LD_LIBRARY_PATH_modified']=True
        elif change=='missing_licence':value['source_license_files_retained']=False
        elif change=='different_SIF':value['physics_SIF_sha256']='0'*64
        else:
            p=tmp_path/'audit-codec/libexec/ffmpeg';p.chmod(0o600);p.write_bytes(b'not-ELF');p.chmod(0o400)
            value['files']['libexec/ffmpeg']=dict(bytes=7,sha256=a.sha(b'not-ELF'))
        path.chmod(0o600);path.write_bytes(a.raw(value));path.chmod(0o400);m['execution']['sealed_inputs'][c.MANIFEST]=a.sha(path.read_bytes())
    with pytest.raises(PermissionError):c.validate(tmp_path,m)
