"""Readonly audit-codec overlay; the original physics SIF stays unchanged.

Only the ffmpeg/ffprobe subprocess gets its private library search path. No
LD_LIBRARY_PATH is exported to MuJoCo, the model, S1 or the parent process.
The original audit owns all encoding arguments and independent full decoding.
"""
import os
from pathlib import Path
import re
import shlex

from scripts.pi0294.native_dataset import sha
from scripts.pi0292.durable import read
from . import mix_accounting as a

MANIFEST='plans/VANDA_AUDIT_CODEC_TOOLCHAIN.private.json'
SCHEMA='V5_CD_ORIGINAL_AUDITOR_EXISTING_FFMPEG_OVERLAY_V1'
GLIBC={'libc.so.6','libm.so.6','libdl.so.2','librt.so.1','libpthread.so.0','libresolv.so.2','libutil.so.1'}


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_AUDIT_CODEC_'+why)


def wrapper(tool_root,name):
    root=Path(tool_root)
    need(root.is_absolute() and name in ('ffmpeg','ffprobe'),'ONLY_ORIGINAL_AUDIT_EXECUTABLES')
    libs=str(root/'lib')+':/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu'
    return ('#!/bin/sh\nexec /lib64/ld-linux-x86-64.so.2 --library-path '+shlex.quote(libs)+' '+
            shlex.quote(str(root/'libexec'/name))+' "$@"\n').encode()


def validate(root,manifest):
    root=Path(root);path=root/MANIFEST
    need(path.is_file() and not path.is_symlink() and not path.stat().st_mode&0o222
         and manifest['execution']['sealed_inputs'].get(MANIFEST)==sha(path),'SEALED_ACTUAL_TOOLCHAIN_METADATA')
    value=read(path);tools=root/'audit-codec';files=value['files']
    need(value['schema']==SCHEMA and value['source_kind']=='READONLY_EXISTING_INSTALLED_UBUNTU_PACKAGES'
         and value['source_ffmpeg_version'].startswith('ffmpeg version 6.1.1-3ubuntu5')
         and value['physics_SIF_sha256']=='2cae97d18c6a41450bb102e3bd54819d0c30552605d222d73ffc387c6ec14c8c'
         and value['parent_LD_LIBRARY_PATH_modified'] is False
         and type(files) is dict and 5<=len(files)<=512
         and {'bin/ffmpeg','bin/ffprobe','libexec/ffmpeg','libexec/ffprobe'}<=set(files),
         'EXISTING_CODEC_AND_UNMODIFIED_PHYSICS_RUNTIME')
    need(tools.is_dir() and not tools.is_symlink(),'REAL_TOOLCHAIN_DIRECTORY')
    total=0
    for name,record in files.items():
        p=Path(name);need(not p.is_absolute() and len(p.parts)==2 and p.parts[0] in ('bin','lib','libexec','licenses')
            and '..' not in p.parts and re.fullmatch(r'[A-Za-z0-9_.+-]+',p.name),'EXACT_READONLY_CODEC_MEMBER')
        need(p.name not in GLIBC and not p.name.startswith('ld-linux'),'CONTAINER_SYSTEM_LOADER_AND_GLIBC_UNCHANGED')
        need((tools/p.parts[0]).is_dir() and not (tools/p.parts[0]).is_symlink(),'NO_TOOLCHAIN_DIRECTORY_LINK_ESCAPE')
        need(p.parts[0] not in ('bin','libexec') or p.name in ('ffmpeg','ffprobe'),'NO_ADDITIONAL_EXECUTABLE_ENTRY')
        need(p.parts[0]!='lib' or re.fullmatch(r'lib[A-Za-z0-9_.+-]+\.so(?:\.[A-Za-z0-9_.+-]+)?',p.name),'ONLY_ACTUAL_NAMED_CODEC_DSO')
        file=tools/p;st=file.lstat()
        import stat
        need(stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid() and not st.st_mode&0o222
             and st.st_size==record['bytes'] and sha(file)==record['sha256'],'EXACT_INSTALLED_PACKAGE_BYTES_'+name)
        total+=st.st_size
        if p.parts[0] in ('lib','libexec'):
            with file.open('rb') as f:header=f.read(20)
            need(len(header)==20 and header[:6]==b'\x7fELF\x02\x01' and int.from_bytes(header[18:20],'little')==62,
                 'ACTUAL_AMD64_ELF_NOT_CODE_OR_MODEL_PAYLOAD')
    need(total<=512*1024**2 and all((tools/'bin'/n).read_bytes()==wrapper(tools,n) for n in ('ffmpeg','ffprobe')),
         'FINITE_ORIGINAL_AUDIT_ONLY_WRAPPERS')
    need(os.environ.get('PATH','').split(':')[0]==str(tools/'bin'),'ONLY_THIS_AUDITOR_SUBPROCESS_PATH')
    need(value['source_license_files_retained'] is True,'EXISTING_DISTRIBUTION_LICENSE_BYTES_RETAINED')
    need(any(n.startswith('licenses/ffmpeg') for n in files),'ACTUAL_FFMPEG_DISTRIBUTION_COPYRIGHT_MEMBER')
    return value
