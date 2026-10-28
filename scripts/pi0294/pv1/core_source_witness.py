"""Exact read-only source custody after one complete native hash audit.

The certificate must be byte-pinned in the new request AND its registered row.
It changes no resource transaction, source bytes, RPC timeout or science gate.
Only stable kernel file/directory identities replace repeated full byte reads.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

POLICY='V5_CD_NATIVE_FULL_HASH_THEN_EXACT_KERNEL_SOURCE_CUSTODY_V1'
HELPER='scripts/pi0294/pv1/core_source_witness.py'
MAX_FILES=2048
KERNEL_TIMESTAMP_CLOSE_NS=2_000_000_000


def need(value,why):
    if not value:raise PermissionError('CORE_SOURCE_CUSTODY_'+why)


def raw(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def sha(value):return hashlib.sha256(value).hexdigest()


def identity(path,*,directory=False):
    s=path.lstat()
    need((stat.S_ISDIR(s.st_mode)if directory else stat.S_ISREG(s.st_mode))
         and s.st_uid==os.getuid()and not s.st_mode&0o222,'OWN_IMMUTABLE_REGULAR_OR_DIRECTORY')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid]


def members(code,seal):
    code=Path(code);seal=Path(seal);sb=seal.read_bytes();doc=json.loads(sb)
    files=doc['files'];need(type(files)is dict and 0<len(files)<=MAX_FILES,'BOUNDED_COMPLETE_SEAL')
    need(code.is_absolute()and not code.is_symlink(),'EXACT_SOURCE_ROOT')
    names={}
    directories={'.'}
    for name,pin in files.items():
        relative=Path(name)
        need(type(name)is str and relative.as_posix()==name and not relative.is_absolute()
             and '..'not in relative.parts and relative.parts and type(pin)is str
             and len(pin)==64 and all(c in'0123456789abcdef'for c in pin),'EXACT_SEALED_MEMBER')
        names[name]=code/relative
        directories.update(p.as_posix()for p in relative.parents if p!=Path('.'))
    return doc,sb,names,{name:code if name=='.'else code/name for name in sorted(directories)}


def make(code,seal,*,expected_seal_sha256):
    """Full hash, before/after kernel custody, no certificate write or GO."""
    code,seal=Path(code),Path(seal);doc,sb,files,dirs=members(code,seal)
    need(sha(sb)==expected_seal_sha256,'EXACT_SEAL_BYTES')
    directory_state={name:identity(path,directory=True)for name,path in dirs.items()}
    seal_state=identity(seal);states={};created=time.time_ns()
    for name,path in files.items():
        before=identity(path);need(sha(path.read_bytes())==doc['files'][name],'FULL_HASH:'+name)
        need(identity(path)==before,'UNCHANGED_DURING_FULL_HASH:'+name);states[name]=before
    need(identity(seal)==seal_state and seal.read_bytes()==sb
         and all(identity(path,directory=True)==directory_state[name]for name,path in dirs.items()),
         'WHOLE_SOURCE_STABLE_DURING_HASH')
    # Some filesystems coarsen ctime within a clock tick. Never certify files
    # from that open tick: a rewrite+restoredmtime could share their ctime.
    need(all(state[4]<created-KERNEL_TIMESTAMP_CLOSE_NS for state in [*states.values(),*directory_state.values(),seal_state]),
         'NATIVE_CTIME_WINDOW_MUST_BE_CLOSED_BEFORE_FULL_AUDIT')
    return dict(policy=POLICY,source_sha=doc['source_sha'],source_seal_sha256=sha(sb),
        created_wall_ns=created,kernel_timestamp_close_ns=KERNEL_TIMESTAMP_CLOSE_NS,
        complete_full_hash_performed=True,files=states,directories=directory_state,seal_identity=seal_state,
        source_root=str(code),source_seal=str(seal),native_or_model_admission=False)


def verify(code,seal,certificate_bytes,*,expected_certificate_sha256,expected_seal_sha256):
    """Reject any replacement/chmod/rewrite/symlink, including restoredmtime."""
    need(type(certificate_bytes)is bytes and sha(certificate_bytes)==expected_certificate_sha256,
         'REGISTERED_CERTIFICATE_BYTES')
    code,seal=Path(code),Path(seal);doc,sb,files,dirs=members(code,seal);c=json.loads(certificate_bytes)
    need(c.get('policy')==POLICY and c.get('complete_full_hash_performed')is True
         and c.get('source_sha')==doc['source_sha']and sha(sb)==expected_seal_sha256==c.get('source_seal_sha256')
         and c.get('source_root')==str(code)and c.get('source_seal')==str(seal)
         and set(c.get('files',{}))==set(files)and set(c.get('directories',{}))==set(dirs)
         and c.get('seal_identity')==identity(seal)
         and c.get('kernel_timestamp_close_ns')==KERNEL_TIMESTAMP_CLOSE_NS
         and type(c.get('created_wall_ns'))is int and time.time_ns()>=c['created_wall_ns']
         and all(state[4]<c['created_wall_ns']-KERNEL_TIMESTAMP_CLOSE_NS for state in [*c.get('files',{}).values(),*c.get('directories',{}).values(),c.get('seal_identity',[])]),
         'SAME_COMPLETE_FULL_HASH_CERTIFICATE')
    need(all(identity(path,directory=True)==c['directories'][name]for name,path in dirs.items()),
         'UNCHANGED_ALL_SOURCE_DIRECTORIES')
    need(all(identity(path)==c['files'][name]for name,path in files.items()),
         'UNCHANGED_ALL_SOURCE_KERNEL_IDENTITIES')
    return dict(policy=POLICY,source_sha=doc['source_sha'],source_seal_sha256=sha(sb),
        certificate_sha256=expected_certificate_sha256,verified_files=len(files),read_only=True)

CERTIFICATE='plans/CORE_REMOTE_SOURCE_CUSTODY.private.json'
FIELD='remote_source_custody_sha256'


def row_contract(row):
    value=row.get(FIELD)
    if value is None:return None
    need(type(value)is str and len(value)==64 and all(c in'0123456789abcdef'for c in value)
         and row.get('evaluation_candidate')in('C50','D25'),'EXPLICIT_NEW_CD_CERTIFICATE')
    return value


def request_contract(manifest,request):
    pin=row_contract(request);sealed=manifest.get('execution',{}).get('sealed_inputs',{})
    need((CERTIFICATE not in sealed)if pin is None else sealed.get(CERTIFICATE)==pin,
         'SAME_PRESEALED_REQUEST_AND_CERTIFICATE')
    return pin
