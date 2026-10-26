"""Exact inference-only references to the two already closed Vanda endpoints.

No tensor copying, hardlinking, chmod, training or native permission. Symlinks
are allowed ONLY for these nine sealed input members; ordinary output and
source validators continue to reject links. The original inode metadata is
unchanged, unlike a hardlink view that would change checkpoint ctime/nlink.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import stat

from . import mix_accounting as a, mix_evaluation_transfer as delivery
from . import mix_vanda_cohost as cohost


def need(ok, reason):
    if not ok:
        raise PermissionError('V5_VANDA_REFERENCE_'+reason)


def prepare(global_bytes, capture_bytes, small_bytes, *, arm):
    old = delivery.prepare(capture_bytes,small_bytes,arm=arm)
    capture = json.loads(capture_bytes)
    current = cohost.endpoint(global_bytes,capture)
    _,_,audit = delivery.endpoint(capture_bytes)
    complete_bytes = base64.b64decode(old['private_restoration']['original_complete_base64'],validate=True)
    complete = json.loads(complete_bytes)
    root = Path(audit['checkpoint_root'])
    files = {}
    for name in old['public_plan']['files']:
        need(not Path(name).is_absolute() and '..' not in Path(name).parts
             and not any(x in name.lower() for x in ('optimizer','scheduler','rng','sampler')), 'NINE_INFERENCE_MEMBERS_ONLY')
        rows = [x for x in audit['streamed_witness']['paths'] if Path(x['path']) == root/name]
        need(len(rows) == 1 and rows[0]['sha256'] == complete['files'][name], 'ACTUAL_AUDITED_MEMBER:'+name)
        files[name] = dict(path=str(root/name),sha256=complete['files'][name],bytes=rows[0]['bytes'])
    # Public transfer metadata rewrites private processor paths and is 242
    # bytes shorter for these endpoints. A local reference keeps the ORIGINAL
    # nine members; its sizes come from their independent full-state audit.
    need(len(files) == 9 and 0 < sum(x['bytes'] for x in files.values()) < 16*1024**3,
         'ORIGINAL_COMPLETE_NINE_MEMBER_PAYLOAD')
    return dict(schema='V5_CD_VANDA_CLOSED1024_READONLY_INFERENCE_REFERENCE_V1',arm=arm,
        checkpoint_root=str(root),checkpoint_manifest_sha256=current['checkpoint_manifest_sha256'],
        handoff=current,files=files,original_complete_base64=base64.b64encode(complete_bytes).decode(),
        tensor_copies=0,source_hardlinks=0,source_chmod=False,source_full_resume_retained=True,
        inference_view_full_resume_payload=False,model_loaded=False,native_admission=False)


def metadata(path):
    p = Path(path); st = p.lstat()
    need(stat.S_ISREG(st.st_mode) and p.resolve() == p and st.st_uid == os.getuid(), 'OWN_CANONICAL_REGULAR_SOURCE')
    return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,mode=stat.S_IMODE(st.st_mode),nlink=st.st_nlink)


def audit_reference(plan):
    need(plan['schema'] == 'V5_CD_VANDA_CLOSED1024_READONLY_INFERENCE_REFERENCE_V1'
         and plan['arm'] in ('C50','D25') and len(plan['files']) == 9
         and plan['tensor_copies'] == plan['source_hardlinks'] == 0 and plan['source_chmod'] is False,
         'EXPLICIT_NINE_MEMBER_READONLY_PLAN')
    rows = {}
    root = Path(plan['checkpoint_root'])
    for name,member in plan['files'].items():
        path = root/name
        need(str(path) == member['path'] and path.is_relative_to(root) and '..' not in Path(name).parts,
             'EXACT_SOURCE_MEMBER_PATH')
        before = metadata(path); need(before['bytes'] == member['bytes'], 'EXACT_MEMBER_SIZE')
        h = hashlib.sha256()
        fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            st = os.fstat(stream.fileno())
            need((st.st_dev,st.st_ino) == (before['device'],before['inode']), 'SAME_OPENED_INODE')
            for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
        need(metadata(path) == before and h.hexdigest() == member['sha256'], 'FULL_CURRENT_SHA_AND_NO_CONCURRENT_SOURCE_WRITE')
        rows[name] = dict(before,sha256=h.hexdigest())
    return dict(schema='V5_CD_VANDA_NINE_INFERENCE_REFERENCE_STREAM_AUDIT_V1',arm=plan['arm'],
        reference_plan_sha256=a.sha(a.raw(plan)),members=rows,model_loaded=False,new_tensors_written=0)


def create_view(view, plan, audit):
    """Called only by a separately admitted, registered cold input packer."""
    view = Path(view)
    need(view.is_absolute() and not view.exists() and not view.is_symlink(), 'NEW_EXACT_REGISTERED_INPUT_VIEW')
    need(audit['reference_plan_sha256'] == a.sha(a.raw(plan)) and audit['arm'] == plan['arm']
         and set(audit['members']) == set(plan['files']), 'FULL_STREAMED_CURRENT_REFERENCE_AUDIT')
    before = {}
    for name,member in plan['files'].items():
        before[name] = metadata(member['path'])
        need(before[name] == {k:v for k,v in audit['members'][name].items() if k != 'sha256'}
             and audit['members'][name]['sha256'] == member['sha256'], 'UNCHANGED_AUDITED_SOURCE_BEFORE_VIEW')
    view.mkdir(mode=0o700)
    for name,member in plan['files'].items():
        dest = view/name; dest.parent.mkdir(exist_ok=True,parents=True,mode=0o700)
        dest.symlink_to(member['path'])
    complete = base64.b64decode(plan['original_complete_base64'],validate=True)
    need(a.sha(complete) == plan['checkpoint_manifest_sha256'], 'ORIGINAL_COMPLETE_MANIFEST_BYTES')
    fd = os.open(view/'CHECKPOINT_COMPLETE.json',os.O_WRONLY|os.O_EXCL|os.O_CREAT|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb') as f: f.write(complete); f.flush(); os.fsync(f.fileno())
    for name,member in plan['files'].items():
        need(metadata(member['path']) == before[name] and (view/name).resolve() == Path(member['path']),
             'REFERENCE_DID_NOT_TOUCH_SOURCE_CTIME_MODE_OR_INODE')
    return dict(status='EXACT_NINE_MEMBER_REFERENCE_VIEW_CREATED_NO_MODEL',view=str(view),
                plan_sha256=a.sha(a.raw(plan)),new_tensor_bytes=0,source_inodes_unchanged=True,
                full_resume_payload_in_view=False,native_admission=False)


def inference_member(root,checkpoint,name,digest):
    """Narrow original-loader seam for these nine sealed local input links.

    It does not make arbitrary external checkpoint members legal. The native
    entry has separately validated the live allocation, phase and actor. The
    original loader still streams and compares the complete member SHA.
    """
    from . import mix_vanda_eval_identity as identity
    root,checkpoint=Path(root),Path(checkpoint);m=json.loads((root/'RUN_MANIFEST.json').read_bytes())
    need(identity.cold_profile(m) and checkpoint==root/'model_views'/m['candidate'],'ONLY_NEW_TYPED_NINE_MEMBER_VIEW')
    name_path=Path(name);need(not name_path.is_absolute() and '..' not in name_path.parts,'EXACT_MEMBER_NAME')
    plan_path=root/'plans/VANDA_READONLY_INFERENCE_REFERENCE.private.json';plan_bytes=plan_path.read_bytes()
    need(not plan_path.is_symlink() and not plan_path.stat().st_mode&0o222
         and m['execution']['sealed_inputs'].get(str(plan_path.relative_to(root)))==a.sha(plan_bytes),'IMMUTABLE_PRESEALED_REFERENCE_PLAN')
    plan=json.loads(plan_bytes);member=plan['files'].get(name)
    need(plan['schema']=='V5_CD_VANDA_CLOSED1024_READONLY_INFERENCE_REFERENCE_V1'
         and plan['arm']==m['candidate'] and len(plan['files'])==9
         and plan['tensor_copies']==plan['source_hardlinks']==0 and plan['source_chmod'] is False
         and type(member) is dict and member['sha256']==digest
         and plan['checkpoint_manifest_sha256']==a.sha((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes()),
         'EXACT_AUDITED_ORIGINAL_INFERENCE_MEMBER')
    p=checkpoint/name
    need(p.is_symlink() and p.resolve()==Path(member['path'])
         and Path(member['path']).parent==Path(plan['checkpoint_root'])/name_path.parent
         and metadata(member['path'])['bytes']==member['bytes'],'EXACT_ORIGINAL_CANONICAL_READONLY_INPUT')
    return p
