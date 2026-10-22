"""Opt-in byte-identical frozen-shard sharing after a complete native save.

No checkpoint, tensor, optimizer state or parent file is retired. Only a newly
saved duplicate shard is replaced with a hard link to its exact fixed parent.
The complete checkpoint manifest remains byte-for-byte unchanged.
"""
import json
import os
from pathlib import Path
import stat
import struct
import shutil
import time

from scripts.pi0294.native_dataset import sha
from .authority import AUTHORITY, PARENT
from .source_admission import VERSION

POLICY = 'plans/DEC1_CHECKPOINT_STORAGE_POLICY.json'


def fail(reason):
    raise PermissionError('DEC1_CHECKPOINT_STORAGE_'+reason)


def _read(path):
    return json.loads(Path(path).read_bytes())


def _owned_file(path, root=None):
    path=Path(path)
    if (path.is_symlink() or not stat.S_ISREG(path.stat().st_mode)
            or path.stat().st_uid!=os.getuid()
            or root is not None and not path.resolve().is_relative_to(Path(root).resolve())):
        fail('OWN_REGULAR_NON_SYMLINK_MEMBER_ONLY')
    return path


def _once(path, value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    Path(path).chmod(0o400)
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _header(path):
    with Path(path).open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if not 0<length<=8*1024**2:fail('BOUNDED_SAFETENSORS_HEADER')
        header=json.loads(stream.read(length))
    return {name:value for name,value in header.items() if name!='__metadata__'}


def configured(root):
    """Legacy scopes have precisely their original save behavior."""
    root=Path(root);m=_read(root/'RUN_MANIFEST.json')
    sealed=m['execution']['sealed_inputs']
    if POLICY not in sealed:return False
    policy=_owned_file(root/POLICY,root)
    if policy.stat().st_mode&0o222 or sha(policy)!=sealed[POLICY]:
        fail('ORIGINAL_SEALED_OPT_IN_POLICY_REQUIRED')
    a=m['authority'];p=_read(policy)
    if (a.get('training_admission_version')!=VERSION or a.get('amendments')!=[AUTHORITY]
            or p.get('schema')!='DEC1_EXACT_FROZEN_SHARD_SHARING_V1'
            or p.get('authority_release')!=AUTHORITY
            or p.get('parent_checkpoint_manifest_sha256')!=PARENT
            or p.get('checkpoint_retirement') is not False or p.get('parent_chmod') is not False
            or p.get('maximum_shared_shards')!=1 or len(p.get('members',{}))!=1):
        fail('EXPLICIT_DEC1_EXACT_PARENT_NO_RETIREMENT_OR_PERMISSION_CHANGE')
    folder=root/'evidence/DEC1_CHECKPOINT_SHARING'
    if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
        fail('OWN_STORAGE_RECEIPTS_ONLY')
    for intent in folder.glob('*.INTENT.json'):
        _owned_file(intent,root)
        receipt=intent.with_name(intent.name.removesuffix('.INTENT.json')+'.RESULT.json')
        if not receipt.is_file():fail('UNSETTLED_STORAGE_INTENT_READONLY_RECONCILIATION_REQUIRED')
        _owned_file(receipt,root);i=_read(intent);r=_read(receipt)
        if (r.get('status')!='EXACT_FROZEN_SHARDS_SHARED_FULL_STATE_UNCHANGED'
                or r.get('authority_release')!=AUTHORITY
                or any(r.get(k)!=i.get(k) for k in ('job_id','checkpoint_complete_sha256','checkpoint_relative'))
                or r.get('new_optimizer_updates')!=0 or r.get('new_model_loads')!=0
                or r.get('checkpoint_retirement') is not False
                or {v.get('name'):v.get('sha256') for v in r.get('members',[])}!=i.get('members')):
            fail('UNSETTLED_STORAGE_INTENT_INCOMPLETE_OR_MISMATCHED_RECEIPT')
    return True


def share_after_native_save(root, checkpoint, guard):
    root=Path(root);checkpoint=Path(checkpoint)
    if not configured(root):return dict(status='NOT_CONFIGURED_OR_LEGACY_SAVE_UNCHANGED')
    guard()
    m=_read(root/'RUN_MANIFEST.json');policy=_read(root/POLICY)
    if (not os.environ.get('PBS_JOBID') or checkpoint.is_symlink()
            or not checkpoint.resolve().is_relative_to((root/'training').resolve())
            or checkpoint.parent.parent!=root/'training'
            or checkpoint.parent.name not in ('A_CONTROL','B_COVERAGE')):
        fail('NEW_OWN_NATIVE_TRAINING_CHECKPOINT_ONLY')
    complete_path=_owned_file(checkpoint/'CHECKPOINT_COMPLETE.json',checkpoint)
    complete_sha=sha(complete_path);complete=_read(complete_path)
    if (complete.get('full_resume') is not True
            or type(complete.get('global_step')) is not int or not 0<complete['global_step']<=4096
            or checkpoint.name!='checkpoint-'+str(complete.get('global_step'))
            or complete.get('origin_job_id')!=os.environ['PBS_JOBID']
            or complete.get('origin_source_sha')!=m['code']['vla_commit']):
        fail('CURRENT_JOB_COMPLETE_FULL_STATE_BEFORE_SHARING')
    inputs=_read(root/'plans/PV1_TRAIN_INPUTS.private.json')
    parent=Path(inputs['CORR_parent_checkpoint'])
    if parent.is_symlink() or sha(_owned_file(parent/'CHECKPOINT_COMPLETE.json',parent))!=PARENT:
        fail('PROTECTED_ORIGINAL_PARENT_IDENTITY')
    parent_complete=_read(parent/'CHECKPOINT_COMPLETE.json')
    inventory=_read(root/'plans/INITIAL_CORR_PARAMETER_INVENTORY.json')
    destinations=[]
    for name,expected in policy['members'].items():
        relative=Path(name)
        if relative.name!=name or not name.endswith('.safetensors'):
            fail('EXPLICIT_TOP_LEVEL_SAFETENSORS_MEMBER_ONLY')
        source=_owned_file(parent/name,parent);target=_owned_file(checkpoint/name,checkpoint)
        before=source.stat();duplicate=target.stat()
        if (parent_complete['files'].get(name)!=expected or complete['files'].get(name)!=expected
                or before.st_dev!=duplicate.st_dev or before.st_size!=duplicate.st_size
                or sha(source)!=expected or sha(target)!=expected):
            fail('EXACT_COMPLETE_SAME_FILESYSTEM_BYTES_NOT_APPROXIMATE_WEIGHTS')
        tensors=_header(source)
        if (not tensors or _header(target)!=tensors
                or any(not key.startswith('backbone.') or key not in inventory
                       or inventory[key].get('requires_grad') is not False for key in tensors)):
            fail('EVERY_TENSOR_KNOWN_FROZEN_BACKBONE_NOT_MIXED_TRAINABLE_SHARD')
        destinations.append((name,source,target,before,duplicate))
    folder=root/'evidence/DEC1_CHECKPOINT_SHARING'
    if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
        fail('OWN_STORAGE_RECEIPTS_ONLY')
    folder.mkdir(parents=True,exist_ok=True)
    label=checkpoint.parent.name+'-'+checkpoint.name
    intent=folder/(label+'.INTENT.json');receipt=folder/(label+'.RESULT.json')
    if receipt.exists():
        _owned_file(receipt,root)
        prior=_read(receipt)
        if prior['checkpoint_complete_sha256']!=complete_sha:fail('PRIOR_RECEIPT_CHANGED_IDENTITY')
        if not all(os.path.samefile(s,t) and sha(t)==policy['members'][n] for n,s,t,_,_ in destinations):
            fail('PRIOR_SHARED_MEMBER_CHANGED')
        return prior
    if intent.exists():fail('UNSETTLED_STORAGE_INTENT_READONLY_RECONCILIATION_REQUIRED')
    if any(os.path.samefile(s,t) or d.st_nlink!=1 for _,s,t,_,d in destinations):
        fail('PRESERVE_UNREGISTERED_EXISTING_LINKS')
    _once(intent,dict(authority_release=AUTHORITY,checkpoint_complete_sha256=complete_sha,
        checkpoint_relative=checkpoint.relative_to(root).as_posix(),members=policy['members'],
        job_id=os.environ['PBS_JOBID'],epoch=time.time(),checkpoint_retirement=False))
    guard();rows=[]
    for name,source,target,before,duplicate in destinations:
        # Byte and inode witnesses fence the exact new duplicate, not a path
        # discovered by glob. Never chmod a link: that would change the parent.
        current=source.stat();candidate=target.stat()
        if ((current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)!=
                (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
                or (candidate.st_dev,candidate.st_ino,candidate.st_size,candidate.st_mtime_ns,candidate.st_nlink)!=
                (duplicate.st_dev,duplicate.st_ino,duplicate.st_size,duplicate.st_mtime_ns,1)):
            fail('MEMBER_CHANGED_AFTER_INTENT_NO_REPLACEMENT')
        temporary=target.with_name(target.name+'.DEC1-share-'+str(os.getpid()))
        os.link(source,temporary,follow_symlinks=False)
        os.replace(temporary,target)
        fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if (not os.path.samefile(source,target) or sha(target)!=policy['members'][name]
                or source.stat().st_mode!=before.st_mode or sha(complete_path)!=complete_sha):
            fail('POST_REPLACEMENT_FULL_BYTES_PARENT_MODE_AND_MANIFEST')
        rows.append(dict(name=name,sha256=policy['members'][name],logical_bytes=before.st_size,
            parent_checkpoint_manifest_sha256=PARENT,
            preexisting_parent_inode=dict(device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns),
            old_duplicate_allocated_bytes=duplicate.st_blocks*512,new_incremental_payload_bytes=0,
            parent_mode_unchanged=True,full_checkpoint_manifest_unchanged=True))
    result=dict(status='EXACT_FROZEN_SHARDS_SHARED_FULL_STATE_UNCHANGED',
        authority_release=AUTHORITY,job_id=os.environ['PBS_JOBID'],checkpoint_complete_sha256=complete_sha,
        checkpoint_relative=checkpoint.relative_to(root).as_posix(),members=rows,
        shared_file_is_not_an_independent_backup=True,checkpoint_retirement=False,
        new_optimizer_updates=0,new_model_loads=0,completed_epoch=time.time())
    _once(receipt,result)
    guard()
    return result


def prewrite_headroom(root):
    """Optional own-root physical upper; never walk the external data bank."""
    if not configured(root):return
    root=Path(root);policy=_read(root/POLICY);bound=policy.get('prewrite_headroom')
    if bound is None:return
    if (set(bound)!={'resident_upper_bytes','full_save_upper_bytes','metadata_upper_bytes','filesystem_reserve_bytes'}
            or any(type(v) is not int or v<=0 for v in bound.values())
            or bound['full_save_upper_bytes']>bound['resident_upper_bytes']
            or bound['filesystem_reserve_bytes']<40*1024**3):
        fail('PRESEALED_POSITIVE_FULL_SAVE_AND_FINAL_RESERVE')
    parent=Path(_read(root/'plans/PV1_TRAIN_INPUTS.private.json')['CORR_parent_checkpoint'])
    existing=set()
    for name,expected in policy['members'].items():
        p=_owned_file(parent/name,parent)
        if sha(p)!=expected:fail('PREWRITE_PROTECTED_PARENT_CHANGED')
        s=p.stat();existing.add((s.st_dev,s.st_ino))
    seen=set();resident=0;input_aliases=[]
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in list(dirs):
            p=Path(folder)/name
            if p.is_symlink():
                relative=p.relative_to(root).as_posix()
                if relative!='model_inputs/nvidia/Cosmos-Reason2-2B':
                    fail('PREWRITE_OWN_ROOT_NO_DIRECTORY_ESCAPE:'+relative)
                # The mature loader's one sealed input alias is not output.
                # Validate it with the same selector, and never walk its cache.
                binding='NEURAL_BINDINGS.private.json'
                b=_owned_file(root/binding,root)
                sealed=_read(root/'RUN_MANIFEST.json')['execution']['sealed_inputs']
                if b.stat().st_mode&0o222 or sealed.get(binding)!=sha(b):
                    fail('PREWRITE_MODEL_INPUT_REQUIRES_SEALED_BINDING')
                from scripts.pi0294.pv1.model_inputs import preflight
                preflight(root,_read(b))
                dirs.remove(name);input_aliases.append(relative)
            elif p.stat().st_uid!=os.getuid():
                fail('PREWRITE_OWN_ROOT_NO_DIRECTORY_ESCAPE:'+p.relative_to(root).as_posix())
        for name in names:
            p=_owned_file(Path(folder)/name,root);s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen and key not in existing:resident+=max(s.st_size,s.st_blocks*512)
            seen.add(key)
    required=bound['full_save_upper_bytes']+bound['metadata_upper_bytes']
    if resident+required>bound['resident_upper_bytes']:fail('OWN_RESIDENT_PLUS_COMPLETE_TRANSIENT_SAVE_BOUND')
    if shutil.disk_usage(root).free<required+bound['filesystem_reserve_bytes']:
        fail('ACTUAL_FILESYSTEM_FULL_SAVE_AND_FINAL_RESERVE')
    return dict(status='PASS_OWN_RESIDENT_AND_COMPLETE_SAVE_HEADROOM',resident_bytes=resident,
        full_save_and_metadata_upper_bytes=required,approved_input_aliases_not_traversed=input_aliases)


def wrap_trainer(base, root, guard):
    if not configured(root):return base

    class SharedFrozenShardTrainer(base):
        def _save_checkpoint(self, model, trial):
            checkpoint=Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            if (checkpoint.parent.parent!=Path(root)/'training'
                    or checkpoint.parent.name not in ('A_CONTROL','B_COVERAGE')
                    or checkpoint.is_symlink()
                    or not checkpoint.resolve().is_relative_to((Path(root)/'training').resolve())):
                fail('ONLY_OWN_TRAINING_OUTPUT_BEFORE_NATIVE_WRITER')
            if checkpoint.exists():
                # A repeated save must not truncate a parent-linked shard.
                # Preserve the first complete state, just as the mature
                # interrupted-state saver already does; never repair partials.
                guard()
                complete=_read(_owned_file(checkpoint/'CHECKPOINT_COMPLETE.json',checkpoint))
                if (complete.get('full_resume') is not True
                        or complete.get('global_step')!=self.state.global_step
                        or self.consumed!=self.state.global_step*32 or not complete.get('files')):
                    fail('PRESERVE_EXISTING_PARTIAL_OR_DIFFERENT_NATIVE_STATE')
                for name,expected in complete['files'].items():
                    relative=Path(name)
                    if relative.is_absolute() or '..' in relative.parts:
                        fail('EXISTING_COMPLETE_MEMBER_ESCAPE')
                    if sha(_owned_file(checkpoint/name,checkpoint))!=expected:
                        fail('EXISTING_COMPLETE_MEMBER_CHANGED_NO_OVERWRITE')
                return
            if 'prewrite_headroom' in _read(Path(root)/POLICY):
                guard();prewrite_headroom(root)
            super()._save_checkpoint(model,trial)
            share_after_native_save(root,checkpoint,guard)

    return SharedFrozenShardTrainer
