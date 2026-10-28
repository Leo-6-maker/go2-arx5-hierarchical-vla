"""Retain the nonzero closed prefix and bound all new output by its paid gross.

No tensors are streamed, copied, removed or credited here. Old zero-effect
metadata retains its separate full-content portable proof. The nonzero prefix
is tied to the original independent full-stream witness and native filesystem.
"""
from copy import deepcopy
import json,os,stat,time
from pathlib import Path
from . import minimal_remaining_contract as r,minimal_execution as e
from . import minimal_portable_inventory as portable,minimal_retry_physical_guard as original
from .minimal_remaining_inputs import Inputs,PLAN,own_body

SCHEMA='ARX5_MINIMAL_RETAINED_NONZERO_PREFIX_AND_NEW_OUTPUT_CUSTODY_V1'
from .minimal_retry_runtime_guard import CONTROL_GROWTH_RESERVE
need=e.need
PAYLOAD={'.safetensors','.pt','.pth','.ckpt','.bin'}


def identity(value):
    return portable.digest(dict(value,allocated_bytes=None,
        rows=[{k:v for k,v in row.items()if k not in('device','allocated_bytes')}for row in value['rows']]))


def prefix_inventory(root,*,witnesses=None):
    root=Path(root);need(root.is_absolute()and root.resolve()==root and root.is_dir(),
        'REMAINING_PHYSICAL_EXACT_REAL_PREFIX_ROOT')
    rows=[];inodes={};links={}
    for path in[root,*sorted(root.rglob('*'))]:
        s=path.lstat();name='.'if path==root else path.relative_to(root).as_posix()
        if stat.S_ISLNK(s.st_mode):
            need(name=='model_inputs/nvidia/Cosmos-Reason2-2B','REMAINING_PHYSICAL_ONLY_ORIGINAL_INPUT_ALIAS')
            links[name]=os.readlink(path);continue
        need(s.st_uid==os.getuid()and(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),
            'REMAINING_PHYSICAL_OWN_REGULAR_PREFIX')
        if witnesses is not None and stat.S_ISREG(s.st_mode)and path.suffix in PAYLOAD:
            old=witnesses.get(str(path))
            need(old is not None and stat.S_IMODE(s.st_mode)==0o400
                and(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(old[k]for k in('device','inode','bytes','mtime_ns','ctime_ns')),
                'REMAINING_PHYSICAL_EVERY_RETAINED_TENSOR_HAS_REAL_UNCHANGED_FULL_STREAM_WITNESS')
        key=s.st_dev,s.st_ino
        inodes[key]=max(inodes.get(key,0),s.st_blocks*512)
        rows.append(dict(path=name,device=s.st_dev,inode=s.st_ino,bytes=s.st_size,allocated_bytes=s.st_blocks*512,
            links=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,uid=s.st_uid,mode=stat.S_IMODE(s.st_mode),
            kind='file'if stat.S_ISREG(s.st_mode)else'directory'))
    return dict(schema=SCHEMA,root=str(root),filesystem=portable.filesystem(root),rows=rows,
        input_symlinks=links,allocated_bytes=sum(inodes.values()),
        scientific_payload_full_stream_witness_required=True,other_closed_control_metadata_stat_only=True)


def validate_custody(p,row,*,prefix_binding):
    from . import minimal_remaining_execution as execution
    need(p['schema']==SCHEMA and p['request_id']==row['request_id']and p['submission_nonce']==row['submission_nonce']
        and p['run_uuid']==row['run_uuid']==execution.RUN and p['saved_prefix_binding']==prefix_binding
        and p['prefix_closed_lease_sha256']==r.authority.LEASES['science']
        and p['paid_output_ceiling']==execution.PAID_OUTPUT and p['old_metadata_ceiling']==256*1024**2
        and p['additional_output_charge_bytes']==0,
        'REMAINING_PHYSICAL_EXACT_NONZERO_PREFIX_AND_ORIGINAL_PAID_GROSS')
    need(type(p['prefix_initial_allocated_bytes'])is int and 0<=p['prefix_initial_allocated_bytes']<=p['paid_output_ceiling'],
        'REMAINING_PHYSICAL_INITIAL_PREFIX_ALLOCATION_IS_NOT_A_REFUND')
    expected=['arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02',
        'arx5-minimal-raw-parent-first512-retry-01','arx5-minimal-raw-parent-first512-retry-02']
    need([x['run_uuid']for x in p['old_metadata_roots']]==expected
        and Path(p['prefix_root']).name=='arx5-minimal-raw-parent-first512-retry-03'
        and p['scientific_payload_full_stream_witness_required']is True
        and p['other_closed_control_metadata_stat_only']is True
        and p['full_streamed_prefix_witnesses_sha256']and p['raw_parent_audit_sha256'],
        'REMAINING_PHYSICAL_EXACT_FOUR_OLD_METADATA_ROOTS_AND_NONZERO_FIFTH')
    for key in('prefix_identity_sha256','prefix_inventory_sha256','old_metadata_portable_sha256','old_metadata_identity_sha256'):
        need(type(p.get(key))is str and len(p[key])==64,'REMAINING_PHYSICAL_EXACT_REFERENCED_INPUT_SHA')
    return p


def prepare(prefix,request,old_physical):
    """Current login-side inputs only; no new root, registration or permission."""
    from . import minimal_remaining_execution as execution
    need(isinstance(prefix,Inputs)and prefix.root is None,'REMAINING_PHYSICAL_ORIGINAL_CLOSED_LOGIN_INPUTS')
    prefix.verify_current();snapshot=prefix_inventory(prefix.science,witnesses=prefix.paths)
    p=dict(schema=SCHEMA,request_id=request['request_id'],submission_nonce=request['submission_nonce'],run_uuid=request['run_uuid'],
        saved_prefix_binding=deepcopy(prefix.binding),prefix_closed_lease_sha256=r.authority.LEASES['science'],
        paid_output_ceiling=execution.PAID_OUTPUT,old_metadata_ceiling=256*1024**2,additional_output_charge_bytes=0,
        prefix_root=str(prefix.science),prefix_filesystem=snapshot['filesystem'],prefix_initial_allocated_bytes=snapshot['allocated_bytes'],
        prefix_inventory_sha256=e.sha(e.raw(snapshot)),prefix_identity_sha256=identity(snapshot),
        raw_parent_filesystem=portable.filesystem(prefix.raw_checkpoint),
        old_metadata_roots=deepcopy(old_physical['old_roots']),prior_physical_custody_sha256=e.sha(e.raw(old_physical)),
        old_metadata_portable_sha256=old_physical['portable_inventory']['portable_sha256'],
        old_metadata_identity_sha256=portable.identity_digest(old_physical['portable_inventory']['value']),
        full_streamed_prefix_witnesses_sha256=prefix.result['full_streamed_witnesses_sha256'],
        raw_parent_audit_sha256=prefix.result['raw_parent_post_F_audit_sha256'],
        scientific_payload_full_stream_witness_required=True,other_closed_control_metadata_stat_only=True)
    validate_custody(p,request,prefix_binding=prefix.binding)
    plan=dict(prefix.plan,portable_prefix_filesystem=p['prefix_filesystem'],portable_raw_parent_filesystem=p['raw_parent_filesystem'])
    return dict(custody=p,prefix_storage_inventory=snapshot,readonly_input_plan=plan,
        new_native_permission=False,tensor_bytes_restreamed=0)


def check_proof(proof,p,row,*,now_epoch,observed_epoch):
    need(0<=now_epoch-observed_epoch<=120 and proof['observed_epoch']==observed_epoch
        and proof['status']=='ACTUAL_REMAINING_NONZERO_PREFIX_AND_NEW_PHYSICAL_BOUNDS_NO_PERMISSION'
        and proof['custody_sha256']==row['physical_custody_sha256']==e.sha(e.raw(p))
        and proof['prefix_identity_sha256']==p['prefix_identity_sha256']
        and proof['old_metadata_identity_sha256']==p['old_metadata_identity_sha256']
        and proof['paid_output_ceiling']==p['paid_output_ceiling']
        and proof['old_metadata_ceiling']==p['old_metadata_ceiling']
        and type(proof['prefix_allocated_bytes'])is int and proof['prefix_allocated_bytes']>=p['prefix_initial_allocated_bytes']
        and type(proof['new_allocated_bytes'])is int and proof['new_allocated_bytes']>=0
        and type(proof['old_metadata_allocated_bytes'])is int and 0<=proof['old_metadata_allocated_bytes']<=p['old_metadata_ceiling']
        and type(proof['next_output_upper_bytes'])is int and proof['next_output_upper_bytes']>=CONTROL_GROWTH_RESERVE
        and proof['prefix_allocated_bytes']+proof['new_allocated_bytes']+proof['next_output_upper_bytes']<=p['paid_output_ceiling']
        and proof['ledger_changed']is proof['native_permission']is False,
        'REMAINING_PHYSICAL_FRESH_UNIQUE_INODE_CURRENT_PREFIX_PLUS_NEW_GROSS_AND_CLOSE_RESERVE')
    return proof


class Guard:
    def __init__(self,root,request,manifest,original_guard,*,clock=time.monotonic):
        from scripts.pi0294.dec1.a40_training_context import sealed
        from . import minimal_remaining_execution as execution
        self.root=Path(root);self.request=request;self.original=original_guard;self.clock=clock
        self.last=None;self.cache={};self.custody=sealed(root,execution.PHYSICAL)
        self.prefix=None;self.manifest=manifest
        prefix=r.saved_prefix((self.root/r.PREFIX).read_bytes(),(self.root/r.CLOSURE).read_bytes())
        validate_custody(self.custody,request,prefix_binding=prefix)
        need(manifest['execution']['sealed_inputs'].get(execution.PHYSICAL)==request['physical_custody_sha256']
            and e.sha(e.raw(self.custody))==request['physical_custody_sha256'],
            'REMAINING_PHYSICAL_CURRENT_MANIFEST_AND_REQUEST_SEAL')
        self.prefix_snapshot=sealed(root,execution.PREFIX_STORAGE)
        need(e.sha(e.raw(self.prefix_snapshot))==manifest['execution']['sealed_inputs'].get(execution.PREFIX_STORAGE)==self.custody['prefix_inventory_sha256']
            and identity(self.prefix_snapshot)==self.custody['prefix_identity_sha256']
            and self.prefix_snapshot['filesystem']==self.custody['prefix_filesystem']
            and self.prefix_snapshot['root']==self.custody['prefix_root']
            and self.prefix_snapshot['allocated_bytes']==self.custody['prefix_initial_allocated_bytes'],
            'REMAINING_PHYSICAL_SEALED_COMPLETE_PREFIX_STORAGE_INPUT')
        prior=Path(self.custody['prefix_root'])/'plans/MINIMAL_RETRY_PHYSICAL_CUSTODY.private.json'
        old_bytes=own_body(prior,maximum=8*1024**2)
        need(e.sha(old_bytes)==self.custody['prior_physical_custody_sha256'],
            'REMAINING_PHYSICAL_ORIGINAL_OLD_METADATA_BYTES_REFERENCED_NOT_DUPLICATED')
        self.old_inventory=json.loads(old_bytes)['portable_inventory']
        need(self.old_inventory['portable_sha256']==self.custody['old_metadata_portable_sha256']
            and portable.identity_digest(self.old_inventory['value'])==self.custody['old_metadata_identity_sha256'],
            'REMAINING_PHYSICAL_REAL_ORIGINAL_FOUR_METADATA_ROOTS')
        self.metadata=[]
        for item in self.custody['old_metadata_roots']:
            path=self.root.parent/item['run_uuid'];binding=json.loads((path/'OUTPUT_BINDING.private.json').read_bytes())
            need(e.sha((path/'RUN_MANIFEST.json').read_bytes())==item['manifest_sha256']
                and e.sha(Path(binding['lease']).read_bytes())==item['closed_lease_sha256']
                and json.loads(Path(binding['lease']).read_bytes())['status']=='CLOSED',
                'REMAINING_PHYSICAL_ALL_OLD_ZERO_METADATA_ACTUALLY_CLOSED')
            self.metadata.append(str(path))

    def inspect(self,*,next_output_upper_bytes=0):
        p=self.custody;self.original()
        if self.prefix is None:self.prefix=Inputs(self.root,self.manifest,guard=self.original)
        need(str(self.prefix.science)==p['prefix_root'],'REMAINING_PHYSICAL_SAME_AUDITED_PREFIX_ROOT')
        self.prefix.verify_current()
        prior=prefix_inventory(self.prefix.science)
        need(identity(prior)==p['prefix_identity_sha256'],
            'REMAINING_PHYSICAL_CLOSED_PREFIX_FILE_OR_FILESYSTEM_CHANGED')
        metadata=portable.capture(self.metadata,metadata_ceiling=p['old_metadata_ceiling'],file_hash_cache=self.cache)
        portable.verify(self.old_inventory,metadata)
        current,_=original._scan(self.root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
        all_old={(x['device'],x['inode'])for x in prior['rows']}
        for root in self.metadata:
            scanned,_=original._scan(root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
            need(not(all_old&set(scanned)),'REMAINING_PHYSICAL_OLD_ROOTS_HAVE_NO_CROSS_ROOT_ALIAS');all_old.update(scanned)
        need(not(all_old&set(current)),'REMAINING_PHYSICAL_NEW_OUTPUT_NOT_OLD_PAYLOAD_HARDLINK')
        result=dict(status='ACTUAL_REMAINING_NONZERO_PREFIX_AND_NEW_PHYSICAL_BOUNDS_NO_PERMISSION',observed_epoch=time.time(),
            custody_sha256=self.request['physical_custody_sha256'],prefix_identity_sha256=p['prefix_identity_sha256'],
            old_metadata_identity_sha256=p['old_metadata_identity_sha256'],
            old_metadata_allocated_bytes=metadata['value']['allocated_bytes'],old_metadata_ceiling=p['old_metadata_ceiling'],
            prefix_allocated_bytes=max(prior['allocated_bytes'],p['prefix_initial_allocated_bytes']),
            new_allocated_bytes=sum(v['allocated']for v in current.values()),next_output_upper_bytes=next_output_upper_bytes,
            paid_output_ceiling=p['paid_output_ceiling'],ledger_changed=False,native_permission=False)
        check_proof(result,p,self.request,now_epoch=result['observed_epoch'],observed_epoch=result['observed_epoch'])
        self.last=self.clock();self.latest=result;return result

    def __call__(self,*,startup=False):
        value=self.original(startup=startup)
        if self.last is None or self.clock()-self.last>=30:self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        return value

    def reserve(self,next_output_upper_bytes):
        return self.inspect(next_output_upper_bytes=next_output_upper_bytes+CONTROL_GROWTH_RESERVE)
