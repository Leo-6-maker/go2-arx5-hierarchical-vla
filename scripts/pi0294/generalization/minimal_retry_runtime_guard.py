"""Attach bounded physical accounting to an explicitly sealed retry only."""
import hashlib,json,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_execution as e,minimal_retry_physical_guard as physical,minimal_scope as scope

FILE='plans/MINIMAL_RETRY_PHYSICAL_CUSTODY.private.json'
SCHEMA='ARX5_MINIMAL_EXACT_RETRY_PHYSICAL_CUSTODY_V1'
PORTABLE_SCHEMA='ARX5_MINIMAL_SECOND_RETRY_PORTABLE_PHYSICAL_CUSTODY_V1'
CONTROL_GROWTH_RESERVE=64*1024**2


class Guard:
    def __init__(self,root,request,manifest,original,*,clock=time.monotonic):
        self.root=Path(root);self.original=original;self.clock=clock;self.last=None;self.proof=None
        path=self.root/FILE;pin=manifest['execution']['sealed_inputs'].get(FILE)
        e.need(not path.is_symlink() and path.stat().st_mode&0o777==0o400 and pin==sha(path),
            'RETRY_PHYSICAL_ACTUAL_SEALED_CUSTODY_BYTES')
        p=read(path);self.custody=p
        self.portable=p.get('schema')==PORTABLE_SCHEMA
        if self.portable:
            self._portable_init(request,manifest)
            return
        e.need(p.get('schema')==SCHEMA and p.get('request_id')==request['request_id']
            and p.get('submission_nonce')==request['submission_nonce']
            and p.get('run_uuid')==request['run_uuid']==self.root.name
            and self.root.name=='arx5-minimal-raw-parent-first512-retry-01'
            and p.get('old_metadata_ceiling')==e.CONTROL['Hopper_storage_bytes']
            and p.get('paid_output_ceiling')==e.native_resources()['Hopper_storage_bytes']
            and p.get('engineering_metadata_allowance')==scope.components()['ONE_KNOWN_ZERO_EFFECT_COLD_ENGINEERING_REPLACEMENT']['Hopper_storage_bytes']
            and len(p.get('old_inventory_sha256',''))==64,
            'RETRY_PHYSICAL_EXACT_REQUEST_AND_PAID_GROSS_BOUNDS')
        rows=p.get('old_roots');e.need(type(rows)is list and len(rows)==2,'RETRY_PHYSICAL_TWO_CLOSED_ROOTS')
        e.need([x.get('run_uuid')for x in rows]==['arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02'],
            'RETRY_PHYSICAL_EXACT_OWN_OLD_ROOTS')
        self.roots=[]
        for row in rows:
            old=self.root.parent/row['run_uuid'];binding=read(old/'OUTPUT_BINDING.private.json')
            e.need(not old.is_symlink() and sha(old/'RUN_MANIFEST.json')==row['manifest_sha256']
                and sha(binding['lease'])==row['closed_lease_sha256'] and read(binding['lease'])['status']=='CLOSED'
                and binding['run_root']==str(old),'RETRY_PHYSICAL_OLD_ROOT_AND_CLOSED_LEASE')
            self.roots.append(old)

    def inspect(self,*,next_output_upper_bytes=0):
        p=self.custody
        if self.portable:
            from . import minimal_portable_inventory as portable
            actual=portable.capture([str(r)for r in self.roots],metadata_ceiling=p['old_metadata_ceiling'],file_hash_cache=self.hash_cache)
            identity=portable.verify(p['portable_inventory'],actual)
            new,_=physical._scan(self.root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
            old_keys=set()
            for root in self.roots:
                rows,_=physical._scan(root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
                e.need(not(old_keys&set(rows)),'PORTABLE_NO_RETAINED_CROSS_ROOT_ALIAS');old_keys.update(rows)
            e.need(not(old_keys&set(new)),'PORTABLE_NEW_OUTPUT_NOT_OLD_INODE_ALIAS')
            new_bytes=sum(r['allocated']for r in new.values())
            e.need(type(next_output_upper_bytes)is int and next_output_upper_bytes>=0
                and new_bytes+next_output_upper_bytes<=p['paid_output_ceiling'],'PORTABLE_FULL_OUTPUT_AND_CLOSE_RESERVE')
            result=dict(status='ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION',old_inventory_sha256=p['old_inventory_sha256'],
                old_allocated_bytes=actual['value']['allocated_bytes'],new_allocated_bytes=new_bytes,
                next_output_upper_bytes=next_output_upper_bytes,old_metadata_ceiling=p['old_metadata_ceiling'],paid_output_ceiling=p['paid_output_ceiling'],
                old_roots=[str(r)for r in self.roots],new_root=str(self.root),ledger_changed=False,native_permission=False,
                portable_inventory_sha256=p['portable_inventory']['portable_sha256'],
                current_portable_observation_sha256=actual['portable_sha256'],
                portable_identity_sha256=identity['portable_identity_sha256'],
                actual_kernel_device_ids=actual['kernel_device_ids'])
            self.last=self.clock();self.proof=result;return result
        result=physical.check(self.roots,self.root,old_metadata_ceiling=p['old_metadata_ceiling'],
            paid_output_ceiling=p['paid_output_ceiling'],next_output_upper_bytes=next_output_upper_bytes)
        e.need(result['old_inventory_sha256']==p['old_inventory_sha256'],
            'RETRY_PHYSICAL_OLD_RETAINED_INODES_UNCHANGED')
        self.last=self.clock();self.proof=result;return result

    def fresh_inventory(self):
        """Read current closed metadata; never restamp the sealed old capture.

        Installation and native reader checks can outlive the original120s
        snapshot. A new registration therefore consumes fresh filesystem and
        lease reads joined to the immutable original inode inventory.
        """
        proof=self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        if self.portable:
            return dict(status=('ACTUAL_FOUR_CLOSED_PORTABLE_METADATA_READONLY'if len(self.roots)==4 else'ACTUAL_THREE_CLOSED_PORTABLE_METADATA_READONLY'),observed_epoch=time.time(),
                portable_inventory_sha256=self.custody['old_inventory_sha256'],unique_allocated_bytes=proof['old_allocated_bytes'],new_writes=0)
        roots={};joined=[];allocated=0
        for root,expected in zip(self.roots,self.custody['old_roots']):
            binding=read(root/'OUTPUT_BINDING.private.json')
            e.need(sha(root/'RUN_MANIFEST.json')==expected['manifest_sha256']
                and sha(binding['lease'])==expected['closed_lease_sha256']
                and read(binding['lease'])['status']=='CLOSED',
                'RETRY_FRESH_CLOSED_IDENTITY_UNCHANGED')
            rows,links=physical._scan(root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
            values=list(rows.values());allocated+=sum(v['allocated']for v in values)
            joined.extend(dict(root=root.name,**v)for v in values)
            roots[root.name]=dict(path=str(root),manifest_sha256=expected['manifest_sha256'],
                lease_sha256=expected['closed_lease_sha256'],rows=values,input_symlinks=links)
        wire=json.dumps(sorted(joined,key=lambda v:(v['root'],v['path'])),
            sort_keys=True,separators=(',',':')).encode()
        e.need(hashlib.sha256(wire).hexdigest()==self.custody['old_inventory_sha256']
            and allocated==proof['old_allocated_bytes'],'RETRY_FRESH_FULL_OLD_INVENTORY_JOIN')
        return dict(status='ACTUAL_TWO_CLOSED_ROOTS_UNIQUE_INODE_READONLY',
            observed_epoch=time.time(),new_writes=0,roots=roots,
            unique_allocated_bytes=allocated,includes_directory_allocations=True)

    def _portable_init(self,request,manifest):
        from . import minimal_followup_scope as scope,minimal_portable_inventory as portable
        generation=request.get('continuation_generation',1)
        if generation==3:
            from . import minimal_entry_continuation_scope as scope
        p=self.custody
        e.need(request.get('continuation_generation')in(2,3) and self.root.name==request['run_uuid']==scope.RUN
            and p['request_id']==request['request_id']and p['submission_nonce']==request['submission_nonce']
            and p['run_uuid']==scope.RUN and p['scope_sha256']==scope.GRANT_SHA
            and p['old_metadata_ceiling']==e.CONTROL['Hopper_storage_bytes']+(generation-1)*64*1024**2
            and p['paid_output_ceiling']==e.native_resources()['Hopper_storage_bytes']
            and p['engineering_metadata_allowance']==64*1024**2,
            'PORTABLE_EXACT_NEW_SCOPE_REQUEST_AND_PAID_ENVELOPE')
        names=['arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02','arx5-minimal-raw-parent-first512-retry-01']+(['arx5-minimal-raw-parent-first512-retry-02']if generation==3 else [])
        e.need([r['run_uuid']for r in p['old_roots']]==names
            and p['portable_inventory']['portable_sha256']==p['old_inventory_sha256'],
            'PORTABLE_ALL_THREE_CLOSED_ROOTS_AND_FULL_CONTENT_SEAL')
        portable.verify(p['portable_inventory'],p['portable_inventory']);self.roots=[];self.hash_cache={}
        for row in p['old_roots']:
            root=self.root.parent/row['run_uuid'];binding=read(root/'OUTPUT_BINDING.private.json')
            e.need(not root.is_symlink()and sha(root/'RUN_MANIFEST.json')==row['manifest_sha256']
                and sha(binding['lease'])==row['closed_lease_sha256']and read(binding['lease'])['status']=='CLOSED'
                and binding['run_root']==str(root),'PORTABLE_ACTUAL_OLD_ROOT_AND_CLOSED_LEASE')
            self.roots.append(root)

    def __call__(self,*,startup=False):
        result=self.original(startup=startup)
        if self.last is None or self.clock()-self.last>=30:self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        return result

    def reserve(self,next_output_upper_bytes):
        """Recheck actual allocation immediately before a bounded save/export."""
        self.original();return self.inspect(next_output_upper_bytes=next_output_upper_bytes+CONTROL_GROWTH_RESERVE)
