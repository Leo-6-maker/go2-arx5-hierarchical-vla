"""Attach bounded physical accounting to an explicitly sealed retry only."""
import time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_execution as e,minimal_retry_physical_guard as physical,minimal_scope as scope

FILE='plans/MINIMAL_RETRY_PHYSICAL_CUSTODY.private.json'
SCHEMA='ARX5_MINIMAL_EXACT_RETRY_PHYSICAL_CUSTODY_V1'
CONTROL_GROWTH_RESERVE=64*1024**2


class Guard:
    def __init__(self,root,request,manifest,original,*,clock=time.monotonic):
        self.root=Path(root);self.original=original;self.clock=clock;self.last=None;self.proof=None
        path=self.root/FILE;pin=manifest['execution']['sealed_inputs'].get(FILE)
        e.need(not path.is_symlink() and path.stat().st_mode&0o777==0o400 and pin==sha(path),
            'RETRY_PHYSICAL_ACTUAL_SEALED_CUSTODY_BYTES')
        p=read(path);self.custody=p
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
        result=physical.check(self.roots,self.root,old_metadata_ceiling=p['old_metadata_ceiling'],
            paid_output_ceiling=p['paid_output_ceiling'],next_output_upper_bytes=next_output_upper_bytes)
        e.need(result['old_inventory_sha256']==p['old_inventory_sha256'],
            'RETRY_PHYSICAL_OLD_RETAINED_INODES_UNCHANGED')
        self.last=self.clock();self.proof=result;return result

    def __call__(self,*,startup=False):
        result=self.original(startup=startup)
        if self.last is None or self.clock()-self.last>=30:self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        return result

    def reserve(self,next_output_upper_bytes):
        """Recheck actual allocation immediately before a bounded save/export."""
        self.original();return self.inspect(next_output_upper_bytes=next_output_upper_bytes+CONTROL_GROWTH_RESERVE)
