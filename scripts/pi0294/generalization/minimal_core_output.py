"""Finite metadata-only output bound using the existing Core inventory reader."""
import os,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.mix_core_d25_output_guard import inventory
from scripts.pi0294.pv1.mix_core_context import LOCAL_BINDING
from . import minimal_core_accounting as c


class Guard:
    def __init__(self,root,manifest,request,local,*,clock=time.monotonic):
        self.root=Path(root);self.clock=clock;self.last=None;self.proof=None
        row=local[c.FIELD][request['request_id']];c.validate_row(row)
        c.need(row['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json')
            and row['request_sha256']==c.digest(request),'EXACT_PAID_OUTPUT_SCOPE')
        binding=read(self.root/LOCAL_BINDING)
        self.owner=Path(binding['owner_root']);self.physical=self.root.parent.parent/'children'/self.root.name
        c.need(self.root.parent.name=='neural-children'and self.owner.parent==self.root.parent.parent/'control'
            and self.owner.is_dir(),'REGISTERED_OUTPUT_PEERS')
        self.uid=self.root.stat().st_uid;c.need(self.uid==os.getuid()==2009,'ACTUAL_OUTPUT_OWNER')
        self.request_id=request['request_id'];self.limit=c.HOLD['Core_storage_bytes']

    def check(self,*,force=False):
        if not force and self.last is not None and self.clock()-self.last<30:return self.proof
        all_inodes={};counts={}
        for role,root in [('physical',self.physical),('neural',self.root),('owner',self.owner)]:
            rows,allocated=inventory(root,uid=self.uid)
            counts[role]=len(rows)
            for key,n in allocated.items():all_inodes[key]=max(all_inodes.get(key,0),n)
        current=sum(all_inodes.values())
        # Stop before another operation. Two GiB are held inside the total for
        # in-flight bounded RPC/episode writes, stable evidence and closeout.
        c.need(current+2*1024**3<=self.limit,'COMBINED_OUTPUT_AND_INFLIGHT_CLOSEOUT_RESERVE')
        self.proof=dict(schema='ARX5_MINIMAL_CORE_FIXED512_OUTPUT_BOUND_READBACK_V1',
            request_id=self.request_id,current_unique_bytes=current,upper_bytes=self.limit,
            reserved_inflight_closeout_bytes=2*1024**3,entries=counts,
            deleted_bytes=0,model_streams=0,canonical_GLOBAL_is_fresh=False)
        self.last=self.clock();return self.proof
