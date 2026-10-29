"""All retained roots plus a new70GB root share the original paid gross pool."""
import json,time
from pathlib import Path
from . import minimal_execution as e,minimal_saved512_contract as c,minimal_saved512_execution as execution
from . import minimal_remaining_physical as previous,minimal_retry_physical_guard as scan
from .minimal_remaining_inputs import own_body
from .minimal_saved512_inputs import Inputs

SNAPSHOT='plans/MINIMAL_SAVED512_OLD_ROOT_STORAGE.private.json'
CONTROL_GROWTH_RESERVE=previous.CONTROL_GROWTH_RESERVE


def validate_custody(p,r,*,saved_binding):
    e.need(p['schema']==execution.PHYSICAL_SCHEMA and p['saved_binding']==saved_binding
        and p['request_id']==r['request_id']and p['submission_nonce']==r['submission_nonce']and p['run_uuid']==r['run_uuid']==c.RUN
        and p['paid_output_ceiling']==execution.PAID_OUTPUT and p['new_output_bound_bytes']==70000000000
        and p['additional_output_charge_bytes']==0 and p['old_closed_lease_sha256']==c.scope.LEASE
        and Path(p['old_science_root']).name=='arx5-minimal-raw-parent-first512-remainder-02'
        and type(p['old_snapshot_sha256'])is str and len(p['old_snapshot_sha256'])==64
        and type(p['old_snapshot_identity_sha256'])is str and len(p['old_snapshot_identity_sha256'])==64,
        'SAVED512_PHYSICAL_EXACT_NEW_IDENTITY_AND_RETAINED144GB_GROSS')
    cold=p.get('retained_cold_roots')
    e.need(type(cold)is list and len(cold)==1 and cold[0]['run_uuid']=='arx5-minimal-saved-e1-inference-e2-fresh512-01'
        and cold[0]['source_sha']=='ade4dbe39e3def8bec9be6c9608a29105552bbb4'
        and cold[0]['manifest_sha256']=='0299c9e9ea1ec9fcbb6b83295113710881dd38026ed3206e9f8170b689e298ba'
        and all(type(cold[0].get(k))is str and len(cold[0][k])==64 for k in('closed_lease_sha256','snapshot_identity_sha256'))
        and type(cold[0]['allocated_bytes'])is int and 0<=cold[0]['allocated_bytes']<=64*1024**2,
        'SAVED512_EXACT_UNSTARTED_COLD01_RETAINED_NOT_DELETED_OR_HIDDEN')
    e.need(p['prefix_initial_allocated_bytes']+p['old_metadata_ceiling']+p['old_science_initial_allocated_bytes']+70000000000<=execution.PAID_OUTPUT,
        'SAVED512_COMPLETE_PHYSICAL_POOL_NO_REFUND_OR_HIDDEN_NEW_COPY')
    return p


def check_proof(v,p,r,*,now_epoch,observed_epoch):
    e.need(type(observed_epoch)in(int,float)and 0<=now_epoch-observed_epoch<=120
        and v['observed_epoch']==observed_epoch and v['status']=='ACTUAL_SAVED512_ALL_RETAINED_ROOTS_AND_NEW_OUTPUT_BOUNDED_NO_PERMISSION'
        and v['custody_sha256']==r['physical_custody_sha256']and v['old_snapshot_identity_sha256']==p['old_snapshot_identity_sha256']
        and v['ledger_changed']is v['native_permission']is False,
        'SAVED512_FRESH_ACTUAL_COMPLETE_PHYSICAL_PROOF')
    keys=('prefix_allocated_bytes','old_metadata_ceiling','old_science_allocated_bytes','new_allocated_bytes','next_output_upper_bytes','retained_cold_allocated_bytes')
    e.need(all(type(v[k])is int and v[k]>=0 for k in keys)
        and v['prefix_allocated_bytes']>=p['prefix_initial_allocated_bytes']
        and v['old_metadata_ceiling']==p['old_metadata_ceiling']
        and v['old_science_allocated_bytes']>=p['old_science_initial_allocated_bytes']
        and v['retained_cold_allocated_bytes']>=sum(q['allocated_bytes']for q in p['retained_cold_roots'])
        and v['new_allocated_bytes']+v['retained_cold_allocated_bytes']+v['next_output_upper_bytes']<=70000000000
        and sum(v[k]for k in keys)<=p['paid_output_ceiling'],
        'SAVED512_PHYSICAL_GROSS_KEEPS_ALL_OLD_AND_NEW_INODES')
    return v


class Guard:
    def __init__(self,root,request,manifest,original_guard,*,clock=time.monotonic):
        from scripts.pi0294.dec1.a40_training_context import sealed
        self.root=Path(root);self.request=request;self.manifest=manifest;self.original=original_guard;self.clock=clock;self.last=None
        self.custody=sealed(root,execution.PHYSICAL)
        b=c.binding(own_body(self.root/c.RESULT_FILE),own_body(self.root/c.EXIT_FILE))
        validate_custody(self.custody,request,saved_binding=b)
        e.need(e.sha(own_body(self.root/execution.PHYSICAL,maximum=8*1024**2))==manifest['execution']['sealed_inputs'][execution.PHYSICAL]==request['physical_custody_sha256'],
            'SAVED512_PHYSICAL_MANIFEST_AND_REQUEST_EXACT_SEAL')
        self.snapshot=sealed(root,SNAPSHOT);p=self.custody
        e.need(e.sha(e.raw(self.snapshot))==manifest['execution']['sealed_inputs'][SNAPSHOT]==p['old_snapshot_sha256']
            and previous.identity(self.snapshot)==p['old_snapshot_identity_sha256']
            and self.snapshot['root']==p['old_science_root'],'SAVED512_PHYSICAL_COMPLETE_OLD_CURRENT_ROOT_SNAPSHOT')
        self.saved=None;self.old_root=Path(p['old_science_root']);old_manifest=json.loads(own_body(self.old_root/'RUN_MANIFEST.json'))
        old_request=json.loads(own_body(self.old_root/'pbs/alloc-39/REQUEST.json'))
        self.prior=previous.Guard(self.old_root,old_request,old_manifest,lambda **_:None)
        e.need(e.sha(e.raw(self.prior.custody))==p['old_physical_custody_sha256'],
            'SAVED512_KEEP_COMPLETE_ORIGINAL_PREFIX_AND_METADATA_CUSTODY')

    def inspect(self,*,next_output_upper_bytes=0):
        self.original()
        if self.saved is None:self.saved=Inputs(self.root,self.manifest,guard=self.original)
        self.saved.verify_current();p=self.custody
        # verify_current bound every path/stat to the same native backing FS.
        # Kernel device numbers differ between login and compute clients.
        mapped={name:dict(row,device=Path(name).lstat().st_dev)for name,row in self.saved.paths.items()}
        current_old=previous.prefix_inventory(self.old_root,witnesses=mapped)
        e.need(previous.identity(current_old)==p['old_snapshot_identity_sha256'],
            'SAVED512_OLD_SCIENCE_ROOT_CHANGED_OR_LOST_RETAINED_HISTORY')
        retained=self.prior.inspect(next_output_upper_bytes=0)
        cold_allocated=0;cold_inodes=set()
        for item in p['retained_cold_roots']:
            root=self.root.parent/item['run_uuid'];binding=json.loads(own_body(root/'OUTPUT_BINDING.private.json'))
            lease=own_body(Path(binding['lease']),immutable=False)
            e.need(e.sha(own_body(root/'RUN_MANIFEST.json'))==item['manifest_sha256']
                and e.sha(lease)==item['closed_lease_sha256']and json.loads(lease)['status']=='CLOSED'
                and not(root/'control/resident-owner').exists(),'SAVED512_ACTUAL_COLD01_CLOSED_NO_GPU_OWNER_OR_REQUEST')
            snapshot=previous.prefix_inventory(root)
            e.need(previous.identity(snapshot)==item['snapshot_identity_sha256'],'SAVED512_RETAINED_COLD01_FILES_CHANGED')
            cold_allocated+=max(item['allocated_bytes'],snapshot['allocated_bytes'])
            part,_=scan._scan(root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'});cold_inodes.update(part)
        current,_=scan._scan(self.root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
        old_inodes={(x['device'],x['inode'])for x in current_old['rows']}
        for name in[self.prior.custody['prefix_root'],*self.prior.metadata]:
            part,_=scan._scan(name,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'});old_inodes.update(part)
        e.need(not old_inodes.intersection(cold_inodes)and not (old_inodes|cold_inodes).intersection(current),'SAVED512_NO_NEW_HARDLINK_OR_COPY_CREDIT_FOR_OLD_TENSORS')
        v=dict(status='ACTUAL_SAVED512_ALL_RETAINED_ROOTS_AND_NEW_OUTPUT_BOUNDED_NO_PERMISSION',observed_epoch=time.time(),
            custody_sha256=self.request['physical_custody_sha256'],old_snapshot_identity_sha256=p['old_snapshot_identity_sha256'],
            prefix_allocated_bytes=max(retained['prefix_allocated_bytes'],p['prefix_initial_allocated_bytes']),
            old_metadata_ceiling=p['old_metadata_ceiling'],old_science_allocated_bytes=max(current_old['allocated_bytes'],p['old_science_initial_allocated_bytes']),
            new_allocated_bytes=sum(x['allocated']for x in current.values()),retained_cold_allocated_bytes=cold_allocated,next_output_upper_bytes=next_output_upper_bytes,
            ledger_changed=False,native_permission=False)
        check_proof(v,p,self.request,now_epoch=v['observed_epoch'],observed_epoch=v['observed_epoch'])
        self.last=self.clock();self.latest=v;return v

    def __call__(self,*,startup=False):
        v=self.original(startup=startup)
        if self.last is None or self.clock()-self.last>=30:self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        return v
    def reserve(self,next_output_upper_bytes):return self.inspect(next_output_upper_bytes=next_output_upper_bytes+CONTROL_GROWTH_RESERVE)
