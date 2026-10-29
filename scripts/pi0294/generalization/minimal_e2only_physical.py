"""Keep every old root and charge the new root against its existing gross pool.

No cleanup, tensor copy or inode credit. The closed predecessor's original
guard still checks its entire retained lineage and independently hashed inputs.
"""
import json,time
from pathlib import Path
from . import minimal_execution as e,minimal_e2only_contract as c,minimal_e2only_execution as execution
from . import minimal_saved512_physical as previous,minimal_remaining_physical as inventory
from . import minimal_retry_physical_guard as scan
from .minimal_remaining_inputs import own_body

SNAPSHOT='plans/MINIMAL_E2_ONLY_PREDECESSOR_STORAGE.private.json'
CONTROL_GROWTH_RESERVE=previous.CONTROL_GROWTH_RESERVE


def validate_custody(p,r,*,predecessor_binding):
    e.need(p['schema']==execution.PHYSICAL_SCHEMA and p['predecessor_binding']==predecessor_binding==c.binding()
        and p['request_id']==r['request_id']and p['submission_nonce']==r['submission_nonce']
        and p['run_uuid']==r['run_uuid']==c.RUN and p['old_closed_lease_sha256']==c.scope.LEASE
        and Path(p['previous_root']).name=='arx5-minimal-saved-e1-inference-e2-fresh512-03'
        and p['paid_output_ceiling']==execution.PAID_OUTPUT and p['new_output_bound_bytes']==70000000000
        and p['additional_output_charge_bytes']==0
        and type(p['retained_initial_bytes'])is int and p['retained_initial_bytes']>=66671091712
        and p['retained_initial_bytes']+p['new_output_bound_bytes']<=p['paid_output_ceiling'],
        'E2ONLY_EXACT_CLOSED_PREDECESSOR_ALL_GROSS_AND_NEW70GB')
    for key in('previous_manifest_sha256','previous_custody_sha256','previous_snapshot_sha256','previous_snapshot_identity_sha256'):
        e.need(type(p.get(key))is str and len(p[key])==64,'E2ONLY_SEALED_PHYSICAL_'+key)
    return p


def check_proof(v,p,r,*,now_epoch,observed_epoch):
    e.need(type(observed_epoch)in(int,float)and 0<=now_epoch-observed_epoch<=120
        and v['observed_epoch']==observed_epoch and v['status']=='ACTUAL_E2_ONLY_ALL_OLD_GROSS_AND_NEW70GB_NO_PERMISSION'
        and v['custody_sha256']==r['physical_custody_sha256']==e.sha(e.raw(p))
        and v['previous_snapshot_identity_sha256']==p['previous_snapshot_identity_sha256']
        and v['ledger_changed']is v['native_permission']is False,'E2ONLY_FRESH_PHYSICAL_NOT_CPU_PREPARATION')
    for k in('retained_allocated_bytes','new_allocated_bytes','next_output_upper_bytes'):
        e.need(type(v[k])is int and v[k]>=0,'E2ONLY_REAL_NONNEGATIVE_GROSS_'+k)
    e.need(v['retained_allocated_bytes']>=p['retained_initial_bytes']
        and v['new_allocated_bytes']+v['next_output_upper_bytes']<=p['new_output_bound_bytes']
        and v['retained_allocated_bytes']+v['new_allocated_bytes']+v['next_output_upper_bytes']<=p['paid_output_ceiling'],
        'E2ONLY_NO_OLD_COST_REFUND_OR_HIDDEN_CONTROL_TEMP_OR_CHECKPOINT')
    return v


class Guard:
    def __init__(self,root,request,manifest,original_guard,*,clock=time.monotonic):
        from scripts.pi0294.dec1.a40_training_context import sealed
        self.root=Path(root);self.request=request;self.original=original_guard;self.clock=clock;self.last=None
        self.custody=sealed(root,execution.PHYSICAL);p=self.custody
        validate_custody(p,request,predecessor_binding=c.binding())
        e.need(e.sha(own_body(self.root/execution.PHYSICAL))==manifest['execution']['sealed_inputs'][execution.PHYSICAL]
            ==request['physical_custody_sha256'],'E2ONLY_PHYSICAL_OWN_MANIFEST_AND_REQUEST')
        self.old_root=Path(p['previous_root']);oldm=json.loads(own_body(self.old_root/'RUN_MANIFEST.json'))
        oldr=json.loads(own_body(self.old_root/'pbs/alloc-40/REQUEST.json'))
        b=json.loads(own_body(self.old_root/'OUTPUT_BINDING.private.json'));lease=own_body(Path(b['lease']),immutable=False)
        e.need(e.sha(lease)==c.scope.LEASE and json.loads(lease)['status']=='CLOSED'
            and e.sha(own_body(self.old_root/'RUN_MANIFEST.json'))==p['previous_manifest_sha256']
            and oldr['request_id']==c.scope.REQUEST,'E2ONLY_REAL_OLD_LEASE_CLOSED_NOT_OLD_OWNER_RESTART')
        snapshot=sealed(root,SNAPSHOT)
        e.need(e.sha(e.raw(snapshot))==manifest['execution']['sealed_inputs'][SNAPSHOT]==p['previous_snapshot_sha256']
            and inventory.identity(snapshot)==p['previous_snapshot_identity_sha256']
            and snapshot['root']==str(self.old_root),'E2ONLY_COMPLETE_FAILED122_ROOT_PRESERVED')
        self.prior=previous.Guard(self.old_root,oldr,oldm,lambda **_:None)
        e.need(e.sha(e.raw(self.prior.custody))==p['previous_custody_sha256'],'E2ONLY_ENTIRE_OLD_PHYSICAL_LINEAGE')

    def inspect(self,*,next_output_upper_bytes=0):
        self.original();p=self.custody
        old=self.prior.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        snapshot=inventory.prefix_inventory(self.old_root)
        e.need(inventory.identity(snapshot)==p['previous_snapshot_identity_sha256'],
            'E2ONLY_OLD_FAILED_ROOT_CHANGED_OR_HISTORY_LOST')
        current,_=scan._scan(self.root,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
        old_inodes=set()
        roots=[self.old_root,self.prior.old_root,self.prior.prior.custody['prefix_root'],
            *self.prior.prior.metadata,*[self.old_root.parent/q['run_uuid']for q in self.prior.custody['retained_cold_roots']]]
        for name in roots:
            part,_=scan._scan(name,permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'});old_inodes.update(part)
        e.need(not old_inodes.intersection(current),'E2ONLY_NO_TENSOR_HARDLINK_OR_DEDUPLICATION_CREDIT')
        retained=sum(old[k]for k in('prefix_allocated_bytes','old_metadata_ceiling','old_science_allocated_bytes',
            'new_allocated_bytes','retained_cold_allocated_bytes'))
        v=dict(status='ACTUAL_E2_ONLY_ALL_OLD_GROSS_AND_NEW70GB_NO_PERMISSION',observed_epoch=time.time(),
            custody_sha256=self.request['physical_custody_sha256'],previous_snapshot_identity_sha256=p['previous_snapshot_identity_sha256'],
            retained_allocated_bytes=max(retained,p['retained_initial_bytes']),new_allocated_bytes=sum(x['allocated']for x in current.values()),
            next_output_upper_bytes=next_output_upper_bytes,ledger_changed=False,native_permission=False)
        check_proof(v,p,self.request,now_epoch=v['observed_epoch'],observed_epoch=v['observed_epoch'])
        self.last=self.clock();self.latest=v;return v

    def __call__(self,*,startup=False):
        v=self.original(startup=startup)
        if self.last is None or self.clock()-self.last>=30:self.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE)
        return v

    def reserve(self,next_output_upper_bytes):return self.inspect(next_output_upper_bytes=next_output_upper_bytes+CONTROL_GROWTH_RESERVE)
