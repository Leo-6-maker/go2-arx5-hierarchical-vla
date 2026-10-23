"""One Hopper-side campaign ledger across eligible projects and cold children."""
from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
import time

from scripts.pi0294.e1.control_read import read
from scripts.pi0292.durable import write
from .decision import AUTHORITY
from .resources import admit_resources


def check(value,*,final=False,allow_unknown=False):
    from scripts.pi0294.dec1 import b_heartbeat_recovery as b
    b_scope=b.FIELD in value
    from scripts.pi0294.dec1 import b_vanda_recovery2 as vanda2
    vanda2_scope=vanda2.FIELD in value
    if vanda2_scope:vanda2.validate_global(value)
    if b_scope:b.validate_global(value)
    if (type(final) is not bool or value['schema']!='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
            or type(allow_unknown) is not bool or value['authority_release']!=AUTHORITY
            or type(value['unknown_effects']) is not list or value['unknown_effects'] and not allow_unknown
            or type(value['CPU_s']) not in (int,float) or not math.isfinite(value['CPU_s'])
            or not 0<=value['CPU_s']<=160*3600
            or type(value['revision']) is not int or value['revision']<0
            or value['used']['optimizer_updates']!=value['optimizer_updates']
            or value['used']['neural_queries']!=value['generation_queries']
            or value['used']['PBS_submissions']!=len(value['PBS_submissions'])
            or len(value.get('training_runs',{}))>(7 if vanda2_scope else (6 if b_scope else 5))):
        raise PermissionError('PV1_EXACT_SINGLE_CAMPAIGN_COUNTERS_AND_UNKNOWN_EFFECT_FENCE')
    for key in ('new_storage_peak_bytes','new_storage_current_bytes','storage_reserved_bytes'):
        if key in value and (type(value[key]) is not int or value[key]<0):
            raise PermissionError('PV1_NONNEGATIVE_ACTUAL_AND_RESERVED_STORAGE_COUNTERS')
    if value.get('new_storage_current_bytes',0)>value.get('new_storage_peak_bytes',0):
        raise PermissionError('PV1_RESIDENT_STORAGE_CANNOT_EXCEED_RECORDED_PEAK')
    if any(s.get('native_platform') is not None for s in value['PBS_submissions']):
        from scripts.pi0294.dec1.alternate_accounting import validate_items
        validate_items(value)
    if (type(value.get('CPU_reserved_s',0)) not in (int,float)
            or not math.isfinite(value.get('CPU_reserved_s',0)) or value.get('CPU_reserved_s',0)<0
            or value['CPU_s']+value.get('CPU_reserved_s',0)>160*3600):
        raise PermissionError('PV1_ALL_GPU_SITE_CPU_RESERVATIONS_INSIDE_ORIGINAL_CAP')
    admit_resources(value['used'],value['reserved'],{},final=final,b_recovery_global=value if b_scope else None)
    return value


class Ledger:
    role='hopper'
    def __init__(self,root,*,final=False):
        self.root=Path(root);self.final=final
        self.parent=self.root.parent.parent
        self.path=self.parent/'HOPPER_RESOURCE_LEDGER.json'
        if self.path.is_symlink():raise PermissionError('PV1_NO_FOREIGN_SHARED_LEDGER_SYMLINK')
        a=read(self.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
        if check(read(self.path),final=final,allow_unknown=True)['campaign_run_uuid']!=a['run_uuid']:
            raise PermissionError('PV1_SAME_PARENT_CAMPAIGN_LEDGER_NOT_A_NEW_BUDGET')

    @contextmanager
    def transaction(self):
        with self.path.with_suffix('.owner.lock').open('a') as handle:
            until=time.monotonic()+1
            while True:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>=until:raise TimeoutError('PV1_BOUNDED_CAMPAIGN_COUNTER_LOCK_NO_UNCHARGED_EFFECT')
                    time.sleep(.01)
            value=check(read(self.path),final=self.final,allow_unknown=True)
            if value['campaign_run_uuid']!=read(self.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')['run_uuid']:
                raise PermissionError('PV1_COUNTER_CAMPAIGN_IDENTITY_CHANGED')
            yield value
            value['used']['optimizer_updates']=value['optimizer_updates']
            value['used']['neural_queries']=value['generation_queries']
            check(value,final=self.final,allow_unknown=True)
            value['revision']+=1
            write(self.path,value)

    def charge(self,key,count):
        if key not in ('generation_queries','model_loads') or type(count) is not int or count<0:
            raise ValueError('PV1_KNOWN_NONNEGATIVE_NEURAL_CHARGE_ONLY')
        with self.transaction() as value:
            if value['unknown_effects']:raise PermissionError('PV1_NO_NEW_NEURAL_EFFECT_WHILE_UNKNOWN')
            value[key]=value.get(key,0)+count

    def cpu(self,cost):
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost<0:
            raise ValueError('PV1_FINITE_CPU_COST')
        with self.transaction() as value:value['CPU_s']+=cost

    def guard(self,*,closeout=False):
        if type(closeout) is not bool:raise PermissionError('PV1_EXPLICIT_READ_ONLY_CLOSEOUT_FLAG')
        a=read(self.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
        cutoff=a['hard_closeout_epoch'] if closeout else a['stop_new_physics_epoch']
        if not a['T0_epoch']<=time.time()<cutoff:raise TimeoutError('PV1_ORIGINAL_CAMPAIGN_CUTOFF')
        if not closeout and any(p.exists() for p in [self.parent/'STOP.json',self.root/'STOP.json',self.root/'SOURCE_SAFETY_STOP_FROM_CORE.json']):
            raise PermissionError('PV1_SOURCE_OR_CAMPAIGN_STOP')
        binding=read(self.root/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
        if (lease['status']!='ACTIVE' or lease['run_uuid']!=self.root.name or lease['authority_release']!=AUTHORITY
                or time.time()>=lease['deadline_epoch']):
            raise PermissionError('PV1_NEURAL_LEDGER_REQUIRES_LIVE_CHILD_LEASE')
        return check(read(self.path),final=self.final,allow_unknown=closeout)
