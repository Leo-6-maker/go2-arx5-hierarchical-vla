"""Scientific512 identity on the unchanged Core process/device reader."""
from . import minimal_core_accounting as c

need=c.need
from scripts.pi0294.pv1.mix_core_identity import STATUS,OWNER_STARTED
OWNER_ROLE=c.OWNER_ROLE


def current_owner(*args,**kwargs):
    from scripts.pi0294.pv1.mix_core_identity import current_owner as original
    return original(*args,**kwargs)


def cold_profile(m):
    need(c.typed(m) and m.get('scope')==c.ROLE and m.get('execution',{}).get('profile')==c.PROFILE
        and 'platform_training_version' not in m,'EXPLICIT_RAW512_NEURAL_ROLE')
    return True


def request_identity(m,r,manifest_sha256):
    cold_profile(m);c.request(r)
    from .minimal_core_sharing import manifest
    manifest(m,r)
    need(m.get('schema')=='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
        and m.get('owner')=='EXISTING_LOCAL_PC' and m.get('owner_user')=='liuyu-nfs'
        and m.get('owner_uid')==2009 and m['execution'].get('uid')==2009
        and m['execution'].get('host')=='corelab-4080s'
        and m['authority'].get('protocol_ref')==c.campaign.TASK_SOURCE
        and m['authority'].get('scope_grant_sha256')==c.GRANT_SHA
        and m['execution'].get('sealed_inputs',{}).get(c.GRANT_FILE)==c.GRANT_SHA,
        'STANDARD_MANIFEST_AND_EXACT_DIRECT_SCIENCE_SCOPE')
    need(r['run_uuid']==m.get('run_uuid') and r['run_manifest_sha256']==manifest_sha256
        and r['source_sha']==m.get('code',{}).get('vla_commit')
        and m['limits'].get('T_CD')==r['T_CD']
        and m['limits'].get('operation_deadline_epoch')==r['deadline_epoch'],
        'SAME_REGISTERED_SOURCE_REQUEST_AND_DECLARED_CLOCK')
    return r


def check_clock(r,*,started_epoch,now_epoch):
    c.request({k:v for k,v in r.items()if k!='request_sha256'})
    need(type(started_epoch) in (int,float) and type(now_epoch) in (int,float)
        and r['created_epoch']-1<=started_epoch<=now_epoch<r['deadline_epoch']
        and r['T_CD']<=started_epoch<r['deadline_epoch']-28860,
        'ACTUAL_FULL_EIGHT_HOUR_PROCESS_WINDOW')


def require_device(request,allocation,*,visible,count,name,memory_bytes):
    from scripts.pi0294.pv1.mix_core_identity import STATUS
    r,v=request,allocation
    need(c.typed(r) and c.typed(v) and v.get('status')==STATUS
        and r.get('stage')==v.get('stage')==c.plan.STAGE
        and r.get('evaluation_candidate')==v.get('evaluation_candidate')=='E1'
        and type(r.get('checkpoint_step')) is int and r['checkpoint_step']==512
        and r.get('checkpoint_manifest_sha256')==v.get('checkpoint_manifest_sha256')==c.CHECKPOINT
        and r.get('statistics_sha256')==v.get('statistics_sha256')==c.plan.R0
        and type(r.get('optimizer_updates')) is int and r['optimizer_updates']==0
        and visible==r.get('GPU_uuid') and type(count) is int and count==1
        and name=='NVIDIA GeForce RTX 4080 SUPER' and type(memory_bytes) is int and memory_bytes>=30*1024**3,
        'UNCHANGED_PHYSICAL_DEVICE_GATE_FOR_EXPLICIT_RAW512')
    return True
