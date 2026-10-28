"""Explicit bounded segmented continuation; old long-allocation scopes unchanged."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.pi0294.pv1.future_budget import _read
from scripts.pi0294.native_dataset import sha
from .lost_save_recovery import validate_adopted as original_adopted,LINEAGE,RECIPE

RELEASE='3335c8620c29d8d58f6b5e19075f3943e2dd1904'
GRANT_SHA='2eb61986e4d87ca428ec386e703df323c8480916c5ddb0ea9abaef3bf26f5615'
PLAN='plans/DEC1_SEGMENTED_RECOVERY_AUTHORIZATION.json'
FIELD='source_segmented_recovery'
CPU_PROFILE='A40_SEGMENT_22500S_PROCESS_CPU54000_V1'
RATE=9.010780406137929
CLOSURE_SHA='f5b4342813cc1e3e0c98496a407aecadc5549214e2e3c4065a7ed4ca660d40cb'


def fail(reason):raise PermissionError('DEC1_SEGMENT_'+reason)


def grant(raw):
    value,digest=_read(raw)
    if (digest!=GRANT_SHA or value.get('schema')!='DEC1_A40_SEGMENTED_RECOVERY_AUTHORIZATION_V1'
            or value.get('status')!='HUMAN_AUTHORIZED_BOUNDED_SEGMENTED_CONTINUATION'):
        fail('EXACT_HUMAN_FIXED_SEGMENT_AUTHORIZATION')
    return value


def enabled(row):
    if row.get('segmented_authorization_release') is None:
        if any(k in row for k in ('segmented_grant_sha256','segment_stop_step')):fail('NO_UNBOUND_SEGMENT')
        return False
    if (row.get('segmented_authorization_release')!=RELEASE or row.get('segmented_grant_sha256')!=GRANT_SHA):
        fail('EXACT_SEGMENT_RELEASE_AND_GRANT')
    return True


def limits(row):
    if not enabled(row):fail('EXPLICIT_SEGMENT_ONLY')
    block=row.get('training_block',{})
    arm=row.get('arm',block.get('arm'));start=row.get('start_native_step',row.get('expected_native_start_step'))
    stop=row.get('segment_stop_step',block.get('stop_after_step'))
    from . import b_heartbeat_recovery as b
    extra=[]
    if b.enabled(row):
        b.lineage(row);extra=[('B_COVERAGE',1024,2048)]
    if ((arm,start,stop) not in (('A_CONTROL',32,2048),('A_CONTROL',2048,4096),
            ('B_COVERAGE',0,2048),('B_COVERAGE',2048,4096),*extra)
            or row.get('CPU_envelope_profile')!=CPU_PROFILE):
        fail('ONLY_FOUR_AUTHORIZED_EXACT_SEGMENTS_NO_RESTART_OR_ROLLBACK')
    return start,stop


def validate_adopted(h,d):
    original_adopted(h,d);marker=h.get(FIELD)
    if (type(marker) is not dict or marker!=d.get(FIELD) or marker.get('authority_release')!=RELEASE
            or marker.get('grant_sha256')!=GRANT_SHA or marker.get('initial_recovery_step')!=32
            or marker.get('original_T0_epoch')!=1792348196):
        fail('BOTH_NATIVE_ADOPTED_MARKERS_NOT_JUST_A_PLAN')
    return marker


def adopt(h,d,*,grant_bytes,closure_bytes,now_epoch):
    grant(grant_bytes);_,closure_sha=_read(closure_bytes)
    original_adopted(h,d)
    if (FIELD in h or FIELD in d or closure_sha!=CLOSURE_SHA
            or h['optimizer_updates']!=1056 or d['optimizer_updates']!=1056
            or h.get('training_runs',{}).get(LINEAGE)!=dict(last_native_step=32,recipe_sha256=RECIPE)
            or h['unknown_effects']!=[] or h['pending_updates']!=[] or d['unknown_effects']!=[]
            or any(h.get('reserved',{}).values()) or h.get('CPU_reserved_s',0) or h.get('storage_reserved_bytes',0)
            or d.get('GPU_reserved_s',0) or d.get('CPU_reserved_s',0)
            or not 1792348196<=now_epoch<1792348196+144*3600
            or len(d['new_submissions'])!=4 or len(h['PBS_submissions'])!=15
            or any(x['status'] not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for x in h['PBS_submissions'])):
        fail('ONCE_SETTLED_FULL_STATE32_NO_UNKNOWN_OR_RESERVED_EFFECT')
    h,d=deepcopy(h),deepcopy(d)
    marker=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,initial_recovery_step=32,
        original_T0_epoch=1792348196,closed_native_receipt_sha256=closure_sha,adopted_epoch=now_epoch)
    h[FIELD]=deepcopy(marker);d[FIELD]=deepcopy(marker);validate_adopted(h,d)
    return h,d


def admitted(root,manifest,request):
    if not enabled(request):return False
    limits(request);p=Path(root)/PLAN
    if (p.is_symlink() or p.stat().st_mode&0o222 or sha(p)!=GRANT_SHA
            or manifest.get('authority',{}).get('segmented_authorization_release')!=RELEASE
            or manifest['execution']['sealed_inputs'].get(PLAN)!=GRANT_SHA
            or request.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA):
        fail('EXACT_MANIFEST_REQUEST_AND_COLD_SEALED_SEGMENT_POLICY')
    grant(p.read_bytes())
    native=Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name
    if native.read_bytes()!=p.read_bytes():fail('SAME_NATIVE_SEGMENT_AUTHORIZATION_BYTES')
    return True


def phase(root,request,*,name,stop,resume_relative):
    start,end=limits(request);arm=request['training_block']['arm']
    if start<1024:
        if name=='BEFORE_PROBES':valid=stop==1024 and resume_relative==request['training_block']['resume_checkpoint_relative']
        elif name=='AFTER_PROBES':valid=stop==end and resume_relative==f'training/{arm}/checkpoint-1024'
        else:valid=False
    elif start==1024:
        from . import b_heartbeat_recovery as b
        valid=(b.enabled(request) and b.lineage(request)==b.LINEAGE and name=='B_RECOVERY_REENTRY'
            and stop==2048 and resume_relative=='training/B_COVERAGE/checkpoint-1024')
    else:valid=name=='SEGMENT_REENTRY' and stop==end and resume_relative==f'training/{arm}/checkpoint-2048'
    if not valid:fail('EXACT_SEGMENT_PHASE_NO_1024_PROBE_SKIP')
    if name in ('AFTER_PROBES','SEGMENT_REENTRY','B_RECOVERY_REENTRY'):
        from .long_training import probe_receipt
        probe_receipt(root,arm)


def execute(*,start_step,stop_step,train,probe,unload,load,guard,checkpoint,resume_checkpoint=None,recovery_request=None):
    if start_step==1024:
        from . import b_heartbeat_recovery as b
        if (recovery_request is None or not b.enabled(recovery_request) or b.lineage(recovery_request)!=b.LINEAGE
                or limits(recovery_request)!=(1024,2048) or stop_step!=2048 or resume_checkpoint is None):
            fail('EXACT_NEW_B1024_IMPORT_NOT_LEGACY_PROBE_SKIP')
        final=train(stop=2048,phase='B_RECOVERY_REENTRY',checkpoint=resume_checkpoint)
        return final,[final]
    if stop_step not in (2048,4096) or start_step not in (0,32,2048):fail('EXACT_EXECUTION_SEGMENTS')
    if start_step==2048:
        if stop_step!=4096 or resume_checkpoint is None:fail('FULL_STATE2048_REENTRY_REQUIRED')
        final=train(stop=4096,phase='SEGMENT_REENTRY',checkpoint=resume_checkpoint)
        return final,[final]
    if stop_step!=2048:fail('FIRST_SEGMENT_STOPS2048')
    first=train(stop=1024,phase='BEFORE_PROBES',checkpoint=resume_checkpoint)
    records=[first]
    if first['global_step']!=1024:return first,records
    unload();guard();load(checkpoint,trainable=False);records.append(probe())
    unload();guard();load(checkpoint,trainable=True)
    final=train(stop=2048,phase='AFTER_PROBES',checkpoint=checkpoint);records.append(final)
    return final,records
