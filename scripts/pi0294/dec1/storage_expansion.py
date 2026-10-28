"""Explicit storage-only continuation; no cost reset, retirement or dispatch."""
from copy import deepcopy
import math
from pathlib import Path
import stat

from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.future_budget import _read
from . import segmented_training as seg
from . import lost_save_recovery as recovery
from .authority import AUTHORITY, PARENT_AUTHORITY

RELEASE='1abb20973d84836b29baebf1d73466377c98759c'
GRANT_SHA='fd8761efd5a0c8a49b2e5f5c378965502c23a201f7fbd9c37eb681e5430862cd'
PLAN='plans/VAP1_STORAGE_ONLY_AUTHORIZATION.json'
FIELD='source_storage_expansion'
CLOSURE_SHA='540100c29f9a4ea3715029e16902c6687034bfc62a3386bbcd1b436e82b0f959'
CONTROL_CLOSURE_SHA='1bcbfb52dfe7f569b19ea814ec0215667fa0b86d72df8f09a5e1d0d272e24f02'
JOB='1474599.stdct-mgmt-02'
REQUEST='75e12289-7b99-4f46-bd69-bbe64ff9cad0'
T0=1792348196
GIB=1024**3


def fail(reason):
    raise PermissionError('DEC1_STORAGE_EXPANSION_'+reason)


def grant(raw):
    value,digest=_read(raw)
    if (digest!=GRANT_SHA or value.get('schema')!='VAP1_STORAGE_ONLY_AUTHORIZATION_V1'
            or value.get('status')!='HUMAN_AUTHORIZED_STORAGE_ONLY_CONTINUATION'
            or value.get('original_segmented_authority')!=seg.RELEASE
            or value.get('storage_cap_bytes')!={'DEC1':[176*GIB,352*GIB],'global':[250*GIB,419*GIB]}
            or value.get('original_T0_epoch')!=T0 or value.get('retirement_credit_bytes')!=0
            or value.get('all_checkpoints_preserved') is not True):
        fail('EXACT_HUMAN_STORAGE_ONLY_GRANT_NO_RETIREMENT_CREDIT')
    return value


def enabled(row):
    if row.get('storage_authorization_release') is None:
        if 'storage_grant_sha256' in row or 'storage_authorization_release' in row:
            fail('NO_PARTIAL_OR_UNBOUND_STORAGE_AUTHORIZATION')
        return False
    if (row.get('storage_authorization_release')!=RELEASE
            or row.get('storage_grant_sha256')!=GRANT_SHA or not seg.enabled(row)):
        fail('EXACT_STORAGE_RELEASE_AND_ORIGINAL_SEGMENT_REQUIRED')
    return True


def caps(row):
    if enabled(row):
        return 419*GIB,352*GIB
    return 250*GIB,(176 if seg.enabled(row) else 160)*GIB


def _marker(h):
    marker=h.get(FIELD)
    keys={'authority_release','grant_sha256','segmented_authority_release','original_T0_epoch',
        'closed_native_receipt_sha256','closed_control_receipt_sha256','adopted_epoch'}
    if (type(marker) is not dict or set(marker)!=keys
            or marker.get('authority_release')!=RELEASE or marker.get('grant_sha256')!=GRANT_SHA
            or marker.get('segmented_authority_release')!=seg.RELEASE
            or marker.get('original_T0_epoch')!=T0
            or marker.get('closed_native_receipt_sha256')!=CLOSURE_SHA
            or marker.get('closed_control_receipt_sha256')!=CONTROL_CLOSURE_SHA
            or type(marker.get('adopted_epoch')) not in (int,float)
            or not math.isfinite(marker['adopted_epoch'])
            or not T0<=marker['adopted_epoch']<T0+144*3600):
        fail('BOTH_NATIVE_IDENTICAL_STORAGE_MARKERS_AND_ORIGINAL_CLOCK')
    return marker


def validate_adopted(h,d):
    seg.validate_adopted(h,d)
    marker=_marker(h)
    if marker!=d.get(FIELD):fail('BOTH_NATIVE_IDENTICAL_STORAGE_MARKERS_AND_ORIGINAL_CLOCK')
    return marker


def global_cap_from_ledger(h):
    """Read only the adopted global cap; this supplies no scientific scope."""
    from scripts.pi0294.pv1 import evaluation_scope as dev
    if dev.FIELD in h:
        dev.validate_global(h);return dev.caps(h)['Hopper_storage_bytes']
    if FIELD not in h:return 250*GIB
    _marker(h)
    old=h.get(seg.FIELD,{});lost=h.get(recovery.FIELD,{})
    segment_keys={'authority_release','grant_sha256','initial_recovery_step','original_T0_epoch',
        'closed_native_receipt_sha256','adopted_epoch'}
    recovery_keys={'authority_release','grant_sha256','failed_request_id','known_irrecoverable_optimizer_updates',
        'replacement_lineage_key','closed_native_receipt_sha256','adopted_epoch'}
    if (h.get('authority_release')!=PARENT_AUTHORITY
            or h.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or h.get('DEC1_source_pair_platform')!='VANDA_A40'
            or type(old) is not dict or set(old)!=segment_keys
            or old.get('authority_release')!=seg.RELEASE or old.get('grant_sha256')!=seg.GRANT_SHA
            or old.get('initial_recovery_step')!=32 or old.get('original_T0_epoch')!=T0
            or old.get('closed_native_receipt_sha256')!=seg.CLOSURE_SHA
            or type(lost) is not dict or set(lost)!=recovery_keys
            or lost.get('authority_release')!=recovery.RELEASE or lost.get('grant_sha256')!=recovery.GRANT_SHA
            or lost.get('failed_request_id')!=recovery.FAILED_REQUEST
            or lost.get('known_irrecoverable_optimizer_updates')!=1024
            or lost.get('replacement_lineage_key')!=recovery.LINEAGE
            or lost.get('closed_native_receipt_sha256')!=recovery.CLOSURE_SHA
            or any(type(m.get('adopted_epoch')) not in (int,float) or not math.isfinite(m['adopted_epoch'])
                or not T0<=m['adopted_epoch']<=h[FIELD]['adopted_epoch'] for m in (old,lost))
            or h.get('training_runs',{}).get('A_CONTROL')!=dict(last_native_step=1024,recipe_sha256=recovery.RECIPE)
            or h.get('optimizer_updates',0)<3072
            or h.get('optimizer_updates')!=h.get('used',{}).get('optimizer_updates')):
        fail('GLOBAL_CAP_REQUIRES_ORIGINAL_COMPLETE_ADOPTED_MARKERS_AND_CAMPAIGN')
    from . import heartbeat_recovery as heartbeat
    if heartbeat.FIELD in h:
        heartbeat.validate_global(h)
        from . import zero_update_recovery as zero
        if zero.FIELD in h:
            zero.validate_global(h)
            from . import b_heartbeat_recovery as b
            if b.FIELD in h:
                b.validate_global(h)
                from . import b_vanda_recovery2 as vanda2
                if vanda2.FIELD in h:
                    vanda2.validate_global(h);return 680*GIB
                return 615*GIB
            return 550*GIB
        return 485*GIB
    return 419*GIB


def _zero(value):
    if isinstance(value,dict):return all(_zero(v) for v in value.values())
    if isinstance(value,(list,tuple)):return not value
    return type(value) in (int,float) and value==0


def _unreserved(value):
    return all(_zero(v) for k,v in value.items() if 'reserved' in k)


def adopt(h,d,*,grant_bytes,closure_bytes,control_closure_bytes,now_epoch):
    grant(grant_bytes);seg.validate_adopted(h,d)
    closure,closure_sha=_read(closure_bytes);control,control_sha=_read(control_closure_bytes)
    effects=closure.get('effects',{});settled=closure.get('settlement',{})
    gpu_close=control.get('GPU_close',{});control_row=control.get('settled_submission',{})
    rows=[r for r in h['PBS_submissions'] if r.get('request_id')==REQUEST]
    dec_rows=[r for r in d['new_submissions'] if r.get('request_id')==REQUEST]
    b=h.get('training_runs',{}).get('B_COVERAGE')
    if (FIELD in h or FIELD in d or closure_sha!=CLOSURE_SHA or control_sha!=CONTROL_CLOSURE_SHA
            or closure.get('status')!='CLOSED_SEGMENT_FULL_STATE_PRESERVED'
            or closure.get('job_id')!=JOB or closure.get('request_id')!=REQUEST
            or closure.get('global_step')!=2048 or closure.get('recoverable_full_state') is not True
            or closure.get('retired_or_deleted_files')!=0
            or effects.get('arm')!='A_CONTROL' or effects.get('start_native_step')!=32
            or effects.get('last_native_step')!=2048 or effects.get('optimizer_updates_committed')!=2016
            or effects.get('pending_updates')!=[] or effects.get('unknown_query_intents')!=[]
            or effects.get('recipe_sha256')!=recovery.RECIPE
            or effects.get('effect_journal_complete') is not True
            or settled.get('unknown_effects')!=[] or settled.get('updates')!=3072
            or control.get('status')!='CLOSED_CPU_SEGMENT2048_CONTROL_AFTER_KNOWN_EFFECTS_SETTLEMENT'
            or control.get('job_id')!=JOB or control.get('unknown_effects')!=[]
            or control.get('new_A_recovery_step')!=2048
            or control.get('old_A1024_costs_preserved') is not True
            or gpu_close.get('lease_status')!='CLOSED' or gpu_close.get('terminal_receipt_sha256')!=closure_sha
            or gpu_close.get('terminal_receipt')!=closure
            or len(rows)!=1 or rows[0]!=control_row or rows[0].get('status')!='NATIVE_TERMINAL'
            or rows[0].get('job_id')!=JOB or rows[0].get('committed_optimizer_updates')!=2016
            or len(dec_rows)!=1 or dec_rows[0].get('status')!='NATIVE_TERMINAL' or dec_rows[0].get('job_id')!=JOB
            or h.get('authority_release')!=PARENT_AUTHORITY or d.get('authority_release')!=AUTHORITY
            or h['optimizer_updates']!=h['used']['optimizer_updates'] or h['optimizer_updates']!=3072
            or d['optimizer_updates']!=3072
            or h.get('training_runs',{}).get(recovery.LINEAGE)!=dict(last_native_step=2048,recipe_sha256=recovery.RECIPE)
            or b is not None and b!=dict(last_native_step=0,recipe_sha256=recovery.MATCHED_RECIPES['B_COVERAGE'])
            or h.get('unknown_effects')!=[] or d.get('unknown_effects')!=[]
            or h.get('pending_updates')!=[] or d.get('pending_updates',[])!=[]
            or not _unreserved(h) or not _unreserved(d)
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for r in h['PBS_submissions'])
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for r in d['new_submissions'])
            or type(now_epoch) not in (int,float) or not math.isfinite(now_epoch)
            or not T0<=max(closure.get('closed_epoch',T0),control.get('closed_epoch',T0))<=now_epoch<T0+144*3600):
        fail('ONCE_ACTUAL_A2048_SETTLED_GPU_AND_CONTROL_CLOSED_NO_RESERVED_UNKNOWN_OR_RESET')
    result,dec=deepcopy(h),deepcopy(d)
    marker=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,segmented_authority_release=seg.RELEASE,
        original_T0_epoch=T0,closed_native_receipt_sha256=closure_sha,
        closed_control_receipt_sha256=control_sha,adopted_epoch=now_epoch)
    result[FIELD]=deepcopy(marker);dec[FIELD]=deepcopy(marker)
    validate_adopted(result,dec)
    return result,dec


def _readonly(path):
    path=Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode&0o222:
        fail('REGULAR_READONLY_SEALED_GRANT_NOT_SYMLINK')
    return path


def admitted(root,manifest,request):
    if not enabled(request):return False
    seg.admitted(root,manifest,request)
    path=_readonly(Path(root)/PLAN)
    if (sha(path)!=GRANT_SHA
            or manifest.get('authority',{}).get('storage_authorization_release')!=RELEASE
            or manifest.get('authority',{}).get('storage_grant_sha256')!=GRANT_SHA
            or manifest['execution']['sealed_inputs'].get(PLAN)!=GRANT_SHA
            or request.get('sealed_inputs',{}).get(PLAN)!=GRANT_SHA):
        fail('EXACT_MANIFEST_REQUEST_AND_SEALED_STORAGE_GRANT')
    raw=path.read_bytes();grant(raw)
    native=_readonly(Path(root).parent.parent/'authority/DEC1'/Path(PLAN).name)
    if native.read_bytes()!=raw:fail('SAME_NATIVE_STORAGE_AUTHORIZATION_BYTES')
    return True
