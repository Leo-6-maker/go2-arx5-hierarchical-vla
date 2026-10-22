"""One explicit lost-save lineage; cumulative costs never become progress."""
from copy import deepcopy

from scripts.pi0294.pv1.future_budget import _read
from .authority import AUTHORITY, PARENT_AUTHORITY

RELEASE='1d84e4cfa4f10c84d0cb33c42b06129b048c90c4'
GRANT_SHA='ef7cb5feb3395e653873affe189fb7a4bdb5e38b1cbed099f5180e91f4d6a28c'
PLAN='plans/DEC1_LOST_SAVE_RECOVERY_AUTHORIZATION.json'
LINEAGE='A_CONTROL_SAVE_RECOVERY1'
FAILED_REQUEST='e24d2ce8-b243-4daf-b9f3-f68c9e1ce19f'
CLOSURE_SHA='2a8cb4b37194557810ec239b5944d343a54a20313ea825d1fc5ecd5783bff2e7'
RECIPE='eeed62acb199bb9c4ad56b0ecc2a0ca1ed31cb7fe9c8b50a208a6f488c7479f7'
FIELD='source_lost_save_recovery'


def fail(reason):raise PermissionError('DEC1_LOST_SAVE_'+reason)


def grant(raw):
    value,digest=_read(raw)
    if (digest!=GRANT_SHA or value.get('schema')!='DEC1_A40_LOST_SAVE_RECOVERY_AUTHORIZATION_V1'
            or value.get('status')!='HUMAN_AUTHORIZED_BOUNDED_RECOVERY'):
        fail('EXACT_FIXED_HUMAN_AUTHORIZATION_NOT_GENERIC_RESTART')
    return value


def lineage(row):
    release=row.get('recovery_authorization_release');arm=row.get('arm',row.get('training_block',{}).get('arm'))
    if release is None:
        if 'training_lineage_key' in row or 'recovery_adoption_sha256' in row:fail('NO_UNBOUND_REPLACEMENT_LINEAGE')
        return arm
    expected=LINEAGE if arm=='A_CONTROL' else 'B_COVERAGE'
    if (release!=RELEASE or arm not in ('A_CONTROL','B_COVERAGE')
            or row.get('recovery_adoption_sha256')!=GRANT_SHA or row.get('training_lineage_key')!=expected):
        fail('ONE_EXACT_REPLACEMENT_A_OR_ORIGINAL_B_LINEAGE')
    return expected


def validate_adopted(h,d):
    marker=h.get(FIELD)
    if (type(marker) is not dict or marker!=d.get(FIELD)
            or marker.get('authority_release')!=RELEASE or marker.get('grant_sha256')!=GRANT_SHA
            or marker.get('failed_request_id')!=FAILED_REQUEST
            or marker.get('known_irrecoverable_optimizer_updates')!=1024
            or marker.get('replacement_lineage_key')!=LINEAGE
            or marker.get('closed_native_receipt_sha256')!=CLOSURE_SHA
            or h.get('authority_release')!=PARENT_AUTHORITY or d.get('authority_release')!=AUTHORITY
            or h.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or d.get('campaign_run_uuid')!=h.get('campaign_run_uuid')
            or h.get('DEC1_source_pair_platform')!='VANDA_A40'
            or h['optimizer_updates']!=h['used']['optimizer_updates']
            or h['optimizer_updates']<1024 or d['optimizer_updates']<1024
            or h.get('training_runs',{}).get('A_CONTROL')!=dict(last_native_step=1024,recipe_sha256=RECIPE)):
        fail('ADOPTED_BOTH_NATIVE_LEDGERS_AND_OLD_PROGRESS_PRESERVED')
    rows=[r for r in h['PBS_submissions'] if r.get('request_id')==FAILED_REQUEST]
    dr=[r for r in d['new_submissions'] if r.get('request_id')==FAILED_REQUEST]
    if (len(rows)!=1 or rows[0].get('status')!='NATIVE_TERMINAL'
            or rows[0].get('job_id')!='1474116.stdct-mgmt-02'
            or rows[0].get('committed_optimizer_updates')!=1024
            or rows[0].get('terminal_native_sha256')!='43c0ffec6b3d3adff703d561b7489580d0b9a723f464bace2da23ee1d6955d0b'
            or rows[0].get('terminal_effect_summary_sha256')!='e5b707657d517773c9f2521dd1f57d0d252935f1ad2a04dc95b04952678644c9'
            or len(dr)!=1 or dr[0].get('status')!='NATIVE_TERMINAL'
            or dr[0].get('job_id')!=rows[0].get('job_id')):
        fail('EXACT_CLOSED_FAILED_JOB_COSTS_AND_EFFECTS_PRESERVED')
    return marker


def adopt(value,dec1_value,*,grant_bytes,closure_bytes,now_epoch):
    grant(grant_bytes);closure,digest=_read(closure_bytes)
    h,d=deepcopy(value),deepcopy(dec1_value)
    if (FIELD in h or FIELD in d or not 1792348196<=now_epoch<1792348196+144*3600
            or h['optimizer_updates']!=1024 or d['optimizer_updates']!=1024
            or h.get('unknown_effects')!=[] or h.get('pending_updates')!=[] or d.get('unknown_effects')!=[]
            or any(h.get('reserved',{}).values()) or h.get('CPU_reserved_s',0) or d.get('GPU_reserved_s',0)
            or any(r.get('status') not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for r in h['PBS_submissions'])
            or digest!=CLOSURE_SHA or closure.get('recoverable_full_state') is not False
            or closure.get('checkpoint_complete_files')!=[]
            or closure.get('job_id')!='1474116.stdct-mgmt-02'
            or closure.get('effects',{}).get('request_id')!=FAILED_REQUEST
            or closure['effects'].get('optimizer_updates_committed')!=1024
            or closure.get('settlement',{}).get('unknown_effects')!=[]):
        fail('ONCE_CLOSED_KNOWN_LOSS_NO_ACTIVE_RESERVATION_OR_UNKNOWN_EFFECT')
    marker=dict(authority_release=RELEASE,grant_sha256=GRANT_SHA,failed_request_id=FAILED_REQUEST,
        known_irrecoverable_optimizer_updates=1024,replacement_lineage_key=LINEAGE,
        closed_native_receipt_sha256=CLOSURE_SHA,adopted_epoch=now_epoch)
    h[FIELD]=deepcopy(marker);d[FIELD]=deepcopy(marker)
    validate_adopted(h,d)
    return h,d
