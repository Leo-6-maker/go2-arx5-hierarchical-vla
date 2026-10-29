"""Source-only own512 HF control carved from the two existing Eval escrows.

Pure original-GLOBAL proposals and uploader admission; no new scheduler,
reservation, model permission or Core dependency. The original package stays
byte-identical. Its exact overlay consumes the once-paid source control while
retaining both complete Core panels, all old costs and final protections.
"""
import base64
from copy import deepcopy
import json
import math
from pathlib import PurePosixPath
import re

from . import minimal_execution as e, minimal_remaining_execution as remaining
from . import minimal_inference_transfer as transfer

FIELD='ARX5_minimal_first512_source_model_only_delivery'
SCHEMA='ARX5_MINIMAL_OWN512_SOURCE_DELIVERY_CONTROL_ONCE_V1'
MANIFEST='ARX5_MINIMAL_OWN512_SOURCE_ONLY_CONTROL_MANIFEST_V1'
LOCAL=FIELD+'_noncanonical_LOCAL_projection'
CONTROL=e._vector(dict(Hopper_CPU_s=900,local_CPU_s=60,
    Hopper_storage_bytes=64*1024**2,transfer_bytes=32*1024**2))
need=e.need


def model_identity(plan):
    p=plan['public_plan'];arm=p.get('candidate')
    need(p.get('schema')==transfer.SCHEMA and arm in e.training.ARMS
        and type(p.get('checkpoint_step'))is int and p['checkpoint_step']==512
        and p.get('cursor')==16384 and p.get('scheduler_horizon')==1024
        and p.get('statistics_sha256')==e.R0 and p.get('full_resume_payload')is False
        and type(p.get('payload_bytes'))is int and 0<p['payload_bytes']<=e.scope.INFERENCE_BYTES
        and p.get('numerical_tensor_changes')==0,'DELIVERY_ONLY_ORIGINAL_AUDITED_OWN512_FULL_INFERENCE')
    for key in('official_inference_manifest_sha256','wrapped_checkpoint_manifest_sha256'):
        need(type(p.get(key))is str and re.fullmatch('[0-9a-f]{64}',p[key]),'DELIVERY_FULL_ENDPOINT_SHA')
    need(sum(v['bytes']for v in p['files'].values())==p['payload_bytes']
        and set(plan['public_rewritten_metadata_base64'])==set(transfer.original.REWRITTEN),
        'DELIVERY_EXACT_PAYLOAD_AND_TWO_METADATA_REBINDS')
    return dict(candidate=arm,checkpoint_step=512,checkpoint_manifest_sha256=p['official_inference_manifest_sha256'],
        wrapped_checkpoint_manifest_sha256=p['wrapped_checkpoint_manifest_sha256'],statistics_sha256=e.R0,
        files=deepcopy(p['files']),payload_bytes=p['payload_bytes'],
        rewritten_metadata_sha256={n:e.sha(base64.b64decode(v,validate=True))
            for n,v in plan['public_rewritten_metadata_base64'].items()})


def source_manifest(body,identity):
    need(type(body)is bytes,'DELIVERY_ACTUAL_SOURCE_CONTROL_MANIFEST_BYTES');m=json.loads(body)
    need(m.get('schema')==MANIFEST and m.get('candidate')==identity['candidate']
        and m.get('checkpoint_manifest_sha256')==identity['checkpoint_manifest_sha256']
        and m.get('model_transfer_only')is True and type(m.get('CPU_upper_s'))is int and m['CPU_upper_s']==600
        and type(m.get('walltime_upper_s'))is int and m['walltime_upper_s']==7200
        and type(m.get('source_preparation_CPU_upper_s'))is int and m['source_preparation_CPU_upper_s']==300
        and m.get('scientific_effect_limits')==dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0)
        and all(type(v)is int for v in m['scientific_effect_limits'].values()),
        'DELIVERY_FINITE_REGISTERED_SOURCE_ONLY_NO_NEURAL_OR_CORE_PERMISSION')
    for k,n in(('source_sha',40),('parent_manifest_sha256',64),('compiled_plan_sha256',64),('worker_sha256',64)):
        need(type(m.get(k))is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',m[k]),'DELIVERY_PINNED_CONTROL_'+k)
    parent,root,checkpoint=(PurePosixPath(m[k])for k in('canonical_parent','control_root','checkpoint_root'))
    need(all(p.is_absolute()and '..'not in p.parts for p in(parent,root,checkpoint))
        and root.parent==parent/'control'and root.name==m.get('run_uuid')
        and checkpoint.parts[-2:]==('inference',identity['candidate'])
        and m.get('repo_id')=='leoyuer/pi0294-public-artifacts'
        and m.get('HF_prefix')=='arx5-first512/'+identity['candidate']+'/'+identity['checkpoint_manifest_sha256'],
        'DELIVERY_EXACT_NEW_CONTROL_ROOT_AND_FIXED_PUBLIC_MODEL_NAMESPACE')
    return m


def validate(g):
    entries=g.get(FIELD,{})
    need(type(entries)is dict and set(entries)<=set(e.training.ARMS),'DELIVERY_TWO_ORIGINAL_ARMS_ONLY')
    if not entries:return entries
    index,row=remaining.basic(g)
    need(row['status']==e.STATES[2]and row.get('scientific_endpoint_complete')is True
        and index['status']=='COMPLETE'and g['pending_updates']==g['unknown_effects']==[],
        'DELIVERY_ONLY_KNOWN_COMPLETED_REMAINDER_NO_GLOBAL_UNKNOWN')
    for arm,m in entries.items():
        identity=m['model_identity'];cp=index['terminal']['audit']['endpoints'][arm]
        need(m.get('schema')==SCHEMA and m.get('scope_grant_sha256')==e.authority.GRANT_SHA
            and m.get('status')=='AUTHORIZED_SOURCE_UPLOAD_NOT_EVAL'
            and identity['candidate']==arm and identity['checkpoint_step']==512
            and identity['checkpoint_manifest_sha256']==cp['official_inference_manifest_sha256']
            and identity['wrapped_checkpoint_manifest_sha256']==cp['checkpoint_manifest_sha256']
            and m['training_row_sha256']==e.sha(e.raw(row))
            and m['training_marker_sha256']==e.sha(e.raw(index))
            and m['original_package_sha256']==e.sha(e.raw(g[e.package.FIELD]))
            and m['paid_source_control']==CONTROL and m['additional_reservation']=={}
            and m['Core_permission_or_LOCAL_write']is False,'DELIVERY_EXACT_ORIGINAL_SCOPE_ROW_AND_PAID_SOURCE_SLICE')
        body=base64.b64decode(m['source_manifest_base64'],validate=True);manifest=source_manifest(body,identity)
        need(e.sha(body)==m['source_manifest_sha256']and manifest['compiled_plan_sha256']==m['compiled_plan_sha256']
            and type(m['paid_epoch'])in(int,float)and math.isfinite(m['paid_epoch'])
            and row['settled_epoch']<=m['paid_epoch']<row['deadline_epoch']-28860-7200,
            'DELIVERY_IMMUTABLE_MANIFEST_AND_ORIGINAL_COMPLETE_EVAL_WINDOW')
        before=m['before']
        need(g['CPU_s']>=before['CPU_s']+CONTROL['Hopper_CPU_s']
            and g['new_storage_current_bytes']>=before['new_storage_current_bytes']+CONTROL['Hopper_storage_bytes']
            and g['new_storage_peak_bytes']>=max(before['new_storage_peak_bytes'],g['new_storage_current_bytes']),
            'DELIVERY_PAID_SOURCE_HISTORY_CURRENT_AND_PEAK_NEVER_REFUNDED')
    return entries


def consume_remaining(g,value):
    out=deepcopy(value)
    for _ in validate(g):e.subtract(out,CONTROL)
    return out


def budget_LOCAL(g,local):
    entries=validate(g)
    if not entries:return local
    v=deepcopy(local);paid=v.setdefault(LOCAL,{})
    for arm,m in entries.items():
        proof=dict(source_manifest_sha256=m['source_manifest_sha256'],costs=CONTROL,
            canonical_LOCAL_written=False,source_only_not_Core_admission=True)
        if arm in paid:need(paid[arm]==proof,'DELIVERY_ONCE_UNCHANGED_LOCAL_PROJECTION');continue
        v['CPU_s']['local_PC']+=CONTROL['local_CPU_s'];v['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes'];paid[arm]=proof
    return v


def propose(global_bytes,historical_LOCAL_bytes,*,lease_bytes,settled_bytes,owner_observation,
            small_bytes,arm,source_manifest_bytes,now_epoch):
    g=json.loads(global_bytes);remaining.validate(g)
    plan=transfer.prepare(global_bytes,lease_bytes,settled_bytes,owner_observation,small_bytes,arm=arm)
    identity=model_identity(plan);manifest=source_manifest(source_manifest_bytes,identity)
    need(manifest['compiled_plan_sha256']==e.sha(e.raw(plan))and arm not in validate(g)
        and type(now_epoch)in(int,float)and math.isfinite(now_epoch),
        'DELIVERY_ONE_EXACT_MODEL_COMPILED_PLAN_AND_NEW_CONTROL_MANIFEST')
    before=e.scope.first512_forecast(global_bytes,historical_LOCAL_bytes)
    need(before['cap_changes_required']=={},'DELIVERY_EXISTING_FULL_FUTURE_FITS_BEFORE_SOURCE_CONTROL')
    v=deepcopy(g);index,row=remaining.basic(g)
    v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['CPU_reserved_s']-=CONTROL['Hopper_CPU_s']
    v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes'];v['storage_reserved_bytes']-=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    v.setdefault(FIELD,{})[arm]=dict(schema=SCHEMA,status='AUTHORIZED_SOURCE_UPLOAD_NOT_EVAL',
        scope_grant_sha256=e.authority.GRANT_SHA,training_row_sha256=e.sha(e.raw(row)),
        training_marker_sha256=e.sha(e.raw(index)),original_package_sha256=e.sha(e.raw(g[e.package.FIELD])),
        model_identity=identity,compiled_plan_sha256=e.sha(e.raw(plan)),
        source_manifest_base64=base64.b64encode(source_manifest_bytes).decode(),source_manifest_sha256=e.sha(source_manifest_bytes),
        paid_source_control=deepcopy(CONTROL),additional_reservation={},Core_permission_or_LOCAL_write=False,
        paid_epoch=now_epoch,before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')},
        before_GLOBAL_sha256=e.sha(global_bytes),historical_LOCAL_sha256=e.sha(historical_LOCAL_bytes))
    v['revision']+=1;validate(v)
    from scripts.pi0294.pv1.neural_ledger import check
    check(v);after=e.scope.first512_forecast(e.raw(v),historical_LOCAL_bytes)
    need(after['cap_changes_required']=={}and all(after['dimensions'][k]['projected']==r['projected']
        for k,r in before['dimensions'].items()),'DELIVERY_ALL14_UNCHANGED_NOT_NEW_BUDGET_OR_DROPPED_CORE_ESCROW')
    return dict(proposed_global=v,compiled_plan=plan,source_control_manifest=manifest,
        full_future_proof=after,original_onceIO_required=True,Core_LOCAL_written=False,
        new_GPU_PBS_queries_updates_physics=0,native_admission=False)


def upload_admission(g,plan,*,now_epoch,checkpoint_complete_bytes,source_control_manifest_bytes,
                     source_checkpoint_root):
    remaining.validate(g);identity=model_identity(plan);m=validate(g).get(identity['candidate'],{})
    need(m.get('model_identity')==identity and m.get('compiled_plan_sha256')==e.sha(e.raw(plan))
        and e.sha(source_control_manifest_bytes)==m.get('source_manifest_sha256'),
        'DELIVERY_NATIVE_UPLOADER_EXACT_ONCE_PAID_MODEL_AND_REGISTERED_MANIFEST')
    manifest=source_manifest(source_control_manifest_bytes,identity)
    need(source_checkpoint_root==manifest['checkpoint_root']
        and e.sha(checkpoint_complete_bytes)==identity['checkpoint_manifest_sha256']
        and m['paid_epoch']<=e.original.number(now_epoch)<remaining.basic(g)[1]['deadline_epoch']-28860-7200,
        'DELIVERY_CURRENT_ORIGINAL_WINDOW_AND_EXACT_SOURCE_EXPORT_BYTES')
    # The receiver revalidates this same closure through the original restore
    # dispatch. The uploader still streams every source byte before/after and
    # verifies the one returned fixed HF revision with the mature protocol.
    return dict(status='MINIMAL_OWN512_SOURCE_ONLY_IO_WITHIN_EXISTING_EVAL_ESCROW',
        model_identity=identity,source_CPU_upper_s=600,model_or_physics_admission=False,
        new_GPU_PBS_queries_updates=0,stage_completed=False,
        commit_intent_schema='ARX5_MINIMAL_OWN512_MODEL_ONLY_PUBLIC_HF_COMMIT_ONCE_INTENT_V1')
