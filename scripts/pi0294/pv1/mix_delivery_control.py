"""C/D model-only transfer within the existing Eval control and IO envelope.

Pure proposals, no new reservation or scientific admission. Native transfer
keeps the original uploader, source/destination streaming and once Journal.
"""
from copy import deepcopy
import json
import base64

from . import mix_accounting as a, mix_core_accounting as c
from . import mix_evaluation_transfer as t, successor_cd as s

FIELD = 'V5_CD_model_only_delivery'


def need(ok, reason):
    if not ok:
        raise PermissionError('V5_CD_MODEL_DELIVERY_' + reason)


def model_identity(plan):
    p = plan['public_plan']
    need(p.get('schema') == 'V5_CD_CLOSED1024_MODEL_ONLY_TRANSFER_PLAN_V1'
         and p.get('task') == s.TASK and p.get('candidate') in ('C50', 'D25')
         and type(p.get('checkpoint_step')) is int and p['checkpoint_step'] == 1024
         and p.get('statistics_sha256') == a.STATS
         and p.get('full_resume_payload') is False, 'ONLY_AUDITED_1024_INFERENCE')
    return dict(candidate=p['candidate'], checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],
        checkpoint_step=1024, statistics_sha256=p['statistics_sha256'],
        files=deepcopy(p['files']), payload_bytes=p['payload_bytes'],
        rewritten_metadata_sha256={n:a.sha(base64.b64decode(v,validate=True))
            for n,v in plan['public_rewritten_metadata_base64'].items()})


def prepay(g, l, *, capture_bytes, small_bytes, arm, now_epoch):
    # Recompile complete audit/closure, not a caller-supplied PASS/plan.
    plan = t.prepare(capture_bytes, small_bytes, arm=arm)
    capture = json.loads(capture_bytes)
    terminal = {k:base64.b64decode(v,validate=True)
        for k,v in capture['terminal_capture_base64'].items()}
    need(capture['global_base64'] == base64.b64encode(a.raw(g)).decode(),
         'EXACT_CURRENT_GLOBAL_CLOSED_CAPTURE')
    result = c.prepay_control(g,l,terminal_capture=terminal,arm=arm,now_epoch=now_epoch)
    v,w = result['proposed_global'],result['proposed_LOCAL']
    stage = c.key(arm)
    need(stage not in g[a.PACKAGE].get(FIELD,{}), 'ONE_DELIVERY_NOT_BLIND_RETRY')
    identity = model_identity(plan)
    remaining = c.remaining_stage()
    need(identity['payload_bytes'] < remaining['Core_storage_bytes']
         and 2*identity['payload_bytes'] < remaining['transfer_bytes'],
         'MODEL_IO_INSIDE_ORIGINAL_FULL_STAGE_NOT_EXTRA_BUDGET')
    marker = dict(schema='V5_CD_MODEL_ONLY_EXISTING_STAGE_DELIVERY_V1',
        task=s.TASK, scope_grant_sha256=s.GRANT_SHA, stage=stage,
        status='AUTHORIZED_NOT_DELIVERED', model_identity=identity,
        source_control_CPU_upper_s=600, physical_Core_metadata_CPU_upper_s=300,
        existing_paid_site_control_CPU_s=7200,
        existing_physical_Core_stage_CPU_s=93900,
        before_GLOBAL_sha256=a.sha(a.raw(g)), before_LOCAL_sha256=a.sha(a.raw(l)),
        observed_epoch=now_epoch, additional_reservation={},
        stage_completed=False, model_or_physics_admission=False,
        IO_charged_by_original_stage_terminal_once=True)
    for value in (v,w):
        value[a.PACKAGE].setdefault(FIELD,{})[stage] = deepcopy(marker)
    c.checked(v);c.paired(v,w)
    result['delivery_marker'] = marker
    return result


def upload_admission(g, plan, *, now_epoch, checkpoint_complete_bytes,
                     source_control_manifest_bytes=None,source_checkpoint_root=None):
    """Native uploader still verifies every source tensor before/after upload."""
    if plan['public_plan'].get('candidate')in('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
        from scripts.pi0294.generalization.minimal_delivery_control import upload_admission as minimal_upload
        return minimal_upload(g,plan,now_epoch=now_epoch,checkpoint_complete_bytes=checkpoint_complete_bytes,
            source_control_manifest_bytes=source_control_manifest_bytes,source_checkpoint_root=source_checkpoint_root)
    c.checked(g);p=a.package_view(g);arm=plan['public_plan']['candidate'];stage=c.key(arm)
    marker=p.get(FIELD,{}).get(stage,{})
    need(marker.get('schema')=='V5_CD_MODEL_ONLY_EXISTING_STAGE_DELIVERY_V1'
         and marker.get('task')==s.TASK and marker.get('stage')==stage
         and marker.get('scope_grant_sha256')==s.GRANT_SHA
         and marker.get('status')=='AUTHORIZED_NOT_DELIVERED'
         and marker.get('model_identity')==model_identity(plan), 'ACTUAL_AUTHORIZED_EXACT_MODEL')
    need(stage in p.get('prepaid_control',{})
         and p['prepaid_control'][stage]['costs']==c.CONTROL
         and p['completed_stages']==c.before_stage(arm,g)
         and all(r['status']in a.TERMINAL for r in g['PBS_submissions'])
         and not any(r['status']!='NATIVE_TERMINAL'for r in g.get(a.CORE_ROWS,{}).values()),
         'CLOSED_TRAINING_AND_ONCE_PAID_CONTROL_BEFORE_EVAL')
    need(s.validated(g)['T_CD']<=a.number(now_epoch)<s.validated(g)['deadline_epoch']-28860,
         'ORIGINAL_WINDOW_AND_FULL_EVAL_CLOSE_MARGIN')
    cp=json.loads(checkpoint_complete_bytes);identity=marker['model_identity']
    need(a.sha(checkpoint_complete_bytes)==identity['checkpoint_manifest_sha256']
         and type(cp.get('global_step'))is int and cp['global_step']==1024
         and cp.get('full_resume')is True,'ACTUAL_COMPLETE_SOURCE_BYTES_NOT_PREPARED_ALIAS')
    need(g['unknown_effects']==g['pending_updates']==[], 'NO_UNKNOWN_EFFECTS')
    return dict(status='MODEL_ONLY_IO_INSIDE_EXISTING_EVAL_ENVELOPE',
        model_identity=identity, source_CPU_upper_s=600, model_or_physics_admission=False,
        new_GPU_PBS_queries_updates=0, stage_completed=False)
