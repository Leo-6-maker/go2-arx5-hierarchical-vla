"""Exact first512 Vanda + original D25 Core join in the existing GLOBAL.

Cold policy only; not native admission, a second budget, or generic sharing.
Initial first512 registration remains serial so the sealed live C50 owner is
never given an unknown PBS row. A new D25 request/manifest must bind this exact
adopted main row before the usual full native process/resource qualification.
"""
from copy import deepcopy
import json
from . import minimal_execution as e

VERSION='ARX5_FIRST512_VANDA_PLUS_ORIGINAL_D25_CORE_COLD_V1'
KEY='minimal_first512_cross_scope'
LOCAL_MIRROR='ARX5_minimal_first512_LOCAL_main_inventory_mirror'


def binding(g):
    mark=e.validate(g);e.need(mark is not None,'CONCURRENT_MAIN_NOT_REGISTERED')
    row=e.rows(g)[0]
    e.need(row['status']==e.STATES[1]and row['job_id']is not None,
           'CONCURRENT_REAL_BOUND_MAIN_PBS_NOT_QSUB_UNKNOWN')
    return binding_identity(row)


def binding_identity(row):
    """Stable sealed pair identity; this alone never permits new concurrency."""
    e.need(row['platform_training_version']==e.VERSION and row['status']in e.STATES[1:]
        and row['job_id']is not None,'CONCURRENT_EXACT_BOUND_OR_ALREADY_AUDITED_TERMINAL_MAIN')
    return dict(schema=VERSION,main_request_id=row['request_id'],main_request_sha256=row['request_sha256'],
        main_job_id=row['job_id'],main_source_sha=row['source_sha'],
        scope_grant_sha256=e.authority.GRANT_SHA,main_scope=e.scope.SCOPE,
        Core_candidate='D25',Core_conditions=24,Core_noise=294101,
        maximum_Vanda_GPUs=1,maximum_Core_GPUs=1,
        global_pending_and_unknown_fences_preserved=True,
        allocation_local_pending_updates_are_not_removed_or_globalized=True)


def current_native_view(g,Core_record,*,queue_proof,terminal_proofs,independent_job_bytes,now_epoch):
    """Read the real main F during independent audit without calling it UNKNOWN.

    Only a new D25 capsule sealed to this main may use this join. A missing
    queue handle still requires exact fresh native F/request/nonce history;
    it is never inferred terminal, omitted, refunded or marked settled here.
    The existing Core owner must collect those original raw native proofs.
    """
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.dec1 import vanda_campaign as q
    from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
    check(g);rows=e.rows(g);e.need(len(rows)==1,'CONCURRENT_VIEW_ONE_ADOPTED_MAIN')
    main=rows[0];proof=binding_identity(main)
    from scripts.pi0294.pv1.mix_core_accounting import VERSION as Core_VERSION
    e.need(Core_record.get(KEY)==proof and Core_record.get('evaluation_candidate')=='D25'
        and Core_record.get('platform_evaluation_version')==Core_VERSION
        and Core_record.get('native_platform')=='CORELAB_RTX4080'
        and Core_record.get('source_sha')!='8fba44f38aa5c6858614d8bd4e34ad7f969dd40c'
        and Core_record['created_epoch']>=g[e.FIELD]['registered_epoch']
        and g['pending_updates']==g['unknown_effects']==[],
        'CONCURRENT_VIEW_ONLY_EXPLICIT_COLD_D25_AND_ALL_GLOBAL_FENCES')
    current=q.queue_rows(queue_proof,now_epoch=now_epoch)
    terminals=q.terminal_rows(terminal_proofs,ledger=g,current_rows=current,now_epoch=now_epoch)
    campaign,outside,unknown=q.classify(g,current+terminals)
    independent=checked_outside(current+terminals,outside,independent_job_bytes)
    e.need(not unknown and all(v['job_id']==main['job_id']and v['state']in('Q','R')for v in campaign),
        'CONCURRENT_VIEW_COMPLETE_REAL_QUEUE_AND_OWN_DISTINCT_NATIVE_TERMINAL')
    if main['status']==e.STATES[1]:
        e.need(len(campaign)+len(terminals)==1,'CONCURRENT_VIEW_OWN_MAIN_IS_ACTUALLY_PRESENT_OR_F')
    else:e.need(not campaign and not terminals,'CONCURRENT_VIEW_SETTLED_MAIN_HAS_NO_NEW_ALLOCATION')
    return dict(schema='ARX5_FIRST512_D25_CURRENT_NATIVE_QUEUE_OR_KNOWN_F_V1',main_binding=proof,
        complete_account_queue=queue_proof,native_terminal_history=terminal_proofs,
        campaign_jobs=campaign,user_confirmed_independent_jobs=independent,
        unknown_campaign_jobs=[],GLOBAL_sha256=e.sha(e.raw(g)),observed_epoch=now_epoch,
        known_native_F_waiting_for_independent_main_audit=bool(terminals),
        GLOBAL_or_LOCAL_mutated=False,new_model_or_update_or_physics_permission=False)


def validate_pair(g,pbs_rows,Core_rows):
    """Allow precisely the two different scopes/sites, never two trainers."""
    e.need(len(pbs_rows)==len(Core_rows)==1 and e.FIELD in g,
           'CONCURRENT_EXACT_ONE_MAIN_AND_ONE_CORE_ROLE')
    proof=binding(g);main=pbs_rows[0];core=Core_rows[0]
    from scripts.pi0294.pv1.mix_core_accounting import VERSION as Core_VERSION
    e.need(main['request_id']==proof['main_request_id']
        and main['platform_training_version']==e.VERSION
        and core.get('evaluation_candidate')=='D25'
        and core.get('platform_evaluation_version')==Core_VERSION
        and core.get('native_platform')=='CORELAB_RTX4080'
        and core.get(KEY)==proof
        and core['source_sha']!= '8fba44f38aa5c6858614d8bd4e34ad7f969dd40c'
        and core['created_epoch']>=g[e.FIELD]['registered_epoch']
        and g['pending_updates']==g['unknown_effects']==[],
        'CONCURRENT_EXPLICIT_NEW_D25_SCOPE_SOURCE_AND_ALL_GLOBAL_FENCES')
    return proof


def permit_Core(g,record,manifest=None):
    """Reused at registration and each original premodel/heartbeat proof."""
    active=[r for r in g['PBS_submissions']if r['status']not in e.original.TERMINAL]
    if not active:return None
    proof=validate_pair(g,active,[record])
    if manifest is not None:
        e.need(manifest['authority'].get(KEY)==proof,
               'CONCURRENT_PHYSICAL_PEER_MANIFEST_BOUND_BEFORE_NATIVE_REGISTRATION')
    return proof


def validate_LOCAL(g,l):
    active=[r for r in g['PBS_submissions']if r['status']not in e.original.TERMINAL]
    if not active:return None
    proof=binding(g);mirror=l.get(LOCAL_MIRROR,{})
    e.need(mirror.get('schema')==VERSION and mirror.get('main_binding')==proof
        and mirror.get('only_original_once_control_costs')==e.CONTROL
        and any(x==active[0]for x in l['PBS_submissions']),
        'CONCURRENT_ACTUAL_LOCAL_MAIN_INVENTORY_MUST_BE_CAUGHT_UP_BEFORE_CORE')
    return proof


def propose_LOCAL(global_bytes,local_bytes,*,now_epoch):
    """One pure canonical inventory/control proposal; original onceIO required.

    The new package is held in GLOBAL already. Do not debit the old Core holds
    or prepay the same Vanda control again. Keep all14 forecast totals exactly
    equal to the original conservative control projection. No copied LOCAL is
    called current, and no model/Core permission follows from this proposal.
    """
    from scripts.pi0294.pv1 import mix_core_accounting as core
    from scripts.pi0294.pv1.neural_ledger import check
    from .minimal_scope import first512_forecast
    g,l=json.loads(global_bytes),json.loads(local_bytes);check(g,allow_unknown=True)
    proof=binding(g);row=e.rows(g)[0]
    e.need(LOCAL_MIRROR not in l and g['PBS_submissions'][-1]==row
        and l['PBS_submissions']==g['PBS_submissions'][:-1]
        and g['unknown_effects']==g['pending_updates']==[]
        and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[]
        and all(x['status']=='NATIVE_TERMINAL'for x in g.get(core.FIELD,{}).values())
        and g[e.FIELD]['registered_epoch']<=now_epoch<row['deadline_epoch'],
        'CONCURRENT_ONCE_COLD_CORE_BOUNDARY_AND_EXACT_OLD_NATIVE_LOCAL_BYTES')
    before=first512_forecast(global_bytes,local_bytes)
    v=e.budget_LOCAL(g,l);v['PBS_submissions']=deepcopy(g['PBS_submissions'])
    v[LOCAL_MIRROR]=dict(schema=VERSION,main_binding=proof,
        before_LOCAL_sha256=e.sha(local_bytes),GLOBAL_sha256=e.sha(global_bytes),
        only_original_once_control_costs=e.CONTROL,observed_epoch=now_epoch,
        new_budget=False,new_model_queries_updates_or_physics=0)
    core.paired(g,v);validate_LOCAL(g,v)
    after=first512_forecast(global_bytes,e.raw(v))
    e.need(before['cap_changes_required']==after['cap_changes_required']=={}
        and all(before['dimensions'][k]['projected']==after['dimensions'][k]['projected']
                for k in before['dimensions']), 'CONCURRENT_ALL14_DIMENSIONS_NO_NEW_COST_OR_DROPPED_HOLD')
    return dict(proposed_LOCAL=v,full_future_proof=after,original_onceIO_required=True,
                native_admission=False,new_GPU=0,new_PBS=0,new_models=0)
