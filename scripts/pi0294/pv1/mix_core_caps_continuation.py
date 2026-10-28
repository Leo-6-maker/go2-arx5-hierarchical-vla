"""One cold continuation of the bound, zero-effect old-cap precedence failure.

Original paid history, counters, gross IO and predecessor grants remain intact.
Pure proposal: native original onceIO and complete fresh admission remain required.
"""
import json
from copy import deepcopy
from . import mix_accounting as a,successor_cd as s
FIELD='V5_CD_exact54b19dfb_zero_model_cap_precedence_continuation'
FAILED='54b19dfb-1cda-4031-831d-3605c29c69f0'
SOURCE='4980a78fa62869c5198e614453f4b7c782e8d884'

def need(ok,why):
    if not ok:raise PermissionError('V5_CD_CAP_PRECEDENCE_CONTINUATION_'+why)

def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=prior.continuation(g);r=g[c.FIELD][FAILED]
    need(parent is not None and parent['replacement_request_id']==FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==SOURCE
         and r['native_handle']and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_BOUND_ZERO_MODEL_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_MODEL_CAP_PRECEDENCE_COLD_CONTINUATION_V1'
         and m['failed_request_id']==FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-06'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','model_load_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and all(physical[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-04'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    return m

def propose(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    need(FIELD not in g and FIELD not in l and prior.continuation(g)['replacement_request_id']==FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_54B19DFB_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
         and p['request_id']==FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_MODEL_CAP_PRECEDENCE_COLD_CONTINUATION_V1',failed_request_id=FAILED,
        terminal_native_sha256=g[c.FIELD][FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-06',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50')
    for val in(v,w):
        val[FIELD]=deepcopy(m)
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)

RPC_FIELD='V5_CD_exactafc1532b_zero_model_registered_RPC_affinity_continuation'
RPC_FAILED='afc1532b-732c-47a8-9f0e-ccc74c2306b9'
RPC_SOURCE='64bf3308bbd3b799e03ab9f8218741d8d5b0d480'

def validate_RPC(g):
    m=g.get(RPC_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate(g);r=g[c.FIELD][RPC_FAILED]
    need(parent is not None and parent['replacement_request_id']==RPC_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==RPC_SOURCE
         and r['native_handle']and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_BOUND_ZERO_MODEL_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_MODEL_REGISTERED_RPC_AFFINITY_COLD_CONTINUATION_V1'
         and m['failed_request_id']==RPC_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-07'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=RPC_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','model_load_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and all(physical[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-05'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    return m

def propose_RPC(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    need(RPC_FIELD not in g and RPC_FIELD not in l and validate(g)['replacement_request_id']==RPC_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_AFC1532B_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
         and p['request_id']==RPC_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_MODEL_REGISTERED_RPC_AFFINITY_COLD_CONTINUATION_V1',failed_request_id=RPC_FAILED,
        terminal_native_sha256=g[c.FIELD][RPC_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-07',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50')
    for val in(v,w):
        val[RPC_FIELD]=deepcopy(m)
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_RPC(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(RPC_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)

PEER_FIELD='V5_CD_exact1aeacefb_zero_model_physical_peer_identity_continuation'
PEER_FAILED='1aeacefb-20cc-4077-aad8-a959f0268f78'
PEER_SOURCE='69b0f0828c72a79e27a77c536cf5194555718af0'

def validate_peer(g):
    m=g.get(PEER_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate_RPC(g);r=g[c.FIELD][PEER_FAILED]
    need(parent is not None and parent['replacement_request_id']==PEER_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==PEER_SOURCE
         and r['native_handle']and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_BOUND_ZERO_MODEL_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_MODEL_PHYSICAL_PEER_IDENTITY_COLD_CONTINUATION_V1'
         and m['failed_request_id']==PEER_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-08'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=PEER_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','model_load_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and all(physical[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-06'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    return m

def propose_peer(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    need(PEER_FIELD not in g and PEER_FIELD not in l and validate_RPC(g)['replacement_request_id']==PEER_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_1AEACEFB_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
         and p['request_id']==PEER_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_MODEL_PHYSICAL_PEER_IDENTITY_COLD_CONTINUATION_V1',failed_request_id=PEER_FAILED,
        terminal_native_sha256=g[c.FIELD][PEER_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-08',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50')
    for val in(v,w):
        val[PEER_FIELD]=deepcopy(m)
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_peer(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(PEER_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)

DESKTOP_FIELD='V5_CD_exactd6a4e1a8_zero_model_recorded_desktop_epoch_continuation'
DESKTOP_FAILED='d6a4e1a8-5a2e-4581-b7cd-651dc7108fd1'
DESKTOP_SOURCE='f4c00131ce5011f619233a0eaa4f67d123ebc1c9'

def validate_recorded_desktop(g):
    m=g.get(DESKTOP_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate_peer(g);r=g[c.FIELD][DESKTOP_FAILED]
    need(parent is not None and parent['replacement_request_id']==DESKTOP_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==DESKTOP_SOURCE
         and r['native_handle']and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_BOUND_ZERO_MODEL_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_MODEL_RECORDED_DESKTOP_EPOCH_COLD_CONTINUATION_V1'
         and m['failed_request_id']==DESKTOP_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-09'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=DESKTOP_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','model_load_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and all(physical[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-07'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    return m

def propose_recorded_desktop(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    need(DESKTOP_FIELD not in g and DESKTOP_FIELD not in l and validate_peer(g)['replacement_request_id']==DESKTOP_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_D6A4E1A8_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
         and p['request_id']==DESKTOP_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_MODEL_RECORDED_DESKTOP_EPOCH_COLD_CONTINUATION_V1',failed_request_id=DESKTOP_FAILED,
        terminal_native_sha256=g[c.FIELD][DESKTOP_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-09',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50')
    for val in(v,w):
        val[DESKTOP_FIELD]=deepcopy(m)
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_recorded_desktop(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(DESKTOP_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)

BOUND_FIELD='V5_CD_exact75c63c6e_zero_model_post600_bound_age_continuation'
BOUND_FAILED='75c63c6e-c3f6-41cc-adae-a62971b26e74'
BOUND_SOURCE='945595ccd6fbca8f50a51fde963085f9c286be39'

def validate_bound_age(g):
    m=g.get(BOUND_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate_recorded_desktop(g);r=g[c.FIELD][BOUND_FAILED]
    need(parent is not None and parent['replacement_request_id']==BOUND_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==BOUND_SOURCE
         and r['native_handle']and r['query_attempts']==r['model_load_attempts']==0
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_BOUND_ZERO_MODEL_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_MODEL_POST600_BOUND_AGE_COLD_CONTINUATION_V1'
         and m['failed_request_id']==BOUND_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-10'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=BOUND_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','model_load_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and all(physical[k]==0 for k in('main_starts','engine_starts','live_and_recovery_integrations','shadow_integrations'))
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-08'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    return m

def propose_bound_age(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    from scripts.pi0294.generalization import minimal_gpu_repair
    need(minimal_gpu_repair.validate(g)is not None,'APPROVED_148_HOURS_ACTUALLY_ADOPTED')
    need(BOUND_FIELD not in g and BOUND_FIELD not in l and validate_recorded_desktop(g)['replacement_request_id']==BOUND_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_75C63C6E_BOUND_ZERO_MODEL_THREE_CLOSED_READONLY'
         and p['request_id']==BOUND_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_MODEL_POST600_BOUND_AGE_COLD_CONTINUATION_V1',failed_request_id=BOUND_FAILED,
        terminal_native_sha256=g[c.FIELD][BOUND_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-10',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50')
    for val in(v,w):
        val[BOUND_FIELD]=deepcopy(m)
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_bound_age(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(BOUND_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)


def engineering(files):
    import base64
    hashes={'result':'7815bd5af536302ba0b7c6994f083d9aa634e9d853c82a8c67a6dae4ea52a7e5',
      'adjudication':'ac5e01c6a776e12ec4808570b2faf992e8af57a4a8fe72671a9fb565249687ca',
      'audit':'34d3be9c641bee23dc310cc018118690f4e419915a9dd3b739f58ae250ec28bc',
      'RPC_log':'ee6bff19660cb552c211047f39aec0dd1c80ea3f73f41c95b2a33400757056e7'}
    need(set(files)==set(hashes),'EXACT_ORIGINAL_ENGINEERING_RECEIPTS')
    raw={k:base64.b64decode(v['base64'],validate=True)for k,v in files.items()}
    need(all(a.sha(raw[k])==v['sha256']==hashes[k]for k,v in files.items()),'ORIGINAL_RAW_ENGINEERING_BYTES')
    result,adj,audit=(json.loads(raw[k])for k in('result','adjudication','audit'))
    need(adj['status']=='ENGINEERING_INCOMPLETE'and result['physics_steps']==result['model_calls']==0
      and result['physics_starts']==1 and audit['status']=='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'
      and audit['adjudication']==adj and audit['scientific_policy_outcome_valid']is False
      and audit['errors']==[]and all(v is True for v in audit['checks'].values())
      and b'ValueError: not allowed to raise maximum limit'in raw['RPC_log'],'NO_VALID_POLICY_OUTCOME_OR_QUERY_REPLAY')
    return audit

LIMIT_FIELD='V5_CD_exacted3880cd_zero_integration_RPC_CPU_limit_continuation'
LIMIT_FAILED='ed3880cd-2a21-4855-84aa-ec8d943104cb'
LIMIT_SOURCE='090b5d301a031804500d72e9863386a9bd6aa384'

def validate_rpc_limit(g):
    m=g.get(LIMIT_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate_bound_age(g);r=g[c.FIELD][LIMIT_FAILED]
    need(parent is not None and parent['replacement_request_id']==LIMIT_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==LIMIT_SOURCE
         and r['native_handle']and r['query_attempts']==0 and r['model_load_attempts']==1
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_ONE_MODEL_ZERO_INTEGRATION_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_INTEGRATION_RPC_CPU_LIMIT_COLD_CONTINUATION_V1'
         and m['failed_request_id']==LIMIT_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-11'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=LIMIT_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['model_load_attempts']==1 and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and physical['main_starts']==physical['engine_starts']==1 and physical['live_and_recovery_integrations']==physical['shadow_integrations']==0
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-09'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    engineering(m['engineering'])
    need(type(m['engineering_replacements_used_before'])is int and 0<=m['engineering_replacements_used_before']<4
         and m['engineering_replacements_used_after']==m['engineering_replacements_used_before']+1
         and m['engineering_replacements_used_after']<=g[a.PACKAGE]['engineering_replacements_used']<=4,'ORIGINAL_FINITE_ENGINEERING_POOL')
    return m

def propose_rpc_limit(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    from scripts.pi0294.generalization import minimal_gpu_repair
    need(minimal_gpu_repair.validate(g)is not None,'APPROVED_148_HOURS_ACTUALLY_ADOPTED')
    need(LIMIT_FIELD not in g and LIMIT_FIELD not in l and validate_bound_age(g)['replacement_request_id']==LIMIT_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_ED3880CD_ONE_MODEL_ZERO_INTEGRATION_THREE_CLOSED_READONLY'
         and p['request_id']==LIMIT_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_INTEGRATION_RPC_CPU_LIMIT_COLD_CONTINUATION_V1',failed_request_id=LIMIT_FAILED,
        terminal_native_sha256=g[c.FIELD][LIMIT_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-11',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    engineering(p['engineering']);before=g[a.PACKAGE]['engineering_replacements_used'];need(before<4,'ORIGINAL_ENGINEERING_POOL_NOT_EXHAUSTED')
    m.update(engineering=deepcopy(p['engineering']),engineering_replacements_used_before=before,engineering_replacements_used_after=before+1)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50');restore['episode_starts']-=1;restore['engine_starts']-=1
    for val in(v,w):
        val[LIMIT_FIELD]=deepcopy(m)
        val[a.PACKAGE]['engineering_replacements_used']=before+1
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_rpc_limit(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(LIMIT_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)


def production_engineering(files):
    import base64
    hashes={'result': '54d890a27ab716c2e7a357243eac6663136558880f881b8f4ba6e96066855f6a', 'adjudication': '09a5de7540a60fe701e656fc0234b8083e7a6088efd38cf0abcd9b970ecad90f', 'audit': 'e19919083626dad8ab8d05a830e5dd1a8e54fd11e74ecd0c6cad864af8444b25', 'RPC_log': 'b782cc2053583fcdf17be666ff94b4d19467f837f8731086313530c8cdcc8078'}
    need(set(files)==set(hashes),'EXACT_ORIGINAL_ENGINEERING_RECEIPTS')
    raw={k:base64.b64decode(v['base64'],validate=True)for k,v in files.items()}
    need(all(a.sha(raw[k])==v['sha256']==hashes[k]for k,v in files.items()),'ORIGINAL_RAW_ENGINEERING_BYTES')
    result,adj,audit=(json.loads(raw[k])for k in('result','adjudication','audit'))
    need(adj['status']=='ENGINEERING_INCOMPLETE'and result['physics_steps']==result['model_calls']==0
      and result['physics_starts']==1 and audit['status']=='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'
      and audit['adjudication']==adj and audit['scientific_policy_outcome_valid']is False
      and audit['errors']==[]and all(v is True for v in audit['checks'].values())
      and b'PermissionError: PV1_PRESEALED_DIRECTORY_ASSISTANCE_REQUIRED'in raw['RPC_log'],'NO_VALID_POLICY_OUTCOME_OR_QUERY_REPLAY')
    return audit

CONTROL_FIELD='V5_CD_exactfb11b53c_zero_integration_production_RPC_control_continuation'
CONTROL_FAILED='fb11b53c-6d49-44da-b655-b90ea6d69d3a'
CONTROL_SOURCE='fb54da77d489b8000057440332a4d8e222cd8373'

def validate_production_control(g):
    m=g.get(CONTROL_FIELD)
    if m is None:return None
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior
    parent=validate_rpc_limit(g);r=g[c.FIELD][CONTROL_FAILED]
    need(parent is not None and parent['replacement_request_id']==CONTROL_FAILED
         and r['status']==c.STATES[3]and r['native_exit_status']==1 and r['source_sha']==CONTROL_SOURCE
         and r['native_handle']and r['query_attempts']==0 and r['model_load_attempts']==1
         and r['unknown_query_intents']==[]and r['reserved_neural_queries']==0,
         'ACTUAL_CLOSED_ONE_MODEL_ZERO_INTEGRATION_PREDECESSOR')
    # A registered replacement may already exist; only its immutable identity is
    # checked here. The proposal separately rejects every prior UUID/nonce.
    need(m['schema']=='V5_CD_ZERO_INTEGRATION_PRODUCTION_RPC_CONTROL_COLD_CONTINUATION_V1'
         and m['failed_request_id']==CONTROL_FAILED and m['terminal_native_sha256']==r['terminal_native_sha256']
         and m['scope_grant_sha256']==s.GRANT_SHA and m['maximum_continuations']==1
         and m['new_PBS']==0 and m['new_models_queries_physics_valid_conditions']==0
         and m['all_prior_rows_costs_unknowns_caps_and_clock_preserved']is True
         and m['replacement_run_uuid']=='v5-cd-c50-n294101-12'
         and c.canonical_uuid(m['replacement_request_id'])and c.canonical_uuid(m['replacement_submission_nonce'])
         and m['replacement_request_id']!=CONTROL_FAILED and m['replacement_submission_nonce']!=r['submission_nonce'],
         'NEW_EXACT_CUSTODY')
    proof=m['native_proof'];effects=m['effects'];physical=m['physical'];clock=s.validated(g)
    need(c.digest(proof)==r['terminal_native_sha256']and c.digest(effects)==r['terminal_effect_summary_sha256']
         and proof['native_exit_status']==1 and proof['recorded_process_terminal']is True
         and proof['terminal_inventory_stable']is True and effects['effect_journal_complete']is True
         and all(effects[k]==0 for k in('query_attempts','optimizer_updates_committed','new_physics','new_episodes'))
         and effects['model_load_attempts']==1 and effects['unknown_query_intents']==[]and physical['reservations']=={}and physical['unknown_effects']==[]
         and physical['main_starts']==physical['engine_starts']==1 and physical['live_and_recovery_integrations']==physical['shadow_integrations']==0
         and m['T_CD']==clock['T_CD']and m['deadline_epoch']==clock['deadline_epoch'],
         'ORIGINAL_TERMINAL_EFFECTS_COUNTERS_AND_CLOCK')
    need(set(m['closed_leases'])=={'GPU','physical','control'},'ALL_THREE_REAL_LEASES')
    for role,lease in m['closed_leases'].items():
        need(lease['status']=='CLOSED'and lease['terminal_native_sha256']==r['terminal_native_sha256']
             and lease['all_actual_workers_terminal']is True and lease['authority_release']==s.TASK_SOURCE
             and lease['run_uuid']==('v5-cd-c50-eval-owner-10'if role=='control'else r['child_run_uuid']),
             'ACTUAL_PREDECESSOR_LEASE_CLOSED')
    production_engineering(m['engineering'])
    need(type(m['engineering_replacements_used_before'])is int and 0<=m['engineering_replacements_used_before']<4
         and m['engineering_replacements_used_after']==m['engineering_replacements_used_before']+1
         and m['engineering_replacements_used_after']<=g[a.PACKAGE]['engineering_replacements_used']<=4,'ORIGINAL_FINITE_ENGINEERING_POOL')
    return m

def propose_production_control(global_bytes,local_bytes,*,closed_readback,new_request_id,new_nonce,now_epoch):
    from . import mix_core_accounting as c,mix_core_prebind_terminal as prior,mix_core_premodel_repair as repair
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l);p=closed_readback
    from scripts.pi0294.generalization import minimal_gpu_repair
    need(minimal_gpu_repair.validate(g)is not None,'APPROVED_148_HOURS_ACTUALLY_ADOPTED')
    need(CONTROL_FIELD not in g and CONTROL_FIELD not in l and validate_rpc_limit(g)['replacement_request_id']==CONTROL_FAILED
         and all(r['status']==c.STATES[3]for r in g[c.FIELD].values())and g['unknown_effects']==g['pending_updates']==[],
         'COLD_ALL_REAL_TERMINALS_NO_CURRENT_EFFECT')
    need(p['status']=='ACTUAL_FB11B53C_ONE_MODEL_ZERO_INTEGRATION_CLOSURE_READONLY'
         and p['request_id']==CONTROL_FAILED and p['host']=='corelab-4080s'and p['uid']==2009
         and p['GLOBAL_sha256']==a.sha(global_bytes)and p['LOCAL_sha256']==a.sha(local_bytes)
         and 0<=now_epoch-p['observed_epoch']<=120,
         'FRESH_COMPLETE_ACTUAL_DUAL_BYTES_AND_LEASES')
    need(not any(r.get('request_id')==new_request_id or r.get('submission_nonce')==new_nonce
         for r in[*g[c.FIELD].values(),*g['PBS_submissions'],*g.get('Core_native_GPU_allocations',{}).values()]),'NEW_IDENTITY_NO_REPLAY')
    clock=s.validated(g);need(clock['T_CD']<=now_epoch<clock['deadline_epoch']-28860,'FULL_ORIGINAL_PANEL_WINDOW')
    m=dict(schema='V5_CD_ZERO_INTEGRATION_PRODUCTION_RPC_CONTROL_COLD_CONTINUATION_V1',failed_request_id=CONTROL_FAILED,
        terminal_native_sha256=g[c.FIELD][CONTROL_FAILED]['terminal_native_sha256'],scope_grant_sha256=s.GRANT_SHA,
        maximum_continuations=1,new_PBS=0,new_models_queries_physics_valid_conditions=0,
        all_prior_rows_costs_unknowns_caps_and_clock_preserved=True,
        replacement_request_id=new_request_id,replacement_submission_nonce=new_nonce,
        replacement_run_uuid='v5-cd-c50-n294101-12',closed_leases=deepcopy(p['leases']),
        native_proof=deepcopy(p['native_proof']),effects=deepcopy(p['effects']),physical=deepcopy(p['physical']),
        closed_readback_sha256=a.sha(a.raw(p)),T_CD=clock['T_CD'],deadline_epoch=clock['deadline_epoch'],created_epoch=now_epoch)
    production_engineering(p['engineering']);before=g[a.PACKAGE]['engineering_replacements_used'];need(before<4,'ORIGINAL_ENGINEERING_POOL_NOT_EXHAUSTED')
    m.update(engineering=deepcopy(p['engineering']),engineering_replacements_used_before=before,engineering_replacements_used_after=before+1)
    v,w=deepcopy(g),deepcopy(l);restore=repair.allocation(g,'C50');restore['episode_starts']-=1;restore['engine_starts']-=1
    for val in(v,w):
        val[CONTROL_FIELD]=deepcopy(m)
        val[a.PACKAGE]['engineering_replacements_used']=before+1
        for k,x in restore.items():val[a.PACKAGE]['remaining_resources'][k]+=x
    for k in('H200_s','neural_queries','optimizer_updates'):v['reserved'][k]+=restore[k]
    v['CPU_reserved_s']+=restore['Hopper_CPU_s'];v['storage_reserved_bytes']+=restore['Hopper_storage_bytes']
    w['GPU_reserved_s']+=restore['H200_s']
    for k,site in [('Hopper_CPU_s','Hopper'),('Core_CPU_s','CoreLab'),('local_CPU_s','local_PC')]:w['CPU_reserved_s'][site]+=restore[k]
    v['revision']+=1;validate_production_control(v);c.checked(v);c.paired(v,w)
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    full=first512_forecast(a.raw(v),a.raw(w));need(not full['cap_changes_required'],'ENTIRE_FUTURE_MATRIX_FINAL:'+str(full['cap_changes_required']))
    need(all(v[k]==value for k,value in g.items()if k not in(CONTROL_FIELD,'revision',a.PACKAGE,'reserved','CPU_reserved_s','storage_reserved_bytes')),'ALL_OLD_COSTS_AND_ROWS_RETAINED')
    return dict(proposed_global=v,proposed_LOCAL=w,full_future_proof=full,new_scientific_admission=False,original_onceIO_required=True)
