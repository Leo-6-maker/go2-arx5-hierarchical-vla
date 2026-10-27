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
