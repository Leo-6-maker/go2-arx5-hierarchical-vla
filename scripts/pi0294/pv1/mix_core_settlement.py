"""C/D closeout through original native terminal checks and existing ledgers.

Pure byte proposals only. The resident owner supplies real process/unit exit,
physical-counter bytes, raw independent history and actual dual closed leases.
Metadata/HF IO is inside the original stage, never a second delivery invoice.
"""
from copy import deepcopy
import json,re
from . import mix_accounting as a,mix_core_accounting as c,successor_cd as s
from .evaluation_core_accounting import terminal_values
from .successor_core_context import LEDGER_SCHEMA

FIELD='V5_CD_actual_Core_stage_projection'


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_CORE_CLOSE_'+why)


def physical(physical_bytes,row):
    need(type(physical_bytes)is bytes,'ACTUAL_PHYSICAL_COUNTER_BYTES')
    p=json.loads(physical_bytes)
    need(p.get('schema')==LEDGER_SCHEMA and p.get('run_uuid')==row['child_run_uuid']
         and p.get('authority_release')==s.TASK_SOURCE and p.get('unknown_effects')==[],
         'SAME_REAL_PHYSICAL_PEER_NOT_OLD_AB')
    amounts={name:a.number(p.get(key),integer=name!='physical_CPU_s')for name,key in
        [('physical_CPU_s','Core_CPU_s'),('episodes','main_starts'),('constructors','engine_starts'),
         ('live','live_and_recovery_integrations'),('shadow','shadow_integrations')]}
    need(amounts['physical_CPU_s']<=93900 and amounts['episodes']<=26
         and amounts['constructors']<=2108 and amounts['live']+amounts['shadow']<=832000
         and type(p.get('reservations'))is dict,'ORIGINAL_COMPLETE_PHYSICAL_ENVELOPE')
    return p,amounts


def sync_LOCAL(g,l,*,request_id,physical_bytes):
    """Mirror paid actual neural and physical costs before a fresh full proof.

    GLOBAL neural effects already paid through its original RPC. LOCAL mirrors
    them once; physical reserved full-stage uppers include current child holds.
    Caller must CAS the one canonical LOCAL and reread GLOBAL after this proposal.
    """
    a.validate_global(g);c.validate_row(g[c.FIELD][request_id]);r=g[c.FIELD][request_id]
    need(l[a.PACKAGE]==g[a.PACKAGE] and s.validated(l)==s.validated(g), 'EXACT_ADOPTED_PACKAGE')
    oldrow=l.get(c.FIELD,{}).get(request_id);need(type(oldrow)is dict,'ACTUAL_LOCAL_BOUND_STAGE')
    # A GLOBAL pre-effect charge can advance these fields before LOCAL mirrors.
    mutable={'status','charged_effects','charged_CPU_s','charged_query_attempts','charged_model_loads','reserved_neural_queries'}
    need(all(oldrow.get(k)==v for k,v in r.items()if k not in mutable)
         and all(k in r for k in oldrow),'NO_IDENTITY_OR_TERMINAL_REPLACEMENT')
    need(r['status']in c.STATES[1:3], 'ONLY_REAL_BOUND_ACTIVE_STAGE')
    p,amounts=physical(physical_bytes,r);q,loads,cpu=c.progress(r)
    amounts.update(neural_CPU_s=cpu,queries=q,model_loads=loads)
    previous=l.get(FIELD,{}).get(request_id,{k:0 for k in amounts})
    need(set(previous)==set(amounts)and all(amounts[k]>=previous[k]for k in amounts),
         'NO_COST_COUNTER_ROLLBACK_OR_SECOND_DEBIT')
    v=deepcopy(l);delta={k:amounts[k]-previous[k]for k in amounts}
    for site,cost in [('Hopper',delta['neural_CPU_s']),('CoreLab',delta['neural_CPU_s']+delta['physical_CPU_s'])]:
        need(v['CPU_reserved_s'][site]>=cost,'ACTUAL_CPU_ESCROW:'+site)
        v['CPU_reserved_s'][site]-=cost;v['CPU_s'][site]+=cost
    v['neural_queries']+=delta['queries'];v['episode_starts']+=delta['episodes'];v['engine_starts']+=delta['constructors']
    v['physics_ticks']['live']+=delta['live'];v['physics_ticks']['shadow']+=delta['shadow']
    held=v['Core_native_GPU_reserved']
    for name,cost in [('episode_starts',delta['episodes']),('constructors',delta['constructors']),('physics_ticks',delta['live']+delta['shadow'])]:
        need(held[name]>=cost,'ORIGINAL_PHYSICAL_HOLD:'+name);held[name]-=cost
    v[c.FIELD][request_id]=deepcopy(r);v.setdefault(FIELD,{})[request_id]=amounts
    v.setdefault('V5_CD_current_Core_pending',{})[request_id]=deepcopy(p['reservations'])
    c.paired(g,v)
    return v


def propose(global_bytes,local_bytes,*,request_id,native_proof,effects,physical_bytes,
            delivery_receipt_bytes,now_epoch):
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g)
    before=g[c.FIELD][request_id];need(before['status']in c.STATES[1:3],'ONCE_KNOWN_BOUND_TERMINAL')
    # Synchronize actual costs, not a caller's claimed zero pending snapshot.
    local=sync_LOCAL(g,l,request_id=request_id,physical_bytes=physical_bytes)
    p,phys=physical(physical_bytes,before)
    need(p['reservations']=={},'ALL_REAL_PHYSICAL_RESERVATIONS_TERMINAL')
    amounts=terminal_values(before,request_id=request_id,native_proof=native_proof,effects=effects,now_epoch=now_epoch)
    need(not amounts['unknown'],'UNKNOWN_QUERY_EFFECTS_REQUIRE_INDEPENDENT_RECONCILIATION_NO_SUCCESS')
    delivery=json.loads(delivery_receipt_bytes);arm=before['evaluation_candidate']
    need(delivery.get('status')=='ACTUAL_CLOSED_'+arm+'_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED'
         and delivery.get('checkpoint_manifest_sha256')==before['checkpoint_manifest_sha256']
         and delivery.get('full_resume_files_present')is False
         and delivery.get('new_GPU_model_queries_physics')==0
         and type(delivery.get('HF_revision'))is str and re.fullmatch('[0-9a-f]{40}',delivery['HF_revision'])
         and len([n for n in delivery.get('files',{})if n.endswith('.safetensors')])==3,
         'ACTUAL_FIXED_REVISION_DELIVERY_NO_SECOND_INVOICE')
    v=deepcopy(g);row=v[c.FIELD][request_id];gpu,cpu,q,loads=(amounts[k]for k in ('gpu','cpu','queries','loads'))
    oldq,oldloads,oldcpu=c.progress(before)
    need(q==oldq and loads==oldloads,'EVERY_MODEL_QUERY_PAID_BEFORE_EFFECT')
    v['used']['H200_s']+=gpu;v['CPU_s']+=cpu-oldcpu;v['CPU_reserved_s']-=c.CPU_UPPER-oldcpu
    v['reserved']['H200_s']-=c.GPU_UPPER;v['reserved']['neural_queries']-=before['reserved_neural_queries']
    from . import mix_core_premodel_repair as repair
    covered=repair.output_credit(g,before)
    v['storage_reserved_bytes']-=before['additional_output_upper_bytes']-covered
    v['new_storage_current_bytes']+=max(0,amounts['output']-covered);v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    row.update(status=c.STATES[3],native_exit_status=native_proof['native_exit_status'],
        conservative_GPU_s=gpu,conservative_CPU_s=cpu,query_attempts=q,model_load_attempts=loads,
        unknown_query_intents=[],reserved_neural_queries=0,terminal_native_sha256=c.digest(native_proof),
        terminal_effect_summary_sha256=c.digest(effects),settled_epoch=now_epoch)
    need(request_id not in local.get('V5_CD_Core_terminal_settlements',{}),'ONE_NATIVE_FINANCIAL_TERMINAL')
    local['GPU_reserved_s']-=c.GPU_UPPER;local['GPU_allocated_s']+=gpu
    local['CPU_reserved_s']['Hopper']-=c.CPU_UPPER-oldcpu;local['CPU_s']['Hopper']+=cpu-oldcpu
    local['CPU_reserved_s']['CoreLab']-=c.CPU_UPPER-oldcpu+93900-phys['physical_CPU_s'];local['CPU_s']['CoreLab']+=cpu-oldcpu
    allocation=repair.allocation(g,arm);held=local['Core_native_GPU_reserved']
    for name,unused in [('episode_starts',26-phys['episodes']),('constructors',2108-phys['constructors']),('physics_ticks',832000-phys['live']-phys['shadow']),('CoreLab_output_bytes',allocation['Core_storage_bytes']),('transfer_bytes',allocation['transfer_bytes'])]:
        need(held[name]>=unused,'ENTIRE_ORIGINAL_REMAINING_ESCROW:'+name);held[name]-=unused
    # Retain the original conservative stage IO/storage charge; actual delivered
    # byte/CPU witnesses stay explicit and never masquerade as that upper.
    local['network_transfer_accounted_bytes']+=allocation['transfer_bytes']
    local['Core_native_GPU_current_storage_bytes']=local.get('Core_native_GPU_current_storage_bytes',0)+allocation['Core_storage_bytes']
    local[c.FIELD][request_id]=deepcopy(row)
    local['V5_CD_Core_CPU_substages'][c.key(arm)].update(status='SETTLED_KNOWN_NATIVE_TERMINAL')
    receipt=dict(schema='V5_CD_CORE_KNOWN_NATIVE_DUAL_FINANCIAL_TERMINAL_V1',request_id=request_id,
        source_sha=before['source_sha'],checkpoint_manifest_sha256=before['checkpoint_manifest_sha256'],
        native_handle=before['native_handle'],GPU_s=gpu,neural_CPU_s=cpu,physical_CPU_s=phys['physical_CPU_s'],
        query_attempts=q,model_load_attempts=loads,physical=phys,
        Core_output_charged_upper_bytes=allocation['Core_storage_bytes'],transfer_charged_upper_bytes=allocation['transfer_bytes'],
        original_gross_GPU_output_upper_bytes=before['additional_output_upper_bytes'],
        same_panel_GPU_output_upper_already_paid=covered,
        original_gross_Core_output_upper_bytes=c.remaining_stage()['Core_storage_bytes'],
        original_gross_transfer_upper_bytes=c.remaining_stage()['transfer_bytes'],
        actual_model_delivery_receipt_sha256=a.sha(delivery_receipt_bytes),delivery_GPU_site_CPU_already_in_7200s_control=True,
        physical_counter_sha256=a.sha(physical_bytes),terminal_native_sha256=c.digest(native_proof),
        original_GLOBAL_sha256=a.sha(global_bytes),original_LOCAL_sha256=a.sha(local_bytes),observed_epoch=now_epoch,
        stage_completed=False,leases_closed=False,new_scientific_admission=False)
    local.setdefault('V5_CD_Core_terminal_settlements',{})[request_id]=receipt
    c.checked(v);c.paired(v,local)
    return dict(proposed_global=v,proposed_LOCAL=local,receipt=receipt)


def complete_stage(global_bytes,local_bytes,*,request_id,population_bytes,evidence,
                   GPU_lease_bytes,physical_lease_bytes,now_epoch):
    """Only independently re-audited original24 plus both real closures advance.

    Normal policy/S1 failures are valid results. No success-rate threshold,
    replacement seed, data change or new training permission is introduced.
    """
    from .baseline import partial_history
    from .dataset import POPULATION
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    r=g[c.FIELD][request_id];arm=r['evaluation_candidate'];key=c.key(arm)
    need(r['status']==c.STATES[3]and r['native_exit_status']==0
         and r['unknown_query_intents']==[]and l['unknown_effects']==[]
         and g[a.PACKAGE]['completed_stages']==c.before_stage(arm,g), 'TRUE_SUCCESSFUL_NATIVE_CLOSE_BOUNDARY')
    settlement=l.get('V5_CD_Core_terminal_settlements',{}).get(request_id)
    need(type(settlement)is dict and settlement['terminal_native_sha256']==r['terminal_native_sha256']
         and settlement['checkpoint_manifest_sha256']==r['checkpoint_manifest_sha256'], 'ACTUAL_ONCE_DUAL_FINANCIAL_TERMINAL')
    need(a.sha(population_bytes)==POPULATION,'ORIGINAL_UNSCREENED_PANEL_POPULATION')
    history=partial_history(population_bytes,candidate=arm,evidence=evidence,
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'])
    need(history['valid_conditions']==24,'ALL_TWENTY_FOUR_VALID_OUTCOMES_NOT_PARTIAL')
    for raw in (GPU_lease_bytes,physical_lease_bytes):
        need(type(raw)is bytes,'REAL_CLOSED_LEASE_BYTES')
        lease=json.loads(raw)
        need(lease.get('status')=='CLOSED'and lease.get('run_uuid')==r['child_run_uuid']
             and lease.get('authority_release')==s.TASK_SOURCE
             and lease.get('terminal_native_sha256')==r['terminal_native_sha256']
             and lease.get('all_actual_workers_terminal')is True
             and type(lease.get('closed_epoch'))in(int,float)
             and r['settled_epoch']<=lease['closed_epoch']<=now_epoch,
             'OWN_REAL_DUAL_CLOSURE_AFTER_FINANCIAL_TERMINAL')
    proof=dict(schema='V5_CD_COMPLETE_ORIGINAL_DEV24_AUDITED_DUAL_CLOSED_V1',request_id=request_id,
        candidate=arm,checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],
        terminal_native_sha256=r['terminal_native_sha256'],history=history,
        GPU_lease_sha256=a.sha(GPU_lease_bytes),physical_lease_sha256=a.sha(physical_lease_bytes),
        all_valid_failures_preserved=True,observed_epoch=now_epoch)
    v,w=deepcopy(g),deepcopy(l)
    for value in (v,w):
        package=value[a.PACKAGE];package['completed_stages'].append(key)
        package.setdefault('closed_Eval_stages',{})[key]=deepcopy(proof)
    c.checked(v);c.paired(v,w)
    return dict(proposed_global=v,proposed_LOCAL=w,receipt=proof)
