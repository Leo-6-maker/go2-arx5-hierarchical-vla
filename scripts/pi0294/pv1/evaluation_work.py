"""Derive V4's remaining regular DEV work from the original byte-bound history.

This is the work/cost input to fresh admission, not a budget grant or a model
release. It never calls the legacy complete-B0 training forecast. The native
owner still must enumerate all historical attempts, collect current costs,
preserve every hold and check all original clocks and resource dimensions.
"""
from scripts.pi0294.dec1.authority import BANK
from .baseline import partial_history
from .evaluation_accounting import output_upper
from .evaluation_platform import ENDPOINTS
from .future_budget import CAPS,FINAL,_vector

GIB=1024**3
MATRIX=_vector(dict(H200_s=86580,PBS_submissions=3,optimizer_updates=0,
    episode_starts=88,neural_queries=7040,physics_ticks=2816000,engine_starts=7128,
    Core_CPU_s=193968.05627,Hopper_CPU_s=85520,local_CPU_s=6600,
    transfer_bytes=6174015488,Hopper_storage_bytes=9442971836,
    Core_storage_bytes=17968398336,local_storage_bytes=0))


def remaining(population_bytes,*,evidence_by_candidate):
    if type(evidence_by_candidate) is not dict or set(evidence_by_candidate)!=set(ENDPOINTS):
        raise PermissionError('PV1_V4_ALL_THREE_ORIGINAL_CANDIDATES_EXPLICIT_HISTORY')
    histories={};jobs=[]
    for candidate in ENDPOINTS:
        evidence=evidence_by_candidate[candidate]
        if type(evidence) not in (list,tuple):
            raise PermissionError('PV1_V4_ORIGINAL_ATTEMPT_BYTE_RECORDS_REQUIRED')
        history=partial_history(population_bytes,candidate=candidate,evidence=evidence)
        histories[candidate]=history
        for label,cards in history['remaining_panels'].items():
            if not cards:continue
            noise={c['noise_bank'] for c in cards}
            if len(noise)!=1:raise PermissionError('PV1_V4_ONE_ORIGINAL_NOISE_PANEL_PER_COLD_ALLOCATION')
            n=len(cards)
            jobs.append(dict(work_id='DEV:'+candidate+':N'+str(next(iter(noise))),candidate=candidate,
                native_platform='VANDA_A40',
                original_panel_label=label,noise_bank=next(iter(noise)),
                condition_ids=[c['condition_id'] for c in cards],attempt_ids=[c['id'] for c in cards],
                conditions=n,walltime_s=28800,accounting_pad_s=60,process_CPU_envelope_s=80000,
                GPU_site_output_upper_bytes=output_upper(n),model_loads_upper=2,
                TRAIN_compatibility_queries_upper=17,repeat_original_fixed_probes_authorized=False))
    cases=sum(j['conditions'] for j in jobs);count=len(jobs);prep=600 if count else 0
    resources=_vector(dict(H200_s=count*28860,PBS_submissions=count,optimizer_updates=0,
        episode_starts=cases,neural_queries=80*cases+17*count,physics_ticks=32000*cases,
        engine_starts=81*cases+2*count,Core_CPU_s=3600*cases+7200*count+prep,
        Hopper_CPU_s=80000*count+7200*count+prep,local_CPU_s=7200*count+prep,
        transfer_bytes=sum(j['GPU_site_output_upper_bytes'] for j in jobs)+(GIB if count else 0),
        Hopper_storage_bytes=sum(j['GPU_site_output_upper_bytes'] for j in jobs),
        Core_storage_bytes=cases*768*1024**2+count*512*1024**2,local_storage_bytes=count*64*1024**2))
    return dict(schema='PI0294_V4_PRIMARY_DEV_REMAINING_REGULAR_WORK_V1',
        status='BYTE_VALIDATED_WORK_UPPER_NOT_FRESH_NATIVE_ADMISSION',new_bank_manifest_sha256=BANK,
        histories=histories,jobs=jobs,resources=resources,
        original_matrix_protection=dict(MATRIX),original_final_protection=_vector(FINAL),
        all_valid_outcomes=sum(h['valid_conditions'] for h in histories.values()),
        exhausted_engineering_conditions=sum(h['exhausted_engineering_conditions'] for h in histories.values()),
        complete_baseline_required_for_this_independent_work=False,
        original_training_and_publication_gates_unchanged=True,
        legacy_resource_names_are_accounting_projections_not_physical_H200=True,
        resource_caps_changed=False,new_engineering_attempt_pool_authorized=False,
        eight_hour_panels_not_a_completion_or_throughput_guarantee=True,
        remote_inventory_completeness_and_fresh_costs_still_required=True,
        native_or_resource_admission=False,new_integrations=0,new_model_queries=0,new_training_updates=0)


def diagnostic(work,*,used,reserved,caps):
    """Arithmetic only; caps are an explicit scenario, never adopted by this call."""
    if (work.get('schema')!='PI0294_V4_PRIMARY_DEV_REMAINING_REGULAR_WORK_V1'
            or set(caps)!=set(CAPS) or work.get('native_or_resource_admission') is not False
            or work.get('original_matrix_protection')!=MATRIX
            or work.get('original_final_protection')!=_vector(FINAL)):
        raise PermissionError('PV1_V4_COMPLETE_SCENARIO_AND_ORIGINAL_PROTECTION')
    used,reserved,caps=_vector(used),_vector(reserved),_vector(caps)
    future=_vector(work['resources']);final=_vector(FINAL)
    rows={}
    for key in CAPS:
        total=used[key]+reserved[key]+future[key]+MATRIX[key]+final[key]
        rows[key]=dict(cap=caps[key],used=used[key],reserved=reserved[key],S2_upper=future[key],
            matrix=MATRIX[key],final=final[key],projected=total,shortfall=max(0,total-caps[key]))
    return dict(schema='PI0294_V4_PURE_RESOURCE_SCENARIO_V1',dimensions=rows,
        shortfalls={k:r['shortfall'] for k,r in rows.items() if r['shortfall']},
        mathematical_fit=all(r['shortfall']==0 for r in rows.values()),
        caps_are_only_scenario_not_authority=True,active_work_credit_not_applied=True,
        native_or_resource_admission=False)
