"""Dependency-free final-plan and original-command consumption identities.

These sidecars never change the V6 dataset or its loss. A plan certificate is
not a task result;renewal ACKs and equal values from another query are not
consumption evidence.
"""
import hashlib
import json
import math

SCHEMA='SLA_FINAL_PLAN_DUAL_MASK_SIDECAR_V6_1'
PLAN_SHA='6e6636577d53b6b25a92b1e36cf2e619e765eab3'
DATA_SHA='29fd45ff31a2031f70f5d32402e6bfefe2f8b644e8e81ad1d2003d4537a1a6f5'
STATS_SHA='024951afc22e03586635f17b5ee8f473aead1960f107d8768384643844afb9e1'
STAGES=('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _trajectory(value):
    value=value.tolist()if hasattr(value,'tolist')else value
    if(len(value)!=40 or any(len(row)!=10 or any(type(x)not in(int,float)or not math.isfinite(x)for x in row)for row in value)):
        raise ValueError('FORTY_FINITE_PHYSICAL_EEF10_POINTS_REQUIRED')
    return [[float(x)for x in row]for row in value]


def final_plan(program,*,query_id,native_step,planner_sha256,adapter_sha256,normalizer_sha256,full_H40_valid):
    if(not isinstance(query_id,str)or not query_id or type(native_step)is not int or native_step<5703
            or any(not isinstance(x,str)or len(x)!=64 for x in(planner_sha256,adapter_sha256))
            or normalizer_sha256 is not None and(not isinstance(normalizer_sha256,str)or len(normalizer_sha256)!=64)
            or type(full_H40_valid)is not bool):raise ValueError('EXPLICIT_QUERY_CLOCK_AND_IMMUTABLE_CONTRACT_IDENTITIES')
    raw=_trajectory(program.decoded_H40);installed=_trajectory(program.reference_path)
    core=dict(schema=SCHEMA,query_id=query_id,native_step=native_step,final_H40=raw,
        installed_reference_H40=installed,planner_sha256=planner_sha256,adapter_sha256=adapter_sha256,
        normalizer_sha256=normalizer_sha256,command_period_ticks=25,physics_dt_s=.002,
        full_H40_valid=full_H40_valid)
    # Certification arrives after the immutable physical plan is recorded.
    # Its status cannot change the original query/command identities.
    pin=identity({k:v for k,v in core.items()if k!='full_H40_valid'})
    core.update(plan_sha256=pin,supervision_mask=[full_H40_valid]*40,
        command_ids=[identity(dict(plan_sha256=pin,origin_query_id=query_id,k=k,
            start_tick=native_step+25*k,end_tick=native_step+25*(k+1)))for k in range(40)],
        consumed_mask=[False]*40,task_success_claim=False)
    return core


def consumed_mask(plan,events):
    """Fold original per-atom confirmations;no future-query value matching."""
    mask=[False]*40
    for event in events:
        if event.get('kind')!='NATIVE_ATOM':continue
        k=event['k']
        if (type(k)is not int or not 0<=k<40 or event['origin_query_id']!=plan['query_id']
                or event['command_id']!=plan['command_ids'][k]
                or event['native_start_step']!=plan['native_step']+25*k
                or event['native_end_step']!=plan['native_step']+25*(k+1)):
            raise ValueError('ORIGINAL_QUERY_COMMAND_AND_PHYSICAL_INTERVAL_REQUIRED')
        if(event['actual_ticks']==25 and event['solver_policies_with_exact_installed_target']is True
                and event['gripper_command_consumed']is True):mask[k]=True
    return mask


def selection(manifest):
    """Fix stage and seam coverage without reading any model prediction."""
    if (manifest['dataset_id']!='T_PICK_HOLD_N20_ANCHOR_EQUIVALENT_CONSUMED_REFERENCE_V6'
            or manifest['statistics']['sha256']!=STATS_SHA or len(manifest['episodes'])!=20
            or manifest['supervised_windows']!=1220):raise ValueError('EXACT_V6_N20_POPULATION_AND_FROZEN_STATISTICS')
    episodes=sorted(manifest['episodes'],key=lambda e:(e['layout'],e['restart']))
    by_layout={}
    for e in episodes:by_layout.setdefault(e['layout'],e)
    varied=list(by_layout.values());chosen=[];used=set()
    def take(e,a,kind):
        key=(e['id'],a)
        if key in used:return False
        if not 0<=a<61:raise ValueError('COMPLETE_TRAIN_H40_WINDOW_REQUIRED')
        used.add(key);chosen.append(dict(id=e['id']+':'+str(a).zfill(2),episode=e['id'],
            layout=e['layout'],anchor=a,native_step=5703+250*a,
            stage=e['expert_stage_audit_only'][a],selection_kind=kind));return True
    for j,stage in enumerate(STAGES):
        for i in range(2):
            for shift in range(len(varied)):
                e=varied[(2*j+i+shift)%len(varied)]
                candidates=[a for a,s in enumerate(e['expert_stage_audit_only'])if s==stage]
                if any(take(e,a,'STAGE_'+stage)for a in candidates):break
            else:raise ValueError('MISSING_STAGE_'+stage)
    joins=[('PREGRASP','DESCEND'),('DESCEND','CLOSE'),('CLOSE','LIFT'),('LIFT','HOLD')]
    for j,(before,after)in enumerate(joins):
        found=False
        for shift in range(len(varied)):
            e=varied[(j+2+shift)%len(varied)];st=e['expert_stage_audit_only']
            for a in range(len(st)-1):
                if st[a]==before and st[a+1]==after and take(e,a,'JOIN_'+before+'_'+after):found=True;break
            if found:break
        if not found:raise ValueError('MISSING_DECLARED_REPLAN_JOIN')
    for e in varied:
        st=e['expert_stage_audit_only']
        if any(take(e,a,'GRIP_CLOSE_EVENT')for a,s in enumerate(st)if s=='CLOSE'):break
    else:raise ValueError('MISSING_DISTINCT_CLOSE_EVENT')
    for e in varied:
        if take(e,57,'POLICY_TO_RECORDED_TERMINAL_HOLD'):break
    if len(chosen)!=16 or len(used)!=16:raise ValueError('ALL_SIXTEEN_DISTINCT_WINDOWS_REQUIRED')
    rp=[dict(chosen[2*j],RP8_kind='STAGE_'+stage)for j,stage in enumerate(STAGES)]
    rp.extend(dict(chosen[j],RP8_kind=chosen[j]['selection_kind'])for j in(11,12,13))
    return dict(schema='SLA_FIXED_S16_RP8_INPUTS_V6_1',plan_sha=PLAN_SHA,data_manifest_sha256=DATA_SHA,
        statistics_sha256=STATS_SHA,S16=chosen,RP8=rp,inference_noise_ids=[0,1],
        fixed_before_S16_candidate_outputs=True,refit_statistics=False,
        selected_episodes=len({x['episode']for x in chosen}),selected_layouts=len({x['layout']for x in chosen}),
        model_loads=0,optimizer_updates=0,physical_requests=0,oracle_is_student=False)
