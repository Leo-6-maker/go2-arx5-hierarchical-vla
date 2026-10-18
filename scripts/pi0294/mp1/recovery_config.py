"""Fixed existing-model evaluation recovery, charged only to the adopted MP1."""
SCHEMA='MP1_E1_EVAL_RECOVERY_STAGE_V1'
MODE='MP1_FIXED_E1_EVAL_RECOVERY'
CAPS=dict(main_starts=16,recovery_forks=0,live_and_recovery_integrations=256000,
    shadow_integrations=256000,generation_queries=1280,optimizer_updates=0)
CPU_CAPS=dict(core=43200,hopper=28800)
GPU_CAP=21600
PBS_CAP=2
ADDITIONAL_PBS_AUTHORIZATION=dict(schema='MP1_E1_RECOVERY_ONE_ADDITIONAL_PBS_V1',
    card_sha256='d3c093f5fe4772d731e480e4e97ba6a129abbc4578c6a3d47147d90d073134ec')
CHECKPOINTS={'M2-BC':'edc0f390776056f9b22cd07d152c8fd5e273556f1d242f8e4f220b3034f9ffe6',
    'M2-CORR':'5275bb0c188a1d2148418d0cd7593dd712d2563e74b1907148563e5582a01f9a'}


def is_recovery(stage):return stage.get('schema')==SCHEMA


def family_PBS_cap(stage):
    value=stage.get('additional_PBS_authorization')
    if value is None:return PBS_CAP
    if not is_recovery(stage) or value!=ADDITIONAL_PBS_AUTHORIZATION:
        raise PermissionError('EXACT_HUMAN_ONE_ADDITIONAL_PBS_AUTHORIZATION')
    return 3


def prior_usage(stage):
    value=stage.get('prior_recovery_usage')
    revised=family_PBS_cap(stage)==3
    bounds=dict(GPU_s=GPU_CAP,PBS=family_PBS_cap(stage),**CAPS,CPU_core_s=CPU_CAPS['core'],CPU_hopper_s=CPU_CAPS['hopper'])
    if value is None:
        if revised:raise PermissionError('ADDITIONAL_PBS_REQUIRES_BOTH_SETTLED_PRIOR_ALLOCATIONS')
        return {k:0 for k in bounds}
    if set(value)!=set(bounds) or any(type(value[k]) is not int or not 0<=value[k]<=limit for k,limit in bounds.items()):
        raise PermissionError('EXACT_CUMULATIVE_RECOVERY_FAMILY_PRIOR_COSTS')
    if value['PBS']!=(2 if revised else 1) or value['optimizer_updates'] or value['recovery_forks']:
        raise PermissionError('ONLY_ONE_SETTLED_RECOVERY_ALLOCATION_CAN_CONTINUE')
    return value


def limits(stage,*,root=None):
    if stage.get('schema')=='MP1_SOURCE_CONFIRM_V1':
        if root is None:raise PermissionError('CONFIRMATION_REQUIRES_ITS_SEALED_SEPARATE_ROOT')
        from .confirmation_config import limits as confirmation_limits
        return confirmation_limits(root)
    if stage.get('schema')=='MP1_SAVED_M1_T2_SHADOW_DIAGNOSIS_V1':
        raise PermissionError('SAVED_T2_DIAGNOSTIC_HAS_NO_GPU_OR_PBS_AUTHORITY')
    if not is_recovery(stage):
        cap=4
        if root is not None:
            from .n1_limits import resolve
            n1=resolve(root)
            if n1 is not None:
                return dict(mode='MP1_FIXED_SOURCE_PAIR',GPU_s=n1['source']['GPU_s'],PBS=n1['source']['PBS'],
                    output_bytes=140*1024**3,job_wall_s_max=n1['source']['job_wall_s_max'])
            from .cold_boundary05 import effective_PBS_cap
            cap=effective_PBS_cap(root)
        return dict(mode='MP1_FIXED_SOURCE_PAIR',GPU_s=43200,PBS=cap,output_bytes=140*1024**3)
    prior=prior_usage(stage);remaining=GPU_CAP-prior['GPU_s']
    if family_PBS_cap(stage)==3:remaining=min(remaining,7200)
    return dict(mode=MODE,GPU_s=remaining,PBS=family_PBS_cap(stage)-prior['PBS'],output_bytes=2*1024**3)


def validate_checkpoint_bindings(bindings):
    if set(bindings)!=set(CHECKPOINTS):raise PermissionError('ONLY_TWO_FIXED_EXISTING_M2_CHECKPOINTS')
    for label,digest in CHECKPOINTS.items():
        v=bindings[label]
        if v['checkpoint_manifest_sha256']!=digest or v['global_step']!=512:
            raise PermissionError('ORIGINAL_M2_512_WEIGHTS_NO_TRAINING_OR_REPLACEMENT')


def plan(cards,prior):
    layouts={'PI0294/TRAIN/F0/000','PI0294/TRAIN/F1/000'}
    if len(cards)!=4 or {(c['scene_group'],c['task']) for c in cards}!={(s,t) for s in layouts for t in ['T1','T2']} or any(c['split']!='TRAIN' for c in cards):
        raise PermissionError('ONLY_ORIGINAL_TRAIN4_NO_DEV')
    panels={}
    for model in CHECKPOINTS:
        originals=[r for r in prior if r['model'] in [model,model+'-REMAINDER-03']]
        if any(r['pipeline_trustworthy'] or r['status']!='ENGINEERING_INCOMPLETE' or r['live_native_failures'] for r in originals):
            raise PermissionError('NO_REPLAY_VALID_OR_UNRESOLVED_SAFETY_OUTCOME')
        label='E1REC-'+model+'-512-TRAIN4';rows=[]
        for original in sorted(cards,key=lambda c:(c['scene_group'],c['task'])):
            c={k:v for k,v in original.items() if not k.startswith(('relay_request_','e1_','mp1_'))}
            c.update(id='PI0294-MP1-'+label+'-'+c['scene_group'].split('/')[-2]+'-000-'+c['task'],mp1_model_label=label)
            rows.append(c)
        panels[label]=dict(branch=model,checkpoint_step=512,split='TRAIN',cards=rows,teacher_takeover=False,
            original_S1=True,complete_task_duration_s=32,prior_E1_outcomes_preserved=True)
    return dict(schema='MP1_EXISTING_E1_M2_EVALUATION_RECOVERY_V1',panels=panels,regular_starts=8,
        engineering_replacement_per_condition_max=1,optimizer_updates=0,valid_policy_outcomes_repeated=False,
        original_action_seed_namespace='PI0294',new_DEV_labels=False,independent_confirmation_included=False)
