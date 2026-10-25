"""Cold C/D peers assembled from the original qualified Core asset template.

No transport, registry, reservation, model load or physics. The existing owner
must publish/seal sources, perform each native once operation and fresh admission.
The original template supplies assets/reference, never a candidate/lease/clock.
"""
import base64
from copy import deepcopy
import json,re
from pathlib import Path
from . import mix_accounting as a,mix_core_accounting as c,mix_core_context as context
from . import mix_core_identity as identity,mix_evaluation as evaluation,successor_cd as s
from .successor_core_context import SCHEMA,EVAL_ROLE
from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA


def need(ok,why):
    if not ok:raise PermissionError('V5_CD_CORE_ASSEMBLY_'+why)


def assemble(template,*,global_bytes,local_bytes,training_capture_bytes,arm,source_sha,
             source_seal_bytes,owner_root,owner_manifest_bytes,canonical_LOCAL_ledger,
             remote_binding,run_name,request_id,nonce,GPU_uuid,now_epoch):
    g,l=json.loads(global_bytes),json.loads(local_bytes);c.checked(g);c.paired(g,l)
    capture=json.loads(training_capture_bytes);proof=context.captured_endpoint(capture)
    marker=s.validated(g);key=c.key(arm)
    need(proof['arm']==arm and key in g[a.PACKAGE]['prepaid_control']
         and g[a.PACKAGE]['completed_stages']==c.before_stage(arm), 'REAL_CLOSED_SAME_ARM_AND_PAID_CONTROL')
    need(type(source_sha)is str and re.fullmatch('[0-9a-f]{40}',source_sha)
         and c.canonical_uuid(request_id)and c.canonical_uuid(nonce)
         and re.fullmatch(r'v5-cd-(c50|d25)-n294101-[0-9]{2}',run_name)
         and ('-c50-'if arm=='C50'else'-d25-')in run_name,'EXPLICIT_NEW_IDENTITY_NO_AB_ALIAS')
    base=deepcopy(template['rows']['A_CONTROL']);need(set(base)=={'core','hopper'},'ORIGINAL_TWO_PHYSICAL_ROLES')
    frozen=base['core']['frozen'];need(frozen==base['hopper']['frozen']and len(frozen)==15
         and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA,
         'ORIGINAL_FIFTEEN_NUMERICAL_ASSETS')
    seal=json.loads(source_seal_bytes)
    need(seal['source_sha']==source_sha and all(seal['files'].get(n)==pin for n,pin in frozen.items()),
         'REAL_SOURCE_SEAL_INCLUDES_ORIGINAL_OVERLAY_BYTES')
    for role,row in base.items():
        need(all(a.sha(base64.b64decode(value,validate=True))==row['manifest_template']['execution']['sealed_inputs'].get(n)
             for n,value in row['files'].items()), 'ORIGINAL_TEMPLATE_COMPLETE_SMALL_BYTES')
    parent=Path(base['core']['root']).parent.parent
    roots={role:str(parent/('children'if role=='core'else'neural-children')/run_name)for role in base}
    om=json.loads(owner_manifest_bytes)
    need(Path(owner_root).parent==parent/'control'and om['scope']==identity.OWNER_ROLE
         and om['output_root']==owner_root and om['code']['vla_commit']==source_sha
         and om['authority']['protocol_ref']==s.TASK_SOURCE
         and om['authority']['scope_grant_sha256']==s.GRANT_SHA
         and om['limits']['T_CD']==marker['T_CD']and om['limits']['operation_deadline_epoch']==marker['deadline_epoch'],
         'ACTUAL_NEW_OWNER_CUSTODY_WITH_ADOPTED_WINDOW')
    local_binding=dict(canonical_LOCAL_ledger=canonical_LOCAL_ledger,owner_root=owner_root,
        owner_manifest_sha256=a.sha(owner_manifest_bytes))
    need(Path(canonical_LOCAL_ledger).is_relative_to(parent/'control')and Path(canonical_LOCAL_ledger).name=='LOCAL_RESOURCE_LEDGER.private.json',
         'SAME_CANONICAL_LOCAL_NO_COPIED_BUDGET')
    rows={};population=None
    for role,old in base.items():
        root=Path(roots[role]);m=deepcopy(old['manifest_template']);b=deepcopy(old['binding'])
        # Original model/DEV history/old queue and authority cards cannot grant
        # the new stage. Keep only the exact assets, data and raw reference.
        keep={'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json','plans/INITIAL_CORR_PARAMETER_INVENTORY.json',
            'plans/FROZEN_SOURCE_PAIR_RECIPE.json','plans/PV1_TRANSPORT_FIXTURE.json',
            'reference/RECORDED_OBSERVATION.json','reference/RAW_REFERENCE_BINDING.private.json',
            'reference/REFERENCE_PROOF_FROM_CORE.json','evidence/LIVE_READINESS.json',
            'plans/PV1_HF_BANK_DELIVERY.json','INPUT_BINDINGS.private.json','plans/ASSET_IDENTITY.private.json',
            'evidence/COUNTER_NATIVE_QUALIFICATION.json','plans/PV1_TRAIN_INPUTS.private.json','NEURAL_BINDINGS.private.json'}
        files={n:base64.b64decode(v,validate=True)for n,v in old['files'].items()if n in keep or n.startswith('inference_dataset/')}
        population=files['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json']
        from .dataset import POPULATION
        need(a.sha(population)==POPULATION,'EXACT_ORIGINAL_POPULATION_BYTES')
        groups=json.loads(population)['groups']
        files['V5_CD_SCOPE_BINDING.json']=a.raw(marker)
        files[context.CAPTURE]=training_capture_bytes;files[context.LOCAL_BINDING]=a.raw(local_binding)
        files['plans/SOURCE_EVALUATION_PLAN.json']=a.raw(evaluation.candidate_plan(groups,arm))
        files['plans/V5_CD_PAIRED_EVALUATION_PLAN.json']=a.raw(evaluation.plan(groups))
        files['PV1_STAGE_BINDING.json']=a.raw(evaluation.stage(arm));files['plans/PRIOR_EVALUATION_OUTCOMES.json']=a.raw([])
        fixture=json.loads(files['plans/PV1_TRANSPORT_FIXTURE.json']);fixture['run_uuid']=run_name
        files['plans/PV1_TRANSPORT_FIXTURE.json']=a.raw(fixture)
        files['plans/EVALUATION_CHECKPOINT.private.json']=a.raw(dict(candidate=arm,global_step=1024,
            checkpoint=str(Path(roots['hopper'])/'model_views'/arm),checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],full_resume_files_present=False))
        if role=='hopper':
            neural=json.loads(files['NEURAL_BINDINGS.private.json']);neural['dataset']=str(root/'inference_dataset/ORIGINAL_CORR')
            files['NEURAL_BINDINGS.private.json']=a.raw(neural)
            rb=deepcopy(remote_binding);rb['physical_peer_root']=roots['core']
            need(rb['remote_seal_sha256']==a.sha(source_seal_bytes)and rb['remote_user']=='e1192548'
                 and rb['remote_host']=='vanda.nus.edu.sg','EXACT_ACTUAL_REMOTE_SOURCE_FOR_ORIGINAL_GLOBAL_RPC')
            files['plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json']=a.raw(rb)
        b.update(run_uuid=run_name,role=(EVAL_ROLE if role=='core'else c.ROLE)+'__'+run_name,
            run_root=str(root),registered_parent=str(root.parent),lease=str(Path(b['lease']).parent/(parent.name+'-'+run_name+('-physics'if role=='core'else'-neural')+'.json')))
        m.update(schema=SCHEMA if role=='core'else'PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',
            scope=EVAL_ROLE if role=='core'else c.ROLE,candidate=arm,run_uuid=run_name,output_root=str(root),
            owner='EXISTING_LOCAL_PC',owner_user='liuyu-nfs',owner_uid=2009)
        if role=='hopper':m['platform_evaluation_version']=c.VERSION
        m['authority']=dict(protocol_ref=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA)
        m['code'].update(vla_commit=source_sha,capsule_relative='capsules/'+source_sha+'/code',
            source_seal_relative='capsules/'+source_sha+'/SOURCE_SEAL.json',source_seal_sha256=a.sha(source_seal_bytes))
        m['execution'].update(private_path_role=b['role'],sealed_inputs={n:a.sha(v)for n,v in files.items()})
        m['limits'].pop('PV1_T0_epoch',None);m['limits'].pop('PV1_hard_close_epoch',None)
        m['limits'].update(T_CD=marker['T_CD'],operation_deadline_epoch=marker['deadline_epoch'])
        if role=='core':
            m['execution'].pop('output_binding_sha256',None)
            m['limits'].update(CPU_s=93900,engine_starts=2108,physics_ticks=832000,policy_episodes=26,
                output_bytes=c.remaining_stage()['Core_storage_bytes'])
        else:m['execution']['output_binding_sha256']=a.sha(a.raw(b))
        mb=a.raw(m)
        if role=='core':b['run_manifest_sha256']=a.sha(mb)
        rows[role]=dict(root=str(root),source_sha=source_sha,manifest=m,manifest_bytes_base64=base64.b64encode(mb).decode(),
            manifest_sha256=a.sha(mb),binding=b,files={n:base64.b64encode(v).decode()for n,v in files.items()},
            frozen=deepcopy(frozen),fixture_plan=fixture,plan_sha256=a.sha(files['plans/PV1_TRANSPORT_FIXTURE.json']))
    record=dict(schema='V5_CD_CORE_NATIVE_GPU_ALLOCATION_V1',campaign_run_uuid=parent.name,
        native_platform='CORELAB_RTX4080',platform_evaluation_version=c.VERSION,owner_user='liuyu-nfs',
        authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,stage='V5_MATCHED_DEV24',noise_bank=294101,
        remaining_conditions=24,checkpoint_step=1024,statistics_sha256=a.STATS,parent_checkpoint_manifest_sha256=a.PARENT,
        owner_uid=2009,hostname='corelab-4080s',physical_GPU_count=1,physical_GPU_kind='RTX_4080_SUPER',walltime_s=28800,
        accounting_pad_s=60,reserved_GPU_s=c.GPU_UPPER,CPU_upper_s=c.CPU_UPPER,declared_process_CPU_limit_s=79760,
        active_cpu_limit=4,minimum_GPU_memory_bytes=30*1024**3,requested_memory_bytes=48*1024**3,optimizer_update_limit=0,
        reserved_optimizer_updates=0,PBS_submissions_consumed=0,query_limit=2097,charges_DEC1_subcaps=False,new_robot_physics=0,
        legacy_GPU_seconds_field_is_projection=True,additional_output_upper_bytes=c.remaining_stage()['Hopper_storage_bytes'],
        evaluation_candidate=arm,source_sha=source_sha,request_id=request_id,submission_nonce=nonce,
        run_manifest_sha256=rows['hopper']['manifest_sha256'],checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],
        training_handoff_sha256=a.sha(a.raw(proof)),planned_panel_sha256=rows['hopper']['manifest']['execution']['sealed_inputs']['plans/SOURCE_EVALUATION_PLAN.json'],
        cold_input_audit_sha256=a.sha(a.raw(dict(source_sha=source_sha,source_seal_sha256=a.sha(source_seal_bytes),manifest_sha256={k:v['manifest_sha256']for k,v in rows.items()}))),
        T_CD=marker['T_CD'],deadline_epoch=marker['deadline_epoch'],created_epoch=now_epoch,child_run_uuid=run_name,
        native_handle=None,GPU_uuid=GPU_uuid,status=c.STATES[0],reserved_neural_queries=2097)
    request={k:v for k,v in record.items()if k not in ('status','native_handle','child_run_uuid','reserved_neural_queries')}
    request.update(schema='V5_CD_CORE_NATIVE_GPU_REQUEST_V1',task=s.TASK,run_uuid=run_name,
        allocation_number=1 if arm=='C50'else 2,native_unit=run_name+'-neural.service',optimizer_updates=0,
        Teacher_takeover=False,normalizer_refit=False,evaluation_data_for_training=False,LOCKBOX_opened=False,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1')
    request_sha=a.sha(a.raw(request));record['request_sha256']=request_sha
    c.validate_row(record)
    identity.request_identity(rows['hopper']['manifest'],request,rows['hopper']['manifest_sha256'])
    return dict(schema='V5_CD_COLD_CORE_PAIRED_PEERS_ASSEMBLED_NOT_REGISTERED_V1',rows=rows,record=record,
        request=request,request_sha256=request_sha,population_base64=base64.b64encode(population).decode(),
        owner_manifest_sha256=a.sha(owner_manifest_bytes),native_registration=False,model_or_physics_admission=False)
