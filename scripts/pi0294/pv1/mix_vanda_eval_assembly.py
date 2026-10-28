"""Pure same-PBS assembly of the original C/D source evaluation peers.

No native filesystem, registry, reservation, process or model operation lives
here. Registered owners must separately perform every actual once operation.
"""
import base64
from copy import deepcopy
import json
from pathlib import Path
import re

from . import mix_accounting as a,mix_vanda_cohost as cohost,successor_cd as s
from . import mix_vanda_eval_identity as identity,mix_vanda_eval_context as context
from . import mix_core_accounting as core_accounting
from . import mix_vanda_eval_accounting as accounting,mix_vanda_reference as reference
from . import mix_evaluation as evaluation,mix_vanda_codec as codec
from .evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
from . import evaluation_account_jobs as outside


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_ASSEMBLY_'+why)


def assemble(template,*,global_bytes,historical_LOCAL_bytes,training_capture_bytes,reference_plan,
             source_sha,source_seal_bytes,parent,registry,environment_registry,profiles,
             native_neural_bindings,native_training_inputs,asset_plan,codec_toolchain,codec_preparation_root,codec_closed_lease_bytes,
             queue_limit_bytes,independent_job_bytes,owner_code,programs,
             request_id,nonce,run_name,now_epoch,cpu_slot_policy=None):
    """Build immutable bytes without request/owner-manifest hash cycles."""
    from .mix_vanda_affinity import POLICY
    need(cpu_slot_policy in (None,POLICY),'EXPLICIT_CPU_SLOT_POLICY')
    g=json.loads(global_bytes);accounting.checked(g);marker=cohost.validated(g)
    need(marker is not None and a.sha(historical_LOCAL_bytes)==cohost.first.LOCAL_SHA,
         'EXACT_ADOPTED_SITE_AND_HISTORICAL_LOCAL_NOT_FRESH')
    cap=json.loads(training_capture_bytes);proof=cohost.endpoint(global_bytes,cap);arm=proof['arm']
    need(arm in ('C50','D25') and reference_plan['arm']==arm
         and reference_plan['checkpoint_manifest_sha256']==proof['checkpoint_manifest_sha256']
         and cap['global_base64']==base64.b64encode(global_bytes).decode()
         and cap['handoff']==proof,'FRESH_CLOSED_SAME_MODEL_AND_REAL_GLOBAL')
    need(re.fullmatch('[0-9a-f]{40}',source_sha) and core_accounting.canonical_uuid(request_id)
         and core_accounting.canonical_uuid(nonce),'EXPLICIT_SOURCE_AND_UNIQUE_REQUEST')
    need(run_name.startswith('v5-cd-'+arm.lower()+'-n294101-') and len(run_name)<=128,'EXACT_ORIGINAL_PANEL')
    parent=Path(parent);need(parent.is_absolute()and parent.name=='20261018T182956Z-localpc-pv1','EXISTING_VANDA_CAMPAIGN')
    roots=dict(core=str(parent/'children'/run_name),hopper=str(parent/'neural-children'/run_name),
               owner=str(parent/'control'/(run_name+'-owner')))
    peers={role:dict(root=root,role={'core':identity.PHYSICAL_ROLE,'hopper':identity.NEURAL_ROLE,'owner':identity.OWNER_ROLE}[role])for role,root in roots.items()}
    base=deepcopy(template['rows']['A_CONTROL']);need(set(base)=={'core','hopper'},'EXISTING_ORIGINAL_PEERS')
    frozen=base['core']['frozen'];seal=json.loads(source_seal_bytes)
    need(frozen==base['hopper']['frozen']and len(frozen)==15
         and a.sha(json.dumps(frozen,sort_keys=True).encode())==ORIGINAL_NUMERICAL_TABLE_SHA
         and seal['source_sha']==source_sha and all(seal['files'].get(n)==h for n,h in frozen.items()),
         'ORIGINAL_FIFTEEN_NUMERICAL_BYTES_WITH_EXPLICIT_OVERLAY')
    need(set(profiles)=={'core','hopper'} and all(profiles[role]['source_release']==source_sha
         and profiles[role]['PBS_required']is True for role in profiles),'EXPLICIT_SITE_ENVIRONMENT_PROFILES')
    need(len(asset_plan['files'])==26 and a.sha(json.dumps(asset_plan['files'],sort_keys=True,separators=(',',':')).encode())==context.ASSET_TABLE_SHA,
         'ALL_ORIGINAL_NATIVE_ASSETS')
    need(codec_toolchain['schema']==codec.SCHEMA and codec_toolchain['parent_LD_LIBRARY_PATH_modified']is False
         and codec_toolchain['physics_SIF_sha256']=='2cae97d18c6a41450bb102e3bd54819d0c30552605d222d73ffc387c6ec14c8c',
         'ACTUAL_ORIGINAL_SIF_AND_SEPARATE_AUDIT_CODEC')
    codec_root=Path(codec_preparation_root);closed=json.loads(codec_closed_lease_bytes)
    need(codec_root.parent==parent/'control' and codec_root.name=='v5-cd-c50-vanda-codec-cold-01'
         and closed['status']=='CLOSED' and closed['run_uuid']==codec_root.name
         and closed['all_actual_workers_terminal']is True and closed['tools_retained_as_readonly_future_inputs']is True
         and closed['limits']==dict(GPU=0,PBS=0,models=0,constructors=0,physics=0), 'ACTUAL_CLOSED_FINITE_AUDIT_TOOL_INPUTS_ONLY')
    tools=deepcopy(codec_toolchain);tools.update(delivery_preparation_root=str(codec_root),delivery_preparation_closed_lease_sha256=a.sha(codec_closed_lease_bytes))
    for name in ('ffmpeg','ffprobe'):
        wrapper=codec.wrapper(Path(roots['core'])/'audit-codec',name)
        tools['files']['bin/'+name]=dict(bytes=len(wrapper),sha256=a.sha(wrapper),source_path='EXPLICIT_ORIGINAL_AUDITOR_SUBPROCESS_ONLY_WRAPPER')
    keep={'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json','plans/INITIAL_CORR_PARAMETER_INVENTORY.json',
          'plans/FROZEN_SOURCE_PAIR_RECIPE.json','plans/PV1_TRANSPORT_FIXTURE.json',
          'reference/RECORDED_OBSERVATION.json','reference/RAW_REFERENCE_BINDING.private.json',
          'reference/REFERENCE_PROOF_FROM_CORE.json','evidence/LIVE_READINESS.json','plans/PV1_HF_BANK_DELIVERY.json'}
    rows={};sites={}
    for role,old in base.items():
        need(all(a.sha(base64.b64decode(b))==old['manifest_template']['execution']['sealed_inputs'][n]
                 for n,b in old['files'].items()),'EXACT_ORIGINAL_TEMPLATE_BYTES')
        root=Path(roots[role]);files={n:base64.b64decode(b)for n,b in old['files'].items()if n in keep}
        population=files['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'];groups=json.loads(population)['groups']
        from .dataset import POPULATION
        need(a.sha(population)==POPULATION,'ORIGINAL_POPULATION_NO_SUBSET')
        fixture=json.loads(files['plans/PV1_TRANSPORT_FIXTURE.json']);fixture['run_uuid']=run_name
        files.update({'plans/PV1_TRANSPORT_FIXTURE.json':a.raw(fixture),context.CAPTURE:training_capture_bytes,
            context.PEERS:a.raw(peers),'V5_CD_SCOPE_BINDING.json':a.raw(s.validated(g)),
            'plans/LAST_CONFIRMED_CORE_LOCAL_HISTORICAL.private.json':historical_LOCAL_bytes,
            'plans/SOURCE_EVALUATION_PLAN.json':a.raw(evaluation.candidate_plan(groups,arm)),
            'plans/V5_CD_PAIRED_EVALUATION_PLAN.json':a.raw(evaluation.plan(groups)),
            'plans/PRIOR_EVALUATION_OUTCOMES.json':a.raw([]),'PV1_STAGE_BINDING.json':a.raw(evaluation.stage(arm,colocated=True)),
            'plans/VANDA_READONLY_INFERENCE_REFERENCE.private.json':a.raw(reference_plan),
            'plans/EVALUATION_CHECKPOINT.private.json':a.raw(dict(candidate=arm,global_step=1024,checkpoint=str(Path(roots['hopper'])/'model_views'/arm),
                checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],full_resume_files_present=False)),
            'plans/PV1_TRAIN_INPUTS.private.json':a.raw(native_training_inputs),
            'NEURAL_BINDINGS.private.json':a.raw(native_neural_bindings),
            'INPUT_BINDINGS.private.json':a.raw(dict(asset_root=asset_plan['root'],runtime_python='/opt/wbc-python/bin/python',new_dataset=native_training_inputs['new_dataset'])),
            'plans/ASSETS.private.json':a.raw(asset_plan),'plans/ORIGINAL_ASSET_SHA_TABLE.json':a.raw(asset_plan['files']),
            'plans/VANDA_QUEUE_LIMIT.json':queue_limit_bytes})
        confirmation_name=outside.FIVE_PLAN if a.sha(independent_job_bytes)==outside.FIVE_SHA else outside.PLAN
        need(a.sha(independent_job_bytes) in (outside.FIVE_SHA,outside.SHA),'ORIGINAL_HUMAN_OTHER_JOB_IDENTITY_BYTES')
        files[confirmation_name]=independent_job_bytes
        if role=='core':
            files[codec.MANIFEST]=a.raw(tools)
            files['plans/VANDA_AUDIT_CODEC_PREPARATION_CLOSED_LEASE.private.json']=codec_closed_lease_bytes
        m=deepcopy(old['manifest_template']);m.update(schema=context.SCHEMA,scope=peers[role]['role'],candidate=arm,run_uuid=run_name,
            output_root=str(root),owner='EXISTING_LOCAL_PC',owner_user='e1192548',owner_uid=22182,evaluation_peer_platform=identity.VERSION)
        if role=='hopper':m['platform_evaluation_version']=identity.VERSION
        else:m.pop('platform_evaluation_version',None)
        m['authority']=dict(protocol_ref=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,T_CD=marker['T_CD'],deadline_epoch=marker['deadline_epoch'])
        m['code'].update(vla_commit=source_sha,capsule_relative='capsules/'+source_sha+'/code',source_seal_relative='capsules/'+source_sha+'/SOURCE_SEAL.json',
                         source_seal_sha256=a.sha(source_seal_bytes),frozen_numerical_files=frozen,frozen_numerical_overlay=seal.get('frozen_numerical_overlay',{}))
        binding=dict(run_uuid=run_name,run_root=str(root),registered_parent=str(root.parent),uid=22182,role=m['scope']+'__'+run_name,
            registry=registry,lease=str(Path(registry).parent/'leases'/(run_name+('-physics'if role=='core'else'-neural')+'.json')))
        m['execution']=dict(profile=profiles[role]['name'],environment_registry=environment_registry,environment_profile_sha256=a.sha(a.raw(profiles[role])),
            private_path_role=binding['role'],output_binding_sha256=a.sha(a.raw(binding)),sealed_inputs={n:a.sha(v)for n,v in files.items()},
            control_budget_profile='PV1_FINITE_POLL250MS_CALL150000_V1',owner_heartbeat_policy='PV1_LOCAL_OWNER_HEARTBEAT_ONLY_V1',agent_liveness_required=False)
        if cpu_slot_policy is not None:m['execution']['CPU_slot_policy']=cpu_slot_policy
        m['limits']=dict(T_CD=marker['T_CD'],operation_deadline_epoch=marker['deadline_epoch'],CPU_s=93900 if role=='core'else 80000,
            cpus=[4,5]if role=='core'else[0,1,2,3],active_cores=2 if role=='core'else 4,rpc_cpu=6 if role=='core'else 7,
            engine_starts=2108 if role=='core'else 0,physics_ticks=832000 if role=='core'else 0,policy_episodes=26 if role=='core'else 0,
            training_updates=0,output_bytes=cohost.storage_contract()['original_gross_upper_bytes'])
        rows[role]=dict(root=str(root),manifest=m,manifest_sha256=a.sha(a.raw(m)),binding=binding,files=files)
        code=root/m['code']['capsule_relative'];sites[role]=dict(root=str(root),source_sha=source_sha,manifest_sha256=rows[role]['manifest_sha256'],
            code=str(code),python=native_neural_bindings['python']if role=='hopper'else'/opt/wbc-python/bin/python',
            PV1_worker_manifest=str(code.parent/'WORKER_RUN_MANIFEST.json'),fixture_plan=fixture,plan_sha256=a.sha(files['plans/PV1_TRANSPORT_FIXTURE.json']))
    cfg=dict(source_sha=source_sha,source_code=str(Path(roots['owner'])/'capsules'/source_sha/'code'),sites=sites,
        container=profiles['core']['container'],assets_root=asset_plan['root'],programs=programs,
        outside_confirmation_base64=base64.b64encode(independent_job_bytes).decode(),canonical_LOCAL_sync='PENDING',
        historical_LOCAL_sha256=a.sha(historical_LOCAL_bytes),agent_liveness_required=False)
    owner_binding=dict(uid=22182,run_uuid=Path(roots['owner']).name,run_root=roots['owner'],registered_parent=str(parent/'control'),
        role=identity.OWNER_ROLE+'__'+run_name,registry=registry,lease=str(Path(registry).parent/'leases'/(run_name+'-owner.json')))
    om=dict(schema=context.SCHEMA,scope=identity.OWNER_ROLE,run_uuid=Path(roots['owner']).name,output_root=roots['owner'],
        owner_uid=22182,authority=deepcopy(rows['hopper']['manifest']['authority']),code=dict(vla_commit=source_sha,capsule_relative='capsules/'+source_sha+'/code'),
        execution=dict(configuration_sha256=a.sha(a.raw(cfg)),owner_code_sha256={n:a.sha(b)for n,b in owner_code.items()},
                       output_binding_sha256=a.sha(a.raw(owner_binding)),agent_liveness_required=False),limits=dict(CPU_s=7200,T_CD=marker['T_CD'],operation_deadline_epoch=marker['deadline_epoch']))
    if cpu_slot_policy is not None:
        cfg['CPU_slot_policy']=cpu_slot_policy
        om['execution']['CPU_slot_policy']=cpu_slot_policy
        om['execution']['configuration_sha256']=a.sha(a.raw(cfg))
    request=dict(platform_evaluation_version=identity.VERSION,native_platform='VANDA_A40',campaign_run_uuid=parent.name,
        authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,project='CFP04-CF-007',job_name=identity.JOB_NAME,owner_user='e1192548',owner_uid=22182,
        stage='V5_MATCHED_DEV24',noise_bank=294101,remaining_conditions=24,checkpoint_step=1024,parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,reserved_H200_s=28800,accounting_pad_s=60,CPU_upper_s=173900,neural_CPU_upper_s=80000,
        physical_CPU_upper_s=93900,active_cpu_limit=8,query_limit=2097,optimizer_update_limit=0,reserved_optimizer_updates=0,optimizer_updates=0,walltime_s=28800,
        python=native_neural_bindings['python'],declared_process_CPU_limit_s=79760,Teacher_takeover=False,
        normalizer_refit=False,evaluation_data_for_training=False,LOCKBOX_opened=False,
        joint_CPU_upper_s=173900,numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',source_renderer='MESA_LLVMPIPE_CPU_EGL',source_gl_GPU_visible=False,
        model_payload_copies=0,allocation_number=1,submit_count=g['used']['PBS_submissions']+1,owner_manifest_sha256=a.sha(a.raw(om)),
        additional_output_upper_bytes=cohost.storage_contract()['original_gross_upper_bytes'],legacy_GPU_seconds_field_is_projection=True,charges_DEC1_subcaps=False,
        T_CD=marker['T_CD'],deadline_epoch=marker['deadline_epoch'],created_epoch=now_epoch,evaluation_candidate=arm,request_id=request_id,submission_nonce=nonce,run_uuid=run_name,
        checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],source_sha=source_sha,run_manifest_sha256=rows['hopper']['manifest_sha256'],
        planned_panel_sha256=a.sha(rows['hopper']['files']['plans/SOURCE_EVALUATION_PLAN.json']),training_handoff_sha256=a.sha(a.raw(proof)),
        cold_input_audit_sha256=a.sha(a.raw(dict(source_seal_sha256=a.sha(source_seal_bytes),closed_capture_sha256=a.sha(training_capture_bytes),
            peer_manifests={k:v['manifest_sha256']for k,v in rows.items()}))),queue_limit_relative='plans/VANDA_QUEUE_LIMIT.json',
        queue_limit_sha256=a.sha(queue_limit_bytes),sealed_inputs=deepcopy(rows['hopper']['manifest']['execution']['sealed_inputs']))
    record=dict(request,child_run_uuid=run_name,request_sha256=a.sha(a.raw(request)),physical_manifest_sha256=rows['core']['manifest_sha256'],
        job_id=None,status=accounting.STATES[0],reserved_neural_queries=2097)
    repair=cohost.repair.validated(g)
    if arm=='C50':record['prepaid_panel_IO_credit_sha256']=a.sha(a.raw(repair))
    identity.request_identity(rows['hopper']['manifest'],request,rows['hopper']['manifest_sha256'])
    proposed=accounting.register(g,record,request=request,capture=cap,physical_manifest_bytes=a.raw(rows['core']['manifest']),now_epoch=now_epoch)
    return dict(schema='V5_CD_VANDA_COLD_ORIGINAL_PEERS_PURE_PACKET_V1',rows=rows,owner=dict(root=roots['owner'],manifest=om,binding=owner_binding,config=cfg,files=owner_code),
        request=request,record=record,proposed_global=proposed,native_registration_ready_after_owner_fresh_full_queue_and_project=False,native_registered=False,new_GPU=0,new_PBS=0,new_model=0,new_physics=0)


def bind_actual_cold_readback(packet,*,audit_bytes,global_bytes,now_epoch):
    """Finalize the request only after actual immutable peer input readback.

    The packer's deterministic input fingerprint is not native cold evidence.
    This separate boundary avoids treating prepared bytes as an audit receipt.
    The original once owner must also read the exact native audit file itself.
    """
    result=deepcopy(packet);audit=json.loads(audit_bytes);r=result['request'];g=json.loads(global_bytes)
    accounting.checked(g)
    need(audit['schema']=='V5_CD_VANDA_ACTUAL_COLD_PAIRED_INPUT_READBACK_V1'
         and audit['status']=='PASS_ACTUAL_REGISTERED_IMMUTABLE_COLD_INPUTS_NO_MODEL_OR_PHYSICS'
         and audit['source_sha']==r['source_sha'] and audit['run_uuid']==r['run_uuid']
         and audit['GLOBAL_sha256']==a.sha(global_bytes)
         and audit['peer_manifest_sha256']=={role:row['manifest_sha256']for role,row in result['rows'].items()}
         and audit['all_source_files_full_SHA']is True and audit['all_sealed_inputs_full_SHA']is True
         and audit['all_fifteen_original_numerical_files']is True and audit['original_26_assets_full_SHA']is True
         and audit['nine_original_model_inodes_unchanged']is True
         and audit['actual_registry_and_own_ACTIVE_leases']is True
         and 0<=now_epoch-audit['observed_epoch']<=120
         and all(audit[k]==0 for k in ('new_GPU','new_PBS','new_model','new_constructor','new_physics')),
         'ACTUAL_FRESH_NATIVE_COLD_READBACK_NOT_THE_PURE_PACKET')
    cap=json.loads(result['rows']['hopper']['files'][context.CAPTURE]);need(base64.b64decode(cap['global_base64'])==global_bytes,'SAME_ACTUAL_PRE_REGISTRATION_GLOBAL')
    r.update(cold_input_audit_sha256=a.sha(audit_bytes),created_epoch=now_epoch)
    record=result['record'];record.update(cold_input_audit_sha256=r['cold_input_audit_sha256'],created_epoch=now_epoch,request_sha256=a.sha(a.raw(r)))
    result['proposed_global']=accounting.register(g,record,request=r,capture=cap,physical_manifest_bytes=a.raw(result['rows']['core']['manifest']),now_epoch=now_epoch)
    result.update(actual_cold_readback_sha256=a.sha(audit_bytes),native_registration_ready_after_owner_fresh_full_queue_and_project=True)
    return result
