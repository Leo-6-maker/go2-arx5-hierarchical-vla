"""Independent post-F first512 state/output audit, without neural execution.

Reuse original full-stream witnesses and actual serialized Trainer state. No
model construction, CUDA, generation, optimizer step or GLOBAL write occurs.
This is called by the original login owner only after exact native F/finals.
"""
import hashlib,json,os,subprocess
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1.readonly_checkpoint_hash import StreamHashes
from scripts.pi0294.e1.trainer import check_trace
from . import minimal_execution as e,minimal_terminal as terminal,minimal_training as training
from . import minimal_resume as resume,minimal_inference_export as export,minimal_probes as language

need=e.need


def serialized_state(path):
    """CPU-map trusted own hash-verified state, preserving serialized devices.

    Comparing a CPU-mapped optimizer to its saved CUDA identity must retain
    each storage's original device tag. Guessing that every tensor was CUDA
    would mislabel AdamW's CPU step tensors and the saved CPU RNG.
    """
    import torch
    need(not torch.cuda.is_initialized(),'AUDIT_NO_CUDA_INITIALIZATION')
    origins={}
    def cpu_map(storage,location):
        need(location in('cpu','cuda:0'),'AUDIT_ONLY_ORIGINAL_SINGLE_GPU_SERIALIZED_DEVICE')
        key=int(storage._cdata)
        need(key not in origins or origins[key]==location,'AUDIT_ONE_SERIALIZED_DEVICE_PER_STORAGE')
        origins[key]=location;return storage
    value=torch.load(path,map_location=cpu_map,weights_only=False,mmap=True)
    def tree(v):
        if isinstance(v,torch.Tensor):
            row=resume.tree_identity(v);key=int(v.untyped_storage()._cdata)
            need(key in origins,'AUDIT_ACTUAL_SERIALIZED_TENSOR_STORAGE')
            row['device']=origins[key];return row
        if isinstance(v,dict):
            need(all(type(k)in(str,int)for k in v),'AUDIT_EXPLICIT_STATE_KEYS')
            return dict(kind='dict',items=[dict(key_type=type(k).__name__,key=k,value=tree(v[k]))
                for k in sorted(v,key=lambda k:(type(k).__name__,repr(k)))])
        if isinstance(v,(tuple,list)):return dict(kind=type(v).__name__,items=[tree(x)for x in v])
        return resume.tree_identity(v)
    identity=tree(value)
    need(not torch.cuda.is_initialized(),'AUDIT_SERIALIZATION_DID_NOT_CREATE_CUDA_CONTEXT')
    return value,dict(sha256=e.training.study.digest(identity),serialized_devices=sorted(set(origins.values())))


def own_members(folder,complete,stream):
    """Every declared output is own, immutable, regular and physically read."""
    root=Path(folder);need(root.is_dir()and not root.is_symlink(),'AUDIT_OWN_CLOSED_CHECKPOINT_DIRECTORY')
    actual={}
    for p in root.rglob('*'):
        need(not p.is_symlink(),'AUDIT_NO_CHECKPOINT_ALIAS')
        if p.is_file():actual[p.relative_to(root).as_posix()]=p
    need(set(actual)==set(complete['files'])|{'CHECKPOINT_COMPLETE.json'},'AUDIT_NO_UNLISTED_OR_MISSING_OUTPUT')
    for name,path in actual.items():
        need(path.stat().st_uid==os.getuid()and path.stat().st_mode&0o777==0o400,
            'AUDIT_OWN_IMMUTABLE_COMPLETE_MEMBER:'+name)
        actual_sha=stream(path)
        if name!='CHECKPOINT_COMPLETE.json':need(actual_sha==complete['files'][name],'AUDIT_COMPLETE_MEMBER_BYTES:'+name)
    return {name:complete['files'][name]for name in complete['files']}


def optimizer_state(cp,step,contract,stream):
    """Inspect original AdamW/scheduler/RNG files after full member SHA audit."""
    import torch
    saved=read(cp/resume.FILENAME)
    opt,oi=serialized_state(cp/'optimizer.pt');sched,si=serialized_state(cp/'scheduler.pt');rng,ri=serialized_state(cp/'rng_state.pth')
    need(saved['schema']==resume.SCHEMA and saved['step']==step and saved['cursor']==step*32
        and saved['scheduler_horizon']==1024 and saved['optimizer_sha256']==oi['sha256']
        and saved['scheduler_sha256']==si['sha256'] and saved['rng_sha256']==ri['sha256'],
        'AUDIT_SERIALIZED_FULL_OPTIMIZER_SCHEDULER_RNG_NOT_REPORTED_FLAGS')
    names=[r['name']for r in contract['trainable']];groups=saved['optimizer_parameter_groups']
    need(names==saved['optimizer_parameter_names'] and len(groups)==len(opt['param_groups'])
        and set(n for row in groups for n in row)==set(names)
        and len([n for row in groups for n in row])==len(names)
        and [len(row)for row in groups]==[len(row['params'])for row in opt['param_groups']]
        and opt['state'] and sched['last_epoch']==step and {'python','numpy','cpu','cuda'}<=set(rng),
        'AUDIT_EXACT_OPTIMIZER_PARAMETER_ORDER_AND_REAL_SINGLE_CUDA_RNG')
    steps=set()
    for value in opt['state'].values():
        s=value['step'];steps.add(int(s.item()if hasattr(s,'item')else s))
        need(all(k in value and bool(torch.isfinite(value[k]).all())for k in('exp_avg','exp_avg_sq')),
            'AUDIT_FINITE_REAL_ADAMW_MOMENTS')
    need(steps=={step},'AUDIT_EVERY_OPTIMIZER_STATE_AT_OWN_TERMINAL_STEP')
    return dict(optimizer_steps=sorted(steps),scheduler_last_epoch=sched['last_epoch'],
        optimizer_state_sha256=oi['sha256'],scheduler_state_sha256=si['sha256'],rng_state_sha256=ri['sha256'],
        rng_keys=sorted(rng),optimizer_parameter_groups=groups)


def checkpoint(root,spec,request,data,stream):
    step=spec['optimizer_limit'];cp=root/'training'/spec['id']/('checkpoint-'+str(step));complete=read(cp/'CHECKPOINT_COMPLETE.json')
    need(complete['global_step']==step and complete['full_resume']is True and complete['session_terminal']is True
        and complete['training_identity']==spec['arm'] and complete['qualification_only']is(not spec['formal_candidate'])
        and complete['scheduler_horizon']==1024 and complete['origin_job_id']==request['job_id']
        and complete['origin_source_sha']==request['source_sha'],'AUDIT_ACTUAL_OWN_STEP2_OR_FIXED512_FULL_STATE')
    members=own_members(cp,complete,stream)
    required={'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json',
        'statistics.json','ADAPTER_PARAMETER_CONTRACT.json','FROZEN_PARAMETER_IDENTITY.json','MODEL_PARAMETER_STATE_IDENTITY.json',resume.FILENAME}
    need(required<=set(members)and any(n.endswith('.safetensors')for n in members)
        and any(n.startswith('processor/')for n in members)and members['statistics.json']==e.R0,
        'AUDIT_COMPLETE_MODEL_PROCESSOR_AND_ORIGINAL_R0')
    recipe=read(root/'plans/PRIMARY_LEARNING_RECIPES.json')['recipes'][spec['arm']]
    training.validate(recipe,read(root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json'))
    saved=read(cp/'SAMPLER_STATE.json');state=read(cp/'trainer_state.json');full=read(cp/resume.FILENAME)
    expected=data.make_sampler(data).state_dict(step*32)
    need(saved['sampler']==full['sampler']==expected and saved['recipe']==recipe
        and saved['manifest_sha256']==data.manifest_sha and saved['statistics_sha256']==e.R0
        and state['global_step']==step and state['max_steps']==1024,
        'AUDIT_FULL_MATCHED_CURSOR_RECIPE_AND_UNRESTARTED_LR_HORIZON')
    trace=check_trace(root/'training'/spec['id']/'session/worker_trace',data,0,step*32)
    need(trace==saved['worker_trace'],'AUDIT_ACTUAL_FULL_CONSUMPTION_TRACE_NOT_ONLY_CURSOR_LABEL')
    contract=read(cp/'ADAPTER_PARAMETER_CONTRACT.json');frozen=read(cp/'FROZEN_PARAMETER_IDENTITY.json')
    need(contract['arm']==full['arm']==spec['arm']and contract['target_category']==10
        and contract['module_mode']==training.MODE,'AUDIT_EXACT_TARGET10_AND_MATCHED_MODULE_MODE')
    from .minimal_adapter import LEAVES,SHARED,SCHEMA,UPSTREAM
    names={r['name']for r in contract['trainable']};targets={n+'.'+k for n in LEAVES for k in('target_W','target_b')}
    need(contract['schema']==SCHEMA and contract['upstream_commit']==UPSTREAM and targets<=names
        and (names==targets if spec['arm']==training.ARMS[0]else all(n in targets or n.startswith(SHARED)for n in names))
        and not any('position_embedding'in n for n in names),'AUDIT_EXACT_PARAMETER_WHITELIST_NOT_VLM_OR_POSITION')
    state_proof=optimizer_state(cp,step,contract,stream)
    run=read(root/'training'/spec['id']/'session/TRAINING_RUN.json')
    need(run['global_step']==step and run['cursor']==step*32 and run['frozen_tensors_unchanged']is True
        and run['checkpoint_manifest_sha256']==stream(cp/'CHECKPOINT_COMPLETE.json')
        and run['full_model_parameter_state']==read(cp/'MODEL_PARAMETER_STATE_IDENTITY.json'),
        'AUDIT_ACTUAL_NATIVE_TRAINER_CURRENT_PARAMETER_AND_FROZEN_BASELINE')
    if not spec['formal_candidate']:
        q=read(cp/'QUALIFICATION_PARAMETER_UPDATE.json');changed=q['changed_trainable_parameters']
        need(q['optimizer_calls']==2 and q['cursor']==64 and q['arm']==spec['arm']
            and q['frozen_tensors_unchanged']is True and q['gradient_rows']
            and set(changed)<=names and all(q['gradient_rows'][n].get('present')is True
                and q['gradient_rows'][n].get('finite')is True for n in targets)
            and all(any(q['gradient_rows'][n]['maximum_absolute']>0 for n in targets if n.startswith('action_head.'+part+'.'))
                for part in('state_encoder','action_encoder','action_decoder'))
            and all(any(n.startswith('action_head.'+part+'.')for n in changed)
                for part in('state_encoder','action_encoder','action_decoder')),
            'AUDIT_REAL_TARGET_GRADIENTS_AND_TWO_WARMUP_CALL_PARAMETER_UPDATES')
        if spec['arm']==training.ARMS[1]:
            need(any(n.startswith(('action_head.model.','action_head.vlln.','action_head.vl_self_attention.'))for n in changed),
                'AUDIT_REAL_E2_SHARED_PARAMETER_UPDATE')
    return dict(checkpoint_manifest_sha256=stream(cp/'CHECKPOINT_COMPLETE.json'),global_step=step,cursor=step*32,
        scheduler_horizon=1024,raw_parent_files_identity=training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        full_resume=True,qualification_only=not spec['formal_candidate'],state_proof=state_proof,
        trace=trace,frozen_parameter_identity=frozen,full_streamed_members_sha256=e.sha(e.raw(members)))


def draw(path,stream,*,seed,input_fingerprint):
    import numpy as np
    digest=stream(path)
    with np.load(path,allow_pickle=False)as z:
        need(z['physical_H40'].shape==(40,10)and z['normalized'].shape==(1,40,132)
            and np.isfinite(z['physical_H40']).all()and np.isfinite(z['normalized']).all()
            and int(z['noise_seed'])==seed and str(z['observation_fingerprint'])==input_fingerprint,
            'AUDIT_ACTUAL_SAVED_FULL132_NOISE_AND_INPUT')
    return digest


def audit(root,request_path,*,native,inventory_sha256):
    """After native F, independently read every session and actual output pair."""
    from .minimal_effects import Effects
    from .minimal_train_session import dataset
    from .minimal_model import unchanged_raw_parent
    from scripts.pi0294.e1.model import digest
    root=Path(root);request_path=Path(request_path);r=read(request_path);e.request(r)
    job=native['job_id'];need(os.getuid()==22182 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and native['job_state']=='F'and type(native['Exit_status'])is int,'AUDIT_ORIGINAL_CPU_OWNER_POST_F_ONLY')
    p=subprocess.run(['/opt/pbs/bin/qstat','-x','-f','-F','json',job],capture_output=True,text=True,timeout=25)
    need(p.returncode==0 and not p.stderr,'AUDIT_ACTUAL_F_READ_NOT_TIMEOUT')
    actual=json.loads(p.stdout)['Jobs'][job]
    need(actual==native['raw_native']and actual['job_state']=='F'and actual['Exit_status']==native['Exit_status']
        and actual['Variable_List']['PV1_REQUEST_ID']==r['request_id']
        and actual['Variable_List']['PV1_SUBMISSION_NONCE']==r['submission_nonce'],
        'AUDIT_CURRENT_NATIVE_F_SAME_REQUEST_AND_NONCE')
    d=request_path.parent;final=read(d/'PBS_PROCESS_FINAL.json');wrapper=read(d/'PBS_WRAPPER_FINAL.json')
    need(final['job_id']==wrapper['job_id']==job and final['request_id']==r['request_id']
        and final['source_sha']==r['source_sha']and final['request_sha256']==sha(request_path)
        and wrapper['returncode']==native['Exit_status'],'AUDIT_TRUE_NATIVE_WORKER_AND_WRAPPER_FINAL')
    allocation=read(d/'ALLOCATION_ADMISSION.json')
    ledger=Effects(root,allocation,request_path=request_path,guard=lambda:None)
    # check()/summary() are readonly. Never call guard(), transaction(), load,
    # charge or optimizer methods after native F.
    ledger.check(read(ledger.path));effects=ledger.summary()
    need(effects==final['effects']and effects['pending_updates']==effects['unknown_query_intents']==[],
        'AUDIT_CURRENT_FULL_EFFECTS_AND_ORIGINAL_QUERY_JOURNALS')
    base=dict(request_sha256=sha(request_path),source_sha=r['source_sha'],job_id=job,
        effect_summary_sha256=e.sha(e.raw(effects)),inventory_sha256=inventory_sha256,
        raw_parent_files_identity=training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        new_model_loads=0,new_queries=0,new_updates=0,new_physics=0)
    if native['Exit_status']!=0 or not effects['all_five_sessions_runtime_complete']:
        return dict(base,status=terminal.PARTIAL,preserved_session_states=effects['sessions'])
    stream=StreamHashes();folder=root/'evidence/MINIMAL_FIVE_SESSIONS'
    raw=read(folder/'RAW_PARENT_COMPLETE_STREAM_AUDIT.json');raw_plan=read(root/'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')
    unchanged_raw_parent(raw_plan['checkpoint'],raw)
    data=dataset(root,root/'evidence/independent-terminal-reader')
    capture=read(root/'plans/MINIMAL_SIX_PRECONTACT16_TRAIN_CAPTURE.private.json')
    plan=read(root/'plans/MINIMAL_LANGUAGE24_PANEL.private.json');panel=read(root/'plans/MINIMAL_TWELVE_PANEL.json')
    need(plan==language.prepare(capture,panel),'AUDIT_UNCHANGED_TRAIN_ONLY_LANGUAGE24')
    payloads={c['id']:language.payload_for_case(c,capture,data)for c in plan['cases']}
    wire=lambda p:digest({k:v for k,v in p.items()if k!='snapshot_uuid'})
    first=wire(payloads[plan['cases'][0]['id']]);endpoints={};audited=[];qualified_frozen={}
    for spec in e.native_sessions():
        sid=spec['id'];sf=folder/sid;receipt=read(sf/'SESSION_TERMINAL.json');row=effects['sessions'][sid]
        need(stream(sf/'SESSION_TERMINAL.json')==row['terminal_receipt_sha256']and receipt['request_sha256']==base['request_sha256']
            and receipt['source_sha']==r['source_sha']and receipt['session_id']==sid
            and receipt['optimizer_updates']==spec['optimizer_limit']and receipt['query_attempts']==spec['query_limit'],
            'AUDIT_ACTUAL_SAME_SESSION_COMPLETE_RECEIPT')
        if not spec['formal_candidate']:
            for i in range(15):
                path=sf/'draws'/('zero-'+str(i).zfill(2)+'.npz');draw(path,stream,seed=29400+i,input_fingerprint=first)
                if sid!=e.native_sessions()[0]['id']:
                    export.compare_saved_draws(folder/e.native_sessions()[0]['id']/'draws'/path.name,path)
        if spec['optimizer_limit']:
            cp=checkpoint(root,spec,dict(r,job_id=job),data,stream)
            if not spec['formal_candidate']:
                evidence=receipt['actual_session_evidence'];restore=evidence['restore']
                need(restore['status']=='PASS_ORIGINAL_TRAINER_OWN_QUALIFICATION_FULL_STATE_RESTORE_ONLY'
                    and restore['optimizer_scheduler_model_RNG_sampler_checked']is True and restore['step']==2
                    and restore['cursor']==64 and restore['new_optimizer_updates']==restore['new_queries']==0,
                    'AUDIT_ACTUAL_OWN_FULL_STATE_QUALIFICATION_RESTORE')
                for name in('step2-reference','step2-restored'):
                    draw(sf/'draws'/(name+'.npz'),stream,seed=29400,input_fingerprint=first)
                export.compare_saved_draws(sf/'draws/step2-reference.npz',sf/'draws/step2-restored.npz')
                qualified_frozen[spec['arm']]=cp.pop('frozen_parameter_identity')
            else:
                need(cp.pop('frozen_parameter_identity')==qualified_frozen[spec['arm']],
                    'AUDIT_INDEPENDENT_FRESH_RAW_FORMAL_FROZEN_BASELINE_NOT_QUALIFICATION_INHERITANCE')
                pp=root/'evidence'/(sid+'-512')/'TRAIN_PROBES';probe=read(pp/'RESULT.json')
                declared=read(root/'plans/PV1_FIXED_TRAIN_PROBES.json')
                need(probe['complete']is True and len(probe['records'])==12 and probe['checkpoint_manifest_sha256']==cp['checkpoint_manifest_sha256'],
                    'AUDIT_ALL_ORIGINAL_FIXED_TWELVE_TRAIN_PROBES')
                need(probe['probe_plan_sha256']==sha(root/'plans/PV1_FIXED_TRAIN_PROBES.json')
                    and [x['id']for x in probe['records']]==[x['id']for x in declared['windows']]
                    and all(x['split']=='TRAIN'for x in probe['records']),
                    'AUDIT_SAME_ORIGINAL_TWELVE_PLAN_POPULATION_AND_ORDER')
                stream(pp/'RESULT.json')
                for rowp in probe['records']:need(stream(pp/(rowp['id']+'.npz'))==rowp['arrays_sha256'],'AUDIT_TRAIN_PROBE_BYTES')
                inf=root/'inference'/spec['arm'];saved=read(inf/'CHECKPOINT_COMPLETE.json');own_members(inf,saved,stream)
                need(saved['wrapped_checkpoint_manifest_sha256']==cp['checkpoint_manifest_sha256']and saved['global_step']==512,
                    'AUDIT_ORDINARY_FULL_EXPORT_OF_THIS_OWN512')
                case=plan['cases'][0];reload=sf/'draws/official-reloaded.npz'
                draw(reload,stream,seed=case['noise_seed'],input_fingerprint=wire(payloads[case['id']]))
                export.compare_saved_draws(sf/'draws'/(case['id']+'.npz'),reload)
                cp.update(TRAIN_probes=12,language_probes=24,official_inference_manifest_sha256=stream(inf/'CHECKPOINT_COMPLETE.json'))
                endpoints[spec['arm']]=cp
        if spec['optimizer_limit']==0 or spec['formal_candidate']:
            saved=read(sf/'LANGUAGE24_SAVED.json');need(len(saved['records'])==24,'AUDIT_ALL24_LANGUAGE_DRAWS')
            for case,rowp in zip(plan['cases'],saved['records']):
                path=sf/'draws'/(case['id']+'.npz')
                need(rowp['id']==case['id'] and rowp['input_sha256']==case['semantic_input_sha256']
                    and rowp['arrays_sha256']==draw(path,stream,seed=case['noise_seed'],input_fingerprint=wire(payloads[case['id']])),
                    'AUDIT_ACTUAL_MATCHED_LANGUAGE_INPUT_NOISE_AND_BYTES')
        audited.append(sid)
    witnesses=stream.witnesses()
    return dict(base,status=terminal.AUDIT,audited_session_ids=audited,endpoints=endpoints,
        same15_zero_step_raw_wrapped_arms_exact=True,two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact=True,
        official_full_model_reconstruction_output_pairs_exact=True,frozen_parameters_and_buffers_unchanged=True,
        streamed_witnesses=witnesses,raw_parent_audit_sha256=sha(folder/'RAW_PARENT_COMPLETE_STREAM_AUDIT.json'))
