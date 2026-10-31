"""Frozen book checkpoint on the ten predeclared source DEV conditions.

Reuse the original generator, source engine, Prefix8 capture and raw scorer.
No Teacher plan, stage, object pose or task state is supplied to the model.
"""
import gc,hashlib,json,os,signal,sys,time
from pathlib import Path

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
    entry=parser.parse_args();mb=(Path(entry.root)/'RUN_MANIFEST.json').read_bytes()
    if hashlib.sha256(mb).hexdigest()!=entry.manifest_sha256:raise PermissionError('BOOK_DEV_ENTRY_MANIFEST')
    dependency=json.loads(mb)['dependency_code'];sys.path[:0]=[dependency,dependency+'/src']

import numpy as np
from scripts.pi0292.eval_contract import gate_action,matrix10
from scripts.pi0294.e1.model import payload


class ModelReferenceRejected(RuntimeError):
    pass


def valid_model_condition(audit,completed_generations,gate_rejected):
    """Native safety after a real prediction is a negative policy outcome.

    A failure before any completed prediction remains an engineering failure.
    A safety stop ends the panel; it does not disappear from its denominator.
    """
    return bool(gate_rejected or audit.get('complete_episode',False)
        or(completed_generations>0 and audit.get('original_native_safety_pass')is False))


def restoration_dataset(m,bindings):
    """Old full-state cursor restoration uses its original population only."""
    if m['kind']=='V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING':
        if not bindings.get('parent_dataset')or bindings['parent_dataset']==bindings['dataset']:
            raise PermissionError('RECOVERY_SEPARATE_PARENT_RESTORE_DATA_REQUIRED')
        return bindings['parent_dataset']
    return bindings['dataset']


def generation_statistics(m,bindings):
    if m['kind']=='V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING':return Path(bindings['dataset'])/'statistics.json'
    return Path(m.get('delta_checkpoint',m['checkpoint']))/'statistics.json'


def dev_card(condition):
    # Caller binds the complete frozen TRAIN_DEV_SEAL by its exact digest.
    from scripts.pi0294.multitask.libero_book_teacher import route_card,RELEASE_REVISION
    identifier=condition['id'];index=int(identifier[-3:])
    offset=condition['xy_offset_m']
    if (identifier!=f'BOOK_DEV{index:03}' or not 0<=index<10 or type(offset)is not list or len(offset)!=2
            or any(type(x)not in(int,float)or not np.isfinite(x)or abs(x)>.01 for x in offset)):
        raise ValueError('ORIGINAL_TEN_DISJOINT_PREDECLARED_DEV_CONDITIONS')
    card=route_card(RELEASE_REVISION);dx,dy=condition['xy_offset_m']
    card.update(id=identifier,study_split='DEV',initial_xy_offset_m=[dx,dy])
    initial=np.asarray(card['initial_book_native_world']);initial[:2,3]+=[dx,dy]
    card['initial_book_native_world']=initial.tolist();card['source_base_card']['source_xy']=[.36+dx,-.10+dy]
    return card


class Student:
    stage='STUDENT_NO_PRIVILEGED_STAGE'
    candidate_complete=False
    def __init__(self,instance,identifier,record,guard,*,seed_namespace='BOOK5_DEV10'):
        self.instance,self.identifier,self.record,self.guard=instance,identifier,record,guard
        self.seed_namespace=seed_namespace
        self.index=self.last_plan=-1;self.queries=self.completed_generations=0;self.state=self.images=None

    def observe(self,state):
        self.index+=1
        # Explicit projection is the only state boundary into the student.
        from go2arx5_vla.task_data.codec import pose10
        self.state=pose10(np.asarray(state['tcp']),float(state['gripper_fraction']))
        self.time_s=float(state['time_s'])

    def set_images(self,images,instruction):
        if set(images)!={'body_rgb','wrist_rgb'}:raise ValueError('ACTUAL_TWO_SOURCE_CAMERAS')
        self.images=[images['body_rgb'].copy(),images['wrist_rgb'].copy()];self.instruction=instruction

    def plan(self):
        self.guard()
        if self.index%8 or self.index==self.last_plan or self.images is None:
            raise PermissionError('ONE_FRESH_MODEL_QUERY_PER_REAL_PREFIX8_OBSERVATION')
        self.last_plan=self.index;sequence=self.index//8
        request=payload(self.state,self.images,self.instruction)
        seed=int.from_bytes(hashlib.sha256(f'{self.seed_namespace}/{self.identifier}/{sequence}'.encode()).digest()[:8],'big')%(2**63)
        self.record('intent',dict(sequence=sequence,index=self.index,seed=seed,
            observation_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
            student_state_eef10=self.state.tolist(),language=self.instruction,
            image_SHA256=[r['sha256']for r in request['images']],privileged_student_inputs=False))
        self.queries+=1;value=self.instance.generate(request,seed);self.completed_generations+=1
        gate=gate_action(value['physical_H40'],self.state,self.index*.05,self.identifier+'-'+str(sequence))
        self.record('prediction',dict(sequence=sequence,index=self.index,gate=gate,value=value))
        if gate['decision']=='REJECT':raise ModelReferenceRejected('MODEL_REFERENCE_GATE_REJECT')
        matrices=matrix10(value['physical_H40'])
        return dict(poses=matrices,opening=np.asarray(value['physical_H40'])[:,9].copy())


def episode(root,condition,instance,template,guard,*,student_factory=Student,card_factory=dev_card):
    from PIL import Image
    import mujoco
    from scripts.pi0294.multitask.source_engine import make_engine
    from scripts.pi0294.multitask.capture_core import capture
    from scripts.pi0294.multitask.night_book_audit import audit
    from scripts.pi0294.e1.artifacts import atomic_npz
    from scripts.pi0292.durable import write,read
    from scripts.pi0294.native_dataset import sha
    folder=root/'conditions'/condition['id'];folder.mkdir(parents=True,exist_ok=False)
    card=card_factory(condition);write(folder/'CARD.json',card);(folder/'predictions').mkdir()
    stream=(folder/'ACTUAL_RECORDS.jsonl').open('x');engine=None;result=None;error=None;gate_rejected=False;events={};recorded=0;written=0
    def encode(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    def record(kind,row):
        nonlocal written
        path=folder/'predictions'/(str(row['sequence']).zfill(2)+'_'+kind)
        if kind=='prediction':
            row=dict(row);v=row.pop('value');atomic_npz(path.with_suffix('.npz'),physical_H40=v['physical_H40'],normalized=v['normalized'])
            written+=path.with_suffix('.npz').stat().st_size
            row.update(arrays_sha256=sha(path.with_suffix('.npz')),noise=v['noise'],inference_precision=v['inference_precision'],
                normalized_outside_unit_components=v['normalized_outside_unit_components'])
        if path.with_suffix('.json').exists():raise PermissionError('NO_MODEL_QUERY_OR_RESULT_REPLAY')
        write(path.with_suffix('.json'),row)
        written+=path.with_suffix('.json').stat().st_size
    def condition_guard():
        guard()
        if written>=500*1024**2 or time.monotonic()-started>=900:raise PermissionError('BOOK_DEV_FINITE_CONDITION_OUTPUT_OR_WALL')
    student=student_factory(instance,condition['id'],record,condition_guard)
    def emit(kind,row):
        nonlocal recorded,written
        condition_guard();events[kind]=events.get(kind,0)+1
        if kind=='observation':
            row=dict(row,policy_obs=dict(row['policy_obs']))
            student.set_images({n:row['policy_obs'][n]for n in ('body_rgb','wrist_rgb')},card['instruction'])
            for name in ('body_rgb','wrist_rgb'):
                path=folder/(str(row['index']).zfill(4)+'_'+name+'.png');Image.fromarray(row['policy_obs'][name]).save(path)
                row['policy_obs'][name]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size)
                written+=path.stat().st_size
            recorded+=1
        line=json.dumps(dict(kind=kind,value=row),default=encode,separators=(',',':'),allow_nan=False)+'\n';stream.write(line);written+=len(line.encode())
    started=time.monotonic();CPU=time.process_time()
    try:
        assets=read(template/'ASSET_BINDINGS.private.json')
        definitions={n:(template/'inputs'/n/(n+'.xml')).read_bytes()for n in ('black_book','desk_caddy')}
        engine=make_engine(card,assets['robot_root'],pinned_libero=dict(definitions=definitions,
            asset_paths={n:assets['files'][n]['path']for n in assets['libero_binary_names']}))
        (folder/'NATIVE_SCENE.private.xml').write_text(engine.xml)
        initial=engine.snapshot();images,cal=engine.render(initial)
        if initial['safety_valid']is not True or initial['native_grasp']:raise PermissionError('ORIGINAL_BOOK_NATIVE_INITIAL_SAFETY')
        for name,array in images.items():
            if array.shape!=(256,256,3)or array.dtype!=np.uint8:raise PermissionError('ORIGINAL_BOOK_NATIVE_DUAL_CAMERA')
            Image.fromarray(array).save(folder/('qualification_'+name+'.png'))
            uv=np.asarray(cal[name]['object_points_uv']);optical=np.asarray(cal[name]['T_world_optical'])
            depth=(np.asarray(initial['object_points'])-optical[:3,3])@optical[:3,:3]
            if np.sum(np.all((uv>=0)&(uv<256),axis=1)&(depth[:,2]>0))<4:raise PermissionError('ORIGINAL_BOOK_NATIVE_CAMERA_VIEW')
        recorded+=1
        emit('qualification',dict(actual_state=initial,calibration=cal,camera_pair=True,new_constructor=1,steps=0,
            simulation_time=engine.data.time,original_arm_bounds=engine.arm_bounds,original_finger_q_indices=engine.finger_q,
            original_finger_v_indices=engine.finger_v,robot_geom_ids=[i for i in range(engine.model.ngeom)if int(engine.model.geom_bodyid[i])in engine.robot_bodies],
            arm_geom_ids=[i for i in range(engine.model.ngeom)if(mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,int(engine.model.geom_bodyid[i]))or'').startswith('x5_')],
            obstacle_geom_ids=sorted(engine.obstacles),geom_bindings=engine.geoms))
        home=engine.pose('tool').copy()
        for _ in range(2500):
            condition_guard();engine.step(home,.042/.044);s=engine.snapshot();emit('warmup_tick',dict(native_step=engine.steps,actual_state=s))
            if s['safety_valid']is not True:raise PermissionError('ORIGINAL_BOOK_NATIVE_WARMUP_SAFETY')
        emit('warmup_complete',dict(steps=engine.steps,actual_state=engine.snapshot()))
        result=capture(engine,student,instruction=card['instruction'],guard=condition_guard,emit=emit,duration_s=32.)
        result['model_calls']=student.queries
    except ModelReferenceRejected as exc:
        error=str(exc);gate_rejected=True
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        if error and engine is not None:
            stream.write(json.dumps(dict(kind='terminal_failure',error=error,actual_step=engine.steps,actual_state=engine.snapshot()),default=encode,allow_nan=False)+'\n')
        if engine is not None:engine.close()
        stream.flush();os.fsync(stream.fileno());stream.close()
    terminal=dict(status='RAW_SOURCE_COMPLETE_UNAUDITED'if result else'POLICY_DEV_CONDITION_STOPPED',error=error,result=result,
        actual_integrations=engine.steps if engine else 0,actual_RGB_pairs=getattr(engine,'render_pairs',0)if engine else 0,
        recorded_RGB_pairs=recorded,events=events,model_queries=student.queries,gate_rejected=gate_rejected,
        CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,training_admitted=False,observed_epoch=time.time())
    write(folder/'TERMINAL.json',terminal);a=audit(folder);write(folder/'INDEPENDENT_RAW_AUDIT.json',a)
    valid=valid_model_condition(a,student.completed_generations,gate_rejected)
    hard_safety=a.get('original_native_safety_pass')is False
    return dict(id=condition['id'],xy_offset_m=condition['xy_offset_m'],valid_model_condition=valid,
        success=bool(a['task_success']),gate_rejected=gate_rejected,hard_native_safety=hard_safety,error=error,
        model_queries=student.queries,completed_generations=student.completed_generations,integrations=terminal['actual_integrations'],RGB_pairs=terminal['actual_RGB_pairs'],
        raw_sha256=a['raw_sha256'],audit_sha256=sha(folder/'INDEPENDENT_RAW_AUDIT.json'),terminal_sha256=sha(folder/'TERMINAL.json'))


def run(root,manifest_sha256):
    from scripts.pi0292.durable import read,write,OwnerLock
    from scripts.pi0294.e1.smoke import node_local,put
    from scripts.pi0294.native_dataset import sha,official_import
    from scripts.pi0294.generalization.minimal_model import loader_kwargs,validate_loading
    from scripts.pi0294.generalization.minimal_adapter import ParameterContract
    from scripts.pi0294.e1.model import E1ResidentModel
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    from scripts.pi0294.numerics import MODE,prepare_environment,configure_torch
    root=node_local(root);m=read(root/'RUN_MANIFEST.json');bindings=read(root/'NEURAL_BINDINGS.private.json')
    is_probe=m['kind']=='V7_BOOK_TRAIN16_SAMPLER_DIAGNOSTIC'
    is_recovery_learning=m['kind']=='V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING'
    is_fit=m['kind']in('V7_BOOK_TRAIN16_FIT_DIAGNOSTIC','V7_BOOK5_FINITE_TARGET_LEARNING','V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING')
    is_early_recovery=m['kind']=='V7_BOOK_EARLY_TRAIN_RECOVERY_PANEL'
    is_recovery=m['kind']=='V7_BOOK_TRAIN_RECOVERY_DIAGNOSTIC'or is_early_recovery
    if (sha(root/'RUN_MANIFEST.json')!=manifest_sha256 or m['kind']not in('V7_BOOK_FROZEN_DEV10','V7_BOOK_TRAIN16_SAMPLER_DIAGNOSTIC','V7_BOOK_TRAIN16_FIT_DIAGNOSTIC','V7_BOOK5_FINITE_TARGET_LEARNING','V7_BOOK_TRAIN_RECOVERY_DIAGNOSTIC','V7_BOOK_RECOVERY_FINITE_TARGET_LEARNING','V7_BOOK_EARLY_TRAIN_RECOVERY_PANEL')
            or m['TRAIN_DEV_seal_sha256']!='3d5c4a9ad1a94b6c68f38644e1b9a6dc67f7735806667851049ac250b9cae1cf'):
        raise PermissionError('SEALED_BOOK_DEV10_MANIFEST_AND_FIXED_POPULATION')
    if is_probe:
        from scripts.pi0294.multitask.book_probe import validate_manifest
        validate_manifest(m,read(root/'TRAIN_PROBE_PLAN.json'))
    if is_fit:
        from scripts.pi0294.multitask.book_fit import validate_manifest
        validate_manifest(m,read(root/'TRAIN_FIT_PLAN.json'))
    if is_recovery:
        if is_early_recovery:
            from scripts.pi0294.multitask.book_early_recovery import validate_manifest
        else:
            from scripts.pi0294.multitask.book_recovery import validate_manifest
        validate_manifest(m)
    lease=read(root/'control/LEASE.json')
    if lease['status']!='ACTIVE'or lease['manifest_sha256']!=manifest_sha256:raise PermissionError('BOOK_DEV10_ACTIVE_OWNER_LEASE')
    if not any(r.get('path')==str(root)and r.get('role')==m['role']for r in read(root/'control/REGISTERED_PATHS.json')['entries']):raise PermissionError('BOOK_DEV10_ORIGINAL_REGISTRY')
    for n,h in m['sealed_inputs'].items():
        if sha(root/n)!=h:raise PermissionError('BOOK_DEV10_INPUT_CHANGED:'+n)
    for name in ('source','python','cosmos','dataset','cosmos_hub'):node_local(bindings[name])
    prepare_environment(MODE);official_import(bindings['source'])
    import torch
    # Both environments are existing, immutable user libraries, with identical
    # Python ABI. Native CPU import qualification precedes this registered run.
    sys.path.append(str(node_local(m['physics_packages'])))
    import mujoco
    from transformers import AutoModel
    from scripts.pi0294.multitask.book_checkpoint import load_wrapped
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0292.train_adapter import build_processor
    configure_torch(torch,MODE);torch.set_num_threads(4)
    checkpoint=node_local(m['checkpoint']);complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
    if sha(checkpoint/'CHECKPOINT_COMPLETE.json')!=m['checkpoint_manifest_sha256']or complete['global_step']!=128:raise PermissionError('BOOK_DEV10_EXACT_FIXED128_ENDPOINT')
    for n,h in complete['files'].items():
        if sha(checkpoint/n)!=h:raise PermissionError('BOOK_DEV10_CHECKPOINT_CHANGED:'+n)
    lock=OwnerLock(root.parent/'global-native-GPU-locks'/(m['GPU_uuid']+'.lock'))
    model=None;instance=None;rows=[];loads=0;failure=None;probe=None;fit=None;started=time.monotonic();CPU=time.process_time()
    def guard():
        if time.time()>=m['deadline_epoch']or(root/'STOP.json').exists():raise PermissionError('BOOK_DEV10_FINITE_OWNER_STOP')
        fs=os.statvfs(root)
        if fs.f_bavail*fs.f_frsize<m['protected_remaining_bytes']+m['free_margin_bytes']:raise PermissionError('BOOK_DEV10_PROTECTED_REMAINING_WORK_HEADROOM')
    def stop(sig,frame):raise TimeoutError('BOOK_DEV10_NATIVE_FINITE_END')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
    try:
        guard()
        if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*1024**3:raise PermissionError('BOOK_DEV10_SHARED_GPU_REAL_FREE24GIB')
        put(root/'MODEL_LOAD_INTENT.json',dict(checkpoint_manifest_sha256=m['checkpoint_manifest_sha256'],epoch=time.time()));loads+=1
        model,info=AutoModel.from_pretrained(m['parent_checkpoint'],**loader_kwargs(local_model_name(root,bindings),bindings['cosmos_hub'],COSMOS_REVISION));validate_loading(model,info)
        contract=ParameterContract(model,'E1_ADAPTER_ONLY',10);restore_keys=load_wrapped(model,checkpoint)
        if contract.describe()!=read(checkpoint/'ADAPTER_PARAMETER_CONTRACT.json'):raise PermissionError('BOOK_DEV10_EXACT_WRAPPED_ARCHITECTURE')
        model.to('cuda').eval()
        # Exercise the original Trainer's optimizer/scheduler/RNG restoration
        # before any DEV physics, without train() or optimizer.step().
        from scripts.pi0294.multitask.book_learning import make_trainer_class,recipe,BookSampler
        from scripts.pi0294.native_dataset import NativeDataset,ProcessedCollator
        from transformers import TrainingArguments,TrainerState
        data=NativeDataset(node_local(restoration_dataset(m,bindings)),bindings['source'],bindings['cosmos'],for_optimization=True)
        if is_recovery_learning:
            from scripts.pi0294.multitask.book_fit import recovery_plan
            p=recovery_plan()
            if data.manifest_sha!=p['parent_restore_dataset_manifest_sha256']or data.statistics_sha!=p['parent_restore_statistics_sha256']:
                raise PermissionError('RECOVERY_PARENT_CURSOR_REQUIRES_ORIGINAL_SEALED_POPULATION')
        data.initialize();args=TrainingArguments(output_dir=str(root/'restore_only_no_training'),max_steps=128,
            per_device_train_batch_size=1,gradient_accumulation_steps=32,learning_rate=1e-4,weight_decay=1e-5,
            adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,max_grad_norm=1.,bf16=True,tf32=True,optim='adamw_torch',
            remove_unused_columns=False,eval_strategy='no',save_strategy='no',save_total_limit=None,report_to=[],disable_tqdm=True)
        trainer=make_trainer_class(contract,guard)(model=model,args=args,train_dataset=data,
            data_collator=ProcessedCollator(data.processor),evidence_dir=root/'restore_evidence',recipe=recipe())
        trainer.create_optimizer();trainer.create_scheduler(128)
        saved_optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        trainer.optimizer.load_state_dict(saved_optimizer)
        saved_scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=False)
        trainer.lr_scheduler.load_state_dict(saved_scheduler);trainer._load_rng_state(str(checkpoint))
        trainer.state=TrainerState.load_from_json(str(checkpoint/'trainer_state.json'))
        sampler=read(checkpoint/'SAMPLER_STATE.json')
        if trainer.state.global_step!=128 or sampler['recipe']!=recipe()or sampler['sampler']!=BookSampler(data).state_dict(4096):
            raise PermissionError('BOOK_DEV10_EXACT_FULL_STATE128_AND_SAMPLE_CURSOR4096')
        if not saved_optimizer['state']or any(int(s['step'])!=128 for s in saved_optimizer['state'].values()):
            raise PermissionError('BOOK_DEV10_ACTUAL_ADAMW_UPDATE128_STATE')
        if trainer.lr_scheduler.state_dict()!=saved_scheduler:raise PermissionError('BOOK_DEV10_ACTUAL_SCHEDULER_RESTORE')
        write(root/'ACTUAL_FULL_CHECKPOINT_RESTORE.json',dict(checkpoint_manifest_sha256=m['checkpoint_manifest_sha256'],
            model_restore=restore_keys,optimizer_state_entries=len(saved_optimizer['state']),optimizer_steps_restored=128,
            scheduler_restored=True,RNG_restored=True,sampler_cursor=4096,new_optimizer_updates=0,DEV_started=False,observed_epoch=time.time()))
        if m.get('delta_checkpoint'):
            if is_probe or(is_fit and not is_recovery_learning):raise PermissionError('DELTA_INFERENCE_ONLY_ON_NEW_FIXED_DEV_PANEL')
            producer=node_local(m['delta_producer_root']);certificate=producer/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json'
            from scripts.pi0294.multitask.book_checkpoint import validate_delta_header,load_parent_delta
            delta=node_local(m['delta_checkpoint'])
            delta_complete=validate_delta_header(read(delta/'CHECKPOINT_COMPLETE.json'),m['checkpoint_manifest_sha256'])
            delta_steps=delta_complete['global_step'];delta_cursor=delta_complete['cursor']
            if (read(producer/'control/LEASE.json')['status']!='CLOSED'
                    or sha(producer/'control/LEASE.json')!=m['delta_producer_closed_lease_sha256']
                    or delta!=producer/f'training/checkpoint-{delta_steps}'
                    or sha(certificate)!=m['delta_native_qualification_sha256']):raise PermissionError('BOOK_DELTA_PRODUCER_CLOSED_AND_NATIVE_RESTORE_QUALIFIED')
            qualification=read(certificate)
            if(qualification['checkpoint_manifest_sha256']!=m['delta_checkpoint_manifest_sha256']
                    or qualification['required_parent_checkpoint_manifest_sha256']!=m['checkpoint_manifest_sha256']
                    or not all(qualification[k]for k in('model_reconstruction','scheduler_restored','RNG_restored','frozen_parent_unchanged'))
                    or qualification['optimizer_updates']!=delta_steps or qualification['sampler_cursor']!=delta_cursor):raise PermissionError('BOOK_DELTA_ACTUAL_FULL_STATE_QUALIFICATION')
            reconstruction=load_parent_delta(model,contract,delta,
                manifest_sha256=m['delta_checkpoint_manifest_sha256'],parent_sha256=m['checkpoint_manifest_sha256'])
            write(root/'ACTUAL_PARENT_DELTA_INFERENCE_RECONSTRUCTION.json',dict(**reconstruction,
                producer_native_qualification_sha256=m['delta_native_qualification_sha256'],observed_epoch=time.time()))
        data=saved_optimizer=None
        if not is_probe:trainer=None
        gc.collect();torch.cuda.empty_cache();model.eval().requires_grad_(False)
        instance=E1ResidentModel.__new__(E1ResidentModel);instance.root=root;instance.bindings=bindings;instance.model=model
        statistics=node_local(generation_statistics(m,bindings))
        if is_recovery_learning and sha(statistics)!=recovery_plan()['statistics_sha256']:
            raise PermissionError('RECOVERY_GENERATION_EXACT_NEW_TRAIN_STATISTICS')
        instance.processor=build_processor(bindings['source'],bindings['cosmos'],read(statistics))
        instance.identity=dict(kind='BOOK5_E1_WARMSTART_FIXED128',global_step=128,checkpoint_manifest_sha256=m['checkpoint_manifest_sha256'],statistics_sha256=sha(statistics))
        if m.get('delta_checkpoint'):
            instance.identity=dict(kind='BOOK_PARENT_BOUND_TARGET_DELTA',
                new_optimizer_updates=delta_steps,parent_book_updates=128,
                checkpoint_manifest_sha256=m['delta_checkpoint_manifest_sha256'],parent_checkpoint_manifest_sha256=m['checkpoint_manifest_sha256'],
                statistics_sha256=sha(statistics),
                training_population=delta_complete.get('recipe',{}).get('optimization_population','SIXTEEN_FIXED_TRAIN_WINDOWS_NOT_FORMAL_E2'),
                generation_statistics_changed_from_initial_delta=sha(statistics)!=sha(delta/'statistics.json'))
        original_generate=instance.generate
        def timed_generate(request,seed):
            signal.alarm(max(1,min(60,int(m['deadline_epoch']-time.time()))))
            try:return original_generate(request,seed)
            finally:signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
        instance.generate=timed_generate
        write(root/'MODEL_LOADED.json',dict(identity=instance.identity,observed_epoch=time.time(),mujoco_origin=mujoco.__file__))
        if is_fit:
            from scripts.pi0294.multitask.book_fit import run as run_fit
            fit=run_fit(root,instance,contract,guard);write(root/'TRAIN_FIT_RESULT.json',fit);return
        if is_probe:
            from scripts.pi0294.multitask.book_probe import run as run_probe
            probe=run_probe(root,instance,trainer,guard);write(root/'TRAIN_PROBE_RESULT.json',probe);return
        template=node_local(m['source_template'])
        if sha(template/'RUN_MANIFEST.json')!=m['source_template_manifest_sha256']or read(template/'control/NATIVE_SOURCE_LEASE.json')['status']!='CLOSED':raise PermissionError('BOOK_DEV10_IMMUTABLE_CLOSED_SOURCE_TEMPLATE')
        assets=read(template/'ASSET_BINDINGS.private.json')
        for pin in assets['files'].values():
            if sha(node_local(pin['path']))!=pin['sha256']:raise PermissionError('BOOK_DEV10_ACTUAL_SOURCE_ASSET_CHANGED')
        seal=read(root/'TRAIN_DEV_SEAL.json')
        if sha(root/'TRAIN_DEV_SEAL.json')!=m['TRAIN_DEV_seal_sha256']:raise PermissionError('BOOK_DEV10_FIXED_POPULATION')
        conditions=seal['DEV'];episode_options={}
        if is_recovery:
            if is_early_recovery:
                from scripts.pi0294.multitask import book_early_recovery as recovery
                conditions=recovery.conditions()
            else:
                from scripts.pi0294.multitask import book_recovery as recovery
                conditions=[recovery.CONDITION.copy()]
            episode_options=dict(card_factory=recovery.recovery_card,
                student_factory=lambda *args:recovery.make_student(Student,*args))
        for condition in conditions:
            guard();row=episode(root,condition,instance,template,guard,**episode_options);rows.append(row)
            write(root/'PROGRESS.json',dict(conditions=rows,valid=sum(r['valid_model_condition']for r in rows),successes=sum(r['success']for r in rows),observed_epoch=time.time()))
            if row['hard_native_safety']or not row['valid_model_condition']:
                write(root/'STOP.json',dict(reason='ORIGINAL_NATIVE_SAFETY_OR_ENGINEERING_FAILURE',condition=row['id'],automatic_retry=False));break
    except BaseException as error:
        failure=type(error).__name__+': '+str(error);raise
    finally:
        signal.alarm(0)
        if is_fit:
            progress=read(root/'FIT_PROGRESS.json')if(root/'FIT_PROGRESS.json').exists()else dict(step=0,cursor=0)
            status=('COMPLETE_TRAIN16_FIT_DIAGNOSTIC'if m['kind']=='V7_BOOK_TRAIN16_FIT_DIAGNOSTIC'else'COMPLETE_BOOK5_FINITE_TARGET_LEARNING')if fit is not None else'FAILED_BOOK_FINITE_OPTIMIZATION'
            write(root/'RESULT.json',dict(status=status,error=failure,scientific_kind=m['kind'],
                model_load_attempts=loads,generation_attempts=len(list(root.glob('*/*.INTENT.json'))),
                completed_generations=len(list(root.glob('*/*.RESULT.json'))),optimizer_updates=progress['step'],TRAIN_visits=progress['cursor'],
                physics_integrations=0,DEV_conditions=0,CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,completed_epoch=time.time()))
        elif is_probe:
            write(root/'RESULT.json',dict(status='COMPLETE_TRAIN16_DIAGNOSTIC'if probe is not None else'FAILED_TRAIN16_DIAGNOSTIC',error=failure,
                model_load_attempts=loads,generation_attempts=len(list(root.glob('draw-*.INTENT.json'))),
                completed_generations=len(list(root.glob('draw-*.RESULT.json'))),native_loss_forward_attempts=len(list(root.glob('*-LOSS.INTENT.json'))),
                optimizer_updates=0,physics_integrations=0,DEV_conditions=0,CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,completed_epoch=time.time()))
        else:write(root/'RESULT.json',dict(status=(('COMPLETE_EARLY_TRAIN_RECOVERY_PANEL'if len(rows)==10 else'STOPPED_PARTIAL_EARLY_TRAIN_RECOVERY_PANEL')if is_early_recovery else('COMPLETE_TRAIN_RECOVERY_DIAGNOSTIC'if len(rows)==1 else'FAILED_TRAIN_RECOVERY_DIAGNOSTIC'))if is_recovery else('COMPLETE_FROZEN_DEV10'if len(rows)==10 else'STOPPED_PARTIAL_DEV10'),error=failure,
            scientific_kind=m['kind'],formal_DEV_conditions=0 if is_recovery else sum(r['valid_model_condition']for r in rows),
            student_prefix_is_expert=False if is_recovery else None,learned_full_task_success_claim=False if is_recovery else None,
            evaluated_checkpoint_manifest_sha256=m.get('delta_checkpoint_manifest_sha256',m['checkpoint_manifest_sha256']),
            checkpoint_requires_retained_full_parent=bool(m.get('delta_checkpoint')),
            conditions=rows,valid_conditions=sum(r['valid_model_condition']for r in rows),successes=sum(r['success']for r in rows),
            model_load_attempts=loads,model_queries=sum(r['model_queries']for r in rows),integrations=sum(r['integrations']for r in rows),
            RGB_pairs=sum(r['RGB_pairs']for r in rows),optimizer_updates=0,training_data_admission=False,
            CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,completed_epoch=time.time()))
        model=instance=None;gc.collect();lock.close()


if __name__=='__main__':run(entry.root,entry.manifest_sha256)
