"""Independent official-parent A500 TRAIN4 delivery, using original native consumers."""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .native_camera_transfer_v1 import CAMERA_ID
from .stable_task_scene_v1 import pinned
from .storage import digest,write_json
from .native_arx5_diagnostic_v1 import ORDER
from .native_arx5_training_profile_v1 import PARENT_ID
PROTOCOL='A_TARGET500_TRAIN4_DELIVERY_DIAGNOSTIC_V1'
SCHEMA='A_TARGET500_TRAIN4_DELIVERY_PANEL_V1'
INPUT_SCHEMA='A_TARGET500_TRAIN4_DELIVERY_GROUP_INPUT_V1'
NATIVE_SCHEMA='A_TARGET500_TRAIN4_DELIVERY_NATIVE_ACTUAL_V1'
EXPERIMENT='A_TARGET500_FIXED_TRAIN4_DELIVERY_NOT_AB_DEV_TEST_V1'
PURPOSE='FIXED_A_TARGET500_TRAIN_DELIVERY_NOT_DEV_TEST'
METHODS=('A_TARGET500_PLUS_ENDPOINT_BASIS',)
CONTRACT_SHA='eb180f64101cf3fd2f2ee21573f13c52ebc95c171ddfa5ca3b1e52a2b7215951'
_VERIFIED_PACKAGES={}

def protocol():
    return dict(task.protocol(task.CORE_LOCAL_ROLE),stage=PROTOCOL,
        scene_family='TRAIN01_TRAIN07_TRAIN13_TRAIN25_NATIVE_BODY_DIAGNOSTIC',
        observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST',
        policy='OFFICIAL_PARENT_A_TARGET500_PLUS_FIXED_CURRENT_STATE_LAST_KNOT_BASIS',
        raw_GR00T_not_promoted_by_decoding=True)

def official_files():
    from scripts.pi0294.multitask.raw_parent_delivery import FILES
    return {n:v[0]for n,v in FILES.items()}


def qualification(q,checkpoint_files=None,*,verify_package=True):
    from .native_arx5_training_profile_v1 import recipe,REVISION
    task.need(q.get('schema')=='A_TARGET500_NATIVE_DELIVERY_ADMISSION_V1'
        and q.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and q.get('original_cross_hardware_REJECT_preserved')is True
        and q.get('original_DEV_TEST_admission')is False,'A500_INDEPENDENT_TRAIN_DELIVERY_ADMISSION')
    registration=pinned(q['preregistration'])
    from scripts.pi0292.relay_protocol import digest as object_digest
    task.need(registration['protocol']==PROTOCOL and registration['global_step']==500
        and registration['score_contract']==task.score_contract()
        and registration['score_contract_sha256']==object_digest(task.score_contract())
        and [(r['layout'],r['query_seed_base'])for r in registration['conditions']]==list(ORDER)
        and registration['requested']==4 and registration['maximum_queries']==60
        and registration['clipping']is registration['Teacher_fallback']is False,'A500_FIXED_FOUR_SCORE_SEEDS_NO_FALLBACK')
    camera=closed(q['camera_worker']);capture=closed(q['saved_packet_binding'])
    task.need(camera['camera_coverage_passes']and camera['actual_Processor_calls']==1500
        and camera['final_coverage']['supervised']==1450 and camera['camera_id']==CAMERA_ID
        and capture['passes']and all(r['body_equals_offline']and r['wrist_equals_retained_native']
            and r['bound_shadow_roundtrip']for r in capture['records']),'A500_REUSE_EXISTING_REAL_CAMERA_BOUNDARY')
    admission=pinned(q['training_admission'])
    task.need(admission['passes']and admission['parent_files_identity']==PARENT_ID
        and admission['actual_non_candidate_optimizer_updates']==10,'A500_REAL_NEW_PARENT_TRAINING_QUALIFICATION')
    stages=[closed(p)['science']for p in q['candidate_stages']]
    task.need(len(stages)==2 and [v['global_step']for v in stages]==[250,500]
        and [v['candidate_optimizer_updates']for v in stages]==[250,250]
        and all(v['qualification_only']is False and v['stage']=='A_TARGET'
                and v['parent_files_identity']==PARENT_ID for v in stages)
        and stages[1]['resume']==stages[0]['checkpoint'],'A500_ACTUAL_FRESH250_PLUS_RESUMED250')
    rgb=closed(q['ordinary_RGB_reconstruction'])
    task.need(rgb['schema']=='NATIVE_ARX5_ORDINARY_RGB_ENDPOINT_RECONSTRUCTION_ACTUAL_V1'
        and rgb['passes']and rgb['global_step']==500 and rgb['normalized_and_physical_H40_bit_exact']
        and rgb['original_graph_without_wrappers']and rgb['training_feature_cache_used']is False
        and rgb['package']==q['package']and rgb['checkpoint']==stages[1]['checkpoint'],
        'A500_SAVED_RGB_PROCESSOR_FULL_H40_RECONSTRUCTION')
    package=Path(q['package']['path']);manifest=pinned(dict(path=str(package/'MANIFEST.private.json'),sha256=q['package']['sha256']))
    task.need(manifest['schema']=='NATIVE_ARX5_OFFICIAL_ENDPOINT_INFERENCE_PACKAGE_V1'
        and manifest['global_step']==500 and manifest['parent_files_identity']==PARENT_ID
        and manifest['revision']==REVISION and manifest['ordinary_original_RGB_reconstruction_exact']
        and manifest['recipe']==recipe('A_TARGET',admission['dataset_sha256'],admission['statistics_sha256']),
        'A500_NOT_SFT2000_QUAL4_OR_250_PACKAGE')
    if verify_package:
        def stats():
            return {n:tuple(getattr((package/n).stat(),k)for k in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
                    for n in manifest['files']}
        key=(str(package),q['package']['sha256']);before=stats()
        if _VERIFIED_PACKAGES.get(key)!=before:
            task.need(all((package/n).resolve().is_relative_to(package.resolve())and digest(package/n)==sha
                          for n,sha in manifest['files'].items()),'A500_ALL_IMMUTABLE_PACKAGE_MEMBERS')
            task.need(stats()==before,'A500_PACKAGE_UNCHANGED_DURING_VALIDATION');_VERIFIED_PACKAGES[key]=before
    return q


def policy_contract(value,checkpoint_files=None,*,verify_package=True,expected_global_step=500):
    task.need(set(value)=={'arm','parent_files_identity','package','Native_body_qualification'}
        and value['arm']==METHODS[0]and value['parent_files_identity']==PARENT_ID
        and expected_global_step==500,'A500_EXPLICIT_NEW_OFFICIAL_ENDPOINT_POLICY')
    q=pinned(value['Native_body_qualification'])
    task.need(q['package']==value['package'],'A500_SAME_ENDPOINT_PACKAGE')
    return qualification(q,checkpoint_files,verify_package=verify_package)


def memory_contract(s):
    q=policy_contract(s['target_gr00t_policy'],verify_package=False)
    proof=closed(q['ordinary_RGB_reconstruction'])
    task.need(s.get('memory_execution')=='NATIVE_ARX5_OFFICIAL_ORDINARY_RGB_CORE_V1'
        and s.get('memory_execution_reference')==q['ordinary_RGB_reconstruction']
        and proof['max_allocated_bytes']<=s['allocator_limit_bytes']==q['allocator_limit_bytes']
        and s['official_parent_audit']==q['official_parent_audit'],'A500_OWN_ACTUAL_ORDINARY_MEMORY_PROOF')
    return q


def load(s,root):
    import torch
    from . import native_arx5_training_runtime_v1 as runtime
    from .native_arx5_endpoint_v1 import install,load_processor
    from .fusion_paused_query_bridge import _live
    from scripts.pi0294.generalization.minimal_adapter import consume_for_official_inference
    from scripts.pi0294.e1.model import E1ResidentModel
    producer=pinned(s['producer_native_spec']);root=Path(root);bindings=s['neural_bindings']
    alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True,exist_ok=True)
    alias.symlink_to(bindings['cosmos'],target_is_directory=True)
    torch.cuda.set_per_process_memory_fraction(s['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory)
    spec=dict(s,bindings=bindings,official_parent=s['checkpoint'])
    model,contract,_=runtime.load_official(spec,root,lambda:_live(s,producer))
    package=Path(s['target_gr00t_policy']['package']['path'])
    install(torch.load(package/'trainable.pt',map_location='cpu',weights_only=True),model,contract)
    contract._training_session_used=True;model=consume_for_official_inference(contract)
    task.need(sum(p.numel()*p.element_size()for p in model.parameters())==s['checkpoint_parameter_bytes'],
              'A500_ACTUAL_ORDINARY_PARAMETER_BYTES')
    model.to('cuda').eval();instance=E1ResidentModel.__new__(E1ResidentModel)
    instance.root=root;instance.bindings=bindings;instance.model=model
    instance.processor=load_processor(package/'processor',spec,root)
    instance.identity=dict(policy=s['target_gr00t_policy'],camera=CAMERA_ID,numeric_mode=s['numeric_mode'],
        official_parent_files_identity=PARENT_ID,global_step=500,representation='CURRENT_STATE_PREDICTED_LAST_KNOT_QUINTIC')
    write_json(root/'A_TARGET500_POLICY_LOADED.private.json',dict(identity=instance.identity,
        actual_model_loads=1,ordinary_original_GR00T=True,optimizer_updates=0))
    return instance,torch


def closed(pin):
    value = pinned(pin)
    root = Path(pin['path']).parent
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    task.need(lease['status']=='CLOSED' and lease['worker_exit_code']==worker['exit_code']==0
              and lease['spec_sha256']==digest(root/'EXECUTION_SPEC.json'), 'BODY_REAL_CLOSED0_QUALIFICATION')
    return value

def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    rows = panel.get('assignments') or []
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==PROTOCOL
        and panel.get('protocol')==protocol() and panel.get('B1_global_step')==500
        and panel.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and set(panel.get('policies',{}))==set(METHODS)
        and len(rows)==len({r['native_run'] for r in rows})==4, 'BODY_FIXED_INDEPENDENT_TRAIN4')
    for row,(layout,seed) in zip(rows,ORDER):
        task.need(row['layout']==layout and row['method']==METHODS[0] and row['parent']==scene_id(layout)
            and row['restart']==1 and row['split']=='TRAIN' and row['purpose']==PURPOSE
            and row['command']==task.COMMAND and row['query_seeds']==list(range(seed,seed+15))
            and Path(row['native_run']).name==row['native_run'], 'BODY_NO_RETRY_LAYOUT_OR_NOISE_CHANGE')
    return rows

def selection(inp, panel):
    rows = validate_panel(panel)
    task.need(inp.get('schema')==INPUT_SCHEMA and inp.get('source_sha')==panel['source_sha']
        and inp.get('assignment_slice')==[0,4] and inp.get('population_predecessor') is None,
        'BODY_ONE_FIXED_FOUR_NO_PREFIX_SELECTION')
    return rows

def summarize(panel, outcomes):
    rows = validate_panel(panel)
    task.need(len(outcomes)<=4 and len({r['native_run'] for r in outcomes})==len(outcomes)
        and all(all(r[k]==rows[i][k] for k in ('native_run','layout','method','restart','parent','split','query_seeds'))
                for i,r in enumerate(outcomes)), 'BODY_EVERY_ATTEMPT_ONCE_IN_FIXED_ORDER')
    values = outcomes
    return dict(schema='A_TARGET500_TRAIN4_DELIVERY_SUMMARY_V1', requested=4, completed=len(values),
        complete=len(values)==4, methods={METHODS[0]:dict(requested=4,
            attempted=sum(r['attempted'] for r in values),
            task_successes=sum(r['technical_valid'] and r['task_success'] for r in values),
            valid_task_failures=sum(r['technical_valid'] and not r['task_success'] for r in values),
            technical_failures=sum(not r['technical_valid'] for r in values),
            hard_safety_stop_conditions=sum(r['hard_safety_stop'] for r in values),
            actual_GR00T_queries=sum(r.get('actual_GR00T_queries',0) for r in values),
            actual_native_integrations=sum(r.get('actual_native_integrations',0) for r in values))},
        original_DEV_conditions=0, TEST_conditions=0, qualified_training_demonstrations=0,
        model_updates=0, generalization_claim=False, original_cross_hardware_REJECT_replaced=False)

def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry, scene_id, BASE_SCENE
    from .task_learning_local_policy_v1 import scene_builder_custody
    proof = pinned(spec['task_learning_scene_reference'])
    layout = proof['layout']
    panel = pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    builder = scene_builder_custody(panel['registration']['scene_producers'][layout],
        spec['task_learning_scene_reference'],spec['model_path'])
    task.need(layout in {r[0] for r in ORDER} and spec.get('core_local_protocol')==PROTOCOL
        and spec.get('task_learning_role')==PROTOCOL and spec.get('data_role')=='TRAIN'
        and proof['schema']=='TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1'
        and proof['status']=='STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
        and proof['builder_sha256']==builder and proof['geometry']==geometry(layout)
        and proof['parent']==scene_id(layout)==spec['scene_id']==spec['data_parent']
        and proof['split']=='TRAIN' and proof['saved_startup_pose_checks']==5703
        and proof['robot_contract']['status']=='COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'
        and proof['compiled']['base_scene_sha256']==BASE_SCENE
        and proof['compiled']['fresh_XML_compilation'] is True
        and proof['compiled']['original_robot_contract_exact'] is True
        and proof['compiled']['scene_sha256']==digest(spec['model_path'])
        and proof['physics_integrations']==proof['new_transitions']==0
        and proof['native_execution_authorized'] is False and proof['WBC_changes'] is False
        and proof['physics_dt_s']==.002, 'BODY_SAME_SEALED_WHOLE_ROBOT_SCENE_AND_ORIGINAL_SAFETY')
    matches = [r for r in validate_panel(panel) if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1, 'BODY_ONE_PREREGISTERED_NATIVE_REQUEST')
    row = matches[0]
    task.need(row['model_sha256']==digest(spec['model_path'])
        and spec.get('target_gr00t_policy')==panel['policies'][row['method']]
        and spec.get('target_gr00t_query_seeds')==row['query_seeds']
        and spec.get('target_gr00t_pair_id')==row['native_run']
        and spec.get('observation_camera_profile')==CAMERA_ID
        and spec.get('task_learning_teacher_mode') is None and spec.get('task_learning_control') is None,
        'BODY_EXACT_NEW_OBSERVATION_WITHOUT_TEACHER_FALLBACK')
    return proof

def validate_live(s, *, verify_model=True):
    from scripts.pi0292.relay_protocol import digest as document_hash
    from scripts.pi0294.numerics import MODE
    from .native_arx5_training_profile_v1 import PARENT_ID as PARENT
    from .task_learning_live_pair_v1 import PROFILE, KIND
    task.need(s.get('core_local_protocol')==PROTOCOL and s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL' and s.get('physics') is False
        and s.get('device')=='cuda' and s.get('max_queries')==15
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed') is False
        and s.get('checkpoint_manifest_sha256')==PARENT and s.get('task_learning_grasp')==protocol()
        and s['checkpoint_files']==official_files(),
        'A500_ORIGINAL_NUMERICS_AND_OFFICIAL_PARENT_INVENTORY')
    descriptor = s['producer_native_spec']
    producer = pinned(descriptor)
    task.declared(producer)
    task.need(Path(descriptor['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('task_learning_role')==PROTOCOL and producer.get('core_local_protocol')==PROTOCOL
        and producer.get('target_gr00t_gpu_root')==s['registered_root'] and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and producer.get('target_gr00t_pair_id')==s.get('target_gr00t_pair_id')
        and producer.get('target_gr00t_B1_global_step')==s.get('target_gr00t_B1_global_step')==500
        and s.get('observation_camera_profile')==producer.get('observation_camera_profile')==CAMERA_ID,
        'BODY_EXACT_NEW_NATIVE_NEURAL_PAIR')
    native_scene_admission(producer)
    q = policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model)
    task.need(s['GPU_uuid']==q['GPU_uuid'] and s['allocator_limit_bytes']==q['allocator_limit_bytes'],
              'BODY_SAME_CARD_AND_ACTUALLY_QUALIFIED_MEMORY_LIMIT')
    memory_contract(s)
    return producer

def register_panel(request):
    from contextlib import closing
    import time
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .task_learning_student_registration_v2 import closed_base,actual_site,descriptor
    from .task_learning_scene_v1 import scene_id
    from .task_learning_local_policy_v1 import scene_builder_custody
    from .native_arx5_training_profile_v1 import PARENT_ID as PARENT
    n = closed_base(request['native_base'])
    g = closed_base(request['gpu_base'])
    campaign = Path(n['campaign_root'])
    actual_site(campaign)
    root,source = Path(request['root']),Path(request['source_root'])
    task.need(root.parent==Path(n['registered_root']).parents[2] and source.resolve()==source,
              'BODY_NEW_PANEL_UNDER_ORIGINAL_QUALIFIED_PROJECT')
    task.need(all(digest(source/name)==pin for name,pin in request['source_files'].items()),
              'BODY_SEALED_NEW_RUNTIME_SOURCE')
    admission = pinned(request['Native_body_qualification'])
    policy = dict(arm=METHODS[0],parent_files_identity=PARENT,package=admission['package'],
                  Native_body_qualification=request['Native_body_qualification'])
    policy_contract(policy,g['checkpoint_files'])
    task.need(set(request['scenes'])=={r[0] for r in ORDER},'BODY_EXACT_FOUR_TRAIN_SCENES')
    rows = []
    for layout,seed in ORDER:
        scene = request['scenes'][layout]
        proof = pinned(scene)
        model = Path(scene['path']).parent/'SCENE.private.mjb'
        scene_builder_custody(request['scene_producers'][layout],scene,str(model))
        task.need(proof['layout']==layout and proof['parent']==scene_id(layout)
            and proof['split']=='TRAIN' and proof['compiled']['scene_sha256']==digest(model),
            'BODY_ORIGINAL_TRAIN_LAYOUT_MODEL')
        rows.append(dict(native_run='native-a500-'+layout.lower()+'-r1',layout=layout,
            method=METHODS[0],parent=scene_id(layout),restart=1,split='TRAIN',purpose=PURPOSE,
            command=task.COMMAND,model_sha256=digest(model),query_seeds=list(range(seed,seed+15))))
    panel = dict(schema=SCHEMA,task_learning_role=PROTOCOL,protocol=protocol(),source_sha=request['source_sha'],
        B1_global_step=500,training_seed=20261111,policies={METHODS[0]:policy},assignments=rows,
        preregistration=admission['preregistration'],
        registration={k:request[k] for k in ('source_root','source_files','native_base','gpu_base','scenes',
            'scene_producers','scene_allowlist','scene_allowlist_sha256','memory_execution','memory_execution_reference')})
    validate_panel(panel)
    key=root.name.upper()+'-PANEL'
    journal=Journal(campaign/'control/journal',campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists() and not journal.path(key).exists(),'BODY_UNKNOWN_PANEL_NEVER_RECREATE')
        def effect():
            root.mkdir()
            (root/'control').mkdir()
            write(root/'PANEL.private.json',panel)
            write(root/'control/LEASE.json',dict(status='SEALED_NONCOMPUTE_PANEL',planned_conditions=4,
                panel_sha256=digest(root/'PANEL.private.json'),created_epoch=time.time()))
            registry,pin=snapshot(n['registry'])
            task.need(not any(r.get('path')==str(root) for r in registry['entries']),'BODY_FRESH_PANEL_REGISTRY')
            registry['entries'].append(dict(path=str(root),role=key,scope=SCHEMA,owner=n['owner'],
                source_sha=request['source_sha'],local_lease=str(root/'control/LEASE.json'),executor_node=n['host']))
            once(n['registry'],registry,key+'-REGISTRY',expected_before_sha256=pin)
            return dict(status='A_TARGET500_TRAIN4_PANEL_SEALED_NO_WORKERS',panel=descriptor(root/'PANEL.private.json'))
        return journal.once(key,dict(admission=request['Native_body_qualification'],source_sha=request['source_sha']),effect)
