"""Independent four-TRAIN native-body SFT2000 diagnostic admission.

Reuses original serial workers, scoring, endpoint basis and consumer. No old
cross-hardware gate, DEV, TEST or eight-method population is redefined here.
"""
import json
from pathlib import Path

from . import task_learning_grasp_v1 as task
from .native_camera_transfer_v1 import CAMERA_ID
from .stable_task_scene_v1 import pinned
from .storage import digest, write_json

PROTOCOL = 'NATIVE_BODY_SFT2000_TRAIN4_DIAGNOSTIC_V1'
SCHEMA = 'NATIVE_BODY_SFT2000_TRAIN4_PANEL_V1'
INPUT_SCHEMA = 'NATIVE_BODY_SFT2000_TRAIN4_GROUP_INPUT_V1'
NATIVE_SCHEMA = 'NATIVE_BODY_SFT2000_TRAIN4_NATIVE_ACTUAL_V1'
EXPERIMENT = 'NATIVE_BODY_SFT2000_TRAIN4_NOT_AB_DEV_TEST_V1'
PURPOSE = 'NEW_CAMERA_TRAIN_DIAGNOSTIC_NOT_BC_DEV_TEST'
METHODS = ('SFT2000_PLUS_ENDPOINT_BASIS',)
ORDER = (('TRAIN01',202611110100), ('TRAIN07',202611110700),
         ('TRAIN13',202611111300), ('TRAIN25',202611112500))
CONTRACT_SHA = 'e1fa1d744497a5f961829df2efe7a10ce1f79fe3192417c1c51e7e46e774670c'
PACKAGE_SHA = '7feffa8dd11b5c9ce963f2e0f2f2613196c1a3ae8e5039056377bd5e25e02250'
_VERIFIED_PACKAGES = {}


def protocol():
    return dict(task.protocol(task.CORE_LOCAL_ROLE), stage=PROTOCOL,
        scene_family='TRAIN01_TRAIN07_TRAIN13_TRAIN25_NATIVE_BODY_DIAGNOSTIC',
        observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST',
        policy='FROZEN_SFT2000_PLUS_FIXED_CURRENT_STATE_LAST_KNOT_BASIS',
        raw_GR00T_not_promoted_by_decoding=True)


def closed(pin):
    value = pinned(pin)
    root = Path(pin['path']).parent
    lease = json.loads((root/'control/LEASE.json').read_text())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_text())
    task.need(lease['status']=='CLOSED' and lease['worker_exit_code']==worker['exit_code']==0
              and lease['spec_sha256']==digest(root/'EXECUTION_SPEC.json'), 'BODY_REAL_CLOSED0_QUALIFICATION')
    return value


def qualification(q, checkpoint_files, *, verify_package=True):
    task.need(q.get('schema')=='NATIVE_BODY_SFT2000_DIAGNOSTIC_ADMISSION_V1'
        and q.get('preregistration',{}).get('sha256')==CONTRACT_SHA
        and q.get('package',{}).get('sha256')==PACKAGE_SHA
        and q.get('original_cross_hardware_REJECT_preserved') is True
        and q.get('original_DEV_TEST_admission') is False, 'BODY_SEPARATE_TRAIN4_ADMISSION')
    contract = pinned(q['preregistration'])
    task.need(contract['requested']==4 and [(r['layout'],r['query_seed_base']) for r in contract['conditions']]==list(ORDER),
              'BODY_PREREGISTERED_FOUR_ONLY')
    camera = closed(q['camera_worker'])
    task.need(camera.get('camera_coverage_passes') is True and camera['actual_Processor_calls']==1500
        and camera['final_coverage']['supervised']==1450 and camera['camera_id']==CAMERA_ID,
        'BODY_FULL_ACTUAL_FINAL_PROCESSOR_COVERAGE')
    capture = closed(q['saved_packet_binding'])
    task.need(capture.get('passes') is True and [r['layout'] for r in capture['records']]==[r[0] for r in ORDER]
        and all(r['body_equals_offline'] and r['wrist_equals_retained_native'] and r['bound_shadow_roundtrip']
                for r in capture['records']) and capture['live_native_episodes']==0,
        'BODY_REAL_MESA_PACKET_BINDING_NOT_NATIVE_TASK_RESULT')
    replay = closed(q['same_card_replay'])
    task.need(replay.get('schema')=='NATIVE_BODY_SFT2000_SAME_CARD_REPLAY_V1'
        and replay['package']==q['package'] and replay['GPU_uuid']==q['GPU_uuid']
        and replay['dataset_sha256']==camera['dataset_manifest_sha256']
        and replay['same_card_bit_exact'] and replay['ordinary_full_inference']
        and replay['actual_full_model_generations']==len(replay['records'])==16
        and replay['actual_Processor_decodes']==16 and replay['optimizer_updates']==replay['physics_integrations']==0,
        'BODY_ACTUAL_SFT2000_OWN_PROCESSOR_SAME_CARD_REPLAY')
    if verify_package:
        from .target_gr00t_delivery_v1 import verify
        root = Path(q['package']['path'])
        manifest = pinned(dict(path=str(root/'PACKAGE_MANIFEST.json'),sha256=q['package']['sha256']))
        def identities():
            result = {}
            for name in manifest['files']:
                s = (root/name).stat()
                result[name] = (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            return result
        # Full validation at this process's transfer/version boundary. Repeated
        # query admission checks only unchanged immutable member identities;
        # any change triggers the original full hash/contract verification.
        key = (str(root),q['package']['sha256'],tuple(sorted(checkpoint_files.items())))
        before = identities()
        if _VERIFIED_PACKAGES.get(key) != before:
            manifest = verify(root,q['package']['sha256'],checkpoint_files)
            task.need(identities()==before,'BODY_PACKAGE_CHANGED_DURING_VALIDATION')
            _VERIFIED_PACKAGES[key] = before
        task.need(manifest['global_step']==2000, 'BODY_SFT2000_FIXED_ENDPOINT')
    return q


def policy_contract(value, checkpoint_files, *, verify_package=True, expected_global_step=2000):
    from .target_gr00t_profile_v1 import PARENT
    task.need(set(value)=={'arm','parent_manifest_sha256','package','Native_body_qualification'}
        and value['arm']==METHODS[0] and value['parent_manifest_sha256']==PARENT
        and expected_global_step==2000 and value['package']['sha256']==PACKAGE_SHA,
        'BODY_FROZEN_SFT2000_PLUS_BASIS_ONLY')
    q = pinned(value['Native_body_qualification'])
    task.need(q['package']==value['package'], 'BODY_SAME_FIXED_MODEL_PACKAGE')
    return qualification(q, checkpoint_files, verify_package=verify_package)


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    rows = panel.get('assignments') or []
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==PROTOCOL
        and panel.get('protocol')==protocol() and panel.get('B1_global_step')==2000
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
    return dict(schema='NATIVE_BODY_SFT2000_TRAIN4_SUMMARY_V1', requested=4, completed=len(values),
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
    from .fusion_paused_inference import CHECKPOINT_INVENTORY
    from .target_gr00t_profile_v1 import PARENT
    from .task_learning_live_pair_v1 import memory_contract, PROFILE, KIND
    task.need(s.get('core_local_protocol')==PROTOCOL and s.get('profile')==PROFILE and s.get('kind')==KIND
        and s.get('execution_site')=='CORE_FUSION_NEURAL' and s.get('physics') is False
        and s.get('device')=='cuda' and s.get('max_queries')==15
        and s.get('GR00T_updates')==s.get('WM_updates')==s.get('Actor_updates')==0
        and s.get('numeric_mode')==MODE and s.get('baseline_changed') is False
        and s.get('checkpoint_manifest_sha256')==PARENT and s.get('task_learning_grasp')==protocol()
        and document_hash(s['checkpoint_files'])==CHECKPOINT_INVENTORY,
        'BODY_ORIGINAL_NUMERICS_AND_FROZEN_SFT_PARENT_INVENTORY')
    descriptor = s['producer_native_spec']
    producer = pinned(descriptor)
    task.declared(producer)
    task.need(Path(descriptor['path'])==Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer.get('task_learning_role')==PROTOCOL and producer.get('core_local_protocol')==PROTOCOL
        and producer.get('target_gr00t_gpu_root')==s['registered_root'] and producer.get('source_sha')==s['source_sha']
        and producer.get('target_gr00t_policy')==s.get('target_gr00t_policy')
        and producer.get('target_gr00t_query_seeds')==s.get('target_gr00t_query_seeds')
        and producer.get('target_gr00t_pair_id')==s.get('target_gr00t_pair_id')
        and producer.get('target_gr00t_B1_global_step')==s.get('target_gr00t_B1_global_step')==2000
        and s.get('observation_camera_profile')==producer.get('observation_camera_profile')==CAMERA_ID,
        'BODY_EXACT_NEW_NATIVE_NEURAL_PAIR')
    native_scene_admission(producer)
    q = policy_contract(s['target_gr00t_policy'],s['checkpoint_files'],verify_package=verify_model)
    task.need(s['GPU_uuid']==q['GPU_uuid'] and s['allocator_limit_bytes']==q['allocator_limit_bytes'],
              'BODY_SAME_CARD_AND_ACTUALLY_QUALIFIED_MEMORY_LIMIT')
    memory_contract(s)
    return producer


def load(s, root):
    from scripts.wm02.infer_fusion_paused_vla import _load_frozen
    from .target_gr00t_core_v1 import trim_unused_logits
    from .target_gr00t_delivery_v1 import install
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.neural import COSMOS_REVISION
    package = s['target_gr00t_policy']['package']
    def initialize(model):
        install(model,package['path'],package['sha256'],s['checkpoint_files'])
        return dict(package=package,ordinary_CPU_install=True)
    instance, torch = _load_frozen(s,root,cpu_model_initializer=initialize)
    trim_unused_logits(instance.model)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    instance.processor = Gr00tN1d7Processor.from_pretrained(Path(package['path'])/'processor',
        model_name=local_model_name(s['registered_root'],s['neural_bindings']),local_files_only=True,
        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
            cache_dir=s['neural_bindings']['cosmos_hub'],revision=COSMOS_REVISION))
    instance.processor.eval()
    instance.identity = dict(policy=s['target_gr00t_policy'], Processor=package,
        camera=CAMERA_ID, numeric_mode=s['numeric_mode'], representation='CURRENT_STATE_PREDICTED_LAST_KNOT_QUINTIC')
    write_json(Path(root)/'NATIVE_BODY_POLICY_LOADED.private.json',dict(identity=instance.identity,
        actual_model_loads=1, ordinary_internal_GR00T_weights=True, optimizer_updates=0))
    return instance,torch


def create_admission(request, output, checkpoint_files):
    value = dict(schema='NATIVE_BODY_SFT2000_DIAGNOSTIC_ADMISSION_V1',
        preregistration=request['preregistration'], package=request['package'],
        camera_worker=request['camera_worker'], saved_packet_binding=request['saved_packet_binding'],
        same_card_replay=request['same_card_replay'], GPU_uuid=request['GPU_uuid'],
        allocator_limit_bytes=request['allocator_limit_bytes'],
        original_cross_hardware_REJECT_preserved=True, original_DEV_TEST_admission=False,
        qualified_for='FOUR_PREREGISTERED_TRAIN_DIAGNOSTIC_ATTEMPTS_THROUGH_ORIGINAL_GATES',
        actual_new_model_calls=0, optimizer_updates=0, physics_integrations=0)
    qualification(value,checkpoint_files)
    replay_spec = json.loads((Path(request['same_card_replay']['path']).parent/'EXECUTION_SPEC.json').read_text())
    task.need(replay_spec['allocator_limit_bytes']==request['allocator_limit_bytes'],
              'BODY_ACTUALLY_TESTED_MEMORY_LIMIT')
    write_json(output,value)
    return value


def register_panel(request):
    from contextlib import closing
    import time
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot,once
    from .task_learning_student_registration_v2 import closed_base,actual_site,descriptor
    from .task_learning_scene_v1 import scene_id
    from .task_learning_local_policy_v1 import scene_builder_custody
    from .target_gr00t_profile_v1 import PARENT
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
    policy = dict(arm=METHODS[0],parent_manifest_sha256=PARENT,package=admission['package'],
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
        rows.append(dict(native_run='native-body-sft2000-'+layout.lower()+'-r1',layout=layout,
            method=METHODS[0],parent=scene_id(layout),restart=1,split='TRAIN',purpose=PURPOSE,
            command=task.COMMAND,model_sha256=digest(model),query_seeds=list(range(seed,seed+15))))
    panel = dict(schema=SCHEMA,task_learning_role=PROTOCOL,protocol=protocol(),source_sha=request['source_sha'],
        B1_global_step=2000,training_seed=20261109,policies={METHODS[0]:policy},assignments=rows,
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
            return dict(status='NATIVE_BODY_TRAIN4_PANEL_SEALED_NO_WORKERS',panel=descriptor(root/'PANEL.private.json'))
        return journal.once(key,dict(admission=request['Native_body_qualification'],source_sha=request['source_sha']),effect)
