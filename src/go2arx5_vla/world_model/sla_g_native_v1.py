"""Existing official GR00T A500 on the fixed new SLA DEV16; no new training."""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .native_camera_transfer_v1 import CAMERA_ID
from .stable_task_scene_v1 import pinned
from .storage import digest
from .sla_population_v1 import DEV_OFFSETS
from .sla_native_v1 import closed, scene_custody, DEV_DECLARATION_SHA
from .native_a500_diagnostic_v1 import official_files, load
from .native_arx5_training_profile_v1 import PARENT_ID
PROTOCOL='SLA_DEV16_OFFICIAL_GR00T_A500_ENGINEERING_V1'
SCHEMA=PROTOCOL+'_PANEL'
INPUT_SCHEMA=PROTOCOL+'_GROUP_INPUT'
NATIVE_SCHEMA=PROTOCOL+'_NATIVE_ACTUAL'
EXPERIMENT='EXISTING_OFFICIAL_GR00T_A500_FIXED_NEW_SLA_DEV16_ENGINEERING_REFERENCE'
METHODS=('G_OFFICIAL_GR00T_A500_ENGINEERING_REFERENCE',)
G_DECLARATION_SHA='067322ddf06be8fe9ce426c0e348e6e1daa03f2cb59361f406fdf77384524d47'


def protocol():
    return dict(task.protocol(task.CORE_LOCAL_ROLE),stage=PROTOCOL,
        scene_family='FIXED_NEW_SLA_DEV16_ENGINEERING_REFERENCE',
        observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST',
        policy='EXISTING_OFFICIAL_GR00T_A500_PLUS_ORIGINAL_ENDPOINT_BASIS',
        comparison='ENGINEERING_ONLY_NOT_MATCHED_SPATIAL_CAUSAL_CONTROL',Teacher_fallback=False)


def policy_contract(value,checkpoint_files=None,*,verify_package=True,expected_global_step=500):
    from .native_a500_diagnostic_v1 import qualification
    task.need(set(value)=={'arm','parent_files_identity','package','Native_body_qualification','G_DEV_preregistration'}
        and value['arm']==METHODS[0] and value['parent_files_identity']==PARENT_ID
        and expected_global_step==500,'G_EXISTING_OFFICIAL_FIXED500_IDENTITY')
    registration=pinned(value['G_DEV_preregistration'])
    task.need(value['G_DEV_preregistration']['sha256']==G_DECLARATION_SHA
        and registration['method']==METHODS[0] and registration['method_order']=='S1_THEN_S2_THEN_G'
        and registration['population_declaration_sha256']==DEV_DECLARATION_SHA
        and registration['new_training_updates']==registration['new_training_visits']==0,
        'G_FIXED_BEFORE_NEW_DEV_OUTPUTS_NO_NEW_TRAINING')
    q=qualification(pinned(value['Native_body_qualification']),checkpoint_files,verify_package=verify_package)
    task.need(value['package']==q['package']==registration['reference']['package']
        and value['Native_body_qualification']['sha256']==registration['reference']['native_admission_sha256'],
        'G_REUSE_EXACT_ALREADY_QUALIFIED_A500_PACKAGE')
    return q


def memory_contract(spec):
    q=policy_contract(spec['target_gr00t_policy'],verify_package=False)
    proof=closed(q['ordinary_RGB_reconstruction'])
    task.need(spec['memory_execution']=='NATIVE_ARX5_OFFICIAL_ORDINARY_RGB_CORE_V1'
        and spec['memory_execution_reference']==q['ordinary_RGB_reconstruction']
        and proof['max_allocated_bytes']<=spec['allocator_limit_bytes']==q['allocator_limit_bytes']
        and spec['official_parent_audit']==q['official_parent_audit'],'G_REUSE_ACTUAL_ORDINARY_RGB_MEMORY_QUALIFICATION')
    return q


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    declaration=pinned(panel['preregistration']);g=pinned(panel['G_DEV_preregistration'])
    task.need(panel['preregistration']['sha256']==DEV_DECLARATION_SHA
        and panel['G_DEV_preregistration']['sha256']==G_DECLARATION_SHA
        and declaration['schema']=='SLA_FIXED_DEV16_INPUTS_V1'
        and g['method']==METHODS[0] and panel['population']=='DEV16'
        and panel['schema']==SCHEMA and panel['task_learning_role']==PROTOCOL
        and panel['protocol']==protocol() and panel['B1_global_step']==500
        and set(panel['policies'])==set(METHODS),'G_ONE_FIXED_NEW_DEV16_PANEL')
    disjoint=pinned(panel['scene_disjointness'])
    task.need(disjoint['passes'] and disjoint['old_TEST_scores_read'] is False,'G_SAME_NEW_DEV_STATIC_DISJOINTNESS')
    rows=panel['assignments'];expected=declaration['population']
    task.need(len(rows)==len(expected)==len({r['native_run']for r in rows})==16,'G_FULL_DEV16_NO_PREFIX_SELECTION')
    for row,entry in zip(rows,expected):
        task.need(all(row[k]==entry[k]for k in('layout','restart','query_seeds','parent','split'))
            and row['method']==METHODS[0] and row['parent']==scene_id(row['layout'])
            and row['command']==task.COMMAND and row['purpose']==EXPERIMENT
            and Path(row['native_run']).name==row['native_run'],'G_NO_LAYOUT_RESTART_OR_SEED_CHANGE')
    return rows


def selection(inp,panel):
    rows=validate_panel(panel)
    task.need(inp['schema']==INPUT_SCHEMA and inp['source_sha']==panel['source_sha']
        and inp['assignment_slice']==[0,16] and inp.get('population_predecessor')is None,'G_FULL_PREREGISTERED_DEV16_ONLY')
    return rows


def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry, scene_id, BASE_SCENE
    proof = pinned(spec['task_learning_scene_reference'])
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    layout = proof['layout']
    builder = scene_custody(panel['registration']['scene_producers'][layout],
                                   spec['task_learning_scene_reference'], spec['model_path'])
    task.need(spec['task_learning_role'] == spec['core_local_protocol'] == PROTOCOL and layout in DEV_OFFSETS
        and proof['schema'] == 'TASK_LEARNING_NEAR_WORKSPACE_STATIC_V1'
        and proof['status'] == 'STATIC_ONLY_PASSED_NEW_NATIVE_ADMISSION_REQUIRED'
        and proof['builder_sha256'] == builder and proof['geometry'] == geometry(layout)
        and proof['parent'] == scene_id(layout) == spec['scene_id'] == spec['data_parent']
        and proof['split'] == spec['data_role'] and proof['saved_startup_pose_checks'] == 5703
        and proof['robot_contract']['status'] == 'COURSE_ROBOT_DYNAMICS_AND_NATIVE_SAFETY_PROPERTIES_EXACT'
        and proof['compiled']['base_scene_sha256'] == BASE_SCENE
        and proof['compiled']['fresh_XML_compilation'] and proof['compiled']['original_robot_contract_exact']
        and proof['compiled']['scene_sha256'] == digest(spec['model_path'])
        and proof['physics_integrations'] == proof['new_transitions'] == 0
        and not proof['native_execution_authorized'] and not proof['WBC_changes'] and proof['physics_dt_s'] == .002,
        'SLA_SEALED_FULL_ROBOT_SCENE_ORIGINAL_NATIVE_SAFETY')
    matches = [r for r in validate_panel(panel) if r['native_run'] == Path(spec['registered_root']).name]
    task.need(len(matches) == 1, 'SLA_ONE_REGISTERED_NATIVE_REQUEST')
    row = matches[0]
    task.need(row['model_sha256'] == digest(spec['model_path'])
        and spec['target_gr00t_policy'] == panel['policies'][row['method']]
        and spec['target_gr00t_query_seeds'] == row['query_seeds']
        and spec['target_gr00t_pair_id'] == row['native_run']
        and spec['observation_camera_profile'] == CAMERA_ID
        and spec.get('task_learning_teacher_mode') is None and spec.get('task_learning_control') is None,
        'SLA_EXACT_BODY_WRIST_NO_TEACHER_FALLBACK')
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


def summarize(panel, outcomes):
    rows = validate_panel(panel)
    task.need(len(outcomes) <= len(rows) and len({r['native_run'] for r in outcomes}) == len(outcomes)
        and all(all(r[k] == rows[i][k] for k in ('native_run','layout','method','restart','parent','split','query_seeds'))
                for i, r in enumerate(outcomes)), 'SLA_FIXED_EVERY_REQUEST_ONCE')
    return dict(schema='SLA_G_FIXED_DEV16_ENGINEERING_SUMMARY_V1', population=panel['population'],
        requested=len(rows), completed=len(outcomes), complete=len(outcomes) == len(rows),
        methods={METHODS[0]: dict(requested=len(rows), attempted=sum(r['attempted'] for r in outcomes),
            task_successes=sum(r['technical_valid'] and r['task_success'] for r in outcomes),
            valid_task_failures=sum(r['technical_valid'] and not r['task_success'] for r in outcomes),
            technical_failures=sum(not r['technical_valid'] for r in outcomes),
            hard_safety_stop_conditions=sum(r['hard_safety_stop'] for r in outcomes),
            actual_GR00T_queries=sum(r.get('actual_GR00T_queries', 0) for r in outcomes),
            actual_native_integrations=sum(r.get('actual_native_integrations', 0) for r in outcomes))},
        original_DEV_conditions=0, TEST_conditions=0, qualified_training_demonstrations=0,
        model_updates=0, WBC_changes=False, generalization_claim=False)



def register_panel(request):
    from contextlib import closing
    import time
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from .task_learning_student_registration_v2 import closed_base, actual_site, descriptor
    from .task_learning_scene_v1 import scene_id
    n, g = closed_base(request['native_base']), closed_base(request['gpu_base'])
    campaign = Path(n['campaign_root'])
    actual_site(campaign)
    root, source = Path(request['root']), Path(request['source_root'])
    task.need(root.parent == Path(n['registered_root']).parents[2] and source.resolve() == source,
              'SLA_PANEL_IN_EXISTING_PROJECT')
    task.need(all(digest(source/name) == pin for name, pin in request['source_files'].items()),
              'SLA_SEALED_NEW_SOURCE_CAPSULE')
    admission=pinned(request['Native_body_qualification'])
    method,population=METHODS[0],request['population']
    task.need(population=='DEV16','G_ONLY_NEW_DEV16_NO_TRAIN4_REDO')
    policy=dict(arm=method,parent_files_identity=PARENT_ID,package=admission['package'],Native_body_qualification=request['Native_body_qualification'],G_DEV_preregistration=request['G_DEV_preregistration'])
    policy_contract(policy,g['checkpoint_files'])
    declaration = pinned(request['preregistration'])
    expected_sha=DEV_DECLARATION_SHA
    task.need(request['preregistration']['sha256'] == expected_sha, 'SLA_ORIGINAL_INPUT_PREREGISTRATION')
    inputs=[(r['layout'],r['restart'],r['query_seeds'])for r in declaration['population']]
    task.need(set(request['scenes']) == {r[0] for r in inputs}, 'SLA_EXACT_FIXED_SCENE_POPULATION')
    rows = []
    for layout, restart, seeds in inputs:
        scene = request['scenes'][layout]
        proof = pinned(scene)
        model = Path(scene['path']).parent/'SCENE.private.mjb'
        scene_custody(request['scene_producers'][layout], scene, model)
        rows.append(dict(native_run=f"sla-g-{layout.lower()}-r{restart}",
            layout=layout, restart=restart, query_seeds=seeds, parent=scene_id(layout), method=method,
            model_sha256=digest(model), split=proof['split'], purpose=EXPERIMENT, command=task.COMMAND))
    panel = dict(schema=SCHEMA, task_learning_role=PROTOCOL, protocol=protocol(),
        source_sha=request['source_sha'], B1_global_step=500, training_seed=20261112,
        method=method, population=population, policies={method:policy}, assignments=rows,
        preregistration=request['preregistration'], G_DEV_preregistration=request['G_DEV_preregistration'], scene_disjointness=request['scene_disjointness'],
        registration={k:request[k] for k in ('source_root','source_files','native_base','gpu_base','scenes',
            'scene_producers','scene_allowlist','scene_allowlist_sha256','memory_execution','memory_execution_reference')})
    validate_panel(panel)
    key = root.name.upper()+'-PANEL'
    journal = Journal(campaign/'control/journal', campaign.name)
    with closing(OwnerLock(campaign/'OWNER.lock')):
        task.need(not root.exists() and not journal.path(key).exists(), 'SLA_UNKNOWN_PANEL_NEVER_REPLAY')
        def effect():
            root.mkdir()
            (root/'control').mkdir()
            write(root/'PANEL.private.json', panel)
            write(root/'control/LEASE.json', dict(status='SEALED_NONCOMPUTE_PANEL',
                planned_conditions=len(rows), panel_sha256=digest(root/'PANEL.private.json'), created_epoch=time.time()))
            registry, pin = snapshot(n['registry'])
            task.need(not any(r.get('path') == str(root) for r in registry['entries']), 'SLA_NEW_PANEL_REGISTRY')
            registry['entries'].append(dict(path=str(root), role=key, scope=SCHEMA, owner=n['owner'],
                source_sha=request['source_sha'], local_lease=str(root/'control/LEASE.json'), executor_node=n['host']))
            once(n['registry'], registry, key+'-REGISTRY', expected_before_sha256=pin)
            return dict(status='SLA_PANEL_SEALED_NO_WORKER_STARTED', panel=descriptor(root/'PANEL.private.json'))
        return journal.once(key, dict(policy=policy, preregistration=request['preregistration'], source_sha=request['source_sha']), effect)
