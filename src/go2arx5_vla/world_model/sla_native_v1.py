"""Separate SLA evaluation identity on the existing serial native owner/consumer."""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .native_camera_transfer_v1 import CAMERA_ID
from .stable_task_scene_v1 import pinned
from .storage import digest
from .sla_population_v1 import METHODS, TRAIN_ORDER, DEV_OFFSETS

PROTOCOL = 'SLA_TARGET500_NATIVE_EVALUATION_V1'
SCHEMA = 'SLA_TARGET500_NATIVE_PANEL_V1'
INPUT_SCHEMA = 'SLA_TARGET500_NATIVE_GROUP_INPUT_V1'
NATIVE_SCHEMA = 'SLA_TARGET500_NATIVE_ACTUAL_V1'
EXPERIMENT = 'SLA_FIXED500_NATIVE_TRAIN4_AND_INDEPENDENT_DEV16'
TRAIN_DECLARATION_SHA = '02467287506549711e74de58ef0d7a9c5cf521943c826d8fe79aaa16be0c400b'
DEV_DECLARATION_SHA = '41bfcaf33cfc4d4e0deda69d02849730f7e49097abbe773c6ac5e31b08a2299c'


def protocol():
    return dict(task.protocol(task.CORE_LOCAL_ROLE), stage=PROTOCOL,
        scene_family='SLA_FIXED_TRAIN4_OR_INDEPENDENT_DEV16',
        observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST',
        policy='SLA_BASE_TARGET500_PLUS_ORIGINAL_ENDPOINT_BASIS',
        normalized_container_dimensions=32, sampler_steps=10,
        episode_frame='ACTUAL_POST_HOLD_FIXED_X5_BASE_LINK', Teacher_fallback=False)


def closed(pin):
    value = pinned(pin)
    root = Path(pin['path']).parent
    lease = json.loads((root/'control/LEASE.json').read_bytes())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    task.need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == worker['exit_code'] == 0
              and lease['spec_sha256'] == digest(root/'EXECUTION_SPEC.json'), 'SLA_ACTUAL_CLOSED0_ENDPOINT')
    return value


def qualification(q, *, verify_package=True):
    from .sla_r1_diagnostic_v5 import ADMISSION, qualification as r1_qualification
    if q.get('schema') == ADMISSION:
        return r1_qualification(q, verify_package=verify_package)
    task.need(q.get('schema') == 'SLA_TARGET500_NATIVE_ADMISSION_V1'
        and q.get('method') in METHODS and q.get('old_DEV_TEST_admitted') is False,
        'SLA_DISTINCT_FIXED500_IDENTITY')
    training = pinned(q['training_admission'])
    task.need(training.get('schema') == 'SLA_CUDA_TRAIN_RESTORE_RGB_ADMISSION_V1'
              and training.get('passes') and training.get('noncandidate_updates') == 8,
              'SLA_REAL_GRADIENT_RESTORE_RGB_QUALIFICATION')
    candidate = closed(q['candidate'])
    task.need(candidate['status'] == 'COMPLETED_FINITE_SLA_OPERATION' and not candidate['noncandidate']
        and candidate['step'] == 500 and candidate['cursor'] == 16000
        and candidate['actual']['optimizer_updates'] == 500
        and candidate['actual']['completed_training_visits'] == 16000
        and candidate['actual']['physical_requests'] == 0, 'SLA_REAL_FIXED500_16000_NOT_SOFTWARE_RESULT')
    rgb = closed(q['ordinary_RGB_reconstruction'])
    task.need(rgb.get('schema') == 'SLA_FRESH_ORDINARY_RGB_PACKAGE_ACTUAL_V1' and rgb['passes']
        and rgb['step'] == 500 and rgb['ordinary_RGB_generations'] == 16
        and all(v['bit_exact'] for v in rgb['records']) and rgb['package'] == q['package']
        and not rgb['optimizer_loaded'] and not rgb['training_cache_used'] and not rgb['teacher_loaded']
        and rgb['deployed_LoRA_retained'], 'SLA_FRESH_FULL_RGB_MATCHES_TRAINING_ENDPOINT')
    m = pinned(dict(path=str(Path(q['package']['path'])/'MANIFEST.private.json'), sha256=q['package']['sha256']))
    task.need(m['schema'] == 'SLA_ORDINARY_RGB_INFERENCE_PACKAGE_V1' and m['step'] == 500
        and m['cursor'] == 16000 and not m['qualification_only']
        and m['original_complete_sha256'] == candidate['last_checkpoint']['sha256']
        and m['parent_manifest'] == q['parent_manifest']
        and not m['teacher_required'] and not m['training_cache_required'] and not m['optimizer_required'],
        'SLA_EXACT_DEPLOYMENT_PARENT_AND_ALL_ADAPTERS')
    expected_recipe=('S1_FM32_MICRO1_ACCUM32_DROPOUT_TOKEN_ONLY_V1' if q['method']==METHODS[0]
                     else 'S2_NATIVE_FM32_PLUS_FIXED_SPATIAL_ALPHA_HALF_V1')
    task.need(m['bindings']['recipe']==expected_recipe,'SLA_METHOD_MATCHES_ACTUAL_TRAINING_OBJECTIVE')
    rgb_root=Path(q['ordinary_RGB_reconstruction']['path']).parent
    rgb_spec=json.loads((rgb_root/'EXECUTION_SPEC.json').read_bytes())
    task.need(rgb_spec['GPU_uuid']==q['GPU_uuid'] and rgb_spec['parent_params']==q['parent_params']
        and rgb_spec['python']==q['runtime']['python'] and rgb_spec['upstream_root']==q['runtime']['upstream_root']
        and rgb_spec['JAX_memory_fraction']==q['runtime']['JAX_memory_fraction'],
        'SLA_OWN_ACTUAL_RECONSTRUCTION_DEVICE_ENVIRONMENT_AND_PARENT')
    if q['method']==METHODS[1]:
        spatial=pinned(q['spatial_admission'])
        task.need(spatial['schema']=='SLA_SPATIAL_GRADIENT_CONTROL_ADMISSION_V1' and spatial['passes']
            and spatial['alpha0_all_common_gradients_and_updates_bit_exact']
            and spatial['positive_spatial_gradient_reaches_deployed_VLM_LoRA']
            and spatial['positive_noncandidate_updates']==8 and m['bindings']['alpha']==.5
            and candidate['actual']['spatial_backwards_completed']==16000
            and candidate['actual']['spatial_cache_reads']==16000
            and candidate['actual']['teacher_model_loads']==0,'S2_REAL_MATCHED_SPATIAL_GRADIENT_AND_BUDGET_PROOF')
    if verify_package:
        package = Path(q['package']['path'])
        for row in m['adapter_arrays']:
            p = package/row['file']
            task.need(p.resolve().is_relative_to(package.resolve()) and digest(p) == row['sha256'],
                      'SLA_IMMUTABLE_DEPLOYED_ADAPTER')
        for name, key in [('STATISTICS.json', 'statistics_sha256'), ('paligemma_tokenizer.model', 'tokenizer_sha256')]:
            task.need(digest(package/name) == m[key], 'SLA_SAVED_STATISTICS_AND_TOKENIZER')
    return q


def policy_contract(value, checkpoint_files=None, *, verify_package=True, expected_global_step=500):
    from .sla_r1_diagnostic_v5 import METHODS as R1_METHODS
    if value.get('arm') in R1_METHODS:
        task.need(set(value) == {'arm','package','Native_body_qualification'} and expected_global_step == 500,
                  'DISTINCT_R0_R1_POLICY_FIELDS')
        q = qualification(pinned(value['Native_body_qualification']),verify_package=verify_package)
        task.need(q['package'] == value['package'] and q['method'] ==
            (METHODS[0] if value['arm'] == R1_METHODS[0] else R1_METHODS[1]), 'R0_R1_MATCHED_ACTUAL_PACKAGE')
        return q
    task.need(set(value) == {'arm', 'package', 'Native_body_qualification'}
              and value['arm'] in METHODS and expected_global_step == 500, 'SLA_ONLY_SEPARATE_FIXED500_POLICY')
    q = qualification(pinned(value['Native_body_qualification']), verify_package=verify_package)
    task.need(q['package'] == value['package'] and q['method'] == value['arm'], 'SLA_SAME_METHOD_AND_PACKAGE')
    return q


def memory_contract(s):
    q = policy_contract(s['target_gr00t_policy'], verify_package=False)
    proof = closed(q['ordinary_RGB_reconstruction'])
    task.need(s['memory_execution'] == 'SLA_ORDINARY_RGB_JAX_SINGLE_CORE_V1'
        and s['memory_execution_reference'] == q['ordinary_RGB_reconstruction']
        and s['allocator_limit_bytes'] == q['allocator_limit_bytes']
        and proof['GPU_memory']['peak_bytes_in_use'] <= s['allocator_limit_bytes']
        and proof['GPU_memory']['bytes_limit'] <= s['allocator_limit_bytes'], 'SLA_MEASURED_OWN_ORDINARY_RGB_CAPACITY')
    return q


def validate_panel(panel):
    from .sla_r1_diagnostic_v5 import SCHEMA as R1_SCHEMA, validate_panel as r1_panel
    if panel.get('schema') == R1_SCHEMA:
        return r1_panel(panel)
    from .task_learning_scene_v1 import scene_id
    declaration = pinned(panel['preregistration'])
    task.need(panel['preregistration']['sha256'] == (TRAIN_DECLARATION_SHA if panel['population'] == 'TRAIN4' else DEV_DECLARATION_SHA), 'SLA_UNCHANGED_INPUT_DECLARATION')
    if panel.get('registration',{}).get('native_pause_read_wait') is not None:
        from .sla_native_continuation_v2 import wait_seconds
        wait_seconds(dict(task_learning_role=PROTOCOL,native_pause_read_wait=panel['registration']['native_pause_read_wait']))
    rows = panel.get('assignments') or []
    method = panel.get('method')
    split = panel.get('population')
    task.need(panel.get('schema') == SCHEMA and panel.get('task_learning_role') == PROTOCOL
        and panel.get('protocol') == protocol() and panel.get('B1_global_step') == 500
        and method in METHODS and set(panel.get('policies', {})) == {method}, 'SLA_ONE_FIXED_ENDPOINT_PANEL')
    if split == 'TRAIN4':
        expected = [(v['layout'], 1, list(range(v['query_seed_base'], v['query_seed_base']+15)))
                    for v in declaration['conditions']]
        task.need(declaration['schema'] == 'SLA_FIXED_TRAIN4_INPUTS_V1'
                  and tuple(x[0] for x in expected) == TRAIN_ORDER, 'SLA_FIXED_ORIGINAL_TRAIN4')
    else:
        task.need(split == 'DEV16' and declaration['schema'] == 'SLA_FIXED_DEV16_INPUTS_V1'
            and panel.get('scene_disjointness') is not None, 'SLA_INDEPENDENT_INPUTS_BEFORE_CANDIDATE_OUTPUTS')
        proof = pinned(panel['scene_disjointness'])
        task.need(proof['passes'] and proof['old_TEST_scores_read'] is False, 'SLA_ACTUAL_CATALOG_GEOMETRY_DISJOINTNESS')
        expected = [(v['layout'], v['restart'], v['query_seeds']) for v in declaration['population']]
        task.need(len(expected) == 16 and {v[0] for v in expected} == set(DEV_OFFSETS), 'SLA_COMPLETE_DEV16')
    task.need(len(rows) == len(expected) == len({r['native_run'] for r in rows}), 'SLA_FULL_PREREGISTERED_POPULATION')
    for row, (layout, restart, seeds) in zip(rows, expected):
        task.need(row['layout'] == layout and row['restart'] == restart and row['query_seeds'] == seeds
            and row['method'] == method and row['parent'] == scene_id(layout)
            and row['split'] == ('TRAIN' if split == 'TRAIN4' else 'SLA_DEV')
            and row['command'] == task.COMMAND and row['purpose'] == EXPERIMENT
            and Path(row['native_run']).name == row['native_run'], 'SLA_NO_POPULATION_RETRY_OR_SEED_CHANGE')
    from .sla_native_resource_continuation_v3 import configuration
    configuration(panel)
    return rows


def selection(inp, panel):
    rows = validate_panel(panel)
    from .sla_r1_suffix_v5 import SCHEMA as R1_SUFFIX, selection as r1_suffix
    if inp.get('schema') == R1_SUFFIX:
        return r1_suffix(inp,panel,rows)
    from .sla_dev_continuation_v3 import SCHEMA as DEV_SUFFIX, select as select_dev_suffix
    if inp.get('schema') == DEV_SUFFIX:
        return select_dev_suffix(inp, panel, rows)
    from .sla_native_continuation_v2 import INPUT_SCHEMA as COLD_SCHEMA, prefix
    if inp.get('schema') == COLD_SCHEMA:
        retained = prefix(panel)
        expected = panel.get('resource_continuation') or panel['cold_continuation']
        task.need(len(retained) == (3 if panel.get('resource_continuation') else 2) and inp['source_sha'] == panel['source_sha']
            and inp['assignment_slice'] == [len(retained),len(rows)]
            and inp['population_predecessor'] == expected['result'],
            'SLA_ONLY_NEVER_STARTED_SUFFIX_WITH_IMMUTABLE_FAILURE_PREFIX')
        return rows[len(retained):]
    task.need(inp.get('schema') == INPUT_SCHEMA and inp['source_sha'] == panel['source_sha']
        and inp['assignment_slice'] == [0, len(rows)] and inp.get('population_predecessor') is None
        and panel.get('cold_continuation') is None, 'SLA_FULL_FIXED_PANEL_NOT_SELECTED_PREFIX')
    return rows


def native_scene_admission(spec):
    from .task_learning_scene_v1 import geometry, scene_id, BASE_SCENE
    proof = pinned(spec['task_learning_scene_reference'])
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    layout = proof['layout']
    builder = scene_custody(panel['registration']['scene_producers'][layout],
                                   spec['task_learning_scene_reference'], spec['model_path'])
    task.need(spec['task_learning_role'] == spec['core_local_protocol'] == PROTOCOL
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
    from .task_learning_live_pair_v1 import PROFILE, KIND
    task.need(s['core_local_protocol'] == PROTOCOL and s['profile'] == PROFILE and s['kind'] == KIND
        and s['execution_site'] == 'CORE_FUSION_NEURAL' and s['physics'] is False
        and s['device'] == 'cuda' and s['max_queries'] == 15
        and s['GR00T_updates'] == s['WM_updates'] == s['Actor_updates'] == 0
        and s['numeric_mode'] == 'SLA_PINNED_JAX_BF16_DETERMINISTIC_V1' and s['baseline_changed'] is False
        and s['task_learning_grasp'] == protocol(), 'SLA_SEPARATE_SINGLE_JAX_INFERENCE_IDENTITY')
    descriptor = s['producer_native_spec']
    producer = pinned(descriptor)
    task.declared(producer)
    task.need(Path(descriptor['path']) == Path(producer['registered_root'])/'EXECUTION_SPEC.json'
        and producer['source_sha'] == s['source_sha'] and producer['target_gr00t_gpu_root'] == s['registered_root']
        and producer['task_learning_role'] == producer['core_local_protocol'] == PROTOCOL
        and all(producer[k] == s[k] for k in ('target_gr00t_policy', 'target_gr00t_query_seeds',
            'target_gr00t_pair_id', 'target_gr00t_B1_global_step', 'observation_camera_profile')),
        'SLA_SAME_REGISTERED_NATIVE_NEURAL_PAIR')
    native_scene_admission(producer)
    q = policy_contract(s['target_gr00t_policy'], verify_package=verify_model)
    manifest=pinned(dict(path=str(Path(q['package']['path'])/'MANIFEST.private.json'),sha256=q['package']['sha256']))
    from .sla_r1_diagnostic_v5 import ADMISSION, SCHEMA as R1_SCHEMA, inference_identity
    panel = pinned(dict(path=producer['stable_task_panel'],sha256=producer['stable_task_panel_sha256']))
    if panel.get('schema') == R1_SCHEMA:
        proof = (q['ordinary_RGB_reconstruction'] if q.get('schema') == ADMISSION else
                 panel['registration']['S1_loader_reconstruction'])
        rgb_spec = json.loads((Path(proof['path']).parent/'EXECUTION_SPEC.json').read_bytes())
        inference_identity(s['source_root'],rgb_spec['source_root'],logit=q.get('schema') == ADMISSION)
        if q.get('schema') != ADMISSION:
            _reconstructed_loader(s,q)
        task.need(s['GPU_uuid'] == q['GPU_uuid'] and s['allocator_limit_bytes'] == q['allocator_limit_bytes'],
                  'R1_QUALIFIED_CARD_AND_ALLOCATOR')
        memory_contract(s)
        return producer
    updated = _reconstructed_loader(s, q)
    for name,pin in manifest['bindings']['adapter_source'].items():
        expected = updated.get(name, pin)
        task.need(name in s['source_files'] and s['source_files'][name]==expected
            and digest(Path(s['source_root'])/name)==expected,'SLA_DEPLOYMENT_USES_ACTUAL_TRAINED_AND_QUALIFIED_ADAPTER_CODE')
    task.need(s['GPU_uuid'] == q['GPU_uuid'] and s['allocator_limit_bytes'] == q['allocator_limit_bytes'],
              'SLA_ACTUALLY_QUALIFIED_GPU_AND_MEMORY_LIMIT')
    memory_contract(s)
    return producer


def _reconstructed_loader(spec, qualification):
    producer = pinned(spec['producer_native_spec'])
    panel = pinned(dict(path=producer['stable_task_panel'], sha256=producer['stable_task_panel_sha256']))
    evidence = panel['registration'].get('S1_loader_reconstruction')
    if evidence is None:
        return {}
    from .sla_r1_diagnostic_v5 import SCHEMA as R1_SCHEMA
    task.need((panel['population'] == 'DEV16' and panel['method'] == METHODS[0])
              or panel.get('schema') == R1_SCHEMA,
              'LOADER_RECONSTRUCTION_ONLY_FOR_FIXED_S1_DEV16')
    result = closed(evidence)
    root = Path(evidence['path']).parent
    reconstruction = json.loads((root / 'EXECUTION_SPEC.json').read_bytes())
    source = Path(reconstruction['source_root'])
    seal = pinned(reconstruction['source_seal'])
    original = closed(qualification['ordinary_RGB_reconstruction'])
    memory = result['GPU_memory']
    paths = ('src/go2arx5_vla/sla_adapter/model_v1.py', 'src/go2arx5_vla/sla_adapter/calibration_v1.py')
    task.need(result['schema'] == original['schema'] == 'SLA_FRESH_ORDINARY_RGB_PACKAGE_ACTUAL_V1'
              and result['passes'] is True and result['step'] == 500
              and result['package'] == original['package'] == qualification['package']
              and result['frozen_fingerprint'] == original['frozen_fingerprint']
              and result['ordinary_RGB_generations'] == len(result['records']) == 16
              and [(v['index'], v['noise']) for v in result['records']]
              == [(v['index'], v['noise']) for v in original['records']]
              and all(v['bit_exact'] for v in result['records'])
              and result['actual_parent_loads'] == 1 and result['deployed_LoRA_retained']
              and result['optimizer_updates'] == result['physical_requests'] == 0
              and not any(result[k] for k in ('optimizer_loaded', 'training_cache_used', 'teacher_loaded'))
              and reconstruction['GPU_uuid'] == spec['GPU_uuid'] == qualification['GPU_uuid']
              and reconstruction['JAX_memory_fraction'] == qualification['runtime']['JAX_memory_fraction']
              and reconstruction['parent_params'] == qualification['parent_params']
              and reconstruction['source_sha'] == seal['source_sha']
              and all(seal['files'][path] == digest(source / path) for path in paths)
              and max(memory['peak_bytes_in_use'], memory['bytes_limit']) <= spec['allocator_limit_bytes'],
              'ACTUAL_SAME_PACKAGE_16_BIT_EXACT_RGB_LOADER_QUALIFICATION_REQUIRED')
    return {path: seal['files'][path] for path in paths}


def summarize(panel, outcomes):
    from .sla_r1_diagnostic_v5 import SCHEMA as R1_SCHEMA, summarize as r1_summary
    if panel.get('schema') == R1_SCHEMA:
        return r1_summary(panel,outcomes)
    rows = validate_panel(panel)
    task.need(len(outcomes) <= len(rows) and len({r['native_run'] for r in outcomes}) == len(outcomes)
        and all(all(r[k] == rows[i][k] for k in ('native_run','layout','method','restart','parent','split','query_seeds'))
                for i, r in enumerate(outcomes)), 'SLA_FIXED_EVERY_REQUEST_ONCE')
    return dict(schema='SLA_FIXED_NATIVE_POPULATION_SUMMARY_V1', population=panel['population'],
        requested=len(rows), completed=len(outcomes), complete=len(outcomes) == len(rows),
        methods={panel['method']: dict(requested=len(rows), attempted=sum(r['attempted'] for r in outcomes),
            task_successes=sum(r['technical_valid'] and r['task_success'] for r in outcomes),
            valid_task_failures=sum(r['technical_valid'] and not r['task_success'] for r in outcomes),
            technical_failures=sum(not r['technical_valid'] for r in outcomes),
            hard_safety_stop_conditions=sum(r['hard_safety_stop'] for r in outcomes),
            actual_SLA_queries=sum(r.get('actual_SLA_queries', 0) for r in outcomes),
            actual_native_integrations=sum(r.get('actual_native_integrations', 0) for r in outcomes))},
        original_DEV_conditions=0, TEST_conditions=0, qualified_training_demonstrations=0,
        model_updates=0, WBC_changes=False, generalization_claim=False)


def scene_custody(producer, reference, model):
    """Reuse TRAIN custody; new DEV needs only its own closed static producer."""
    from .task_learning_scene_v1 import MEMBER
    from .task_learning_local_policy_v1 import scene_builder_custody
    proof = pinned(reference)
    from . import sla_dev48_v5 as dev
    if proof['layout']in dev.OFFSETS:return dev.scene_custody(producer,reference,model)
    if proof['layout'] in TRAIN_ORDER:
        return scene_builder_custody(producer, reference, model)
    source = closed(producer)
    root = Path(producer['path']).parent
    actual = json.loads((root/'RESULT.private.json').read_bytes())
    task.need(source['operation'] == 'SLA_DEV_INPUT_STATIC_PREPARATION'
        and source['preregistration']['sha256'] == DEV_DECLARATION_SHA
        and pinned(source['preregistration'])['schema'] == 'SLA_FIXED_DEV16_INPUTS_V1'
        and source['source_files'][MEMBER] == proof['builder_sha256']
        and digest(Path(source['source_root'])/MEMBER) == proof['builder_sha256']
        and actual['passes'] and actual['physics_integrations'] == actual['model_calls'] == actual['optimizer_updates'] == 0
        and actual['scenes'][proof['layout']] == reference
        and digest(model) == proof['compiled']['scene_sha256'], 'SLA_NEW_DEV_STATIC_PRODUCER_CUSTODY_WITHOUT_TEACHER')
    return proof['builder_sha256']


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
    q = qualification(pinned(request['Native_body_qualification']))
    method, population = q['method'], request['population']
    policy = dict(arm=method, package=q['package'], Native_body_qualification=request['Native_body_qualification'])
    declaration = pinned(request['preregistration'])
    expected_sha = TRAIN_DECLARATION_SHA if population == 'TRAIN4' else DEV_DECLARATION_SHA
    task.need(request['preregistration']['sha256'] == expected_sha, 'SLA_ORIGINAL_INPUT_PREREGISTRATION')
    inputs = ([(r['layout'], 1, list(range(r['query_seed_base'], r['query_seed_base']+15)))
               for r in declaration['conditions']] if population == 'TRAIN4' else
              [(r['layout'], r['restart'], r['query_seeds']) for r in declaration['population']])
    task.need(set(request['scenes']) == {r[0] for r in inputs}, 'SLA_EXACT_FIXED_SCENE_POPULATION')
    rows = []
    for layout, restart, seeds in inputs:
        scene = request['scenes'][layout]
        proof = pinned(scene)
        model = Path(scene['path']).parent/'SCENE.private.mjb'
        scene_custody(request['scene_producers'][layout], scene, model)
        rows.append(dict(native_run=f"sla-{method[:2].lower()}-{layout.lower()}-r{restart}",
            layout=layout, restart=restart, query_seeds=seeds, parent=scene_id(layout), method=method,
            model_sha256=digest(model), split=proof['split'], purpose=EXPERIMENT, command=task.COMMAND))
    panel = dict(schema=SCHEMA, task_learning_role=PROTOCOL, protocol=protocol(),
        source_sha=request['source_sha'], B1_global_step=500, training_seed=20261112,
        method=method, population=population, policies={method:policy}, assignments=rows,
        preregistration=request['preregistration'], scene_disjointness=request.get('scene_disjointness'),
        registration={k:request[k] for k in ('source_root','source_files','native_base','gpu_base','scenes',
            'scene_producers','scene_allowlist','scene_allowlist_sha256','memory_execution','memory_execution_reference')})
    if request.get('native_pause_read_wait') is not None:
        panel['registration']['native_pause_read_wait'] = request['native_pause_read_wait']
    if request.get('capacity_wait_before_pair_v3') is not None:
        panel['registration']['capacity_wait_before_pair_v3'] = request['capacity_wait_before_pair_v3']
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
