from pathlib import Path
import time

import pytest

from scripts.pi0292.durable import write
from scripts.wm02 import multitask_stage_cards_v6 as cards
from scripts.wm02 import multitask_student_campaign_v6 as student
from scripts.wm02 import multitask_teacher_campaign_v6 as teacher
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases, teacher_slice
from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, DECODE_ORDER, EPSILON, NATIVE_METHOD, RECIPE, SCHEMA
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin, student_package
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS


def test_partial_pilot_continues_only_unstarted_original_conditions():
    original = [dict(id=f'case{i}') for i in range(48)]
    population = dict(conditions=dict(TEACHER=original))
    prefix = [dict(row,task_success=False) for row in original[:3]]
    assert teacher_slice(population,prefix,6) == original[3:6]
    assert teacher_slice(population,original[:6],10) == original[6:10]
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,prefix,7)
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,[original[0],original[2]],6)
    with pytest.raises(ValueError,match='FIXED_SUFFIX'):
        teacher_slice(population,original[:6],49)


def test_registry_prior_scientific_failure_cannot_be_replaced(tmp_path):
    prior = tmp_path/'old'/'M-TEACHER-case'; (prior/'control').mkdir(parents=True)
    registry = tmp_path/'REGISTRY.json'
    write(registry,dict(entries=[dict(path=str(prior))]))
    write(prior/'control/LEASE.json',dict(status='CLOSED'))
    write(prior/'WORKER_EXIT.private.json',dict(not_started=True))
    require_unstarted_cases(registry,{prior.name})
    write(prior/'RESULT.private.json',dict(physical_requests=1,task_success=False))
    with pytest.raises(PermissionError,match='ALREADY_CONSUMED'):
        require_unstarted_cases(registry,{prior.name})


def test_scale_block_carries_exact_closed_scientific_prefix(tmp_path, monkeypatch):
    from go2arx5_vla.world_model import scale20k_native as scale
    from scripts.wm02 import multitask_panel_suffix as suffix
    from scripts.wm02 import sim_only_campaign_v51 as sim

    arm = 'SCALE-TGT5K'
    ordered = [(dict(id=f'tgt{i}',instruction='HOLD-red'),arm) for i in range(24)]
    assert scale.next_block(dict(condition_stop=6),ordered,0)==ordered[:6]
    assert scale.next_block(dict(condition_stop=12),ordered,6)==ordered[6:12]
    with pytest.raises(ValueError,match='FIXED_BLOCK'):
        scale.next_block(dict(condition_stop=13),ordered,6)
    root = tmp_path/'scope'
    (root/'control').mkdir(parents=True)
    monkeypatch.setattr(suffix,'unit_state',lambda name:dict(pid=0,cgroup_pids=[]))
    monkeypatch.setattr(sim,'fatal_control_violation',lambda row:False)
    sealed = []
    paths = []
    for row,_ in ordered[:6]:
        name = f"{row['id']}-{arm}"
        case = root/'cases'/name
        (case/'control').mkdir(parents=True)
        child = case/'EXECUTION_SPEC.json'
        write(child,dict(id=name))
        paths.append(str(child))
        write(case/'control/LEASE.json',dict(status='CLOSED',spec_sha256=cards.pin(child)['sha256']))
        effects = case/'RESULT.private.json'
        write(effects,dict(source_sha='source',physical_requests=1,actual_native_integrations=5705,
                           model_generations=1,technical_failure=False))
        sealed.append(dict(id=name,role='STUDENT',actual_model_generations=1,native_ticks=5705,
            result_sha256=cards.pin(effects)['sha256'],technical_failure=False,task_success=False,
            score=dict(missing_native_intervals=0,native_step=5705)))
    spec = root/'EXECUTION_SPEC.json'
    parent = dict(scope=scale.SCOPE,plan_commit=scale.PLAN,evaluation_stage=scale.DEV,
        compiled_population={},packages={arm:{}},interface_predecessor={},mobile_world_lock='lock',ROS_DOMAIN_ID=194,
        owner='owner',source_sha='source',registered_root=str(root),condition_stop=6,cases=paths)
    write(spec,parent)
    write(root/'control/LEASE.json',dict(status='CLOSED',spec_sha256=cards.pin(spec)['sha256']))
    terminal = root/'RESULT.private.json'
    result = dict(source_sha='source',exit_code=0,unresolved_case_launches=0,failure=None,
        records=sealed,attempted_case_launches=6,new_verified_native_requests=6,
        model_service=dict(exit_code=0,attempted_generations=6,completed_generations=6))
    write(terminal,result)
    pending = dict(parent,condition_stop=12,
        previous_panel_segments=[dict(spec=cards.pin(spec),result=cards.pin(terminal))])
    assert suffix.verified_prefix(pending,ordered)==sealed
    write(terminal,dict(result,new_verified_native_requests=5))
    pending['previous_panel_segments'][0]['result']=cards.pin(terminal)
    with pytest.raises(ValueError,match='RECONCILED_FINITE'):
        suffix.verified_prefix(pending,ordered)


def test_student_registration_binds_the_worker_and_all_fixed_cases(tmp_path, monkeypatch):
    def sealed(path, data):
        write(path,data)
        return cards.pin(path)

    root = tmp_path/'screen'
    prior = tmp_path/'qualified-interfaces'
    (prior/'control').mkdir(parents=True)
    templates = [sealed(prior/family,dict(mobile_condition=dict(family=family),frozen_files={},
                 native_environment={},python='/qualified/native/python'))['path'] for family in ('F1','F2')]
    interface = sealed(prior/'EXECUTION_SPEC.json',dict(registered_root=str(prior),cases=templates))
    write(prior/'control/LEASE.json',dict(status='CLOSED',spec_sha256=interface['sha256']))
    monkeypatch.setattr(cards,'unit_state',lambda unit:dict(pid=0,cgroup_pids=[]))
    monkeypatch.setattr(teacher,'interface_records',lambda *_:[{},{}])
    monkeypatch.setattr(student,'interface_records',teacher.interface_records)
    # Keep the actual package and server verification. Physics is in a separate
    # runtime preflight and no native or GPU executable is used in this fixture.
    monkeypatch.setattr(cards,'validate',student_package)
    monkeypatch.setattr(student,'validate',student_package)

    clusters = [dict(id=f'{family}-{reset}',family=family,reset=reset,scoring_geometry=dict(sha256=family),
                     scene=dict(path=f'/{family}/{reset}.mjb',sha256='c'*64))
                for family in ('F1','F2') for reset in (0,1)]
    rows = [dict(id=f"{c['id']}-{name}",cluster=c['id'],family=c['family'],reset=c['reset'],
                 instruction=name,noise_seed=i,arm_order=ARMS if i%2==0 else ARMS[::-1])
            for c in clusters for i,name in enumerate(INSTRUCTIONS)]
    population = sealed(tmp_path/'SCENES.json',dict(clusters=clusters,conditions=dict(SCREEN=rows)))
    data = sealed(tmp_path/'TARGET.json',dict(state='committed',statistics=dict(sha256='d'*64),
                                            compiled_population_sha256=population['sha256']))
    parent_asset = dict(path='/verified/parent',sha256='a'*64)
    deadline = time.time()+3600
    allocation = deadline+3600
    lock = str(tmp_path/'neural.lock')
    params = tmp_path/'params'
    params.mkdir()
    site = sealed(tmp_path/'SITE.json',dict(allocation_deadline_epoch=allocation,host='corelab',uid=2009,
                            allocation=dict(neural_owner_lock=lock),paths={}))
    template = sealed(tmp_path/'GPU_TEMPLATE.json',dict(site_binding=site,neural_owner_lock=lock,
        python='/qualified/gpu/python',parent_manifest=parent_asset,parent_params=str(params),
        tokenizer={},initial_function_rule={},upstream_sha='b'*40,GPU_uuid='GPU-TEST',
        JAX_memory_fraction=.3,allocator_limit_bytes=1,minimum_host_available_bytes=1,
        RPC_timeout_s=120,model_service_timeout_s=30,save_and_exit_on_limit=True,
        frame_cache_bytes=16,native_environment=dict(CUDA_VISIBLE_DEVICES='GPU-TEST',PYTHONPATH='old')))
    packages, results, rgb = {}, {}, {}
    for arm in ARMS:
        package = tmp_path/arm
        package.mkdir()
        manifest = sealed(package/'MANIFEST.private.json',dict(schema=SCHEMA,codec=NATIVE_METHOD,
            decode_order=DECODE_ORDER,label_epsilon=EPSILON,step=1000,cursor=32000,parent_manifest=parent_asset,
            statistics_sha256='d'*64,bindings=dict(recipe=RECIPE,codec_version=NATIVE_METHOD,arm=arm,
                inference_route='cached_actual',dataset_manifest_sha256=data['sha256'],
                compiled_population_sha256=population['sha256']),adapter_arrays=[None]*28,
            previous_reference_required=True,teacher_required=False,training_cache_required=False,optimizer_required=False,
            state_gripper_transformed=False,physical_execution_admitted=False,endpoint_basis_applied=False))
        packages[arm] = dict(path=str(package),sha256=manifest['sha256'])
        results[arm] = sealed(package/'TRAINED.json',dict(arm=arm,exit_code=0,package=packages[arm],
            new_updates=1000,new_visits=32000,legacy_C20_visits=6400,teacher_visits=25600,
            actual=dict(training_full_generations=4000),full_restore=dict(bit_exact=True,arrays=87)))
        rgb[arm] = sealed(package/'RGB16.json',dict(package=packages[arm],exit_code=0,passed=True,
            actual_generations=16,new_optimizer_updates=0,new_native_requests=0,
            comparisons=[dict(normalized_exact=True,EEF10_exact=True)]*16))
    registry = tmp_path/'REGISTRY.json'
    write(registry,dict(entries=[]))
    spec = dict(registered_root=str(root),campaign_root=str(root),plan_commit=PLAN,scope=SCOPE,physics=True,
        new_optimizer_updates_budget=0,evaluation_stage='SCREEN',interface_predecessor=interface,
        compiled_population=population,multitask_dataset=data,gpu_template=template,neural_owner_lock=lock,
        packages=packages,endpoints=results,ordinary_RGB_restores=rgb,registry=str(registry),
        source_sha='s'*40,source_root=str(tmp_path),source_files={},source_seal='id',
        trusted_source_parent={},goal_plan_sha=PLAN,owner='owner',host='corelab',uid=2009,
        world_control=str(tmp_path),mobile_world_lock=str(tmp_path/'world.lock'),ROS_DOMAIN_ID=198,
        allocation_deadline_epoch=allocation,campaign_deadline_epoch=deadline,deadline_epoch=deadline,
        sim_only_project_authorization={},native_environment=dict(PYTHONPATH='new'))

    parent = read_pin(cards.register_student_panel(spec))
    _, endpoints = student.preflight(parent)
    assert set(endpoints)==set(ARMS)
    worker = read_pin(parent['resident_server_spec'])
    assert parent['new_physical_requests_budget']==48
    assert len(cards.read(registry)['entries'])==50
    assert len({cards.read(path)['stage_identity_sha256'] for path in parent['cases']})==48
    assert worker['new_complete_generations_budget']==worker['unified_requests_budget']==7680
    assert worker['new_physical_requests_budget']==worker['new_optimizer_updates_budget']==0
    assert worker['python']=='/qualified/gpu/python'
    assert worker['gpu_environment']['PYTHONPATH']=='new' and worker['gpu_environment']['CUDA_VISIBLE_DEVICES']=='GPU-TEST'
    assert read_pin(worker['site_binding'])['allocation']['campaign_root']==str(root)
    assert all(worker['request_bindings_by_arm'][arm]['policy_package_sha256']==packages[arm]['sha256']
               for arm in ARMS)
    assert len(list(Path(worker['ipc_root']).iterdir()))==0
    assert not (Path(worker['registered_root'])/'READY.private.json').exists()
    with pytest.raises(PermissionError,match='NEW_OWNED_CARDS'):
        cards.register_student_panel(spec)
def test_adaptation_batch_two_registers_a_disjoint_target_population():
    from scripts.sla.prepare_target_adaptation import population
    from scripts.wm02.multitask_d2_population import position_id
    from go2arx5_vla.world_model import target_adaptation_teacher as adaptation

    cards = {name:dict(object_positions_world={'red':[.4,.1,.2],'blue':[.7,.1,.2]}) for name in ('F1','F2')}
    first = population(cards,[],[])
    for number, cluster in enumerate(first['clusters']):
        cluster['scene'] = dict(sha256=f'{number:064x}')
    next_batch = population(cards,[first],[],scope=adaptation.SECOND_SCOPE)
    for number, cluster in enumerate(next_batch['clusters'],len(first['clusters'])):
        cluster['scene'] = dict(sha256=f'{number:064x}')
    first_geometry = {(c['family'],position_id(c['card'])) for c in first['clusters']}
    second_geometry = {(c['family'],position_id(c['card'])) for c in next_batch['clusters']}
    assert not first_geometry & second_geometry
    before = adaptation.conditions(first)
    after = adaptation.conditions(next_batch,adaptation.SECOND_SCOPE)
    assert not {row['reset'] for row in before} & {row['reset'] for row in after}
    assert [(row['family'],row['instruction']) for row in before] == [(row['family'],row['instruction']) for row in after]
