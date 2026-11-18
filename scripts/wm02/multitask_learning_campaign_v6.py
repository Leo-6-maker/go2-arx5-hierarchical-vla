"""Owned M target-data, fixed endpoints and independent restore sequence."""
from pathlib import Path
import time
import traceback

from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.sla_adapter.multitask_contract_v6 import ARMS, RECIPE
from go2arx5_vla.sla_adapter.multitask_initialization_v6 import sealed_source
from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs
from go2arx5_vla.world_model.multitask_scope_v6 import PLAN, SCOPE, read_pin
from .multitask_stage_cards_v6 import bind_gpu, pin, read, require_unstarted_cases, verified_teacher_prefix
from .target_followup_v6 import run_child


def preflight(spec):
    arms = tuple(spec['training_arms'])
    if (spec['plan_commit']!=PLAN or spec['scope']!=SCOPE or spec['physics'] or spec['new_physical_requests_budget']
            or arms not in ((ARMS[0],),ARMS,(ARMS[1],))
            or spec['new_optimizer_updates_budget']!=len(arms)*1000
            or spec['new_training_visits_budget']!=len(arms)*32000
            or spec['new_model_generations_budget']!=len(arms)*124
            or (arms==(ARMS[1],) and not all(spec.get(k) for k in ('existing_prepared_result','existing_M_BASE_endpoint')))):
        raise PermissionError('EXACT_PREDECLARED_M_ARMS_AND_GENERATION_BUDGET_REQUIRED')
    now = time.time()
    if not now<spec['deadline_epoch']<=min(now+43200,spec['campaign_deadline_epoch'],spec['allocation_deadline_epoch']-60):
        raise PermissionError('LIVE_FINITE_M_LEARNING_WINDOW_REQUIRED')
    return arms


def child_spec(spec, name):
    root = Path(spec['registered_root'])
    folder = root/name
    dirs(folder)
    child = dict(spec,registered_root=str(folder),campaign_root=str(root),role='SLA_V6_M_'+name,
                 new_optimizer_updates_budget=0,new_training_visits_budget=0,new_model_generations_budget=0)
    child['native_environment'] = dict(spec['native_environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'))
    return folder, child


def prepare_data(spec):
    verified_teacher_prefix(spec['previous_teacher_results'],spec['compiled_population'],world_lock=spec['mobile_world_lock'])
    folder, child = child_spec(spec,'DATA')
    child['teacher_campaigns'] = [dict(result=value,spec=pin(Path(value['path']).parent/'EXECUTION_SPEC.json'))
                                  for value in spec['previous_teacher_results']]
    child['native_environment'].update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    register(spec,[(folder,child)])
    return dict(spec=pin(folder/'EXECUTION_SPEC.json'))


def dataset_result(prepared, spec):
    actual = read_pin(prepared)
    if actual['exit_code'] or actual['status']!='G7_TARGET_TRAIN_READY':
        raise ValueError('SEALED_ACTUAL_QUALIFIED_M_TARGET_DATA_REQUIRED')
    manifest = read_pin(actual['dataset'])
    audit = pin(Path(actual['dataset']['path']).parent/'ACTUAL.private.json')
    if (manifest['state']!='committed'
            or manifest['compiled_population_sha256']!=spec['compiled_population']['sha256']
            or manifest['legacy_C20']!=spec['legacy_C20'] or manifest['audit_sha256']!=audit['sha256']
            or spec['previous_teacher_results']!=[stage['result'] for stage in manifest['teacher_campaigns']]):
        raise ValueError('SEALED_ACTUAL_QUALIFIED_M_TARGET_DATA_AND_STATS_REQUIRED')
    return actual['dataset'],audit


def register_gpu_stage(spec, arm, dataset, audit, *, training=None, M_BASE_initial=None):
    restore = training is not None
    name = ('RESTORE-' if restore else 'TRAIN-')+arm
    require_unstarted_cases(spec['registry'],{name})
    folder, child = child_spec(spec,name)
    child.update(arm=arm,dataset=str(Path(dataset['path']).parent),multitask_dataset=dataset,target_audit=audit,
                 legacy_C20_root=spec['legacy_C20']['dataset'],new_model_generations_budget=16 if restore else 108)
    if restore:
        child['trained_result'] = training
    else:
        c20_bank = Path(spec['C20_TRAIN48_bank']['path'])
        read_pin(spec['C20_TRAIN48_bank'])
        child.update(operation=RECIPE,microbatch=4,training_matmul_precision='highest',planned_candidate_updates=1000,
            new_optimizer_updates_budget=1000,new_training_visits_budget=32000,inference_route='cached_actual',
            aux_samples_per_update=4,aux_sampling_steps=10,previous_campaign=str(c20_bank.parent.parent),
            S1000_transfer_result=spec.get('S1000_transfer_result') if arm=='M-SRC' else None)
        if arm=='M-SRC':
            child['matched_M_BASE_initial'] = M_BASE_initial
    bind_gpu(spec,child,folder)
    register(spec,[(folder,child)])
    return dict(spec=pin(folder/'EXECUTION_SPEC.json'))


def fixed_endpoint(pin_result, arm, data_sha):
    result = read_pin(pin_result)
    if (result['exit_code'] or result['arm']!=arm or result['new_updates']!=1000 or result['new_visits']!=32000
            or not result['full_restore']['bit_exact']):
        raise ValueError('ACTUAL_FIXED1000_COMPLETE_RESTORE_REQUIRED')
    manifest = read_pin(dict(path=str(Path(result['package']['path'])/'MANIFEST.private.json'),sha256=result['package']['sha256']))
    if manifest['bindings']['dataset_manifest_sha256']!=data_sha:
        raise ValueError('ACTUAL_FIXED1000_AND_SAME_G7_TRAIN_POPULATION_REQUIRED')
    return result


def run(spec):
    root = Path(spec['registered_root'])
    journal = Journal(root/'control/journal',root.name)
    actual = dict(source_sha=spec['source_sha'],endpoints={},independent_RGB16={},new_native_requests=0,new_native_ticks=0)
    try:
        arms = preflight(spec)
        data = spec.get('existing_prepared_result')
        if data is None:
            prepared = journal.once('REGISTER-DATA',dict(campaigns=spec['previous_teacher_results']),lambda: prepare_data(spec))
            receipt = run_child(spec,read_pin(prepared['spec']),'--multitask-dataset',journal,'RUN-DATA')
            data = receipt['result']
        dataset,audit = dataset_result(data,spec)
        actual['data_result'] = data
        base = spec.get('existing_M_BASE_endpoint')
        for arm in arms:
            if arm=='M-SRC':
                source = spec.get('S1000_transfer_result')
                if source is None:
                    actual['M-SRC'] = 'SOURCE_GATE_PENDING_UNSTARTED'
                    break
                source_result = read_pin(source)
                if source_result['exit_code'] or not source_result['transfer_gate']['passed']:
                    actual['M-SRC'] = 'SOURCE_NO_TRANSFER_UNSTARTED'
                    actual['source_no_transfer_result'] = source
                    break
                template = read_pin(spec['gpu_template'])
                sealed_source(source,template['parent_manifest'])
                base_initial = fixed_endpoint(base,'M-BASE',dataset['sha256'])['initial_checkpoint']
            else:
                base_initial = None
            if time.time()>=spec['deadline_epoch']-600 or (root/'STOP.json').exists():
                actual['remaining_arm'] = dict(arm=arm,status='FINITE_WINDOW_UNSTARTED')
                break
            training = journal.once('REGISTER-TRAIN-'+arm,dict(dataset=dataset,arm=arm,base_initial=base_initial),
                lambda: register_gpu_stage(spec,arm,dataset,audit,M_BASE_initial=base_initial))
            receipt = run_child(spec,read_pin(training['spec']),'--multitask-train',journal,'RUN-TRAIN-'+arm)
            trained = receipt['result']
            fixed_endpoint(trained,arm,dataset['sha256'])
            actual['endpoints'][arm] = trained
            if arm=='M-BASE':
                base = trained
            restore = journal.once('REGISTER-RESTORE-'+arm,dict(dataset=dataset,training=trained),
                lambda: register_gpu_stage(spec,arm,dataset,audit,training=trained))
            receipt = run_child(spec,read_pin(restore['spec']),'--multitask-restore',journal,'RUN-RESTORE-'+arm)
            checked = read_pin(receipt['result'])
            if not checked['passed'] or checked['actual_generations']!=16:
                raise ValueError('ACTUAL_INDEPENDENT_M_ORDINARY_RGB16_REQUIRED')
            actual['independent_RGB16'][arm] = receipt['result']
            write(root/'PROGRESS.private.json',dict(phase='M_ENDPOINT_AND_RGB16_SEALED',actual=actual,epoch=time.time()))
        actual.update(exit_code=0,status='M_SEALED_ENDPOINT_RESULTS' if len(actual['independent_RGB16'])==len(arms)
                      else 'M_BOUNDED_STAGE_WITH_UNSTARTED_ARMS')
    except BaseException:
        actual.update(exit_code=1,status='PRESERVED_M_LEARNING_TERMINAL_NO_AUTORETRY',failure=traceback.format_exc())
        last = root/'CURRENT_CHILD.private.json'
        if last.exists():
            last_child = read(last)
            actual['last_registered_child'] = last_child
            result = Path(last_child['spec']['path']).parent/'RESULT.private.json'
            if result.is_file():
                actual['last_registered_child_terminal'] = pin(result)
    write(root/'RESULT.private.json',actual)
    close(spec,actual['exit_code'])
    return actual['exit_code']
