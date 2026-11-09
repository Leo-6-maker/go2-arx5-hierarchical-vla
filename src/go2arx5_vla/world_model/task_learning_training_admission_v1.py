"""Actual continuous-data and original-Trainer receipts before a new SFT model."""
import ast,json
from pathlib import Path
from . import target_gr00t_profile_v1 as original
from . import task_learning_training_profile_v1 as task
from .stable_task_scene_v1 import pinned
from .storage import digest

TRAINING='src/go2arx5_vla/world_model/target_gr00t_training_v1.py'
PROFILE='src/go2arx5_vla/world_model/target_gr00t_profile_v1.py'


def sections(path):
    source=Path(path).read_text();result={}
    def visit(nodes,parent=''):
        for node in nodes:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                name=parent+'.'+node.name if parent else node.name
                result[name]=ast.get_source_segment(source,node)
                visit(node.body,name)
    visit(ast.parse(source).body)
    calls=[ast.get_source_segment(source,n)for n in ast.walk(ast.parse(source))
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='TrainingArguments']
    original.need(len(calls)==1,'ONE_ORIGINAL_TRAINING_ARGUMENTS_CONSTRUCTION')
    result['TrainingArguments']=calls[0]
    return result


def unchanged_training_math(qualified,current):
    """Permit task/data/probe metadata changes, retaining the qualified kernels."""
    a,b=Path(qualified['source_root']),Path(current['source_root']);need=original.need
    names=('_load_original_parent','load_checkpoint','semantic_recipe','run.Sampler.index_at',
        'run.TargetTrainer.training_step','run.TargetTrainer.get_train_dataloader',
        'run.TargetTrainer.create_optimizer','run.TargetTrainer.create_scheduler',
        'run.TargetTrainer._load_from_checkpoint','run.TargetTrainer._save_checkpoint',
        'run.Audit.on_pre_optimizer_step','run.Audit.on_step_end','TrainingArguments')
    before,after=sections(a/TRAINING),sections(b/TRAINING)
    need(all(before[n]==after[n]for n in names),'QUALIFIED_NATIVE_TRAINER_LOSS_OPTIMIZER_SAMPLER_AND_RESTORE_MATH_UNCHANGED')
    for name in('scripts/pi0294/native_train.py','scripts/pi0294/numerics.py',
                'scripts/pi0294/generalization/minimal_adapter.py'):
        need(qualified['source_files'][name]==current['source_files'][name]
            and digest(b/name)==current['source_files'][name],
            'UNCHANGED_ORIGINAL_NUMERICAL_AND_MASKED_LOSS_KERNEL_'+name)
    def whitelist(root):
        source=(root/PROFILE).read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='whitelist')
        return ast.get_source_segment(source,node)
    need(whitelist(a)==whitelist(b),'SAME_QUALIFIED_FULL_INTERNAL_ACTION_WHITELIST')
    return dict(status='QUALIFIED_ORIGINAL_TRAINING_MATH_RETAINED',compared_sections=list(names))


def fit4_qualification(proof,current):
    """The old four-window fit is a runtime qualification, never new-task data."""
    from .task_learning_hopper_v1 import qualified_fit4_forecast
    need=original.need;c=pinned(proof['config']);root=Path(c['root'])
    need(Path(proof['config']['path'])==root/'ALLOCATION_CONFIG.json'
        and root.is_relative_to(Path('/scratch/e1192548/Projects/GR00T-WBC/runs/WM02')),
        'ACTUAL_OWN_COMPLETED_HOPPER_QUALIFICATION_ROOT')
    values={}
    for name in('lease','worker','result'):
        rel={'lease':'control/LEASE.json','worker':'WORKER_EXIT.private.json','result':'RESULT.private.json'}[name]
        need(Path(proof[name]['path'])==root/rel,'BOUND_REAL_FIT4_TERMINAL_'+name)
        values[name]=pinned(proof[name])
    lease,worker,result=(values[n]for n in('lease','worker','result'))
    children=result.get('children')or[]
    need(lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==worker.get('exit_code')==0
        and worker.get('failure')is None and lease.get('result_sha256')==proof['result']['sha256']
        and result.get('status')=='HOPPER_ORIGINAL_TRAINER_FIT4_ONLY_SEALED'
        and result.get('fit4_gate',{}).get('passes')is True and len(children)==1,
        'ACTUAL_ORIGINAL_TRAINER_FIT4_COMPLETE_AND_ALL_FOUR_GATES')
    child=children[0];r=child.get('result')or{};restore=r.get('restore')or{}
    need(child.get('exit_code')==0 and r.get('global_step')==200 and r.get('sample_visits')==6400
        and r.get('real_action_parameter_changed')is True and r.get('backbone_frozen')is True
        and r.get('external_adapter_updates')==r.get('native_physics')==0
        and all(restore.get(k)is True for k in('model_parameters_restored','optimizer_restored',
            'scheduler_restored','RNG_restored','fixed_batch_generation_restored'))
        and restore.get('sampler_cursor')==6400,'ACTUAL_NONZERO_FIT_AND_FULL_ENDPOINT_RESTORE')
    full=qualified_fit4_forecast(c)
    need(c['python']==current['python']and c['parent']==current['parent']
        and all(c['bindings'][k]==current['bindings'][k]for k in('source','cosmos','cosmos_hub')),
        'SAME_QUALIFIED_ORIGINAL_PARENT_AND_TRAINING_RUNTIME')
    math=unchanged_training_math(c,current)
    return dict(original_full_restore_job=full['predecessor_job_id'],FIT4_job=lease['job_id'],
        checkpoint_sha256=restore['checkpoint_sha256'],training_math=math,FIT4_weights_used_as_formal_parent=False)


def processor_qualification(proof,spec):
    """Require the actual finite Processor worker, including reserved GPU cost."""
    from .task_learning_hopper_processor_v1 import PROFILE as PROCESSOR_PROFILE,ROOT,configuration
    need=original.need;r=pinned(proof);root=Path(proof['path']).parent
    need(root.is_relative_to(ROOT)and Path(proof['path'])==root/'RESULT.private.json',
        'ACTUAL_OWN_HOPPER_PROCESSOR_WORKER_RESULT')
    lease=json.loads((root/'control/LEASE.json').read_bytes())
    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    config_path=root/'ALLOCATION_CONFIG.json';c=json.loads(config_path.read_bytes())
    data=Path(spec['bindings']['dataset']);admission=data/'PROCESSOR_HOPPER_RECONSTRUCTION.json'
    need(r.get('status')=='PASS_ACTUAL_HOPPER_CONTINUOUS_PROCESSOR_RECONSTRUCTION'
        and r.get('exit_code')==lease.get('worker_exit_code')==worker.get('exit_code')==0
        and r.get('failure')is worker.get('failure')is None and lease.get('status')=='CLOSED'
        and lease.get('result_sha256')==proof['sha256']
        and r.get('job_id')==lease.get('job_id')==worker.get('job_id')
        and r.get('config_sha256')==lease.get('config_sha256')==digest(config_path)
        and c.get('profile')==PROCESSOR_PROFILE and c.get('dataset')==str(data)
        and c.get('pbs_server')==spec.get('pbs_server')
        and c.get('python')==spec.get('python')
        and all(c.get('bindings',{}).get(k)==spec['bindings'][k]for k in('source','cosmos','cosmos_hub'))
        and r.get('manifest_sha256')==spec['recipe']['dataset_sha256']
        and r.get('statistics_sha256')==spec['recipe']['statistics_sha256']
        and r.get('Processor_admission_sha256')==digest(admission)
        and r.get('new_policy_models')==r.get('new_optimizer_updates')==r.get('new_native_integrations')==0
        and r.get('GPU_allocations')==1 and r.get('CUDA_initialized')is False
        and r.get('reserved_GPU_wall_s',0)>0,
        'ACTUAL_CLOSED_ZERO_EXIT_PROCESSOR_BEFORE_FORMAL_MODEL_WITH_RESERVED_GPU_COST')
    configuration(c)
    for name in('scripts/wm02/qualify_target_gr00t_processor_v1.py',
        'src/go2arx5_vla/world_model/task_learning_native_dataset_v1.py',
        'src/go2arx5_vla/world_model/target_gr00t_dataset_v1.py',
        'scripts/pi0292/train_adapter.py','scripts/pi0292/eval_contract.py'):
        need(c['source_files'].get(name)is not None
            and c['source_files'][name]==spec['source_files'].get(name),
            'SAME_ACTUALLY_QUALIFIED_ORIGINAL_PROCESSOR_ROUTE_'+name)
    return dict(job_id=r['job_id'],result_sha256=proof['sha256'],Processor_admission_sha256=digest(admission))


def validate(spec):
    from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
    p=task.validate(spec);need=original.need;a=pinned(spec['task_learning_admission'])
    need(a.get('schema')=='TASK_LEARNING_FORMAL_TRAINING_ADMISSION_V1'
        and a.get('dataset_sha256')==p['dataset_sha256']and a.get('statistics_sha256')==p['statistics_sha256']
        and a.get('parent_manifest_sha256')==original.PARENT,
        'EXACT_NEW_CONTINUOUS_TASK_DATA_AND_PARENT_ADMISSION')
    exported=pinned(a['Core_export']);transfer=pinned(a['data_transfer'])
    need(exported.get('status')=='FORMAL_CONTINUOUS_BUNDLE_CREATED_WAIT_ACTUAL_PROCESSOR'
        and exported.get('manifest_sha256')==transfer.get('manifest_sha256')==p['dataset_sha256']
        and exported.get('statistics_sha256')==transfer.get('statistics_sha256')==p['statistics_sha256']
        and transfer.get('status')=='PASS_PRIVATE_TASK_LEARNING_DATA_TRANSFER',
        'ACTUAL_SEALED_CORE_EXPORT_AND_IDENTICAL_PRIVATE_TRANSFER')
    root=Path(spec['bindings']['dataset']);files=transfer.get('files')or{}
    need(bool(files)and root.resolve()==root,'ACTUAL_CANONICAL_TRANSFERRED_TASK_DATA')
    for name,pin in files.items():
        path=root/name
        need(not Path(name).is_absolute()and '..'not in Path(name).parts
            and path.resolve()==path and path.is_file()and digest(path)==pin,
            'UNCHANGED_TRANSFERRED_TASK_DATA_MEMBER_'+name)
    data=TaskLearningNativeDataset(root,spec['bindings']['source'],spec['bindings']['cosmos'],
        for_optimization=True,optimization_site='HOPPER_SINGLE_GPU')
    need(data.manifest_sha==p['dataset_sha256']and data.statistics_sha==p['statistics_sha256'],
        'ACTUAL_FULL_POPULATION_AND_ORIGINAL_PROCESSOR_ADMISSION')
    processor=json.loads((root/'PROCESSOR_TRAINING_ADMISSION.json').read_text())
    need(processor.get('dataset_family')=='TASK_LEARNING_CONTINUOUS_V1',
        'NEW_REAL_DECISIONS_PROCESSED_NOT_LEGACY_FOUR_WINDOW_ADMISSION')
    runtime=fit4_qualification(a['original_Trainer_FIT4'],spec)
    processor_worker=processor_qualification(a['Hopper_Processor_qualification'],spec)
    for key,schema in(('endpoint_decision','TASK_LEARNING_ENDPOINT_DECISION_V1'),
                       ('first_seed_learning_and_native_gain','TASK_LEARNING_FIRST_SEED_GAIN_DECISION_V1')):
        if key not in spec:continue
        decision=pinned(spec[key])
        need(decision.get('schema')==schema and decision.get('allows_execution')is True
            and decision.get('dataset_sha256')==p['dataset_sha256'],
            'BOUND_ACTUAL_SCIENTIFIC_DECISION_'+key)
        if key=='endpoint_decision':
            need(decision.get('next_endpoint')==p['endpoint']
                and decision.get('generation_id')==spec['generation_id'],'DECLARED_SAME_GENERATION_CONTINUATION_ENDPOINT')
        else:
            need(decision.get('first_seed')==20261109 and bool(decision.get('first_seed_generation_id'))
                and p['seed']in decision.get('allowed_confirmation_seeds',[]),
                'FIRST_SEED_EVIDENCE_FOR_THE_DECLARED_SEQUENTIAL_CONFIRMATION_SEED')
    return dict(status='ACTUAL_TASK_DATA_PROCESSOR_AND_ORIGINAL_TRAINER_ADMITTED',
        manifest_sha256=data.manifest_sha,statistics_sha256=data.statistics_sha,
        optimization_visits_per_population_pass=len(data),runtime_qualification=runtime,
        Hopper_Processor_qualification=processor_worker,
        new_policy_models=0,new_optimizer_updates=0,new_native_integrations=0)
