"""Finite raw four-task recipe and actual same-device native10 admission.

Uses the existing NativeDataset, M32 contract, Trainer and owner. No new
scheduler, usage ledger, loss, decoder, dataset population or statistics fit.
"""
import hashlib
import json
from pathlib import Path

from .shared_task_sampling import recipe as input_recipe,validate_fixed59,TASKS
from .raw_parent_delivery import PARENT_ID

KIND='V7_SHARED_RAW_M32_FINITE_LEARNING'
SCHEMA='SHARED4_RAW_M32_TASK8_TRAINER8192_V2'
DIAGNOSTIC_EPISODES=('PI0294-PV1-TRAIN-F2-020-T1-C04','PI0294-PV1-TRAIN-F2-020-T2-C04',
                     'BOOK_SOURCE_SEED000','CUSTOM_CABINET_PORT08_SOURCE_SEED000')


def recipe(manifest_sha256,statistics_sha256,population_sha256,task_DEV_sha256,*,steps=8192):
    p=input_recipe(manifest_sha256,statistics_sha256,population_sha256,task_DEV_sha256,steps=steps)
    p.update(schema=SCHEMA,diagnostic_seed_namespace='SHARED4_RAW_TRAIN16_V1',
             diagnostic_windows=[dict(id=name,task=task,anchor=a,split='TRAIN')
                 for task,name in zip(TASKS,DIAGNOSTIC_EPISODES)for a in(0,48,200,400)])
    return p


def is_recipe(value):
    try:
        return value==recipe(value['dataset_manifest_sha256'],value['statistics_sha256'],
            value['TRAIN_population_sha256'],value['task_DEV_population_sha256'],steps=value['max_steps'])
    except(KeyError,TypeError,ValueError):return False


def validate_manifest(m,p):
    limits=dict(model_loads=1,optimizer_updates=p.get('max_steps'),physics_integrations=0,
                generation_queries=p.get('generation_queries'),RGB_pairs=0,source_constructors=0)
    if (not is_recipe(p)or m.get('kind')!=KIND or m.get('recipe')!=p
            or m.get('raw_parent_files_identity')!=PARENT_ID
            or any(m.get(k)is not None for k in('checkpoint_manifest_sha256','delta_checkpoint',
                'delta_checkpoint_manifest_sha256','parent_checkpoint','GPU_capacity_qualification',
                'GPU_inference_qualification','GPU_inference_admission'))
            or m.get('limits')!=limits):
        raise PermissionError('EXACT_RAW_SHARED_RECIPE_NO_ADAPTED_PARENT_OR_OLD_CAPACITY_ALIAS')


def validate_data(data,p):
    from .shared_task_data import DATA_SCHEMA,POPULATION_SCHEMA
    from scripts.pi0294.native_dataset import sha
    from scripts.pi0292.durable import read
    root=data.root;m=data.manifest;pop=read(root/'TRAIN_POPULATION.json');dev=read(root/'TASK_DEV_POPULATION.json')
    if (m.get('schema')!=DATA_SCHEMA or m.get('training_admitted')is not True
            or data.manifest_sha!=p['dataset_manifest_sha256']or data.statistics_sha!=p['statistics_sha256']
            or sha(root/'TRAIN_POPULATION.json')!=p['TRAIN_population_sha256']
            or sha(root/'TASK_DEV_POPULATION.json')!=p['task_DEV_population_sha256']
            or m.get('task_DEV_population_sha256')!=p['task_DEV_population_sha256']
            or pop.get('schema')!=POPULATION_SCHEMA or pop.get('episodes')!=m.get('episodes')
            or {t:len(dev.get('tasks',{}).get(t,[]))for t in TASKS}!=dict(T1=24,T2=24,BOOK=10,DOOR=10)
            or any(r.get('split')!='TRAIN'for r in m['episodes'])):
        raise PermissionError('FROZEN_SHARED59_WITH_COMPLETE_ORIGINAL_DEV_REFERENCE')
    validate_fixed59(data.rows)


def qualified_entry(m):
    """A book-only certificate never authorizes this distinct raw model."""
    from .activation_offload import LOW_RESIDENCY_POLICY
    p=m.get('recipe',{});validate_manifest(m,p)
    raw=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
    memory_sha=hashlib.sha256(raw(LOW_RESIDENCY_POLICY)).hexdigest()
    if (m.get('memory_execution_sha256')!=memory_sha
            or m.get('sealed_inputs',{}).get('M32_MEMORY_EXECUTION.json')!=memory_sha
            or m.get('training_GPU_entry_MiB')not in(15360,24576)):
        raise PermissionError('RAW_SHARED_EXPLICIT_V2_MEMORY_AND_MEASURED_DEVICE_ENTRY')
    request=m.get('GPU_training_qualification');admission=m.get('GPU_training_admission')
    from .book_gpu_capacity import TRAINING_REQUEST
    if p['max_steps']==10:
        if request!=TRAINING_REQUEST or admission is not None:
            raise PermissionError('RAW_SHARED_NATIVE10_IS_QUALIFICATION_ONLY')
        return m['training_GPU_entry_MiB']*1024**2
    required={'RUN_MANIFEST.json','RESULT.json','M32_CAPACITY_RESULT.json','ACTUAL_DELTA_CHECKPOINT_RESTORE.json',
              'ACTUAL_ACTIVATION_OFFLOAD.json','RAW_ZERO_STEP_PARAMETERS.json','control/LEASE.json'}
    if (request is not None or not isinstance(admission,dict)
            or admission.get('schema')!='SHARED4_RAW_SAME_GPU_NATIVE10_TRAINING_V1'
            or set(admission.get('files',{}))!=required):
        raise PermissionError('RAW_SHARED_OWN_CLOSED_SAME_GPU_NATIVE10_REQUIRED')
    root=Path(admission['qualification_root']);values={}
    if not root.is_absolute()or root.is_symlink():raise PermissionError('RAW_SHARED_QUALIFICATION_ROOT')
    for name,pin in admission['files'].items():
        path=root/name
        if path.is_symlink()or not path.resolve().is_relative_to(root.resolve()):
            raise PermissionError('RAW_SHARED_NATIVE_CERTIFICATE_PATH')
        body=path.read_bytes()
        if hashlib.sha256(body).hexdigest()!=pin:raise PermissionError('RAW_SHARED_CERTIFICATE_CHANGED:'+name)
        values[name]=json.loads(body)
    q=values['RUN_MANIFEST.json'];result=values['RESULT.json'];capacity=values['M32_CAPACITY_RESULT.json']
    restore=values['ACTUAL_DELTA_CHECKPOINT_RESTORE.json'];lease=values['control/LEASE.json']
    expected=recipe(p['dataset_manifest_sha256'],p['statistics_sha256'],p['TRAIN_population_sha256'],
                    p['task_DEV_population_sha256'],steps=10)
    zero=values['RAW_ZERO_STEP_PARAMETERS.json'];copies=values['ACTUAL_ACTIVATION_OFFLOAD.json']
    if (q.get('recipe')!=expected or q.get('GPU_uuid')!=m.get('GPU_uuid')
            or result.get('status')!='COMPLETE_SHARED_RAW_FINITE_LEARNING'or result.get('error')is not None
            or any(result.get(k)!=v for k,v in dict(model_load_attempts=1,optimizer_updates=10,TRAIN_visits=320,
                generation_attempts=32,completed_generations=32,physics_integrations=0,DEV_conditions=0).items())
            or capacity.get('capacity_pass')is not True or capacity.get('minimum_observed_free_bytes',0)<2*1024**3
            or capacity.get('actual_optimizer_updates')!=10 or capacity.get('saved_and_restored')is not True
            or restore.get('required_raw_parent_files_identity')!=PARENT_ID
            or any(restore.get(k)!=v for k,v in dict(model_reconstruction=True,optimizer_updates=10,
                optimizer_state_entries=42,scheduler_restored=True,RNG_restored=True,sampler_cursor=320,frozen_parent_unchanged=True).items())
            or zero.get('operation')!='COMPARE_FRESH_LOADED_MODEL'or zero.get('raw_parent_files_identity')!=PARENT_ID
            or zero.get('trainable_tensor_count')!=42 or zero.get('trainable_numel')!=77847684
            or copies.get('policy')!=LOW_RESIDENCY_POLICY or copies.get('copied_tensors',0)<=0
            or copies.get('restored_tensors',0)<=0 or lease.get('status')!='CLOSED'
            or lease.get('manifest_sha256')!=admission['files']['RUN_MANIFEST.json']
            or lease.get('result_sha256')!=admission['files']['RESULT.json']):
        raise PermissionError('RAW_SHARED_NATIVE10_FULL_RESTORE_REAL_GENERATIONS_AND_TWO_GIB_MARGIN')
    qualified_entry(q)
    return m['training_GPU_entry_MiB']*1024**2
