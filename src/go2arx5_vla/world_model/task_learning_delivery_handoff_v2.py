"""Read one actual formal SFT terminal and enumerate its private delivery.

No model load, job submission or filesystem mutation occurs here. Parent assets
and optimizer archives stay at their existing sites. Transport verifies the
declared package members while streaming them; the source need not hash the
same multi-GB delta yet again in a login-session metadata query.
"""
import json
from pathlib import Path

from . import task_learning_training_profile_v1 as profile
from .task_learning_grasp_v1 import need
from .storage import digest
from .task_learning_endpoint_decision_v2 import descriptor


def audit_training_root(root, actual_job, *, job_id):
    root=Path(root)
    def read(name):return json.loads((root/name).read_bytes())
    s,c,l,w,r=(read(n)for n in ('EXECUTION_SPEC.json','ALLOCATION_CONFIG.json',
        'control/LEASE.json','WORKER_EXIT.private.json','RESULT.json'))
    p=profile.validate(s)
    need(str(root)==s['registered_root']==c['root'] and root.resolve()==root
        and c['recipe']==s['recipe']and c['source_sha']==s['source_sha']
        and actual_job.get('job_state')=='F'and actual_job.get('Exit_status')==0
        and actual_job.get('Job_Name')=='gr00t_task_sft'
        and (actual_job.get('Job_Owner')or'').split('@')[0]=='e1192548'
        and l.get('status')=='CLOSED'and l.get('worker_exit_code')==w.get('exit_code')==0
        and w.get('failure')is None and l.get('job_id')==w.get('job_id')==s.get('job_id')==job_id
        and l.get('spec_sha256')==digest(root/'EXECUTION_SPEC.json')
        and l.get('config_sha256')==digest(root/'ALLOCATION_CONFIG.json')
        and l.get('result_sha256')==digest(root/'RESULT.json'),
        'ACTUAL_PBS_F0_AND_ORIGINAL_FORMAL_RESULT_JSON_NOT_PRIVATE_RESULT')
    step=p['endpoint'];restore=r.get('restore')or{}
    checkpoint=Path(r['checkpoint'])
    need(checkpoint.is_relative_to(root/'training')and checkpoint.resolve()==checkpoint,
         'OWN_FORMAL_CHECKPOINT_PATH')
    saved=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes())
    need(r.get('status')=='CHECKPOINT_SEALED'and r.get('profile')==profile.PROFILE
        and r.get('source_sha')==s['source_sha']
        and r.get('global_step')==saved.get('global_step')==step
        and r.get('generation_id')==saved.get('generation_id')==s['generation_id']
        and saved.get('recipe')==p and r.get('GR00T_updates')==step-s['start_step']
        and r.get('sample_visits')==(step-s['start_step'])*32
        and r.get('backbone_frozen')is True and r.get('real_action_parameter_changed')is True
        and r.get('external_adapter_updates')==r.get('native_physics')==0
        and all(restore.get(k)is True for k in ('model_parameters_restored','optimizer_restored',
            'scheduler_restored','RNG_restored','fixed_batch_generation_restored'))
        and restore.get('sampler_cursor')==step*32
        and restore.get('checkpoint_sha256')==digest(checkpoint/'CHECKPOINT_COMPLETE.json')
        and read('RESTORE_ACTUAL.json')==restore,
        'ACTUAL_FORMAL_ACTION_UPDATES_FREEZE_AND_COMPLETE_ENDPOINT_RESTORE')
    package=r['inference_package'];folder=Path(package['path'])
    need(folder==root/'inference-package'and digest(folder/'PACKAGE_MANIFEST.json')==package['sha256'],
         'ACTUAL_ENDPOINT_PACKAGE_PATH_AND_MANIFEST')
    manifest=json.loads((folder/'PACKAGE_MANIFEST.json').read_bytes())
    proof=read('PACKAGE_RECONSTRUCTION_ACTUAL.json')
    need(manifest.get('recipe')==p and manifest.get('generation_id')==s['generation_id']
        and manifest.get('source_sha')==s['source_sha']
        and manifest.get('parent_manifest_sha256')==s['parent_manifest_sha256']
        and manifest.get('global_step')==step and manifest.get('checkpoint_sha256')==restore['checkpoint_sha256']
        and manifest.get('training_site')=='HOPPER_SINGLE_GPU'
        and manifest.get('external_inference_adapter')is False
        and proof.get('status')=='PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION'
        and(proof.get('package')or{}).get('sha256')==package['sha256']
        and proof.get('actual_matched_queries')==40
        and proof.get('absolute_component_tolerance')==1e-6
        and set(proof.get('max_abs_error',{}))=={'physical_H40','normalized'}
        and all(0<=v<=1e-6 for v in proof['max_abs_error'].values()),
        'ACTUAL_FORTY_QUERY_ORDINARY_PARENT_DELTA_RECONSTRUCTION')
    files={}
    def add(name,path,pin=None):
        path=Path(path)
        need(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root),
             'OWN_REGULAR_DELIVERY_MEMBER')
        files[name]=dict(source=str(path),sha256=pin or digest(path),bytes=path.stat().st_size)
    for name,pin in manifest['files'].items():
        rel=Path(name)
        need(not rel.is_absolute()and '..'not in rel.parts,'PACKAGE_MEMBER_WITHOUT_PATH_ESCAPE')
        add('package/'+name,folder/name,pin)
    add('package/PACKAGE_MANIFEST.json',folder/'PACKAGE_MANIFEST.json',package['sha256'])
    for name in ('EXECUTION_SPEC.json','ALLOCATION_CONFIG.json','RESULT.json','RESTORE_ACTUAL.json',
                 'WORKER_EXIT.private.json','control/LEASE.json','PACKAGE_RECONSTRUCTION_ACTUAL.json'):
        add('producer/'+name,root/name)
    add('producer/checkpoint/CHECKPOINT_COMPLETE.json',checkpoint/'CHECKPOINT_COMPLETE.json')
    from .task_learning_probes_v1 import artifact,matched_count
    for label,split in [('before','TRAIN'),('before-DEV','DEV'),('restored','TRAIN'),
                        ('restored-DEV','DEV'),('package-reconstructed','TRAIN')]:
        meta=read(label+'/RESULT.json');matched_count(meta,meta)
        need(all(v['split']==split for v in meta['records']),'ALL_REAL_PHASE_PROBES_IN_DELIVERY')
        add('producer/'+label+'/RESULT.json',root/label/'RESULT.json')
        for row in meta['records']:
            add('producer/'+label+'/'+artifact(root/label,row).name,artifact(root/label,row),row['sha256'])
    return dict(schema='TASK_LEARNING_FORMAL_PRIVATE_DELIVERY_INVENTORY_V2',
        status='ACTUAL_FORMAL_ENDPOINT_AND_PACKAGE_READY_FOR_PRIVATE_TRANSFER',global_step=step,
        generation_id=s['generation_id'],source_sha=s['source_sha'],dataset_sha256=p['dataset_sha256'],
        statistics_sha256=p['statistics_sha256'],checkpoint_sha256=restore['checkpoint_sha256'],
        package_sha256=package['sha256'],PBS_job_id=s['job_id'],
        files=files,bytes=sum(v['bytes']for v in files.values()),
        source_result=descriptor(root/'RESULT.json'),
        parent_assets_copied=False,optimizer_archives_copied=False,
        Core_reconstruction='NOT_ESTABLISHED_BY_PRODUCER_RECONSTRUCTION')
