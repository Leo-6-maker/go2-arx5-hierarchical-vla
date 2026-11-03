"""Validate a CLOSED producer's complete state before continuing the same arm."""
import json
from pathlib import Path

def prepare(root,manifest,recipe,dataset):
    from scripts.pi0294.native_dataset import sha
    from .shared_checkpoint import validate_header
    from .shared_task_sampling import TaskBalancedSampler
    request=manifest.get('resume_checkpoint')
    if request is None:return None
    if recipe.get('max_steps')!=8192 or recipe.get('capacity_qualification_only'):
        raise PermissionError('SHARED_ONLY_FORMAL_SAME_ARM_RESUME')
    proof=root/'RESUME_CUSTODY.json'
    if manifest.get('sealed_inputs',{}).get(proof.name)!=sha(proof):
        raise PermissionError('SHARED_SEALED_RESUME_CUSTODY')
    custody=json.loads(proof.read_bytes());checkpoint=Path(request['path']).resolve(strict=True)
    complete=checkpoint/'CHECKPOINT_COMPLETE.json'
    if sha(complete)!=request.get('manifest_sha256'):
        raise PermissionError('SHARED_EXACT_RESUME_CHECKPOINT')
    header=validate_header(json.loads(complete.read_bytes()));step=header['global_step']
    if (header['recipe']!=recipe or step not in (2048,4096,6144)
            or custody.get('science_run_id')!=manifest.get('science_run_id')
            or not custody.get('science_run_id')
            or custody.get('checkpoint_manifest_sha256')!=request['manifest_sha256']):
        raise PermissionError('SHARED_SAME_SCIENTIFIC_ARM_AND_COMMITTED_ENDPOINT')
    values={}
    for name,pin in custody.get('files',{}).items():
        f=proof.parent/'resume_custody'/name
        if f.is_symlink()or not f.resolve().is_relative_to((proof.parent/'resume_custody').resolve())or sha(f)!=pin:
            raise PermissionError('SHARED_RESUME_SOURCE_PROOF_CHANGED')
        values[name]=json.loads(f.read_bytes())
    if set(values)!={'RUN_MANIFEST.json','RESULT.json','LEASE.json','RESTORE.json','TERMINAL.json'}:
        raise PermissionError('SHARED_COMPLETE_PRODUCER_CLOSURE_PROOF')
    source=values['RUN_MANIFEST.json']
    result,lease,restore,terminal=(values[k]for k in ('RESULT.json','LEASE.json','RESTORE.json','TERMINAL.json'))
    if (source.get('science_run_id')!=custody['science_run_id'] or source.get('recipe')!=recipe
            or source.get('run_uuid')!=lease.get('run_uuid')
            or lease.get('manifest_sha256')!=custody['files']['RUN_MANIFEST.json']
            or lease.get('status')!='CLOSED' or lease.get('result_sha256')!=custody['files']['RESULT.json']
            or not step<=result.get('optimizer_updates',-1)<8192
            or terminal.get('effects_closed')is not True or terminal.get('producer_run_uuid')!=lease.get('run_uuid')
            or restore.get('checkpoint_manifest_sha256')!=request['manifest_sha256']
            or any(restore.get(k)!=v for k,v in dict(model_reconstruction=True,optimizer_state_entries=42,
                optimizer_updates=step,scheduler_restored=True,RNG_restored=True,sampler_cursor=step*32,
                frozen_parent_unchanged=True).items())):
        raise PermissionError('SHARED_PREVIOUS_NATIVE_EFFECT_MUST_BE_CLOSED_AND_FULLY_RESTORED')
    for name,pin in header['files'].items():
        f=checkpoint/name
        if f.is_symlink()or not f.resolve().is_relative_to(checkpoint)or sha(f)!=pin:
            raise PermissionError('SHARED_RESUME_MEMBER_CHANGED:'+name)
    saved=json.loads((checkpoint/'SAMPLER_STATE.json').read_bytes())
    if (saved.get('recipe')!=recipe or saved.get('sampler')!=TaskBalancedSampler(dataset,start=step*32).state_dict(step*32)
            or saved.get('dataset_manifest_sha256')!=dataset.manifest_sha
            or saved.get('statistics_sha256')!=dataset.statistics_sha
            or saved.get('total_committed_trace_count',saved.get('worker_trace',{}).get('count'))!=step*32):
        raise PermissionError('SHARED_RESUME_EXACT_SAMPLER_DATA_AND_TRACE_PREFIX')
    return dict(path=str(checkpoint),manifest_sha256=request['manifest_sha256'],step=step,cursor=step*32,
        saved_sampler=saved,science_run_id=manifest['science_run_id'])

def rng_identity_from_saved(saved):
    import hashlib,pickle
    from scripts.pi0294.native_train import tensor_identity
    cuda=saved.get('cuda',[])
    if not isinstance(cuda,(tuple,list)):cuda=[cuda]
    return dict(python=hashlib.sha256(pickle.dumps(saved['python'])).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(saved['numpy'])).hexdigest(),
        torch_cpu=tensor_identity(saved['cpu']),torch_cuda=[tensor_identity(x)for x in cuda])
