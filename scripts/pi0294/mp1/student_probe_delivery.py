"""Private fixed-revision delivery of existing draws for saved-label diagnosis."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.checkpoint_delivery import verify_remote
from .scope import validate,charge_cpu
from .student_label_audit import join

REPO='leoyuer/pi0291-arx5-fixed-base'
PREFIX='SOURCE_S_BC512_HISTORICAL_STUDENT_DRAWS_V1'
MODELS={
    'S-BC-512-TRAIN4':(PREFIX,'SOURCE_STUDENT_PROBE_AUDIT_01','training/S-BC/checkpoint-512',512,
        '1cf83ad430c03c1d953266562cf90f3d0475f49c806701448fe5750d494c4d8c'),
    'S-BC-2048-TRAIN4':('SOURCE_S_BC2048_HISTORICAL_STUDENT_DRAWS_V1','SOURCE_STUDENT_PROBE_AUDIT_BC2048_01','training/S-BC/checkpoint-2048',2048,
        'ac0f46924d540ec5519f8c64b1ec0ea1c7f37bba96df0a21744e6a6a8e88efc2'),
    'S-CORR-512-TRAIN4':('SOURCE_S_CORR512_HISTORICAL_STUDENT_DRAWS_V1','SOURCE_STUDENT_PROBE_AUDIT_CORR512_01','training/S-CORR/checkpoint-512',512,
        '31f0514beb4003e42afc3437aa4781b5df47fc35a6270ba5c4fa081b31102d02')}


def configuration(spec):
    label=spec.get('model_label','S-BC-512-TRAIN4')
    if label not in MODELS:raise PermissionError('ONLY_THREE_EXISTING_IDENTITY_AUDITED_STUDENT_MODELS')
    return (label,*MODELS[label])


def publish(root,spec,token):
    label,prefix_name,audit_id,checkpoint_relative,step,checkpoint_sha=configuration(spec)
    child=root/'children/source-pair-01'
    if sha(child/'RUN_MANIFEST.json')!=spec['child_manifest_sha256']:
        raise ValueError('FIXED_CHILD_SCIENCE_MANIFEST')
    folder=child/'evidence'/label/'probes'
    proof=root/'evidence'/audit_id/'RESULT.json'
    audited=read(proof)
    if (audited['status']!='PASS_TWELVE_SAVED_STUDENT_DRAWS_ORIGINAL_E0_INPUTS_NOISE_AND_CHECKPOINT'
            or audited['probe_states']!=12 or audited['new_model_queries'] or audited['new_physics']
            or sha(proof)!=spec['student_audit_sha256']):
        raise PermissionError('ACTUAL_EXISTING_TWELVE_DRAW_IDENTITY_AUDIT_REQUIRED')
    if label!='S-BC-512-TRAIN4' and (audited.get('model_label')!=label or audited.get('checkpoint_manifest_sha256')!=checkpoint_sha):
        raise PermissionError('THIS_EXACT_NEW_MODEL_IDENTITY_AUDIT_NOT_BC512_RELABELLING')
    results=read(folder/'ON_STUDENT_RESULTS.private.json')
    sources={'ON_STUDENT_RESULTS.private.json':folder/'ON_STUDENT_RESULTS.private.json',
             'STUDENT_PROBES.private.json':child/'plans/STUDENT_PROBES.private.json',
             'HOPPER_INPUT_IDENTITY_AUDIT.json':proof,
             'CHECKPOINT_COMPLETE.json':child/checkpoint_relative/'CHECKPOINT_COMPLETE.json'}
    if label!='S-BC-512-TRAIN4' and sha(sources['CHECKPOINT_COMPLETE.json'])!=checkpoint_sha:
        raise PermissionError('THIS_EXACT_EXISTING_MODEL_CHECKPOINT')
    for row in results['records']:
        name=row['id']+'.npz'
        if Path(name).name!=name or sha(folder/name)!=row['arrays_sha256']:
            raise ValueError('EXACT_SAVED_STUDENT_ARRAY_SOURCE')
        sources[name]=folder/name
    if len(results['records'])!=12 or len(sources)!=16:
        raise ValueError('EXACT_TWELVE_STUDENT_ARRAYS_AND_FOUR_SMALL_IDENTITY_FILES')
    files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in sources.items()}
    if sum(x['bytes'] for x in files.values())>16*1024**2:
        raise PermissionError('STUDENT_DRAW_DELIVERY_16MIB_UPPER')
    output=root/'artifacts'/prefix_name
    output.mkdir(mode=0o700)
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    api=HfApi(token=token);info=api.repo_info(REPO,repo_type='dataset')
    if not info.private:
        raise PermissionError('EXISTING_PRIVATE_EVIDENCE_REPOSITORY_ONLY')
    prefix='PI0294/'+root.name+'/'+prefix_name
    if any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
        raise FileExistsError('FIXED_PREFIX_NO_BLIND_REUPLOAD')
    state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_RETRY',repo_id=REPO,prefix=prefix,files=files)
    write(output/'TRANSFER.private.json',state)
    rev=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
        commit_message='Fixed existing '+label+' student draws for CPU-only same-state label audit',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(sources[n])) for n in files]).oid
    state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=rev)
    write(output/'TRANSFER.private.json',state)
    def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=output/'readback')
    verify_remote(api,REPO,prefix,rev,files,download,token)
    ready=dict(status='READY_FIXED_EXISTING_STUDENT_DRAWS_NOT_NEW_INFERENCE',parent_run=root.name,
        files=files,data_revision=rev,student_states=12,new_model_queries=0,new_physics=0,
        child_manifest_sha256=spec['child_manifest_sha256'],student_audit_sha256=spec['student_audit_sha256'],model_label=label)
    write(output/'READY.json',ready)
    final=api.create_commit(REPO,repo_type='dataset',parent_commit=rev,
        commit_message='READY after private fixed student probe payload byte verification',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
    final_files=dict(files,**{'READY.json':dict(bytes=(output/'READY.json').stat().st_size,sha256=sha(output/'READY.json'))})
    verify_remote(api,REPO,prefix,final,final_files,download,token)
    state.update(status='PASS_PRIVATE_EXISTING_STUDENT_DRAWS_FIXED_REVISION',revision=final,
        READY_sha256=sha(output/'READY.json'),payload_bytes=sum(x['bytes'] for x in files.values()))
    write(output/'TRANSFER.private.json',state)
    return state


def consume(root,spec,token):
    label,prefix_name,audit_id,checkpoint_relative,step,expected=configuration(spec)
    transfer=spec['transfer']
    if (transfer['status']!='PASS_PRIVATE_EXISTING_STUDENT_DRAWS_FIXED_REVISION'
            or transfer['repo_id']!=REPO or transfer['prefix']!='PI0294/'+root.name+'/'+prefix_name
            or len(transfer['files'])!=16 or sum(x['bytes'] for x in transfer['files'].values())>16*1024**2):
        raise PermissionError('FIXED_TWELVE_DRAW_PRIVATE_TRANSFER_ONLY')
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi(token=token);info=api.repo_info(REPO,repo_type='dataset',revision=transfer['revision'])
    if not info.private or info.sha!=transfer['revision']:
        raise PermissionError('PRIVATE_FIXED_REVISION_REQUIRED')
    out=root/'datasets'/prefix_name;out.mkdir(mode=0o700)
    def get(name,checksum):
        if Path(name).name!=name or name in ['.','..']:
            raise ValueError('EXACT_SINGLE_EVIDENCE_FILENAME')
        p=Path(hf_hub_download(REPO,transfer['prefix']+'/'+name,repo_type='dataset',revision=transfer['revision'],token=token,local_dir=out))
        if sha(p)!=checksum:
            raise ValueError('FIXED_STUDENT_DRAW_DELIVERY_MEMBER_HASH')
        return p
    ready=read(get('READY.json',transfer['READY_sha256']))
    if (ready['parent_run']!=root.name or ready['files']!=transfer['files']
            or ready['data_revision']!=transfer['data_revision'] or ready['student_states']!=12
            or ready['new_model_queries'] or ready['new_physics']
            or ready.get('model_label','S-BC-512-TRAIN4')!=label):
        raise ValueError('FIXED_STUDENT_DRAW_READY_IDENTITY')
    for name,row in transfer['files'].items():get(name,row['sha256'])
    payload=out/transfer['prefix']
    old=root.parent/'20261016T150228Z-localpc-e1'
    if read(read(old/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED':
        raise PermissionError('READ_CLOSED_E1_ONLY_NO_REOPEN')
    if sha(payload/'HOPPER_INPUT_IDENTITY_AUDIT.json')!=ready['student_audit_sha256']:
        raise ValueError('SOURCE_AUDIT_IDENTITY')
    identity=read(payload/'HOPPER_INPUT_IDENTITY_AUDIT.json')
    if label!='S-BC-512-TRAIN4' and (identity.get('model_label')!=label or identity.get('checkpoint_manifest_sha256')!=expected):
        raise ValueError('RECEIVED_EXACT_MODEL_IDENTITY_NOT_OTHER_MODEL_ERRORS')
    checkpoint=read(payload/'CHECKPOINT_COMPLETE.json')
    if sha(payload/'CHECKPOINT_COMPLETE.json')!=expected or checkpoint['global_step']!=step:
        raise ValueError('FIXED_EXISTING_STUDENT_CHECKPOINT_IDENTITY')
    files={n:s for n,s in checkpoint['files'].items() if n.endswith('.safetensors') or n in ['config.json','model.safetensors.index.json','statistics.json'] or n.startswith('processor/')}
    from scripts.pi0292.relay_protocol import digest
    plan=read(payload/'STUDENT_PROBES.private.json')
    result=join(old,payload,plan,digest(files))
    result.update(model_label=label,HF_revision=transfer['revision'],HF_READY_sha256=transfer['READY_sha256'],
        all16_private_payload_files_independently_hash_verified_on_CoreLab=True,
        source_student_audit_sha256=ready['student_audit_sha256'],
        old_E1_inputs_or_results_modified=False,source_checkpoint_manifest_sha256=expected)
    write(out/'LABEL_JOIN_RESULT.private.json',result)
    return result


def run(root,binding_relative,token_file):
    root=Path(root)
    binding=root/binding_relative
    if Path(binding_relative).is_absolute() or '..' in Path(binding_relative).parts or binding.is_symlink():
        raise PermissionError('SEALED_SINGLE_STUDENT_DELIVERY_BINDING')
    spec=read(binding)
    role=spec['role']
    if role not in ['publish','consume']:
        raise PermissionError('EXACT_STUDENT_DELIVERY_ROLE')
    m=validate(root,'hopper-neural' if role=='publish' else '4080-cpu-data')
    if (m['execution'].get('student_delivery_binding_sha256')!=sha(binding)
            or spec['parent_run']!=root.name or spec['maximum_CPU_s']!=600
            or spec['repo_id']!=REPO or spec['new_queries'] or spec['new_physics']
            or not spec['created_epoch']<=time.time()<spec['deadline_epoch']<=spec['created_epoch']+600):
        raise PermissionError('BOUNDED_PRIVATE_STUDENT_DELIVERY_AUTHORITY')
    lock=OwnerLock(root/('STUDENT_DELIVERY_'+role.upper()+'.lock'))
    started=time.process_time()
    signal.alarm(max(1,int(spec['deadline_epoch']-time.time())))
    try:
        os.environ['HF_HUB_OFFLINE']='0'
        token=Path(token_file).read_text().strip()
        return (publish if role=='publish' else consume)(root,spec,token)
    finally:
        signal.alarm(0);charge_cpu(root,time.process_time()-started);lock.close()


if __name__=='__main__':
    import resource
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--binding',required=True)
    parser.add_argument('--existing-token-file',required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    def expire(signum,frame):raise TimeoutError('STUDENT_PRIVATE_DELIVERY_600S_ORIGINAL_WALL')
    signal.signal(signal.SIGALRM,expire)
    v=run(args.root,args.binding,args.existing_token_file)
    print(json.dumps(dict(status=v['status'],new_queries=0,new_physics=0)),flush=True)
