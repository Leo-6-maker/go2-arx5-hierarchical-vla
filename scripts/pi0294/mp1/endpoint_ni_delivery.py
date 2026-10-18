"""CPU-only fixed-revision handoff of the sixteen already-generated N1 draws."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.relay_protocol import digest
from scripts.pi0294.checkpoint_delivery import verify_remote
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from . import endpoint_anchor_ni as ni,n1_limits as n1
from .n1_boundary import paths
from .probe_compare import PRECISION
from .scope import validate,charge_cpu

REPO='leoyuer/pi0294-public-artifacts'
NAME='N1_CURRENT_EIGHT_ENDPOINT_NI16_ACTUAL_V1'
SCHEMA='MP1_N1_NI16_CPU_HANDOFF_V1'
RESULT_SHA='e8670aacee9a28b054eeda5d8ba3f37ce1a92b40321a6f57611efea6e0a92971'
FINAL_SHA='d66c045407054d67ddf5b9c420cdf5da0c1ca7b36462ed30ad0e001c1bc7eadd'
MAX_BYTES=16*1024**2


def filename(model,identifier):
    if model not in ni.CHECKPOINTS or identifier not in ni.selected_ids():
        raise PermissionError('EXACT_TWO_ENDPOINTS_AND_ORIGINAL_EIGHT_IDS')
    return model+'--'+identifier+'.npz'


def checked_rows(value,binding):
    expected={(m,i) for m in ni.CHECKPOINTS for i in ni.selected_ids()}
    rows=value['records']
    if (value['status']!='COMPLETE_SIXTEEN_SAME_INPUT_NOISE_ENDPOINT_DRAWS_NOT_POLICY_EVALUATION'
            or len(rows)!=16 or {(r['model'],r['id']) for r in rows}!=expected
            or value['new_neural_queries']!=16 or value['new_physics'] or value['new_optimizer_updates']
            or value['closed_loop_or_task_success_claimed'] or not value['original_eight_BC512_draws_reused']
            or value['native_history_gate_result_sha256']!=ni.GATE_RESULT):
        raise PermissionError('EXACT_ALREADY_GENERATED_NI16_NOT_TASK_EVALUATION')
    for row in rows:
        original=binding['native'][row['id']]
        if (row['noise']!=original['actual_native_noise_fingerprint'] or row['noise_seed']!=original['noise_seed']
                or row['original_query_uuid']!=original['query_uuid']
                or row['observation_fingerprint']!=original['observation_fingerprint']
                or row['checkpoint_manifest_sha256']!=ni.CHECKPOINTS[row['model']]
                or row['new_query'] is not True or row['live_or_shadow_integrations']!=0
                or any(row['inference_precision'].get(k)!=v for k,v in PRECISION.items())):
            raise PermissionError('EXACT_ORIGINAL_INPUT_NOISE_PRECISION_AND_ENDPOINT_WEIGHTS')
        filename(row['model'],row['id'])
    reject_private(value)
    return rows


def checked_array(path,row,weights):
    if path.is_symlink() or sha(path)!=row['arrays_sha256']:
        raise ValueError('ACTUAL_NI_ARRAY_HASH')
    with np.load(path,allow_pickle=False) as z:
        if (z['physical_H40'].shape!=(40,10) or z['normalized'].shape!=(1,40,132)
                or not np.isfinite(z['physical_H40']).all() or not np.isfinite(z['normalized']).all()
                or int(z['noise_seed'])!=row['noise_seed']
                or json.loads(str(z['noise_fingerprint_json'].item()))!=row['noise']
                or json.loads(str(z['inference_precision_json'].item()))!=row['inference_precision']
                or str(z['observation_fingerprint'].item())!=row['observation_fingerprint']
                or str(z['checkpoint_files_fingerprint'].item())!=weights):
            raise ValueError('ACTUAL_NI_ARRAY_NOISE_INPUT_PRECISION_AND_WEIGHT_IDENTITY')


def fixed_binding(child):
    return ni.validate_binding(child,read(child/paths(10)[0]),read(child/'RUN_MANIFEST.json'))


def publish(root,spec,token):
    child=root/'children/source-pair-01'
    if sha(child/'RUN_MANIFEST.json')!=spec['child_manifest_sha256'] or ni.closed_draws(child)!=16:
        raise PermissionError('ACTUAL_COMPLETE_NATIVE_NI16_REQUIRED')
    source=child/'evidence'/ni.LABEL
    if sha(source/'RESULT.json')!=RESULT_SHA or sha(child/'pbs/alloc-10/PBS_PROCESS_FINAL.json')!=FINAL_SHA:
        raise ValueError('EXACT_CLOSED_JOB_NI16_RESULT_AND_FINAL')
    final=read(child/'pbs/alloc-10/PBS_PROCESS_FINAL.json');value=read(source/'RESULT.json')
    if final['job_id']!='673318.hopper-m-02' or final['records']!=[value]:
        raise ValueError('OWN_ALLOCATION10_PROCESS_FINAL_NOT_OLD_JOB')
    rows=checked_rows(value,fixed_binding(child));output=root/'artifacts'/NAME
    output.mkdir(mode=0o700)
    sources={'NI_RESULT.json':source/'RESULT.json'}
    for row in rows:sources[filename(row['model'],row['id'])]=source/row['model']/(row['id']+'.npz')
    index=dict(schema=SCHEMA,authority_release=n1.RELEASE,N1_T0_epoch=1792264081,
        source_sha=read(child/'RUN_MANIFEST.json')['code']['vla_commit'],job_id=final['job_id'],
        result_sha256=RESULT_SHA,PBS_PROCESS_FINAL_sha256=FINAL_SHA,records=rows,
        new_queries_by_delivery=0,new_physics=0,new_training_updates=0,new_Teacher_labels=0,
        original_parent_episodes=4,original_states=8,closed_loop_or_task_success=False)
    reject_private(index);write(output/'ARTIFACT_INDEX.json',index);sources['ARTIFACT_INDEX.json']=output/'ARTIFACT_INDEX.json'
    files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in sources.items()}
    if len(files)!=18 or sum(r['bytes'] for r in files.values())>MAX_BYTES:
        raise PermissionError('ONLY_SIXTEEN_ARRAYS_TWO_SMALL_METADATA_FILES_16MIB')
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    api=HfApi(token=token);info=api.repo_info(REPO,repo_type='dataset')
    if info.private:raise PermissionError('EXISTING_PUBLIC_OWNED_ARTIFACT_REPOSITORY')
    prefix='PI0294/'+root.name+'/'+NAME
    if any(n.startswith(prefix+'/') for n in api.list_repo_files(REPO,repo_type='dataset')):
        raise FileExistsError('FRESH_NI16_PREFIX_NO_BLIND_REUPLOAD')
    state=dict(status='UPLOAD_INTENT_UNKNOWN_NO_RETRY',repo_id=REPO,prefix=prefix,files=files)
    write(output/'TRANSFER.private.json',state)
    rev=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
        commit_message='N1 actual sixteen endpoint same-input-noise diagnostic outputs, not task results',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/'+n,path_or_fileobj=str(p)) for n,p in sources.items()]).oid
    state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING',data_revision=rev);write(output/'TRANSFER.private.json',state)
    def download(*args,**kwargs):return hf_hub_download(*args,**kwargs,local_dir=output/'readback')
    verify_remote(api,REPO,prefix,rev,files,download,token)
    ready=dict(schema=SCHEMA,status='READY_ACTUAL_NI16_NOT_POLICY_EVALUATION',parent_run=root.name,
        files=files,data_revision=rev,result_sha256=RESULT_SHA,states=8,models=2,new_queries=0,new_physics=0)
    write(output/'READY.json',ready)
    final_revision=api.create_commit(REPO,repo_type='dataset',parent_commit=rev,
        commit_message='READY after exact NI16 payload byte verification',
        operations=[CommitOperationAdd(path_in_repo=prefix+'/READY.json',path_or_fileobj=str(output/'READY.json'))]).oid
    final_files=dict(files,**{'READY.json':dict(bytes=(output/'READY.json').stat().st_size,sha256=sha(output/'READY.json'))})
    verify_remote(api,REPO,prefix,final_revision,final_files,download,token)
    state.update(status='PASS_NI16_FIXED_PUBLIC_REVISION',revision=final_revision,READY_sha256=sha(output/'READY.json'),
        payload_bytes=sum(r['bytes'] for r in files.values()))
    write(output/'TRANSFER.private.json',state);return state


def consume(root,spec,token):
    transfer=spec['transfer'];child=root/'children/source-pair-01'
    if (transfer['status']!='PASS_NI16_FIXED_PUBLIC_REVISION' or transfer['repo_id']!=REPO
            or transfer['prefix']!='PI0294/'+root.name+'/'+NAME or len(transfer['files'])!=18
            or sum(r['bytes'] for r in transfer['files'].values())>MAX_BYTES):
        raise PermissionError('EXACT_FIXED_NI16_HANDOFF')
    from huggingface_hub import hf_hub_download
    out=root/'datasets'/NAME;out.mkdir(mode=0o700)
    def get(name,checksum):
        allowed={'READY.json','NI_RESULT.json','ARTIFACT_INDEX.json'}|{filename(m,i) for m in ni.CHECKPOINTS for i in ni.selected_ids()}
        if name not in allowed:raise PermissionError('ONLY_DECLARED_NI16_FILES_NO_MODEL_DOWNLOAD')
        p=Path(hf_hub_download(REPO,transfer['prefix']+'/'+name,repo_type='dataset',revision=transfer['revision'],token=token,local_dir=out))
        if sha(p)!=checksum:raise ValueError('FIXED_REVISION_MEMBER_HASH')
        return p
    ready=read(get('READY.json',transfer['READY_sha256']))
    if (ready['schema']!=SCHEMA or ready['files']!=transfer['files'] or ready['parent_run']!=root.name
            or ready['data_revision']!=transfer['data_revision'] or ready['result_sha256']!=RESULT_SHA
            or ready['states']!=8 or ready['models']!=2 or ready['new_queries'] or ready['new_physics']):
        raise ValueError('EXACT_NI16_READY_IDENTITY')
    members={n:get(n,v['sha256']) for n,v in transfer['files'].items()}
    if sha(members['NI_RESULT.json'])!=RESULT_SHA:raise ValueError('ORIGINAL_NI16_RESULT_BYTES')
    rows=checked_rows(read(members['NI_RESULT.json']),fixed_binding(child))
    index=read(members['ARTIFACT_INDEX.json']);reject_private(index)
    if index['records']!=rows or index['result_sha256']!=RESULT_SHA or index['PBS_PROCESS_FINAL_sha256']!=FINAL_SHA:
        raise ValueError('FIXED_NI16_ARTIFACT_INDEX')
    weights={}
    for model in ni.CHECKPOINTS:
        identities=[read(p)['checkpoint_identity'] for p in (child/'plans').glob(model+'-TRAIN4-ALLOCATION-*/REGISTRY.json')]
        identities=[i for i in identities if i['checkpoint_manifest_sha256']==ni.CHECKPOINTS[model]]
        unique={digest(i['files']) for i in identities}
        if len(unique)!=1:raise PermissionError('EXACT_EXISTING_CORE_ENDPOINT_REGISTRY_WEIGHT_IDENTITY')
        weights[model]=unique.pop()
    for row in rows:checked_array(members[filename(row['model'],row['id'])],row,weights[row['model']])
    result=dict(status='PASS_CORE_NI16_ORIGINAL_INPUT_NOISE_ARRAYS_FIXED_REVISION',repo_id=REPO,
        revision=transfer['revision'],prefix=transfer['prefix'],files=transfer['files'],READY_sha256=transfer['READY_sha256'],
        states=8,models=2,original_parents=4,new_queries=0,new_physics=0,new_labels=0,matched_shadows='NOT_RUN')
    write(out/'CONSUMED.private.json',result);return result


def run(root,binding_relative,token_file):
    root=Path(root);p=Path(binding_relative)
    if p.is_absolute() or '..' in p.parts:raise PermissionError('OWNED_BINDING_RELATIVE_ONLY')
    path=root/p;spec=read(path);role=spec['role']
    if role not in ['publish','consume']:raise PermissionError('ONLY_TWO_CPU_HANDOFF_ROLES')
    m=validate(root,'hopper-neural' if role=='publish' else '4080-cpu-data')
    first=root/n1.FIRST_ADOPTION
    if first.is_symlink() or first.stat().st_mode&0o222 or sha(first)!=n1.ADOPTED_DELTA_SHA:
        raise PermissionError('ORIGINAL_FIRST_N1_ADOPTION')
    delta=n1.validated_delta(read(first),read(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'))
    if (path.is_symlink() or path.stat().st_mode&0o222 or m['execution'].get('endpoint_ni_delivery_binding_sha256')!=sha(path)
            or spec['schema']!=SCHEMA or spec['parent_run']!=root.name or spec['repo_id']!=REPO
            or spec['N1_authority_release']!=n1.RELEASE or spec['N1_adoption_sha256']!=n1.ADOPTED_DELTA_SHA
            or spec['maximum_CPU_s']!=600 or spec['worker_CPU_s']!=420 or spec['registration_CPU_upper_s']!=180
            or spec['new_queries'] or spec['new_physics'] or type(spec['CPU_id']) is not int
            or not delta['autonomy_started_epoch']<=spec['created_epoch']<=time.time()<spec['deadline_epoch']<=min(spec['created_epoch']+900,delta['hard_closeout_epoch'])):
        raise PermissionError('BOUNDED_N1_CPU_ONLY_HANDOFF_MANIFEST_AND_CLOCK')
    if spec['CPU_id'] not in os.sched_getaffinity(0):raise PermissionError('OWN_ALLOWED_DISJOINT_CPU')
    os.sched_setaffinity(0,{spec['CPU_id']});resource.setrlimit(resource.RLIMIT_CPU,(420,420))
    lock=OwnerLock(root/'N1_NI16_DELIVERY.lock');started=time.process_time()
    def expire(signum,frame):raise TimeoutError('N1_NI16_CPU_HANDOFF_FINITE_WALL')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(spec['deadline_epoch']-time.time())))
    try:
        os.environ['HF_HUB_OFFLINE']='0'
        return (publish if role=='publish' else consume)(root,spec,Path(token_file).read_text().strip())
    finally:signal.alarm(0);charge_cpu(root,time.process_time()-started);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--binding',required=True)
    parser.add_argument('--existing-token-file',required=True);args=parser.parse_args()
    result=run(args.root,args.binding,args.existing_token_file)
    print(json.dumps(dict(status=result['status'],new_queries=0,new_physics=0)),flush=True)
