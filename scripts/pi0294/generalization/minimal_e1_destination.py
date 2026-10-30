"""Receive only the already qualified E1@512 fixed public HF package.

No uploader, cross-site SSH, scheduler, model construction or physics. The
existing owner separately registers/adopts this finite node-local IO role.
The original exporter/restorer validates the complete closed E1 evidence.
"""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import time
import urllib.request

ARM='E1_ADAPTER_ONLY'
REVISION='f410cd2b92cd543606d54b46f710197454606d9b'
REPO='leoyuer/pi0294-public-artifacts'
CHECKPOINT='383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d'
PREFIX='arx5-first512/'+ARM+'/'+CHECKPOINT
FIELD='ARX5_qualified_E1_destination_only_IO'
LIMITS=dict(Core_CPU_s=300,active_CPU_cores=1,wall_s=7200,
    Core_peak_bytes=12*1024**3,HTTP_transfer_upper_bytes=13*1024**3,
    models=0,queries=0,updates=0,physics=0,PBS=0)


def raw(v):return(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def need(ok,reason):
    if not ok:raise PermissionError('QUALIFIED_E1_DESTINATION_'+reason)


def put_control(root,name,value):
    """Original exclusive writer, with its actual nested parent prepared."""
    from scripts.pi0294.dec1.source_cpu_once_io import _exclusive
    root=Path(root);relative=PurePosixPath(name)
    need(not relative.is_absolute()and str(relative)==name and '..'not in relative.parts,
         'OWN_CONTROL_MEMBER_ONLY')
    path=root/name
    need(not any(p.is_symlink()for p in(path,*path.parents)),'NO_CONTROL_SYMLINK')
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    _exclusive(path,raw(value))


def expected_members(cfg):
    """Exact existing model identity, not a new model or another HF revision."""
    from .minimal_delivery_control import model_identity
    identity=model_identity(cfg['compiled_plan'])
    need(identity['candidate']==ARM and identity['checkpoint_manifest_sha256']==CHECKPOINT,
         'EXACT_ALREADY_QUALIFIED_E1_512')
    need(cfg['repo_id']==REPO and cfg['revision']==REVISION and cfg['prefix']==PREFIX,
         'ORIGINAL_FIXED_PUBLIC_REVISION')
    need(cfg['limits']==LIMITS,'FINITE_MODEL_ONLY_IO_LIMITS')
    files=dict(identity['files'])
    notices=cfg['notices']
    need(set(notices)=={'LICENSE','NVIDIA_OPEN_MODEL_LICENSE.pdf','Notice','MODEL_CARD.json'},
         'EXACT_ORIGINAL_LICENSE_MEMBERS')
    for name,row in notices.items():
        b=base64.b64decode(row['base64'],validate=True)
        need(len(b)==row['bytes'] and sha(b)==row['sha256'],'ORIGINAL_LICENSE_BYTES')
        files[name]=dict(bytes=row['bytes'],sha256=row['sha256'])
    need(len(files)==14 and sum(r['bytes']for r in identity['files'].values())==12576325049,
         'EXACT_TEN_INFERENCE_MEMBERS')
    for name,row in files.items():
        p=PurePosixPath(name)
        need(not p.is_absolute() and str(p)==name and '..'not in p.parts
             and '\\'not in name and type(row['bytes'])is int and row['bytes']>0,
             'SAFE_PINNED_MEMBER_NAMES_AND_SIZES')
    return files


def closed_source(cfg):
    """Recheck the original frozen evidence, without querying a live site."""
    from .minimal_inference_transfer import closed_endpoint
    b=cfg['compiled_plan']['private_restoration']['closed_evidence']
    proof=closed_endpoint(*(base64.b64decode(b[k],validate=True)for k in
        ('global_base64','lease_base64','settled_base64')),b['owner_observation'],arm=ARM)
    need(proof['endpoint']['official_inference_manifest_sha256']==CHECKPOINT,
         'ORIGINAL_CLOSED_E1_ENDPOINT')
    return proof


def stream_one(url,path,expected,*,guard,progress,open_url=urllib.request.urlopen):
    """One request/one temporary file; a failed or uncertain effect is retained."""
    path=Path(path);temporary=path.with_name(path.name+'.inflight')
    need(not path.exists()and not temporary.exists(),'NO_EXISTING_OR_UNKNOWN_MEMBER_REPLAY')
    total=0;digest=hashlib.sha256();guard(0)
    with open_url(url,timeout=30)as response,temporary.open('xb')as out:
        while True:
            guard(total)
            chunk=response.read(8*1024**2)
            if not chunk:break
            total+=len(chunk);need(total<=expected['bytes'],'RESPONSE_EXCEEDS_PINNED_SIZE')
            digest.update(chunk);out.write(chunk);progress(total)
        out.flush();os.fsync(out.fileno())
    need((total,digest.hexdigest())==(expected['bytes'],expected['sha256']),
         'FULL_RECEIVED_MEMBER_SHA256_AND_SIZE')
    os.rename(temporary,path);path.chmod(0o400)
    return dict(bytes=total,sha256=digest.hexdigest())


def main(root):
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0294.dec1.source_cpu_once_io import _exclusive as put,commit
    from scripts.pi0294.task_path_entry import registered_snapshot,node_local_paths
    root=Path(root);began=time.process_time();started=time.time()
    need(os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
         and not os.environ.get('PBS_JOBID')and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
         'ORIGINAL_CORE_CPU_ONLY_ENTRY')
    need(len(os.sched_getaffinity(0))==1,'ONE_CPU_CORE')
    resource.setrlimit(resource.RLIMIT_CPU,(300,300));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    config_bytes=(root/'MODEL_DELIVERY_CONFIG.private.json').read_bytes();cfg=json.loads(config_bytes)
    mb=(root/'RUN_MANIFEST.json').read_bytes();manifest=json.loads(mb)
    need(manifest['configuration_sha256']==sha(config_bytes)and manifest['run_uuid']==root.name
         and manifest['limits']==LIMITS,'OWN_MANIFEST_AND_EXACT_CONFIG')
    lock=OwnerLock(root/'control/DESTINATION_OWNER.lock')
    phase='COLD_ADMISSION';current=None;result=None;downloaded={};wire=0
    lease_path=root/'control/OWN_LEASE.json'
    def own_lease():
        v=json.loads(lease_path.read_bytes())
        need(v['status']=='ACTIVE'and v['run_uuid']==root.name and v['manifest_sha256']==sha(mb),
             'ORIGINAL_OWN_ACTIVE_LEASE')
        need(time.time()<min(v['deadline_epoch'],cfg['deadline_epoch'],started+LIMITS['wall_s'])
             and not(root/'STOP.json').exists(),'OWN_FINITE_LIFETIME')
        return v
    def guard(member_bytes):
        own_lease()
        # Keep the full old remaining work, the source phase and maintenance
        # margin free. Actual bytes already persisted by this root are included.
        used=cfg['preexisting_root_logical_upper_bytes']+wire+member_bytes
        need(used<=LIMITS['Core_peak_bytes'],'ROOT_STORAGE_UPPER')
        remaining=max(0,LIMITS['Core_peak_bytes']-used)
        fs=os.statvfs(root)
        need(fs.f_bavail*fs.f_frsize>=cfg['other_work_free_floor_bytes']+remaining,
             'REAL_FREE_SPACE_WITH_ALL_OTHER_WORK_RETAINED')
    def save(name,value):put_control(root,name,value)
    try:
        need(not(root/'DESTINATION_NATIVE_STARTED.json').exists(),'ONE_NATIVE_OWNER_NO_REPLAY')
        own_lease();code=root/'entry/code/vla';seal=json.loads((root/'SOURCE_SEAL.json').read_bytes())
        need(seal['source_sha']==manifest['source_sha'],'OWN_IMMUTABLE_SOURCE')
        for n,pin in seal['files'].items():need(sha((code/n).read_bytes())==pin,'EXACT_SOURCE_MEMBER')
        for n,pin in manifest['sealed_inputs'].items():need(sha((root/n).read_bytes())==pin,'EXACT_SEALED_INPUT')
        binding=json.loads((root/'OUTPUT_BINDING.private.json').read_bytes())
        registered_snapshot((root/'control/CANONICAL_REGISTRY_READBACK.json').read_bytes(),
            expected_sha256=binding['registry_snapshot_sha256'],run_root=str(root),role=binding['role'])
        local=json.loads(Path(cfg['LOCAL_path']).read_bytes())
        need(local[FIELD]==json.loads((root/'LOCAL_IO_ADOPTION.json').read_bytes())['marker'],
             'EXACT_ONCE_ADOPTED_DESTINATION_SCOPE')
        node_local_paths(json.loads((root/'RUNTIME_PATHS.private.json').read_bytes()))
        files=expected_members(cfg);source=closed_source(cfg);guard(0)
        save('DESTINATION_NATIVE_STARTED.json',dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
            source_sha=manifest['source_sha'],configuration_sha256=sha(config_bytes),started_epoch=started,
            agent_liveness_required=False,cross_site_runtime_dependency=False,new_GPU_model_physics=0))
        dest=root/'model_view';need(dest.is_dir()and not any(dest.iterdir()),'OWN_EMPTY_MODEL_VIEW')
        phase='FIXED_PUBLIC_REVISION_STREAM';heartbeat=(root/'control/HTTP_PROGRESS.jsonl').open('xb')
        try:
            for name,expected in files.items():
                current=name;path=dest/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                save('download-intents/'+name.replace('/','_')+'.json',dict(member=name,revision=REVISION,
                    bytes=expected['bytes'],sha256=expected['sha256'],automatic_retry=False))
                last=[-1]
                def progress(total):
                    if total//(64*1024**2)!=last[0]:
                        last[0]=total//(64*1024**2)
                        heartbeat.write(raw(dict(member=name,received_bytes=total,observed_epoch=time.time())))
                        heartbeat.flush();os.fsync(heartbeat.fileno())
                url='https://huggingface.co/datasets/'+REPO+'/resolve/'+REVISION+'/'+PREFIX+'/'+name
                received=stream_one(url,path,expected,guard=guard,progress=progress)
                wire+=received['bytes'];need(wire<=LIMITS['HTTP_transfer_upper_bytes'],'HTTP_TOTAL_UPPER')
                downloaded[name]=received;save('download-receipts/'+name.replace('/','_')+'.json',received)
        finally:heartbeat.close()
        phase='ORIGINAL_PRIVATE_METADATA_RESTORATION'
        from .minimal_inference_transfer import restore_original
        plan=cfg['compiled_plan'];public_files={n:dest/n for n in plan['public_plan']['files']}
        restored=restore_original(public_files,plan['private_restoration'],arm=ARM)
        for name in plan['private_restoration']['original_metadata_base64']:
            value=restored[name];need(type(value)is bytes,'RESTORED_ORIGINAL_METADATA_BYTES')
            path=dest/name;os.rename(path,path.with_name(path.name+'.public-transfer-original'));put(path,value)
        complete=base64.b64decode(plan['private_restoration']['original_complete_base64'],validate=True)
        need(sha(complete)==CHECKPOINT,'EXACT_ORIGINAL_COMPLETE_MANIFEST');put(dest/'CHECKPOINT_COMPLETE.json',complete)
        need(not any(n in sys.modules for n in('torch','mujoco','rclpy')),'NO_NEURAL_OR_PHYSICS_IMPORT')
        result=dict(status='ACTUAL_QUALIFIED_E1_FIXED512_CORE_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOAD',
            repo_id=REPO,revision=REVISION,prefix=PREFIX,checkpoint_manifest_sha256=CHECKPOINT,
            model_root=str(dest),files=downloaded,HTTP_payload_bytes=wire,Core_CPU_s=time.process_time()-began,
            original_complete_state_remains_at_source=True,full_resume_files_present=False,
            actual_original_source_job=source['job_id'],E2_partial_sibling_not_promoted=True,
            cross_site_runtime_dependency=False,agent_liveness_required=False,
            new_models_queries_updates_PBS_physics=0,observed_epoch=time.time())
        save('ACTUAL_CORE_MODEL_DELIVERED.json',result)
    except BaseException as exc:
        save('FAILED_OR_UNKNOWN.json',dict(status='FAILED_OR_UNKNOWN_NO_REPLAY',phase=phase,member=current,
            error_type=type(exc).__name__,error=str(exc)if isinstance(exc,PermissionError)else None,
            HTTP_known_complete_payload_bytes=wire,Core_CPU_s=time.process_time()-began,
            automatic_retry=False,new_models_queries_updates_PBS_physics=0,observed_epoch=time.time()))
        raise
    finally:
        lb=lease_path.read_bytes();lease=json.loads(lb)
        need(lease['status']=='ACTIVE'and lease['run_uuid']==root.name,'EXACT_OWN_LEASE_CLOSE')
        lease.update(status='CLOSED',reason='FIXED_E1_DESTINATION_TERMINAL'if result else'FAILED_OR_UNKNOWN_NO_RETRY',
            closed_epoch=time.time(),new_model_queries_updates_physics=0)
        key='QUALIFIED-E1-DESTINATION-ONLY-TERMINAL-CLOSE'
        commit(lease_path,expected_before_sha256=sha(lb),proposed_bytes=raw(lease),request_id=key,
            intent_path=lease_path.with_name(key+'.INTENT.json'),receipt_path=lease_path.with_name(key+'.RECEIPT.json'),
            lock_path=lease_path.with_name('OWN_LEASE.owner.lock'))
        save('DESTINATION_NATIVE_FINAL.json',dict(status='DELIVERED'if result else'FAILED_OR_UNKNOWN',
            phase=phase,Core_CPU_s=time.process_time()-began,wall_s=time.time()-started,
            old_costs_unchanged=True,new_models_queries_updates_PBS_physics=0,observed_epoch=time.time()))
        lock.close()
    print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
