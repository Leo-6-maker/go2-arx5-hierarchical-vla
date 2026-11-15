"""Finite non-preemptive GPU0 follower for the existing full source BOOK seeds."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def run(path):
    path=Path(path);s=json.loads(path.read_bytes());R=Path(s['registered_root']);S=Path(s['source_root'])
    sys.path[:0]=[str(S/'src'),str(S)]
    from scripts.pi0292.durable import OwnerLock,write
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.pilot1_scope import validate
    start=time.time();held=None;child=None;code=1;owned_at=None
    def stop(sig,frame):
        if child and child.poll()is None:child.send_signal(sig)
        raise KeyboardInterrupt('FINITE_OWNED_BOOK_FIRST_FIT_STOP')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def call(root,module,env):
        nonlocal child
        q=json.loads((root/'EXECUTION_SPEC.json').read_bytes())
        argv=[s['python'],'-B','-m',module,str(root/'EXECUTION_SPEC.json')]
        write(root/'LAUNCH_INTENT.private.json',dict(argv=argv,epoch=time.time(),automatic_retry=False))
        with(R/'logs'/(root.name+'.log')).open('xb')as output:
            child=subprocess.Popen(argv,cwd=S,env=env,stdout=output,stderr=subprocess.STDOUT)
            rc=child.wait(timeout=max(1,q['deadline_epoch']-time.time()+30))
        write(root/'CHILD_EXIT.private.json',dict(exit_code=rc));return rc
    try:
        assert not(R/'STARTED.private.json').exists()and digest(__file__)==s['entry_sha256']
        validate(s);os.sched_setaffinity(0,{16,17,18,19})
        write(R/'STARTED.private.json',dict(epoch=time.time(),actual_cpu_affinity=sorted(os.sched_getaffinity(0))))
        base_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(S/'src')+':'+str(S)+':'+s['upstream_root']+'/src:'+s['upstream_root']+'/packages/openpi-client/src')
        base_env.pop('LD_PRELOAD',None)
        # Only metadata is repaired; all labels/RGB are hardlinked unchanged.
        program="""
import json,sys
from pathlib import Path
from scripts.pi0292.durable import write
from scripts.sla.prepare_source_book_v5 import audit_dataset
from go2arx5_vla.sla_adapter.source_book_v5 import deployment_metadata_copy
s=json.loads(Path(sys.argv[1]).read_bytes());r=Path(s['registered_root'])
result=deployment_metadata_copy(s['dataset_parent_root'],s['dataset_parent_manifest_sha256'],r/'dataset')
audit=audit_dataset(r/'dataset',result['manifest_sha256']);assert audit['read_windows']==1281
write(r/'dataset/READBACK.private.json',audit);write(r/'DATASET_ACTUAL.private.json',dict(result,readback=audit))
"""
        with(R/'logs/dataset.out.log').open('xb')as out:
            child=subprocess.Popen([s['python'],'-B','-c',program,str(path)],env=base_env,cwd=S,stdout=out,stderr=subprocess.STDOUT)
            assert child.wait(timeout=180)==0
        data=json.loads((R/'DATASET_ACTUAL.private.json').read_bytes());dataset=R/'dataset'
        while time.time()<s['deadline_epoch']-7200:
            validate(s);states=[]
            for p in s['wait_for_GPU0_predecessors']:
                root=Path(p['root']);assert digest(root/'EXECUTION_SPEC.json')==p['spec_sha256']
                lease=json.loads((root/'control/LEASE.json').read_bytes())
                pid=int(subprocess.check_output(['systemctl','--user','show','wm02-'+root.name,'--property=MainPID','--value'],text=True,timeout=10))
                closed=lease['status']=='CLOSED'and pid==0
                if closed:
                    worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
                    assert worker['exit_code']==lease['worker_exit_code']
                states.append(dict(root_name=root.name,closed=closed,worker_exit=lease.get('worker_exit_code')))
            write(R/'PROGRESS.private.json',dict(state='WAIT_BOTH_EXISTING_SPATIAL_OWNERS_ACTUAL_TERMINAL',epoch=time.time(),
                predecessors=states,GPU_owner_held=False,new_optimizer_updates=0,target_visits=0,source_visits=0,physical_requests=0))
            if all(v['closed']for v in states):break
            time.sleep(60)
        else:raise TimeoutError('FINITE_BOOK_QUEUE_EXPIRED_WITHOUT_GPU_OR_MODEL')
        while time.time()<s['deadline_epoch']-7200:
            validate(s)
            free=int(subprocess.check_output(['nvidia-smi','-i',s['GPU_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15))*1024**2
            ram=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))
            if free>s['allocator_limit_bytes']+1024**3 and ram>24*1024**3:
                try:held=OwnerLock(s['neural_owner_lock'])
                except BlockingIOError:held=None
                if held:break
            write(R/'PROGRESS.private.json',dict(state='WAIT_GPU0_EXISTING_OWNER_AND_CAPACITY',epoch=time.time(),
                GPU_owner_held=False,free_GPU_bytes=free,available_host_bytes=ram,new_optimizer_updates=0,physical_requests=0))
            time.sleep(60)
        else:raise TimeoutError('FINITE_BOOK_GPU0_CAPACITY_WAIT_NO_MODEL')
        owned_at=time.time();validate(s);end=min(time.time()+10800,s['deadline_epoch']-120)
        prior=Path(s['wait_for_GPU0_predecessors'][0]['root'])
        binding=dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2',host=os.uname().nodename,uid=os.getuid(),
            GPU_uuid=s['GPU_uuid'],GPU_name=s['GPU_name'],GPU_headroom_bytes=1024**3,allocator_limit_bytes=s['allocator_limit_bytes'],
            host_headroom_bytes=20*1024**3,allocation_deadline_epoch=end+120,
            paths=dict(parent_params=str(Path(s['parent_params']).resolve(strict=True)),dataset=str(dataset.resolve(strict=True)),
                       compilation_cache_dir=str(Path(s['compilation_cache_dir']).resolve(strict=True))),
            allocation=dict(kind='EXISTING_USER_SERVICE',campaign_root=s['campaign_root'],neural_owner_lock=s['neural_owner_lock']))
        write(R/'SITE_BINDING.private.json',binding)
        env=dict(base_env,CUDA_VISIBLE_DEVICES=s['GPU_uuid'],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            XLA_PYTHON_CLIENT_MEM_FRACTION=str(s['JAX_memory_fraction']),XLA_FLAGS=json.loads(Path(s['initial_function_rule']['path']).read_bytes())['XLA_FLAGS'],
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        candidate=R/'candidate500';candidate.mkdir()
        q=dict(s,registered_root=str(candidate),dataset=str(dataset),dense_manifest_sha256=data['manifest_sha256'],
            target_audit=dict(path=str(dataset/'READBACK.private.json'),sha256=digest(dataset/'READBACK.private.json')),
            site_binding=dict(path=str(R/'SITE_BINDING.private.json'),sha256=digest(R/'SITE_BINDING.private.json')),
            prior_closed_GPU0_training=dict(path=str(prior/'RESULT.private.json'),sha256=digest(prior/'RESULT.private.json')),
            deadline_epoch=end,max_walltime_s=10800,max_gpu_hours=3,model_service_timeout_s=10000,closeout_reserve_s=600,resume=None)
        write(candidate/'EXECUTION_SPEC.json',q)
        write(R/'PROGRESS.private.json',dict(state='RUN_SOURCE_ONLY_BOOK500_CLEAN_BASE',epoch=time.time(),target_visits=0,planned_source_visits=16000,physical_requests=0))
        rc=call(candidate,'scripts.sla.train_source_book_v5',env);assert rc in(0,75),rc
        trained=json.loads((candidate/'RESULT.private.json').read_bytes())
        if rc==75:
            result=dict(schema='SLA_BOOK21_SOURCE_FINITE_PREFIX_V5',exit_code=75,complete500=False,step=trained['step'],
                target_visits=0,source_visits=trained['cursor'],last_checkpoint=trained['last_checkpoint'],actual=trained['actual'])
        else:
            assert trained['complete500']and trained['step']==500 and trained['cursor']==16000
            replay=R/'fresh_RGB';replay.mkdir()
            t=dict(q,registered_root=str(replay),package=trained['package'],reference_probes=str(candidate/'work/step500-RGB/MANIFEST.private.json'))
            write(replay/'EXECUTION_SPEC.json',t)
            assert call(replay,'scripts.sla.reconstruct_source_book_v5',env)==0
            rgb=json.loads((replay/'REPLAY_RESULT.private.json').read_bytes());assert rgb['passes']and rgb['all_bit_exact']and rgb['ordinary_RGB_generations']==16
            result=dict(schema='SLA_BOOK21_SOURCE_FIRST_FIT_ACTUAL_V5',exit_code=0,complete500=True,step=500,target_visits=0,source_visits=16000,
                last_checkpoint=trained['last_checkpoint'],package=trained['package'],actual=trained['actual'],fresh_RGB16_bit_exact=True,
                fresh_RGB_result=dict(path=str(replay/'REPLAY_RESULT.private.json'),sha256=digest(replay/'REPLAY_RESULT.private.json')))
        result.update(source_only=True,target_WBC_qualified=False,physical_requests=0,wait_wall_s=owned_at-start,
            GPU_owned_wall_s=time.time()-owned_at,wall_s=time.time()-start)
        write(R/'RESULT.private.json',result);code=rc
    except BaseException:
        write(R/'FAILED.private.json',dict(error=traceback.format_exc(),no_automatic_retry=True,physical_requests=0))
        traceback.print_exc()
    finally:
        if child and child.poll()is None:
            child.terminate()
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if held:held.close()
        write(R/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.time()-start,
            GPU_owned_wall_s=0 if owned_at is None else time.time()-owned_at))
        lease=json.loads((R/'control/LEASE.json').read_bytes());lease.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time())
        write(R/'control/LEASE.json',lease)
    return code


if __name__=='__main__':raise SystemExit(run(sys.argv[1]))
