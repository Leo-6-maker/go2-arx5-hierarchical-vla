"""Isolated CPU-only PI0286 C1 public full-Trainer archival and byte readback.

Runs outside healthy code/seal and keeps unique local checkpoint untouched.
Uses the exact mature PI0284 upload/readback implementation bound to PI0286.
"""
import argparse,fcntl,hashlib,json,os,resource,time
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
ARM='C1_CORRECT1024'
REPO='leoyuer/pi0246-gate-pass-campaign'
PREFIX='arms/PI0286/C1_CORRECT1024/checkpoint-1024'
HELPER_SHA='126880ca6a0aab20ad556417335500f5b18f26c1f9fbbfdb81027b62049a064c'
EXPECTED={'PI0282_RECOVERY.json','PI0282_SCOPE_PROOF.json','config.json','embodiment_id.json',
 'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json',
 'optimizer.pt','processor_config.json','rng_state.pth','scheduler.pt','statistics.json','trainer_state.json','training_args.bin'}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def configure():
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_HUB_DISABLE_TELEMETRY='1',
      HF_HUB_DISABLE_PROGRESS_BARS='1',HF_XET_HIGH_PERFORMANCE='0',HF_XET_CHUNK_CACHE_SIZE_BYTES='0',
      HF_XET_NUM_CONCURRENT_RANGE_GETS='4',RAYON_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));os.nice(10)


def helper():
    path=ROOT.parent/'PI0284/runtime/archive_trainer.py';source=path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==HELPER_SHA
    old='/scratch/e1192548/Projects/go2-arx5/experiments/PI0284';assert source.count(old)==3
    namespace={'__name__':'pi0286_c1_exact_mature_archive'}
    exec(compile(source.replace(old,str(ROOT)),str(path),'exec'),namespace)
    original=namespace['HfApi']
    class FourThreadApi(original):
        def create_commit(self,*args,**kwargs):
            kwargs['num_threads']=4
            return super().create_commit(*args,**kwargs)
    namespace['HfApi']=FourThreadApi
    return namespace


def ready():
    checkpoint=ROOT/'learning'/ARM/'main/checkpoint-1024';audit=ROOT/'evidence'/f'{ARM}-1024_TRAINING_CPU.json'
    if not checkpoint.is_dir() or not audit.is_file():return None
    proof=json.loads(audit.read_text())
    assert proof['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    assert proof['point']==f'{ARM}-1024' and proof['checkpoint']==str(checkpoint)
    assert {p.name for p in checkpoint.iterdir() if p.is_file()}==EXPECTED
    assert not any(p.is_symlink() for p in checkpoint.rglob('*'))
    state=json.loads((checkpoint/'trainer_state.json').read_text());assert (state['global_step'],state['max_steps'])==(1024,1024)
    assert json.loads((checkpoint/'PI0282_RECOVERY.json').read_text())['cursor']==8192
    assert proof['formal_microbatches']==2048 and proof['chunks']==8192 and len(proof['entries'])==14
    assert {item['path'] for item in proof['entries']}==EXPECTED
    for item in proof['entries']:assert (checkpoint/item['path']).stat().st_size==item['bytes']
    card=json.loads((ROOT/'design'/f'{ARM}_CARD.json').read_text())
    assert card['stage']==ARM and card['updates']==1024 and card['parent_global_update']==6144
    return dict(checkpoint=str(checkpoint),training_CPU_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),card_sha256=hashlib.sha256((ROOT/'design'/f'{ARM}_CARD.json').read_bytes()).hexdigest(),expected_members=14)


def run(preflight=False):
    configure();archive=ROOT/'archive/c1_full_trainer';archive.mkdir(parents=True,exist_ok=True)
    with (archive/'WORKER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace=helper();token=(Path.home()/'.config/huggingface/token').read_text().strip();assert token
        api=namespace['HfApi'](token=token);info=api.model_info(REPO);assert info.private is False
        online=dict(status='PASS_CPU_ONLY_ONLINE_PUBLIC_ARCHIVE_PREFLIGHT',repo_id=REPO,public=True,prefix=PREFIX,
          helper_sha256=HELPER_SHA,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          max_CPU_threads=4,memory_limit_bytes=16*1024**3,offline_training_environment_changed=False,
          local_full_trainer_retained=True,model_queries=0,optimizer_updates=0,physics_calls=0,epoch=time.time())
        write(archive/'ONLINE_PREFLIGHT.json',online);print(json.dumps(online),flush=True)
        if preflight:return
        while True:
            boundary=ready()
            if boundary is not None:break
            write(archive/'WAITING_COMPLETE_C1.json',dict(status='WAIT_ACTUAL_C1_FULL14_AND_NATIVE_TRAINING_CPU_PASS',epoch=time.time(),model_queries=0,optimizer_updates=0))
            time.sleep(30)
        write(archive/'ACTUAL_C1_BINDING.json',{**online,**boundary,'status':'ACTUAL_COMPLETE_C1_CPU_AUDIT_BOUND_BEFORE_ARCHIVE'})
        # Keep the exact mature upload/manifest path; switch at first readback GET.
        class UploadBoundary(Exception):
            pass
        class UploadOnlyNetwork:
            def get(self,*args,**kwargs):
                raise UploadBoundary
        namespace['requests']=UploadOnlyNetwork()
        try:
            namespace['main'](ARM,1024,max_steps=1024,campaign='PI0286',token=token)
        except UploadBoundary:
            assert (ROOT/'evidence'/ARM/'checkpoint-1024.archive_receipt.json').is_file()
        import runpy
        runpy.run_path(str(ROOT/'archive/readback_trainer_parallel.py'))['run'](ARM)
        return


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');a=p.parse_args();run(a.preflight)
