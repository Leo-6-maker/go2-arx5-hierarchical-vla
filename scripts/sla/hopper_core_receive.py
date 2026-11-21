"""Receive sealed public Go2 checkpoints on isolated Core and run one strict RGB16 per endpoint."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_parent_custody import PLAN, digest, sync_dir, write_once
from scripts.sla.hopper_public_checkpoint import PREFIX, REPO, allowed
from scripts.sla.hopper_site import DEADLINE, pinned

SCIENCE='b9581196d0acd02fb748222988e5a5543ce1f232'
KEY=re.compile(r'(MAIN|CONTROL2000)-phase[0-9]{5}')
BASE_PHASES={0,2000,5000,10000,20000}


def receive(root,api):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    repo=api.repo_info(REPO,repo_type='dataset')
    if repo.private or not re.fullmatch('[a-f0-9]{40}',repo.sha):
        raise ValueError('Public repository or revision missing')
    try:
        entries=list(api.list_repo_tree(REPO,repo_type='dataset',path_in_repo=PREFIX,revision=repo.sha))
    except EntryNotFoundError:
        entries=[]
    incoming={}
    def fetch(prefix,name):
        return Path(hf_hub_download(REPO,prefix+'/'+name,repo_type='dataset',revision=repo.sha,cache_dir=root/'cache'))
    for item in entries:
        key,marker=item.path.rsplit('/',1)[-1].rsplit('-',1)
        if (not KEY.fullmatch(key) or not re.fullmatch('[a-f0-9]{12}',marker) or key in incoming
                or item.path!=PREFIX+'/'+key+'-'+marker):
            raise ValueError('Unexpected or duplicate published endpoint')
        incoming[key]=(marker,item.path)
    if len(incoming)>7 or len({int(k[-5:]) for k in incoming if int(k[-5:]) not in BASE_PHASES})>1:
        raise ValueError('Deployment population or matched boundary exceeded')
    for key,(marker,prefix) in sorted(incoming.items()):
        slot=root/'endpoints'/key
        slot.mkdir(exist_ok=True)
        receipt=slot/'RECEIVED.json'
        manifest=fetch(prefix,'CHECKPOINT.json')
        sha=digest(manifest.resolve(strict=True))
        m=read(manifest)
        members=allowed(m)
        if (sha[:12]!=marker or m['training_source']!=SCIENCE or key!=f"{m['role']}-phase{m['phase']:05d}"):
            raise ValueError('Published model or source differs from the fixed lineage')
        if receipt.is_file():
            if read(receipt)['manifest']!=sha or digest(slot/'payload/CHECKPOINT.json')!=sha:
                raise ValueError('Public or private checkpoint identity changed')
            continue
        payload=slot/'payload'
        if not payload.exists():
            with tempfile.TemporaryDirectory(dir=slot) as stage:
                temp=Path(stage)/'payload'
                (temp/'parameters').mkdir(parents=True)
                shutil.copyfile(manifest,temp/'CHECKPOINT.json')
                for name,expected in members.items():
                    shutil.copyfile(fetch(prefix,name),temp/name)
                    if digest(temp/name,flush=True)!=expected:
                        raise ValueError('Public inference member does not match its manifest')
                sync_dir(temp/'parameters')
                sync_dir(temp)
                temp.rename(payload)
                sync_dir(slot)
        if digest(payload/'CHECKPOINT.json')!=sha or any(digest(payload/n)!=v for n,v in members.items()):
            raise ValueError('Local checkpoint custody differs')
        write_once(receipt,dict(repository=REPO,revision=repo.sha,prefix=prefix,manifest=sha,
                    checkpoint=m['checkpoint_sha256'],arrays=len(members),received_epoch=time.time()))
    return repo.sha


def replay(root,slot,cfg):
    import jax
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from scripts.sla.hopper_reference import core_provider,rgb16
    receipt=read(slot/'RECEIVED.json')
    allocation=read(slot/'GPU_START.json')
    end=allocation['end']
    def guard():
        if (time.time()>=end or (root/'STOP').exists() or os.getuid()!=cfg['uid']
                or os.environ['CUDA_VISIBLE_DEVICES']!=cfg['gpu'] or not os.environ.get('INVOCATION_ID')):
            raise PermissionError('Core operator stop, wrong GPU owner or expired lease')
    guard()
    if (jax.default_backend()!='gpu' or len(jax.local_devices())!=1
            or allocation['pid_owner']!=os.getppid() or 'RTX' not in jax.local_devices()[0].device_kind):
        raise PermissionError('Expected one leased Core inference GPU')
    jax.config.update('jax_compilation_cache_dir',str(root/'jax-cache'))
    spec,panel=pinned(cfg['input']),pinned(cfg['panel'])
    target=spec['target_D1']
    source=cfg['model']
    if (target['statistics']['sha256']!=source['state_sha256']
            or digest(Path(cfg['action']['path']))!=cfg['action']['sha256']):
        raise ValueError('Frozen local statistics changed')
    data=AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
    data.bind_state(target['statistics'])
    provider,m=core_provider((slot/'payload').resolve(),receipt['manifest'],source,data,cfg['tokenizer']['path'])
    if m['checkpoint_sha256']!=receipt['checkpoint'] or m['action_sha256']!=cfg['action']['sha256']:
        raise ValueError('Local endpoint custody changed')
    rows,exact=rgb16(provider,data,panel,slot,guard,expected=m['reference'])
    write_once(slot/'RESULT.json',dict(status='CORE_RGB16_PASS' if exact==16 else 'CORE_RGB16_EXACT_GATE_FAILED',
        exact=exact,attempts=16,completed=16,checkpoint=receipt['checkpoint'],manifest=receipt['manifest'],
        first_mismatch=next((x['ordinal'] for x,y in zip(rows,m['reference'],strict=True) if x!=y),None),
        matched={k:sum(x[k]==y[k] for x,y in zip(rows,m['reference'],strict=True)) for k in
                 ('input_sha256','normalized_sha256','decoded_sha256','original_guard_sha256')},
        native_requests=0,native_ticks=0,updates=0))


def dispatch(root,cfg,config_sha,slot):
    from scripts.sla.scale20k_core_prefix import resources
    try:
        lock=OwnerLock(cfg['gpu_lock'])
    except BlockingIOError:
        return 'WAIT_GPU_OWNER'
    with closing(lock):
        capacity=resources(cfg['gpu'])
        fraction=cfg['memory_fraction']
        if capacity['free_bytes']<=capacity['total_bytes']*fraction+2*1024**3 or capacity['host_bytes']<=30*1024**3:
            return 'WAIT_HEADROOM'
        now=time.time()
        end=min(now+3600,cfg['deadline'],DEADLINE)
        if end-now<300 or (root/'STOP').exists():
            return 'STOP'
        receipt=read(slot/'RECEIVED.json')
        write_once(slot/'GPU_START.json',dict(launch=now,end=end,gpu=cfg['gpu'],capacity=capacity,
            reservation_fraction=fraction,pid_owner=os.getpid(),systemd_invocation=os.environ['INVOCATION_ID']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['gpu'],JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION=str(fraction),
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'replay',str(root),config_sha,'--slot',slot.name]
        try:
            with (slot/'worker.log').open('xb') as log:
                proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()))
            reason='worker exited '+str(proc.returncode)
        except subprocess.TimeoutExpired:
            reason='GPU lease expired'
        if not (slot/'RESULT.json').is_file():
            write_once(slot/'RESULT.json',dict(status='CORE_RGB16_INTERRUPTED',reason=reason,checkpoint=receipt['checkpoint'],
                attempted=len(list(slot.glob('*.intent.json'))),completed=len(list(slot.glob('*.generated.json'))),
                native_requests=0,native_ticks=0,updates=0))
        return read(slot/'RESULT.json')['status']


def watch(root,cfg,sha):
    from huggingface_hub import HfApi
    from requests.exceptions import ConnectionError, Timeout
    api=HfApi()
    with closing(OwnerLock(root/'receiver.lock')):
        while time.time()<min(cfg['deadline'],DEADLINE)-60 and not (root/'STOP').exists():
            try:
                rev=receive(root,api)
            except (ConnectionError,Timeout) as exc:
                write(root/'HEARTBEAT.json',dict(status='WAIT_HF_NETWORK',time=time.time(),error=type(exc).__name__))
                time.sleep(150)
                continue
            slots=sorted((root/'endpoints').iterdir())
            state='WAIT_CHECKPOINT'
            for slot in slots:
                if (slot/'RESULT.json').exists():
                    continue
                if (slot/'GPU_START.json').exists():
                    raise RuntimeError('Unresolved launched RGB16 requires reconciliation')
                state=dispatch(root,cfg,sha,slot)
                break
            write(root/'HEARTBEAT.json',dict(status=state,revision=rev,pid=os.getpid(),received=len(slots),
                terminal=sum((p/'RESULT.json').is_file() for p in slots),time=time.time()))
            time.sleep(150)
    write_once(root/'CLOSED.json',dict(status='STOP_OR_DEADLINE',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['watch','replay'])
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    p.add_argument('--slot')
    args=p.parse_args()
    os.umask(0o077)
    cfg=pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha))
    if (cfg['plan']!=PLAN or cfg['uid']!=os.getuid() or cfg['host']!=os.uname().nodename
            or not os.environ.get('INVOCATION_ID') or cfg['deadline']>DEADLINE):
        raise PermissionError('Expected the registered Core service, host, uid and bounded grant')
    if args.action=='watch':
        watch(args.root,cfg,args.config_sha)
    elif args.slot and KEY.fullmatch(args.slot):
        replay(args.root,args.root/'endpoints'/args.slot,cfg)
    else:
        raise ValueError('Invalid endpoint')
