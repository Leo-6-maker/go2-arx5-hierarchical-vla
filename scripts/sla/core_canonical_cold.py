"""Two independent cold processes on the fixed Core production inference path."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock
from scripts.sla.core_canonical_package import CANONICAL_PLAN, PROTOCOL
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned

INITIAL = ('MAIN-phase00000', 'MAIN-phase02000')
SLOTS = ('cold-A', 'cold-B')
MAX_COLD_ATTEMPTS = 64


def pin(path):
    return dict(path=str(path),sha256=digest(path))


class InputAudit:
    def __init__(self, provider, folder):
        self.provider = provider
        self.folder = folder

    def predict(self, *args, **kwargs):
        import jax
        from types import SimpleNamespace
        from go2arx5_vla.sla_adapter.scale20k_package import Provider
        from scripts.sla.hopper_numeric_probe import observation_signatures, signature

        model = self.provider
        seed = kwargs['noise_id']
        def infer(active, frozen, obs, noise):
            record = dict(noise_id=seed,observed=observation_signatures(obs),
                          actual_noise=signature(noise),key=signature(jax.random.PRNGKey(seed)))
            write_once(self.folder/f'{seed}.input.json',record)
            return model.infer(active,frozen,obs,noise)
        audited = SimpleNamespace(active=model.active,frozen=model.frozen,resize=model.resize,
                                  tokenizer=model.tokenizer,statistics=model.statistics,infer=infer)
        return Provider.predict(audited,*args,**kwargs)


def child(root,cfg,arm,slot):
    import jax
    from importlib.metadata import version
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.hopper_reference import core_provider,rgb16
    from scripts.sla.core_canonical_package import validate

    contact = json.loads((root/'OWNER.json').read_bytes())
    if (arm not in INITIAL or slot not in SLOTS or os.getppid()!=contact['pid']
            or time.time()>=cfg['deadline'] or not os.environ.get('INVOCATION_ID')):
        raise PermissionError('Cold run has no registered parent or allocation')
    receiver=pinned(cfg['receiver'])
    package=root/'packages'/arm
    manifest=validate(package,cfg['packages'][arm])
    if (f"{manifest['role']}-phase{manifest['phase']:05d}"!=arm or
            manifest['checkpoint_sha256']!=cfg['checkpoints'][arm]):
        raise ValueError('Sealed endpoint identity changed')
    gpu=jax.local_devices()
    if (jax.default_backend()!='gpu' or len(gpu)!=1 or
            '4080 SUPER' not in gpu[0].device_kind or os.environ.get('CUDA_VISIBLE_DEVICES')!=receiver['gpu']):
        raise PermissionError('Not the registered RTX 4080 SUPER GPU')
    out=root/arm/slot
    out.mkdir()
    profile=dict(gpu=receiver['gpu'],driver=subprocess.check_output(
        ['nvidia-smi','-i',receiver['gpu'],'--query-gpu=driver_version','--format=csv,noheader'],
        text=True,timeout=10).strip(),jax=jax.__version__,jaxlib=version('jaxlib'),
        plugin=version('jax-cuda12-plugin'),cuda=gpu[0].client.platform_version,
        prng=str(jax.config.jax_default_prng_impl),x64=jax.config.jax_enable_x64,
        flags=os.environ.get('XLA_FLAGS'),omp=os.environ.get('OMP_NUM_THREADS'))
    write_once(out/'PROFILE.json',dict(profile=profile,pid=os.getpid()))
    target=pinned(receiver['input'])['target_D1']
    data=AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
    data.bind_state(target['statistics'])
    panel=pinned(receiver['panel'])
    provider,_=core_provider(package,cfg['packages'][arm],receiver['model'],data,receiver['tokenizer']['path'],
                             canonical=True)

    def guard():
        if (time.time()>=cfg['deadline']-15 or (root/'STOP').exists() or
                len(list(root.glob('*/cold-*/*.intent.json')))>=MAX_COLD_ATTEMPTS):
            raise PermissionError('Cold replay reached a finite stop or generation limit')

    predictions,_=rgb16(InputAudit(provider,out),data,panel,out,guard)
    rows=[dict(row,inputs=pinned(pin(out/f"{row['noise_id']}.input.json"))) for row in predictions]
    write_once(out/'RESULT.json',dict(arm=arm,package=cfg['packages'][arm],profile=profile,rows=rows,
        native_requests=0,updates=0))


def drive(root,cfg,sha):
    from scripts.sla.scale20k_core_prefix import resources
    receiver=pinned(cfg['receiver'])
    if (cfg['plan']!=CANONICAL_PLAN or cfg['protocol']!=PROTOCOL or os.getuid()!=receiver['uid']
            or os.uname().nodename!=receiver['host'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receiver['gpu']
            or not os.environ.get('INVOCATION_ID') or not time.time()<cfg['deadline']<=receiver['deadline']):
        raise PermissionError('Wrong Core owner, site profile or deadline')
    with closing(OwnerLock(root/'owner.lock')):
        if (root/'OWNER.json').exists():
            raise PermissionError('Cold evaluation identity was already started')
        write_once(root/'OWNER.json',dict(pid=os.getpid(),invocation=os.environ['INVOCATION_ID'],
                                         registered=sha,deadline=cfg['deadline']))
        # Allow an existing legitimate inference owner to finish within this finite allocation.
        while time.time()<cfg['deadline']-600 and not (root/'STOP').exists():
            try:
                lease=OwnerLock(receiver['gpu_lock'])
            except BlockingIOError:
                time.sleep(10)
                continue
            available=resources(receiver['gpu'])
            if (available['free_bytes'] > available['total_bytes']*receiver['memory_fraction']+2*1024**3
                    and available['host_bytes'] > 36*1024**3):
                break
            lease.close()
            time.sleep(10)
        else:
            raise TimeoutError('No qualified Core GPU capacity in the registered window')
        with closing(lease):
            write_once(root/'AWARD.json',dict(gpu=receiver['gpu'],resource=available,epoch=time.time()))
            for arm in INITIAL:
                (root/arm).mkdir()
                for slot in SLOTS:
                    remaining=cfg['deadline']-time.time()
                    if remaining<=15 or (root/'STOP').exists():
                        raise TimeoutError('Stop or expired cold measurement')
                    with (root/arm/(slot+'.log')).open('xb') as output:
                        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),str(root),sha,arm,slot],
                                       stdout=output,stderr=subprocess.STDOUT,check=True,timeout=remaining-5)
                    if slot=='cold-B':
                        a,b=[json.loads((root/arm/name/'RESULT.json').read_bytes()) for name in SLOTS]
                        same=a==b
                        write_once(root/arm/'COLD_COMPARISON.json',dict(status='COLD_MATCH' if same else 'COLD_MISMATCH',
                            arm=arm,complete=[len(x['rows']) for x in (a,b)],a=pin(root/arm/'cold-A/RESULT.json'),
                            b=pin(root/arm/'cold-B/RESULT.json'),native_admission=False))
                        if not same:
                            break


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    p.add_argument('arm',nargs='?')
    p.add_argument('slot',nargs='?')
    args=p.parse_args()
    os.umask(0o077)
    cfg=pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha))
    for ref in cfg['code']:
        if digest(Path(ref['path']))!=ref['sha256']:
            raise PermissionError('Registered cold code was modified')
    if args.arm is not None:
        child(args.root,cfg,args.arm,args.slot)
    else:
        error=None
        try:
            drive(args.root,cfg,args.config_sha)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        write_once(args.root/'COLD_TERMINAL.json',dict(status='COLD_STAGE_FINISHED' if error is None else 'COLD_STAGE_INTERRUPTED',
            first_failure=error,attempts=len(list(args.root.glob('*/cold-*/*.intent.json'))),
            completed=len(list(args.root.glob('*/cold-*/*.generated.json'))),native_requests=0,updates=0))
        sys.exit(bool(error))
