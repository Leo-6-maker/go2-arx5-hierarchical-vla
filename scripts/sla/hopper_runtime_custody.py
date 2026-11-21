"""Ship the four installed library roots and interpreter used by the VANDA worker."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from hopper_parent_custody import HOST, PLAN, SSH, digest, remote, sync_dir, write_once


def filtered(info):
    parts = Path(info.name).parts
    if (any(x in {'.git', '__pycache__', '.pytest_cache', '.ssh', '.cache'} for x in parts)
            or parts[-1] in {'.netrc', 'direct_url.json'} or info.name.endswith('.pyc')):
        return None
    if info.issym() or info.islnk():
        # Permit packaged soname/interpreter aliases only inside the same library.
        target = info.linkname if info.islnk() else os.path.join(os.path.dirname(info.name), info.linkname)
        if not os.path.normpath(target).startswith(parts[0]+'/'):
            raise ValueError('External link in installed runtime: '+info.name)
    elif not (info.isdir() or info.isfile()):
        raise ValueError('Unexpected runtime member: '+info.name)
    return info


def receive(root, manifest_sha):
    manifest = root/'RUNTIME_BINDING.private.json'
    if digest(manifest) != manifest_sha:
        raise ValueError('Runtime manifest digest changed')
    card = json.loads(manifest.read_bytes())
    if card['plan'] != PLAN or time.time() >= card['deadline_epoch']:
        raise ValueError('Invalid runtime scope or deadline')
    archive = root/'runtime.tar'
    if digest(archive) != card['archive_sha256'] or archive.stat().st_size != card['bytes']:
        raise ValueError('Runtime archive changed')
    incoming = root/'runtime.incoming'
    final = root/'runtime.verified'
    if final.exists() or incoming.exists():
        raise FileExistsError('Reconcile the already started extraction')
    incoming.mkdir(mode=0o700)
    subprocess.run(['tar','--no-same-owner','-xf',str(archive),'-C',str(incoming)],check=True)
    py = incoming/'interpreter/bin/python3.12'
    paths = [str(incoming/k) for k in ('data_dependencies','upstream','packages')]
    env = dict(os.environ,PYTHONPATH=os.pathsep.join(paths),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1')
    script = ('import sys,json,importlib.metadata as m; print(json.dumps([list(sys.version_info[:3]),'
              '{k:m.version(k) for k in '+repr(list(card['versions']))+'}]))')
    version, packages = json.loads(subprocess.check_output([str(py),'-B','-c',script],env=env,text=True))
    if packages != card['versions'] or version != [3,12,14]:
        raise ValueError('Runtime version readback differs from production')
    incoming.rename(final)
    sync_dir(root)
    result = dict(status='REMOTE_TRAINING_INTERPRETER_STAGED',manifest_sha256=manifest_sha,
                  archive_sha256=card['archive_sha256'],bytes=card['bytes'],versions=packages,
                  python=version,job=card['job'],source_revision=card['source_revision'],
                  updates=0,generations=0,native=0,reader_verified=False,optimizer_restored=False)
    write_once(root/'RESULT.private.json',result)
    print(json.dumps(result))


def transfer(root):
    mount = json.loads((root/'MOUNT.private.json').read_bytes())
    if (mount['job'] != os.environ['PBS_JOBID'] or mount['plan'] != PLAN
            or time.time() >= mount['deadline_epoch'] or (root/'STOP').exists()):
        raise ValueError('Runtime transfer owner or scope mismatch')
    for name, expected in mount['scripts'].items():
        if digest(root/name) != expected:
            raise ValueError('Runtime staging source changed')
    base = Path(mount['base'])
    environment = base/'sla-d4-vanda-20261118-02'
    py = (environment/'env/bin/python').resolve(strict=True)
    layers = dict(interpreter=py.parent.parent,packages=environment/'env/lib/python3.12/site-packages',
                  data_dependencies=base/'sla-scale20k-vanda-tgt500-offline-20261119-03/deps',upstream=environment/'upstream')
    package_names = ['numpy','scipy','jax','jaxlib','jax-cuda12-plugin','flax','optax','torch',
                     'av','pyarrow','h5py','pillow','sentencepiece','orbax-checkpoint']
    script = 'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in '+repr(package_names)+'}))'
    env = dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=str(layers['data_dependencies']))
    versions=json.loads(subprocess.check_output([str(environment/'env/bin/python'),'-B','-c',script],env=env,text=True))
    path = root/'runtime.tar'
    with tarfile.open(path,'x',dereference=False) as archive:
        for label, folder in layers.items():
            if (root/'STOP').exists() or time.time() >= mount['deadline_epoch']:
                raise RuntimeError('Runtime transfer stopped or expired')
            archive.add(folder.resolve(strict=True),arcname=label,filter=filtered)
    manifest = root/'RUNTIME_BINDING.private.json'
    write_once(manifest,dict(plan=PLAN,job=mount['job'],source_revision=mount['source_revision'],
               bytes=path.stat().st_size,archive_sha256=digest(path),deadline_epoch=mount['deadline_epoch'],
               versions=versions,roots={k:str(v.resolve(strict=True)) for k,v in layers.items()}))
    receiver=mount['receiver']
    remote('mkdir','-m','700',receiver)
    subprocess.run(['rsync','-rt','--chmod=Fu=rw,Fgo=','-e',shlex.join(SSH),str(path),str(manifest),
                    *[str(root/k) for k in mount['scripts']],HOST+':'+receiver+'/'],check=True)
    cmd=' && '.join('test "$(sha256sum '+shlex.quote(k)+' | cut -d" " -f1)" = '+sha
                    for k,sha in mount['scripts'].items())
    report=remote('bash','-c','set -eu; cd "$1"; '+cmd+' && exec python3 -B hopper_runtime_custody.py receive "$1" "$2"',
                  'bash', receiver,digest(manifest))
    result=json.loads(report)
    if result['manifest_sha256']!=digest(manifest) or result['job']!=mount['job']:
        raise ValueError('Receiver identity mismatch')
    write_once(root/'RESULT.private.json',result)


if __name__ == '__main__':
    os.umask(0o077)
    if sys.argv[1]=='receive':
        receive(Path(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='transfer':
        root=Path.cwd()
        with (root/'OWNER.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                transfer(root)
            except Exception as exc:
                write_once(root/'FAILURE.private.json',dict(error=str(exc),job=os.environ.get('PBS_JOBID')))
                raise
    else:
        raise ValueError('Unknown runtime staging operation')
