"""Finite, CPU-only delivery of the seven already pinned public raw inputs.

This is input staging, never training admission. It uses the existing user's
finite background mechanism; it does not install dependencies or change mounts,
services, credentials, active capsules, experiment leases or other GPU work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import urllib.request

REPO = 'nvidia/GR00T-N1.7-3B'
REVISION = '2fc962b973bccdd5d8ce4f67cc63b264d6886495'
PARENT_ID = '0e493187af087d156f6352d46ce14e049ff18e7c1cb1eabc8671357ab4b96e71'
FILES = {
    'model-00001-of-00002.safetensors': ('8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d',4990519232),
    'model-00002-of-00002.safetensors': ('c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661',1919980184),
    'config.json': ('54c0367060cd310d0b3343fe72a589860a8b6e8173810164a4ffd6253f52e689',2071),
    'processor_config.json': ('85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be',21137),
    'embodiment_id.json': ('abc4a749389837416a102d9455a8c089336f2e417afb18565703b9944974bc35',2145),
    'model.safetensors.index.json': ('407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59',104985),
    'statistics.json': ('c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9',4293096),
}
GIB = 1024**3


def raw(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def spec(root,protected_remaining_bytes,*,now):
    if type(protected_remaining_bytes)is not int or protected_remaining_bytes<0:
        raise ValueError('ACTUAL_EXISTING_REMAINING_WORK_PROTECTION')
    return dict(schema='SHARED_RAW_PARENT_CPU_INPUT_DELIVERY_V1',root=str(root),
        repo=REPO,revision=REVISION,raw_parent_files_identity=PARENT_ID,
        files={n:dict(sha256=p,bytes=b)for n,(p,b)in FILES.items()},
        total_input_bytes=sum(b for p,b in FILES.values()),
        protected_remaining_bytes=protected_remaining_bytes,free_margin_bytes=30*GIB,
        maximum_bytes_per_second=8*1024**2,wall_upper_s=3600,CPU_upper_s=180,
        deadline_epoch=now+3600,created_epoch=now,automatic_retry=False,
        neural_models_GPU_physics_optimizer_updates=0,training_admitted=False)


def exclusive(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(raw(value));f.flush();os.fsync(f.fileno())


def progress(path,value):
    temporary=path.with_name(path.name+'.pending')
    exclusive(temporary,value);os.replace(temporary,path)


def stat_identity(path):
    st=path.stat()
    return dict(dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def download_member(response,target,pin,*,guard,report,throttle):
    """Unknown/corrupt/partial files are retained and never published as valid."""
    target=Path(target);partial=target.with_name(target.name+'.PARTIAL')
    if target.exists()or target.is_symlink():raise PermissionError('NO_RAW_INPUT_OVERWRITE')
    count=0;digest=hashlib.sha256()
    with partial.open('xb')as output:
        while True:
            guard();chunk=response.read(1024**2)
            if not chunk:break
            count+=len(chunk)
            report(count)
            if count>pin['bytes']:raise PermissionError('EXACT_RAW_MEMBER_SIZE_UPPER')
            digest.update(chunk);output.write(chunk);throttle(count)
        output.flush();os.fsync(output.fileno())
    if count!=pin['bytes']or digest.hexdigest()!=pin['sha256']:
        raise PermissionError('EXACT_COMPLETE_RAW_MEMBER_SHA256_AND_BYTES')
    if target.exists()or target.is_symlink():raise PermissionError('NO_RAW_INPUT_OVERWRITE')
    os.rename(partial,target)
    return dict(sha256=pin['sha256'],stat=stat_identity(target))


def run(root,manifest_sha256):
    root=Path(root);mb=(root/'INPUT_MANIFEST.json').read_bytes();m=json.loads(mb)
    if (hashlib.sha256(mb).hexdigest()!=manifest_sha256
            or m!=spec(root,m['protected_remaining_bytes'],now=m['created_epoch'])
            or os.getuid()!=2009 or os.uname().nodename!='corelab-4080s'
            or not root.is_absolute()or root.resolve()!=root
            or not str(root).startswith('/var/cache/users/liuyu-nfs/')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise PermissionError('EXACT_FINITE_NODE_LOCAL_CORE_CPU_INPUT_DELIVERY')
    source=Path(__file__).resolve()
    staged=json.loads((root/'STAGE_COMPLETE.json').read_bytes())
    if staged['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise PermissionError('SEALED_RAW_DELIVERY_SOURCE')
    exclusive(root/'STARTED.json',dict(PID=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        input_manifest_sha256=manifest_sha256,source_sha256=staged['source_sha256'],epoch=time.time(),
        PC_Vanda_GPU_or_forwarded_auth_dependency=False))
    began=time.monotonic();cpu=time.process_time();rows={};completed_bytes=0;current_bytes=0
    result=None;failure=None;observed_transfer_bytes=0
    def guard():
        if time.time()>=m['deadline_epoch']or time.process_time()-cpu>m['CPU_upper_s']:
            raise TimeoutError('FINITE_RAW_INPUT_DELIVERY_END')
        fs=os.statvfs(root)
        remaining=m['total_input_bytes']-completed_bytes-current_bytes
        if fs.f_bavail*fs.f_frsize<m['protected_remaining_bytes']+remaining+m['free_margin_bytes']:
            raise PermissionError('PRESERVE_EXISTING_WORK_AND_ALL_REMAINING_RAW_INPUT_BYTES')
    def stop(sig,frame):raise TimeoutError('FINITE_RAW_INPUT_DELIVERY_SIGNAL')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,int(m['deadline_epoch']-time.time())))
    try:
        guard()
        for name,pin in m['files'].items():
            current_bytes=0;last_report=0.;file_started=time.monotonic()
            exclusive(root/(name+'.INTENT.json'),dict(name=name,sha256=pin['sha256'],bytes=pin['bytes'],epoch=time.time()))
            def report(count):
                nonlocal current_bytes,last_report,observed_transfer_bytes
                observed_transfer_bytes+=count-current_bytes;current_bytes=count
                if time.monotonic()-last_report>=10:
                    progress(root/'PROGRESS.json',dict(file=name,current_file_bytes=count,
                        completed_bytes=completed_bytes,verified_files=len(rows),epoch=time.time()))
                    last_report=time.monotonic()
            def throttle(count):
                delay=count/m['maximum_bytes_per_second']-(time.monotonic()-file_started)
                while delay>0:
                    guard();time.sleep(min(.25,delay))
                    delay=count/m['maximum_bytes_per_second']-(time.monotonic()-file_started)
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
            with urllib.request.urlopen(url,timeout=30)as response:
                rows[name]=download_member(response,root/name,pin,guard=guard,report=report,throttle=throttle)
            completed_bytes+=current_bytes;current_bytes=0
            exclusive(root/(name+'.VERIFIED.json'),rows[name])
        audit=dict(status='ACTUAL_FULL_RAW_PARENT_SEVEN_FILES_VERIFIED',revision=REVISION,
            raw_parent_files_identity=PARENT_ID,files=rows,wall_seconds=time.monotonic()-began,models_loaded=0)
        exclusive(root/'RAW_PARENT_AUDIT.json',audit)
        result='COMPLETE_SEVEN_PINNED_RAW_INPUTS_NOT_TRAINING_ADMISSION'
    except BaseException as error:
        failure=type(error).__name__+': '+str(error);result='FAILED_RAW_INPUT_DELIVERY_NO_AUTOMATIC_RETRY'
    finally:
        signal.alarm(0)
        exclusive(root/'RESULT.json',dict(status=result,error=failure,verified_files=len(rows),
            verified_input_bytes=completed_bytes,actual_HTTP_body_bytes=observed_transfer_bytes,
            CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began,epoch=time.time(),
            input_manifest_sha256=manifest_sha256,neural_models_GPU_physics_optimizer_updates=0,
            training_admitted=False,automatic_retry=False))
    if failure is not None:raise RuntimeError(failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();run(args.root,args.manifest_sha256)
