"""Dependency-free cold metadata entry for the already paid AMD environment.

Never imports NumPy/MuJoCo on the management host. A real Apptainer child
inspects package metadata inside the fixed image; no native package is loaded.
The failed original entry and its immutable source are not modified or replayed.
"""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path

GRANT='73c7d2da1573c5ff440c8833da0f01910bca69ade2188645af866aef88c245e6'
SIF='2cae97d18c6a41450bb102e3bd54819d0c30552605d222d73ffc387c6ec14c8c'
FIELD='GEN1_AMD_layout_CORR1024_authority'
VERSIONS={'mujoco':'3.2.7','numpy':'1.26.4','scipy':'1.11.4','Pillow':'10.4.0','PyOpenGL':'3.1.7','glfw':'2.11.0'}


def need(ok,why):
    if not ok:raise PermissionError('GEN1_AMD_COLD_METADATA_'+why)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def put(p,v):
    b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())


def validate(root):
    r=Path(root);need(os.getuid()==2009 and os.uname().nodename=='corelab-amd-server'
        and r.is_absolute()and not r.is_symlink()and r.stat().st_uid==2009,'ACTUAL_HOST_UID')
    m=read(r/'RUN_MANIFEST.json');b=read(r/'OUTPUT_BINDING.private.json')
    need(m['scope']=='GEN1_AMD_ENVIRONMENT_ONLY_V1'and m['run_uuid']==r.name=='gen1-amd-environment-01'
        and b['run_root']==m['output_root']==str(r)and b['run_manifest_sha256']==sha(r/'RUN_MANIFEST.json'),
        'EXISTING_REGISTERED_ENVIRONMENT_MANIFEST')
    need(m['limits']==dict(Core_CPU_s=1800,LOCAL_CPU_s=1800,constructors=0,physics_ticks=0,
        models=0,PBS=0,retained_output_bytes=2*1024**3,transfer_bytes=2*1024**3),'SAME_PREPAID_ENVIRONMENT_SLICE')
    g=read(r/'plans/ACTUAL_GLOBAL_ENV_COLD02.private.json');mark=g[FIELD];row=mark['environment_preparation']
    receipt=read(r/'plans/ACTUAL_GLOBAL_ENV_COLD02_RECEIPT.private.json');cold=row['cold_metadata_entry']
    need(mark['grant_sha256']==GRANT and row['manifest_sha256']==b['run_manifest_sha256']
        and receipt['after_sha256']==sha(r/'plans/ACTUAL_GLOBAL_ENV_COLD02.private.json')
        and receipt['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
        and cold['entry_sha256']==sha(__file__)and cold['attempt']==2
        and cold['old_failed_metadata_OS_returncode']==1 and cold['no_additional_prepayment']is True,
        'ACTUAL_ONCE_COLD_APPENDIX_NOT_OLD_WRITE_REPLAY')
    old=read(r/'control/METADATA_ABI_OS_EXIT.json')
    need(old['returncode']==1 and old['pid']==cold['old_failed_metadata_PID']
        and not(r/'ENVIRONMENT_METADATA_STARTED.json').exists(), 'ORIGINAL_PREIMPORT_KNOWN_TERMINAL')
    p=Path('/proc')/str(old['pid'])
    need(not p.exists()or(p/'stat').read_text().rsplit(')',1)[1].split()[19]!=old['procstart'],
        'OLD_PROCESS_NOT_RESTARTED')
    lease=read(b['lease']);need(lease['status']=='ACTIVE'and lease['run_uuid']==r.name
        and mark['T_CD']<=time.time()<lease['deadline_epoch']==mark['deadline_epoch'],'SAME_OWN_LEASE_AND_WINDOW')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and'PBS_JOBID'not in os.environ,'NO_GPU_OR_PBS')
    code=r/m['code']['capsule_relative'];seal=read(r/'SOURCE_SEAL.json')
    for name,h in seal['files'].items():need(sha(code/name)==h,'ORIGINAL_SOURCE:'+name)
    for name,h in m['execution']['sealed_inputs'].items():need(sha(r/name)==h,'ORIGINAL_SEALED_INPUT:'+name)
    reg=read(b['registry']);env=read(b['environment'])
    need(sum(x.get('role')==b['role']and x['path']==str(r)for x in reg['entries'])==1
        and sum(x['name']=='amd-gen1-teacher-environment-preparation'and x['manifest']==str(r/'RUN_MANIFEST.json')for x in env['profiles'])==1,
        'ORIGINAL_CANONICAL_ENTRY_REGISTRATION')
    need(sha(r/'runtime/native-wbc-jazzy.sif')==SIF,'ORIGINAL_FULL_SIF')
    return m


def run(root):
    r=Path(root);m=validate(r);start=time.process_time()
    put(r/'control/ENVIRONMENT_COLD02_STARTED.json',dict(epoch=time.time(),pid=os.getpid(),
        procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    text="import importlib.metadata as m,json; names=('mujoco','numpy','scipy','Pillow','PyOpenGL','glfw'); print(json.dumps({n:m.version(n)for n in names}))"
    def limits():
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=subprocess.run(['/usr/bin/apptainer','exec','--cleanenv',str(r/'runtime/native-wbc-jazzy.sif'),
        '/opt/wbc-python/bin/python','-c',text],capture_output=True,text=True,timeout=30,preexec_fn=limits)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    v=dict(status='FAILED_CONTAINER_METADATA_PRESERVED',returncode=result.returncode,stdout=result.stdout,
        stderr=result.stderr,SIF_sha256=SIF,constructors=0,physics_ticks=0,models=0,Teacher_attempts=0,
        new_PBS=0,native_imports=False,source=cold_source(r),epoch=time.time(),
        CPU_s=time.process_time()-start+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
    if result.returncode==0:
        versions=json.loads(result.stdout);need(versions==VERSIONS,'EXACT_SIF_PACKAGE_VERSIONS')
        v.update(status='PASS_ACTUAL_AMD_CONTAINER_METADATA_ONLY',versions=versions)
    put(r/'ENVIRONMENT_COLD02_RESULT.json',v)
    return v


def cold_source(root):
    return read(Path(root)/'plans/ACTUAL_GLOBAL_ENV_COLD02.private.json')[FIELD]['environment_preparation']['cold_metadata_entry']['source_sha']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    v=run(args.root);print(json.dumps(v));raise SystemExit(v['returncode'])
