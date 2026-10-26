"""Finite AMD container metadata qualification through the existing entry.

This entry cannot construct MuJoCo models, run a Teacher, or start training.
The subsequent native stages require distinct sealed roles and allocations.
"""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
from . import amd_adoption as a


def read(p):return json.loads(Path(p).read_bytes())
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def validate(root):
    r=Path(root)
    a.need(r.is_absolute() and r.is_dir() and not r.is_symlink()
           and r.stat().st_uid==os.getuid()==2009
           and os.uname().nodename=='corelab-amd-server','ACTUAL_AMD_HOST_UID')
    m=read(r/'RUN_MANIFEST.json');b=read(r/'OUTPUT_BINDING.private.json')
    a.need(m['schema']=='GEN1_AMD_ENVIRONMENT_PREPARATION_RUN_MANIFEST_V1'
           and m['scope']==a.ROLE and m['run_uuid']==r.name=='gen1-amd-environment-01'
           and m['output_root']==str(r) and b['run_root']==str(r)
           and b['run_manifest_sha256']==file_sha(r/'RUN_MANIFEST.json'),
           'EXPLICIT_ENVIRONMENT_ONLY_ROLE')
    a.need(m['limits']==dict(Core_CPU_s=1800,LOCAL_CPU_s=1800,constructors=0,
           physics_ticks=0,models=0,PBS=0,retained_output_bytes=2*1024**3,
           transfer_bytes=2*1024**3),'APPROVED_FINITE_METADATA_SLICE')
    g=read(r/'plans/ACTUAL_GLOBAL_AFTER_ENVIRONMENT_ALLOCATION.private.json')
    marker=a.validate(g);row=marker['environment_preparation']
    a.need(row['manifest_sha256']==b['run_manifest_sha256'] and row['run_uuid']==r.name
           and row['status']=='PREPAID_ENVIRONMENT_ONLY','REAL_GLOBAL_CLASSIFICATION')
    receipt=read(r/'plans/ACTUAL_GLOBAL_ENVIRONMENT_RECEIPT.private.json')
    a.need(receipt['after_sha256']==file_sha(r/'plans/ACTUAL_GLOBAL_AFTER_ENVIRONMENT_ALLOCATION.private.json')
           and receipt['status']=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',
           'ORIGINAL_ONCE_RECEIPT_AND_EXACT_GLOBAL_BYTES')
    a.need(marker['T_CD']<=time.time()<marker['deadline_epoch'],'ORIGINAL_EXISTING_WINDOW')
    lease=read(b['lease'])
    a.need(lease['status']=='ACTIVE' and lease['run_uuid']==r.name
           and lease['manifest_sha256']==b['run_manifest_sha256']
           and lease['deadline_epoch']==marker['deadline_epoch'],'OWN_ACTIVE_ENVIRONMENT_LEASE')
    reg=read(b['registry']);rows=[x for x in reg['entries']if x.get('role')==a.ROLE]
    a.need(len(rows)==1 and rows[0]['path']==str(r) and rows[0]['run_uuid']==r.name,'UNIQUE_REGISTERED_ROOT')
    env=read(b['environment']);rows=[x for x in env['profiles']if x['name']==a.PROFILE]
    a.need(len(rows)==1 and rows[0]['manifest']==str(r/'RUN_MANIFEST.json')
           and rows[0]['hostnames']==['corelab-amd-server'],'EXPLICIT_AMD_PROFILE')
    code=r/m['code']['capsule_relative'];seal=read(r/'SOURCE_SEAL.json')
    a.need(Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_environment.py').resolve()
           and seal['source_sha']==m['code']['source_sha']
           and file_sha(r/'SOURCE_SEAL.json')==m['code']['source_seal_sha256'],
           'IMMUTABLE_ACTUALLY_IMPORTED_SOURCE')
    for n,h in seal['files'].items():
        q=code/n;a.need(not q.is_symlink() and file_sha(q)==h,'SOURCE_MEMBER:'+n)
    for n,h in m['execution']['sealed_inputs'].items():
        q=r/n;a.need(not q.is_symlink() and file_sha(q)==h,'SEALED_INPUT:'+n)
    for name,key in [('enter.sh','entry_sha256'),('doctor.py','doctor_sha256')]:
        a.need(file_sha(Path(b['registry']).parent/name)==m['execution'][key],'EXISTING_CANONICAL_ENTRY')
    a.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'PBS_JOBID'not in os.environ,
           'CPU_ONLY_NO_GPU_OR_PBS_ALIAS')
    return m


def metadata_abi(root):
    r=Path(root);m=validate(r);start=time.process_time()
    from scripts.pi0292.durable import write,OwnerLock
    with OwnerLock(r/'ENVIRONMENT_METADATA_OWNER.lock'):
        a.need(not(r/'ENVIRONMENT_METADATA_STARTED.json').exists(),'ONCE_NO_METADATA_REPLAY')
        write(r/'ENVIRONMENT_METADATA_STARTED.json',dict(epoch=time.time(),pid=os.getpid(),
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
        image=r/'runtime/native-wbc-jazzy.sif'
        a.need(file_sha(image)==a.SIF_SHA,'ORIGINAL_FULL_SIF_HASH')
        code="import importlib.metadata as m,json; names=('mujoco','numpy','scipy','Pillow','PyOpenGL','glfw'); print(json.dumps({n:m.version(n)for n in names}))"
        def limit():
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            resource.setrlimit(resource.RLIMIT_CPU,(30,30))
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        result=subprocess.run(['/usr/bin/apptainer','exec','--cleanenv',str(image),
            '/opt/wbc-python/bin/python','-c',code],capture_output=True,text=True,
            timeout=40,preexec_fn=limit)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        value=dict(schema='GEN1_AMD_ACTUAL_CONTAINER_METADATA_ABI_V1',returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,source=m['code']['source_sha'],
            CPU_s=time.process_time()-start+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            constructors=0,physics_ticks=0,models=0,Teacher_attempts=0,new_PBS=0,
            native_imports=False,epoch=time.time(),SIF_sha256=a.SIF_SHA)
        if result.returncode==0:
            versions=json.loads(result.stdout)
            a.need(versions=={'mujoco':'3.2.7','numpy':'1.26.4','scipy':'1.11.4',
                'Pillow':'10.4.0','PyOpenGL':'3.1.7','glfw':'2.11.0'},'EXACT_FIXED_PACKAGE_METADATA')
            value['versions']=versions;value['status']='PASS_ACTUAL_AMD_CONTAINER_METADATA_ONLY'
        else:value['status']='FAILED_CONTAINER_START_PRESERVED_NO_PHYSICS'
        write(r/'ENVIRONMENT_METADATA_RESULT.json',value)
        return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();v=metadata_abi(args.root);print(json.dumps(v))
    raise SystemExit(v['returncode'])
