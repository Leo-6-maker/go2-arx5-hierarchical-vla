"""One cold launch after the known pre-submission offline-environment failure.

The original owner and science capsule remain immutable. This version records
its own process/OS witnesses and supplies the same offline flags used by the
successful cold validation and the compute script. It never submits directly.
"""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path

OFFLINE=dict(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
REPAIR='cpu-owner-offline-repair-02'
FAILURE='GEN1_AMD_ADOPTION_EXPORT_CPU_OFFLINE_AND_STOP_FENCES'

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def require(ok,why):
    if not ok:raise PermissionError('GEN1_CPU_OFFLINE_COLD_'+why)
def environment(original):return dict(original,**OFFLINE)
def procstart(pid):
    try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
def once(p,value):
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:
        f.write((json.dumps(value,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())

def previous_boundary(root,*,current=procstart):
    r=Path(root);pin=read(r/'control/CPU_OWNER_BINDING.private.json')
    repair=r/'control'/REPAIR;binding=read(repair/'BINDING.private.json')
    exit=read(r/'control/CPU_OWNER_OS_EXIT.json')
    require(exit['returncode']==1 and exit['control_source']==pin['control_source_sha']
        and 0<exit['site_control_CPU_s']<30,'KNOWN_FINITE_OS1_IN_SHARED_PREPAID_CPU')
    require(sha((r/'control/CPU_OWNER_OS_EXIT.json').read_bytes())==binding['old_OS_exit_sha256']
        and sha((r/'control/CPU_OWNER.stderr').read_bytes())==binding['old_stderr_sha256']
        and (r/'control/CPU_OWNER.stderr').read_text().rstrip().endswith('PermissionError: '+FAILURE),
        'EXACT_RECORDED_OFFLINE_GUARD_FAILURE')
    for name in('CPU_OWNER_WRAPPER_STARTED.json','CPU_OWNER_LAUNCHED_PROCESS.json'):
        row=read(r/'control'/name)
        require(current(row['pid'])!=row['procstart'],'OLD_ACTOR_STILL_ACTIVE')
    require(not(r/'control/original-cpu-owner').exists()
        and not(r/'control/CPU_OWNER_PROCESS.json').exists()
        and not(r/'control/CPU_RUNNING_RELEASE.json').exists()
        and not(r/'pbs/alloc-01').exists(),'NO_OLD_JOURNAL_QSUB_RELEASE_OR_PAYLOAD')
    gbytes=(r.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes()
    require(sha(gbytes)==binding['expected_GLOBAL_sha256'],'EXACT_UNCHANGED_GLOBAL')
    g=json.loads(gbytes)
    rows=[x for x in g['PBS_submissions']if x.get('request_id')==binding['request_id']]
    require(sha(gbytes)==binding['expected_GLOBAL_sha256']and len(rows)==1
        and rows[0]['job_id']is None and rows[0]['run_manifest_sha256']==pin['parent_manifest_sha256'],
        'EXACT_UNCHANGED_GLOBAL_UNBOUND_EXISTING_REQUEST')
    require(sha((r/'RUN_MANIFEST.json').read_bytes())==pin['parent_manifest_sha256']
        and sha((r/'control-code/amd_export_owner.py').read_bytes())==pin['owner_code_sha256']
        and sha((r/'control/CPU_OWNER_CONFIG.private.json').read_bytes())==pin['configuration_sha256']
        and read(read(r/'OUTPUT_BINDING.private.json')['lease'])['status']=='ACTIVE',
        'IMMUTABLE_ORIGINAL_OWNER_SCIENCE_CONFIG_AND_OWN_ACTIVE_LEASE')
    require(binding['old_control_CPU_s']==exit['site_control_CPU_s']
        and binding['additional_prepay']==0 and binding['original_control_upper_CPU_s']==600,
        'NO_OLD_COST_REFUND_OR_SECOND_PREPAY')
    return repair,binding

def run(root):
    require(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-'),
        'ORIGINAL_LOGIN_CPU_CONTROL_ONLY')
    r=Path(root);repair,pin=previous_boundary(r)
    require(Path(__file__).resolve()==(repair/'amd_export_wrapper.py').resolve()
        and sha(Path(__file__).read_bytes())==pin['wrapper_code_sha256'],
        'EXACT_NEW_COLD_ENTRY')
    secret=sys.stdin.readline().rstrip('\n');start=time.process_time()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with(repair/'CPU_OWNER.stdout').open('xb')as out,(repair/'CPU_OWNER.stderr').open('xb')as err:
        child=subprocess.Popen([sys.executable,'-u',str(r/'control-code/amd_export_owner.py'),
            '--root',str(r)],stdin=subprocess.PIPE,stdout=out,stderr=err,
            start_new_session=True,env=environment(os.environ))
        row=dict(pid=child.pid,procstart=procstart(child.pid),
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            host=os.uname().nodename,control_source=pin['control_source_sha'])
        once(repair/'CPU_OWNER_LAUNCHED_PROCESS.json',row)
        child.stdin.write((secret+'\n').encode());child.stdin.close();secret=None
        rc=child.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=time.process_time()-start+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    once(repair/'CPU_OWNER_OS_EXIT.json',dict(row,returncode=rc,
        site_control_CPU_s=cpu,old_control_CPU_s=pin['old_control_CPU_s'],
        combined_failed_and_current_control_CPU_s=cpu+pin['old_control_CPU_s'],
        original_control_upper_CPU_s=600,epoch=time.time(),agent_heartbeat_required=False))
    return rc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    raise SystemExit(run(p.parse_args().root))
