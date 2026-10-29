"""Finish the exact cold S1 import-rejected tail without restaging or charging.

The original standard entry resets PYTHONPATH. Restore the already sealed ROS
search path inside that interpreter; do not edit the entry or runtime capsule.
"""
import hashlib
import json

from scripts.pi0292.durable import OwnerLock
from .task_path_stage import PROGRAM as ORIGINAL_STAGE


PARTIAL_READBACK = r'''
import os,json,hashlib,time,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
root=Path(P['root']);sha=lambda b:hashlib.sha256(b).hexdigest()
files={n:sha((root/n).read_bytes())for n in P['expected_files']}
assert files==P['expected_files']
seal=json.loads((root/'NATIVE_SOURCE_SEAL.json').read_bytes())
assert seal['source_sha']==P['source_sha']and all(sha((root/'native-entry/code/vla'/n).read_bytes())==v for n,v in seal['files'].items())
account=json.loads((root/'control/S1_PREPARATION_ACCOUNT.json').read_bytes())
assert account['starts']==account['integrations']==account['RGB_pairs']==0 and not account['native_attempts_admitted']
assert all(not(root/n).exists()for n in P['absent_tail'])
assert sha(Path(P['registry']).read_bytes())==P['registry_before_sha256']
code=root/'native-entry/code/vla';import sys;sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.dec1.source_cpu_once_io import reconcile
path=root/'control/S1_PREPARATION_ACCOUNT.json';key='S1-NATIVE-COLD-PREPARATION-20261029'
receipt=reconcile(path,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'))
assert receipt['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
hz=os.sysconf('SC_CLK_TCK');boot=int(next(x.split()[1]for x in Path('/proc/stat').read_text().splitlines()if x.startswith('btime ')));live=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit()or int(d.name)==os.getpid():continue
 try:
  if d.stat().st_uid!=os.getuid():continue
  fields=(d/'stat').read_text().rsplit(')',1)[1].split();cmd=(d/'cmdline').read_bytes()
  if fields[0]!='Z'and b'<published-cold-stage>'in cmd and boot+int(fields[19])/hz>=P['intent']['started_epoch']-1:
   live.append(dict(pid=int(d.name),proc_start=fields[19]))
 except FileNotFoundError:pass
print(json.dumps(dict(schema='S1_COLD_IMPORT_REJECTION_EXACT_PARTIAL_V1',intent=P['intent'],observed_epoch=time.time(),
 files=files,registry_sha256=P['registry_before_sha256'],source_seal_verified=True,
 preparation_receipt=receipt,absent_tail=P['absent_tail'],matching_processes=live,
 starts=0,integrations=0,RGB_pairs=0,read_only=True)))
'''


def reconcile_partial_import_failure(journal, proof, expected, transport):
    lock = OwnerLock(journal.root/'RPC_OWNER.lock')
    try:
        state = journal._state(); intent = state['intent']; now = journal.clock()
        if (state['status'] != 'UNKNOWN' or intent['effect'] != 'MUTATION'
                or intent['phase'] != 'S1_NATIVE_COMPRESSED_COLD_CONTINUATION_ONCE'
                or proof.get('schema') != 'S1_COLD_IMPORT_REJECTION_EXACT_PARTIAL_V1'
                or proof.get('intent') != intent or expected['intent'] != intent
                or proof.get('files') != expected['expected_files']
                or proof.get('registry_sha256') != expected['registry_before_sha256']
                or proof.get('absent_tail') != expected['absent_tail']
                or proof.get('source_seal_verified') is not True
                or proof.get('preparation_receipt',{}).get('status') != 'CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
                or proof.get('matching_processes') != [] or proof.get('read_only') is not True
                or any(proof.get(k) != 0 for k in ('starts','integrations','RGB_pairs'))
                or transport.get('returncode') != 1 or transport.get('stdout') != ''
                or "ModuleNotFoundError: No module named 'rclpy'" not in transport.get('stderr','')
                or not intent['started_epoch'] <= proof.get('observed_epoch',-1) <= now
                or not 0 <= now-proof['observed_epoch'] <= 120):
            raise PermissionError('EXACT_IMPORT_REJECTED_PARTIAL_STAGE_TERMINAL_REQUIRED')
        digest=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
        journal._event(intent,'NATIVE_PARTIAL_STAGE_RECONCILED',proof=proof,proof_sha256=digest)
        journal._save(journal.state_path,dict(status='CONFIRMED',intent=intent,
            resolution='EXACT_PARTIAL_STAGE_IMPORT_REJECTED_COSTS_RETAINED',proof_sha256=digest))
        return dict(status='KNOWN_PARTIAL_NO_REPLAY_NO_REFUND',proof_sha256=digest)
    finally:lock.close()


_HEADER = r'''
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(10,10));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
assert os.getuid()==2009 and os.uname().nodename=='corelab-4080s'
began=time.process_time();root=Path(P['root']);registry=Path(P['registry']);before_registry=registry.read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest();raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
assert sha(before_registry)==P['registry_before_sha256']
assert all(0<=time.time()-v['observed_epoch']<=120 for v in P['fresh_observation'].values())
forecast=json.loads((root/'NATIVE_FULL14_FORECAST.json').read_bytes())
assert forecast['shortfalls']=={} and P['fresh_observation']['GLOBAL']['sha256']==forecast['original_forecast']['GLOBAL_sha256']
assert P['fresh_observation']['LOCAL']['sha256']==forecast['original_forecast']['historical_LOCAL_sha256']
for name,pin in P['expected_files'].items():assert sha((root/name).read_bytes())==pin
for name in P['absent_tail']:assert not(root/name).exists()
approval=(root/'S1_HUMAN_APPROVAL.json').read_bytes()
inputs=json.loads((root/'NATIVE_INPUTS.private.json').read_bytes());code=root/'native-entry/code/vla';infra=root/'native-entry/infrastructure'
seal=json.loads((root/'NATIVE_SOURCE_SEAL.json').read_bytes());members=seal['files'];manifest=json.loads((root/'NATIVE_RUN_MANIFEST.json').read_bytes())
assert seal['source_sha']==manifest['code']['vla_commit']==P['source_sha']
assert all(sha((code/n).read_bytes())==pin for n,pin in members.items())
sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.task_path_entry import authority,node_local_paths,ENTRY_FILES
authority(approval,now_epoch=time.time());node_local_paths(inputs['path_roles'])
assert all(sha((infra/n).read_bytes())==pin for n,pin in ENTRY_FILES.items())
assert all(sha(Path(n).read_bytes())==pin for n,pin in inputs['frozen_files'].items())
from scripts.pi0294.dec1.source_cpu_once_io import commit
account_path=root/'control/S1_PREPARATION_ACCOUNT.json'
def put(path,value,mode=0o400):
 p=Path(path);b=value if isinstance(value,bytes)else raw(value)
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==b
put(root/'NATIVE_TAIL_FRESH_FULL14_READBACK.private.json',P['fresh_observation'])
'''
_tail = 'safe_parent_keys='+ORIGINAL_STAGE.split('safe_parent_keys=',1)[1]
_old_probe="probe='import os,sys,json,importlib;mods='"
_new_probe=("probe='import os,sys,json,importlib;sys.path[:0]='+repr([str(code),str(code/'src'),"
            "*inputs['runtime_environment']['PYTHONPATH'].split(':')])+';mods='")
assert _tail.count(_old_probe)==1
TAIL_PROGRAM = _HEADER+_tail.replace(_old_probe,_new_probe).replace(
    "put(root/'NATIVE_COLD_PREPARATION_RESULT.json',result)",
    "result.update(CPU_measurement_scope='THIS_COMPLETION_TAIL_ONLY',prior_partial_costs_retained=True,new_preparation_charge_s=0);put(root/'NATIVE_COLD_PREPARATION_RESULT.json',result)")


def worker_bootstrap(root, episode):
    """Original entry/interpreter, explicit sealed ROS path for owned children."""
    return ("import json,os,sys;from pathlib import Path;root=Path("+repr(str(root))+
        ");i=json.loads((root/'NATIVE_INPUTS.private.json').read_bytes());m=json.loads((root/'NATIVE_RUN_MANIFEST.json').read_bytes());code=root/m['code']['capsule_relative'];"
        "p=':'.join([str(code),str(code/'src'),i['runtime_environment']['PYTHONPATH']]);"
        "os.environ['PYTHONPATH']=p;sys.path[:0]=p.split(':');"
        "from scripts.pi0294.task_path_worker import run;run(str(root),i['native_root'],i['startup_root'],"+repr(episode)+")")


def unit_entry_launcher(root, bootstrap):
    """Keep systemd's actual invocation identity when clearing inherited env.

    The stored environment is the existing sealed whitelist. The invocation
    value comes directly from this newly started native unit, never the PC.
    """
    return ("import json,os;from pathlib import Path;root=Path("+repr(str(root))+
        ");env=json.loads((root/'NATIVE_PROCESS_ENVIRONMENT.private.json').read_bytes());"
        "m=json.loads((root/'NATIVE_RUN_MANIFEST.json').read_bytes());"
        "invocation=os.environ['INVOCATION_ID'];assert len(invocation)==32;"
        "env['INVOCATION_ID']=invocation;os.execve('/bin/bash',['/bin/bash',"
        "str(root/m['entry_relative']/'enter.sh'),'4080-simulation','python','-B','-c',"
        +repr(bootstrap)+"],env)")
