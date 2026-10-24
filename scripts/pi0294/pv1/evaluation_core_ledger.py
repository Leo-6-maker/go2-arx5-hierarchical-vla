"""Core actors debit the original Vanda global ledger through the fixed master.

No local global-ledger substitute, PBS environment alias, fresh SSH fallback or
write retry. The original RPC latch and original once IO own every remote write.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1.owner_transport import Journal
from scripts.pi0294.native_dataset import sha
from . import evaluation_core_accounting as c
from .evaluation_core_scope import validate,member

BINDING='plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json'

REMOTE=r'''
import base64,hashlib,json,os,pathlib,sys,time
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
b=P['binding'];code=Path(b['remote_code']);seal=Path(b['remote_seal'])
assert os.getuid()==22182 and os.environ.get('USER')=='e1192548' and not os.environ.get('PBS_JOBID')
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
assert not code.is_symlink() and code.is_dir() and not seal.is_symlink() and seal.is_file()
assert sha(seal.read_bytes())==b['remote_seal_sha256']
s=json.loads(seal.read_bytes());assert s['source_sha']==P['source_sha']
for name,digest in s['files'].items():
 relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
 p=code/relative;assert p.is_file() and not p.is_symlink() and sha(p.read_bytes())==digest
sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.pv1 import evaluation_core_accounting as c
from scripts.pi0294.dec1.source_cpu_once_io import commit
parent=Path(b['canonical_parent']);assert parent.name==c.scope.CAMPAIGN and parent.is_dir() and not parent.is_symlink()
path=parent/'HOPPER_RESOURCE_LEDGER.json';before=path.read_bytes();g=json.loads(before);c.checked(g)
row=g[c.FIELD][P['request_id']]
assert row['request_sha256']==P['request_sha256'] and row['source_sha']==P['source_sha']
assert row['child_run_uuid']==P['run_uuid'] and row['native_handle']==P['native_handle']
assert row['status'] in c.STATES[1:3]
if P['operation']=='STATE':
 print(json.dumps(dict(global_base64=base64.b64encode(before).decode(),global_sha256=sha(before),observed_epoch=time.time())));raise SystemExit(0)
assert P['operation'] in ('DEBIT','METRICS','CANDIDATE')
if P['operation']=='DEBIT':
 assert P['actor_pid']==row['pid'] and P['actor_start']==row['proc_start']
 g=c.charge(g,request_id=P['request_id'],native_handle=P['native_handle'],
  kind=P['kind'],count=P['count'],effect_id=P['effect_id'])
elif P['operation']=='METRICS':
 cpu=P['CPU_s'];calls=P['RPC_calls']
 assert type(cpu) in (int,float) and 0<=cpu<=7200 and type(calls) is int and 0<=calls<=40000
 control=g['V4_DEV_control_reservations'][P['run_uuid']]
 assert control['source_sha']==P['source_sha'] and control['status']=='RESERVED_NOT_SPENT'
 assert control['control_CPU_upper_s'] in (7200,7800)
 g['CPU_s']+=cpu;g['RPC_calls']=g.get('RPC_calls',0)+calls
else:
 assert P['actor_pid']==row['pid'] and P['actor_start']==row['proc_start']
 assert P['candidate']==row['evaluation_candidate']
 proof=dict(checkpoint_manifest_sha256=row['checkpoint_manifest_sha256'],global_step=4096)
 candidates=g.setdefault('development_candidates',{})
 assert P['candidate'] not in candidates or candidates[P['candidate']]==proof
 assert len(candidates)<6 or P['candidate'] in candidates
 candidates[P['candidate']]=proof
c.checked(g);g['revision']+=1
key='CORE-DEV-'+P['request_id']+'-'+hashlib.sha256(P['effect_id'].encode()).hexdigest()
receipt=commit(path,expected_before_sha256=sha(before),proposed_bytes=raw(g),request_id=key,
 intent_path=path.with_name(key+'.INTENT.private.json'),receipt_path=path.with_name(key+'.RECEIPT.private.json'),
 lock_path=path.with_suffix('.owner.lock'))
print(json.dumps(dict(receipt=receipt,global_base64=base64.b64encode(raw(g)).decode(),global_sha256=sha(raw(g)),observed_epoch=time.time())))
'''


class Ledger:
    role='hopper'  # Original logical wire role; actual execution stays Corelab.
    physical_site='CoreLab'

    def __init__(self,root,request_path=None):
        self.root=Path(root);self.manifest=validate(self.root)[0]
        self.binding=read(member(self.root,BINDING))
        if self.manifest['execution']['sealed_inputs'].get(BINDING)!=sha(self.root/BINDING):
            raise PermissionError('PV1_CORE_EXACT_PRESEALED_REMOTE_GLOBAL_BINDING')
        paths=list((self.root/'pbs').glob('alloc-*/REQUEST.json')) if request_path is None else [Path(request_path)]
        if len(paths)!=1:raise PermissionError('PV1_CORE_ONE_CURRENT_REQUEST_NO_FALLBACK')
        self.request_path=paths[0];self.request=read(self.request_path)
        if (self.request_path.is_symlink() or self.request_path.stat().st_mode&0o222
                or self.request_path.parent.parent.resolve()!=(self.root/'pbs').resolve()
                or self.request['platform_evaluation_version']!=c.VERSION
                or self.request['source_sha']!=self.manifest['code']['vla_commit']
                or self.request['run_manifest_sha256']!=sha(self.root/'RUN_MANIFEST.json')):
            raise PermissionError('PV1_CORE_EXACT_REGISTERED_IMMUTABLE_REQUEST')
        self.request_sha=sha(self.request_path)
        bound=read(self.root/'NATIVE_CORE_PROCESS_BOUND.json');self.native_handle=bound['native_handle']
        if bound['request_sha256']!=self.request_sha or bound['request_id']!=self.request['request_id']:
            raise PermissionError('PV1_CORE_ACTUAL_BOUND_PROCESS_RECORD')
        self.actor_pid=os.getpid();self.actor_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        self.sequence=0
        directory=self.root/'reservations/Core-global-rpc'/str(self.actor_pid)/str(self.actor_start)
        directory.mkdir(parents=True,mode=0o700,exist_ok=True)
        self.journal=Journal(directory,dict(run_uuid=self.root.name,job_id=self.native_handle,
            source_sha=self.request['source_sha'],request_sha256=self.request_sha))
        # Explicit cache only. The canonical resource ledger remains on Vanda.
        self.path=self.root/'evidence/ACTUAL_REMOTE_GLOBAL_SNAPSHOT.private.json'

    def _argv(self):
        b=self.binding;socket=Path(b['master_socket']);parent=socket.parent.stat();s=socket.lstat()
        if (parent.st_uid!=2009 or parent.st_mode&0o077 or not stat.S_ISSOCK(s.st_mode) or s.st_uid!=2009
                or b['remote_user']!='e1192548' or b['remote_host']!='vanda.nus.edu.sg'):
            raise PermissionError('PV1_CORE_OWN_FIXED_MASTER_NO_AUTH_FALLBACK')
        check=subprocess.run(['ssh','-S',str(socket),'-O','check','-l',b['remote_user'],b['remote_host']],
            capture_output=True,text=True,timeout=5)
        if check.returncode:raise ConnectionError('PV1_CORE_CANONICAL_MASTER_NOT_LIVE_NO_RECONNECT')
        python=b['remote_python']
        if type(python) is not str or not Path(python).is_absolute() or any(x in python for x in ('\n','\r','\x00')):
            raise PermissionError('PV1_CORE_PREBOUND_NATIVE_METADATA_INTERPRETER')
        return ['ssh','-T','-S',str(socket),'-o','ControlMaster=no','-o','ProxyCommand=false',
            '-o','CanonicalizeHostname=no','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
            '-o','GSSAPIAuthentication=no','-o','IPQoS=none','-l',b['remote_user'],b['remote_host'],
            shlex.join([python,'-'])]

    def _rpc(self,operation,**values):
        if os.getpid()!=self.actor_pid:raise PermissionError('PV1_CORE_NO_FORKED_OR_REBOUND_GLOBAL_WRITER')
        token=str(self.actor_pid)+':'+str(self.actor_start)+':'+str(self.sequence)+':'+operation
        payload=dict(binding=self.binding,operation=operation,request_id=self.request['request_id'],
            request_sha256=self.request_sha,source_sha=self.request['source_sha'],run_uuid=self.root.name,
            native_handle=self.native_handle,actor_pid=self.actor_pid,actor_start=self.actor_start,
            effect_id=token,**values)
        body='P='+repr(payload)+'\n'+REMOTE
        def invoke(intent,timeout):
            reply=subprocess.run(self._argv(),input=body,capture_output=True,text=True,timeout=timeout)
            record=self.journal.root/('TRANSPORT-'+intent['operation_id']+'.private.json')
            write(record,dict(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr,
                program_sha256=hashlib.sha256(body.encode()).hexdigest(),timeout_s=timeout))
            record.chmod(0o400)
            if reply.returncode:raise ConnectionError('PV1_CORE_REMOTE_GLOBAL_CALL_FAILED:'+str(reply.returncode))
            result=json.loads(reply.stdout);raw=base64.b64decode(result['global_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=result['global_sha256']:
                raise PermissionError('PV1_CORE_ACTUAL_REMOTE_FULL_BYTE_READBACK')
            c.checked(json.loads(raw));return result
        result=self.journal.call(('READ_' if operation=='STATE' else 'WRITE_')+operation,
            'READ_ONLY' if operation=='STATE' else 'MUTATION',invoke)
        raw=base64.b64decode(result['global_base64'],validate=True);self.sequence+=1
        self.raw_state_bytes=raw
        value=json.loads(raw);write(self.path,value);self.path.chmod(0o400)
        return value

    def state(self):return self._rpc('STATE')

    def charge(self,kind,count):
        if kind not in ('generation_queries','model_loads'):raise ValueError('PV1_CORE_KNOWN_MODEL_OR_QUERY_DEBIT')
        self.guard();return self._rpc('DEBIT',kind=kind,count=count)

    def cpu(self,cost):return self._rpc('DEBIT',kind='CPU_s',count=cost)

    def guard(self,*,closeout=False):
        now=time.time();cutoff=c.old.HARD_CLOSE if closeout else c.old.STOP_PHYSICS
        if type(closeout) is not bool or not c.old.T0<=now<cutoff:
            raise TimeoutError('PV1_CORE_ORIGINAL_CUTOFF')
        binding=read(self.root/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
        physical=Path(self.binding['physical_peer_root'])
        if (physical.name!=self.root.name or physical.parent.parent!=self.root.parent.parent
                or lease['status']!='ACTIVE' or lease['run_uuid']!=self.root.name
                or lease['authority_release']!=c.AUTHORITY or lease['owner']!='EXISTING_LOCAL_PC'
                or (not closeout and not now<lease['deadline_epoch'])):
            raise PermissionError('PV1_CORE_OWN_LEASE_AND_SAME_ORIGINAL_PHYSICAL_PEER')
        if not closeout and any(p.exists() for p in [self.root/'STOP.json',self.root.parent.parent/'STOP.json',
                physical/'STOP.json',physical/'SOURCE_PHYSICS_STOP.json',physical/'SOURCE_SAFETY_STOP_FROM_CORE.json']):
            raise PermissionError('PV1_CORE_ORIGINAL_SOURCE_OR_PARENT_STOP')
        return True

    @contextmanager
    def transaction(self):
        """Only mature RPC metrics or the exact existing candidate record."""
        before=self.state();after=deepcopy(before);yield after
        changed={k for k in set(before)|set(after) if before.get(k)!=after.get(k)}
        if not changed:return
        if changed<= {'CPU_s','RPC_calls'}:
            cpu=after.get('CPU_s',0)-before.get('CPU_s',0)
            calls=after.get('RPC_calls',0)-before.get('RPC_calls',0)
            self._rpc('METRICS',CPU_s=cpu,RPC_calls=calls)
        elif changed=={'development_candidates'}:
            candidate=self.request['evaluation_candidate'];expected=deepcopy(before.get('development_candidates',{}))
            expected[candidate]=dict(checkpoint_manifest_sha256=c.old.ENDPOINTS[candidate],global_step=4096)
            if after['development_candidates']!=expected:raise PermissionError('PV1_CORE_EXACT_EXISTING_CANDIDATE_ONLY')
            self._rpc('CANDIDATE',candidate=candidate)
        else:raise PermissionError('PV1_CORE_NO_ARBITRARY_REMOTE_GLOBAL_TRANSACTION')
