"""Same native GLOBAL lock and original physical per-run counters on Vanda.

No remote Core cache, substitute budget, retried mutation or agent heartbeat.
Neural CPU and local RPC metrics are included once in the PBS joint CPU bill.
"""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from .neural_ledger import Ledger as OriginalLedger
from . import mix_vanda_eval_context as context, mix_vanda_eval_accounting as a
from . import mix_accounting as original, mix_core_accounting as core


def need(ok,why):
    if not ok:raise PermissionError('V5_VANDA_EVAL_LEDGER_'+why)


class Ledger(OriginalLedger):
    role='hopper'
    physical_site='VANDA_A40'
    candidate_field=core.CANDIDATES

    def __init__(self,root,request_path=None):
        self.root=Path(root);self.manifest=context.validate(root)[0]
        self.request_path=context.request_path(root) if request_path is None else Path(request_path)
        need(self.request_path==context.request_path(root),'EXACT_SEALED_SINGLE_REQUEST')
        self.request=read(self.request_path);self.request_sha=sha(self.request_path)
        self.request_id=self.request['request_id'];self.job_id=os.environ.get('PBS_JOBID')
        g,row,_=context.row(root)
        need(row['job_id']==self.job_id,'EXACT_NATIVE_BOUND_PBS')
        self.actor_pid=os.getpid();self.actor_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        self.sequence=0
        super().__init__(root)

    def state(self):
        self.raw_state_bytes=self.path.read_bytes();return a.checked(__import__('json').loads(self.raw_state_bytes))

    def guard(self,*,closeout=False):
        need(type(closeout) is bool,'EXPLICIT_CLOSEOUT')
        now=time.time();end=context.deadline(self.root,self.manifest,closeout=closeout)
        binding=read(self.root/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
        need(lease['status']=='ACTIVE' and lease['run_uuid']==self.root.name and lease['deadline_epoch']==end,'OWN_LIVE_GPU_LEASE')
        need(os.environ.get('PBS_JOBID')==self.job_id,'NO_JOB_REBIND')
        if not closeout:
            peers=context.paths(self.root,self.manifest);physical=Path(peers['core']['root'])
            need(not any(p.exists() for p in (self.root/'STOP.json',self.parent/'STOP.json',physical/'STOP.json',
                     physical/'SOURCE_PHYSICS_STOP.json',self.root/'SOURCE_SAFETY_STOP_FROM_CORE.json')),'ALL_ORIGINAL_STOP_FENCES')
            read_live_owner(self.root,self.root.name)
        return True

    def _debit(self,kind,count):
        need(os.getpid()==self.actor_pid and int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])==self.actor_start
             and os.environ.get('PBS_JOBID')==self.job_id,'NO_FORKED_OR_REBOUND_EFFECT_WRITER')
        token=str(self.actor_pid)+':'+str(self.actor_start)+':'+str(self.sequence)+':'+kind
        with super().transaction() as g:
            row=a.item(g,self.request_id)
            if kind!='CPU_s':
                p=row.get('neural_process',{})
                need(p.get('pid')==self.actor_pid and p.get('proc_start')==self.actor_start,'ONLY_BOUND_MODEL_ACTOR_MAY_DEBIT_QUERY_OR_LOAD')
            updated=a.charge(g,request_id=self.request_id,job_id=self.job_id,kind=kind,count=count,effect_id=token)
            g.clear();g.update(updated)
        self.sequence+=1

    def charge(self,kind,count):
        need(kind in ('generation_queries','model_loads'),'ONLY_ORIGINAL_MODEL_AND_QUERY_DEBITS')
        self.guard();return self._debit(kind,count)

    def cpu(self,cost):
        return self._debit('CPU_s',cost)

    @contextmanager
    def transaction(self):
        """The original candidate record or prepaid local RPC metrics only."""
        with super().transaction() as g:
            before=deepcopy(g);yield g
            changed={k for k in set(g)|set(before) if g.get(k)!=before.get(k)}
            if not changed:return
            if changed<={'CPU_s','RPC_calls'}:
                cpu=original.number(g.get('CPU_s',0)-before.get('CPU_s',0))
                calls=original.number(g.get('RPC_calls',0)-before.get('RPC_calls',0),integer=True)
                g['CPU_s']=before['CPU_s'];g['RPC_calls']=before.get('RPC_calls',0)
                metrics=g.setdefault('V5_CD_Vanda_prepaid_control_actual_metrics',{}).setdefault(self.request_id,dict(CPU_s=0,RPC_calls=0))
                need(metrics['CPU_s']+cpu+600<=7200 and metrics['RPC_calls']+calls<=150000,'ORIGINAL_SHARED_PREPAID_CONTROL_BOUND')
                metrics['CPU_s']+=cpu;metrics['RPC_calls']+=calls
            else:
                candidate=self.request['evaluation_candidate'];expected=deepcopy(before.get(self.candidate_field,{}))
                expected[candidate]=dict(checkpoint_manifest_sha256=self.request['checkpoint_manifest_sha256'],global_step=1024)
                need(changed=={self.candidate_field} and g[self.candidate_field]==expected,'ONLY_EXACT_EXISTING_CLOSED_CANDIDATE_RECORD')
