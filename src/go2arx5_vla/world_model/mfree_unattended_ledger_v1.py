"""One finite campaign ledger: atomic claims, shared isolation and snapshots.

Uses the existing durable writer and kernel OwnerLock. Claimed/unknown IDs are
never returned to the pool. Source C20 results retain their original identities.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time
from scripts.pi0292.durable import OwnerLock, write

PLAN='4e926e42a7337ad1d48c00087ef980a8f0157936'
STAGES=(('C20',2,4),('C60',6,12),('C100',10,20))
CELLS=tuple(f'{family}-T{k}'for k in range(5)for family in ('F1','F2'))
ACTIVE={'CLAIMED','REGISTERED','LAUNCHED','RUNNING','UNKNOWN'}

def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ledger:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'ATTEMPTS.private.json';self.lock=self.root/'claim.lock'

    @contextmanager
    def transaction(self):
        # This lock serializes short JSON transactions only, never a worker.
        until=time.monotonic()+10
        while True:
            try:lock=OwnerLock(self.lock);break
            except BlockingIOError:
                if time.monotonic()>until:raise
                time.sleep(.01)
        try:
            value=read(self.path);yield value
            value['revision']+=1;value['updated_epoch']=time.time();write(self.path,value)
        finally:lock.close()

    def create(self,carryover,*,deadline_epoch):
        self.root.mkdir(parents=True,exist_ok=True)
        lock=OwnerLock(self.lock)
        try:
            if self.path.exists():raise FileExistsError('EXISTING_ATTEMPT_LEDGER_NO_REINITIALIZATION')
            rows=[]
            for entry in carryover:
                row=copy.deepcopy(entry)
                if row['cell']not in CELLS or row['state']not in ('QUALIFIED','TASK_FAILED','UNSAFE','TECHNICAL','UNKNOWN'):
                    raise ValueError('ACTUAL_C20_CARRYOVER_STATES_ONLY')
                row['carried_over']=True;rows.append(row)
            if len(rows)>40 or len({r['id']for r in rows})!=len(rows):raise ValueError('UNIQUE_C20_ATTEMPT_CARRYOVER')
            value=dict(schema='MFREE_UNATTENDED_ATTEMPT_LEDGER_V1',plan_sha=PLAN,revision=0,
                deadline_epoch=deadline_epoch,attempts=rows,isolated_families={},blocked_cells={},
                snapshots={},jobs={},supervisor_epochs=[],updated_epoch=time.time())
            write(self.path,value)
        finally:lock.close()

    @staticmethod
    def counts(value,cell):
        rows=[r for r in value['attempts']if r['cell']==cell]
        return dict(used=len(rows),qualified=sum(r['state']=='QUALIFIED'for r in rows),
            active=sum(r['state']in ACTIVE for r in rows))

    @staticmethod
    def stage(value):
        for name,target,cap in STAGES[1:]:
            if name not in value['snapshots']:return name,target,cap
        return None

    def claim(self,worker,*,now=None):
        now=time.time()if now is None else now
        with self.transaction()as value:
            if now>=value['deadline_epoch']-3600:return None
            if any(r['worker']==worker and r['state']in ACTIVE for r in value['attempts']if not r.get('carried_over')):
                raise PermissionError('WORKER_HAS_ACTIVE_OR_UNKNOWN_ATTEMPT_NO_REPLAY')
            stage=self.stage(value)
            if stage is None:return None
            name,target,cap=stage;candidates=[]
            for index,cell in enumerate(CELLS):
                c=self.counts(value,cell)
                if (cell[:2]in value['isolated_families']or cell in value['blocked_cells']
                        or c['used']>=cap or c['qualified']+c['active']>=target):continue
                candidates.append((c['qualified']+c['active'],c['used'],index,cell))
            if not candidates:return None
            _,_,index,cell=min(candidates);used=self.counts(value,cell)['used'];ordinal=used+1
            if ordinal>20 or len(value['attempts'])>=200:raise PermissionError('FIXED_COLLECTION_ATTEMPT_CAP')
            row=dict(id=f'72H-{cell}-A{ordinal:03d}',cell=cell,family=cell[:2],layout_id=cell[3:],
                ordinal=ordinal,reset=10000+index*100+ordinal,seed=int(hashlib.sha256(f'{PLAN}|{cell}|{ordinal}'.encode()).hexdigest()[:8],16),
                stage=name,state='CLAIMED',worker=worker,claimed_epoch=now,physical_requests=0,
                native_integrations=0,technical_successors=0,carried_over=False)
            value['attempts'].append(row);return copy.deepcopy(row)

    def update(self,identity,**fields):
        with self.transaction()as value:
            row=next(r for r in value['attempts']if r['id']==identity)
            if row['state']=='QUALIFIED'and fields.get('state','QUALIFIED')!='QUALIFIED':raise PermissionError('QUALIFIED_RESULT_IMMUTABLE')
            row.update(fields)

    def isolate(self,family,evidence):
        with self.transaction()as value:
            if family not in value['isolated_families']:value['isolated_families'][family]=dict(evidence,epoch=time.time())
            write(self.root/'ISOLATION.private.json',value['isolated_families'])

    def seal_ready(self):
        """Atomic immutable descriptor; only validated CLOSED result pins enter."""
        sealed=[]
        with self.transaction()as value:
            for name,target,cap in STAGES:
                if name in value['snapshots']:continue
                if name=='C20':
                    if any(self.counts(value,c)['qualified']<2 for c in CELLS):raise ValueError('C20_IMPORT_MUST_BE_COMPLETE')
                else:
                    ready=True
                    for cell in CELLS:
                        c=self.counts(value,cell)
                        if c['active']:ready=False
                        if c['qualified']<target and cell[:2]not in value['isolated_families']and cell not in value['blocked_cells']:
                            if c['used']<cap:ready=False
                            else:value['blocked_cells'][cell]=dict(reason='STAGE_ATTEMPT_CAP_WITH_INSUFFICIENT_SUCCESS',stage=name)
                    if not ready:break
                good=[copy.deepcopy(r)for r in value['attempts']if r['state']=='QUALIFIED']
                if name!='C20':
                    # Nested membership retains every earlier immutable result.
                    previous='C20'if name=='C60'else'C60';parent=value['snapshots'][previous]
                    if not set(parent['episode_ids'])<={r['id']for r in good}:raise ValueError('NESTED_EPISODE_MEMBERSHIP')
                path=self.root/f'{name}_SNAPSHOT.private.json'
                if path.exists():raise FileExistsError('SNAPSHOT_ALREADY_EXISTS_RECONCILE_NOT_OVERWRITE')
                descriptor=dict(schema='MFREE_NESTED_CAPTURE_SNAPSHOT_V1',state='SEALED',intended_stage=name,
                    actual_qualified=len(good),dataset_id=f'MFREE_C{len(good)}_FINAL_PLAN_V1'if len(good)==int(name[1:])else f'MFREE_CN{len(good)}_{name}_STAGE_V1',
                    episodes=good,episode_ids=[r['id']for r in good],plan_sha=PLAN,
                    parent_snapshot=None if name=='C20'else value['snapshots']['C20'if name=='C60'else'C60']['pin'],
                    per_cell={cell:self.counts(value,cell)for cell in CELLS},
                    capture_attempts=len(value['attempts']),physical_requests=sum(r.get('physical_requests',0)for r in value['attempts']),
                    native_integrations=sum(r.get('native_integrations',0)for r in value['attempts']),
                    diagnostic_failures=[dict(id=r['id'],state=r['state'],result=r.get('result'))for r in value['attempts']if r['state']not in ('QUALIFIED',*ACTIVE)],
                    isolated_families=value['isolated_families'],blocked_cells=value['blocked_cells'],sealed_epoch=time.time())
                write(path,descriptor);entry=dict(pin=dict(path=str(path),sha256=sha(path)),episode_ids=descriptor['episode_ids'],actual_qualified=len(good),intended_stage=name)
                value['snapshots'][name]=entry;sealed.append(entry)
        return sealed

    def start_epoch(self,pid,start_ticks,cgroup):
        with self.transaction()as value:
            if len(value['supervisor_epochs'])>=3:raise PermissionError('TWO_SUPERVISOR_RESTARTS_MAXIMUM')
            value['supervisor_epochs'].append(dict(pid=pid,start_ticks=start_ticks,cgroup=cgroup,epoch=time.time()))

    def job_update(self,name,**fields):
        with self.transaction()as value:value['jobs'].setdefault(name,{}).update(fields)


def resumable_checkpoint(complete,last_committed_step,*,resumes):
    if (resumes>=2 or complete.get('schema')!='SLA_COMPLETE_PEFT_STATE_V1'
            or len(complete.get('arrays',[]))!=87 or complete['step']!=last_committed_step
            or complete['cursor']!=32*complete['step']or not 0<=complete['step']<1000):
        raise PermissionError('ONLY_COMPLETE_LAST_COMMITTED87_STATE_WITH_TWO_RESUMES_MAX')
    return 1000-complete['step']
