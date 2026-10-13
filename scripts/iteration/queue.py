"""One coordinator owns reservation state. Workers cannot mutate this ledger.

This is a tested state machine, not a replacement for PI0292's serial physics
guard. Unknown or crashed attempts keep their reservations until reconciliation.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, read, write

KEYS=('physics_steps','rgb_pairs','cpu_s','output_bytes')


class Coordinator:
    def __init__(self, root, plan, *, fixture=False):
        if not fixture:raise ValueError('LIVE_ADAPTER_NOT_ADMITTED_PREPARATION_ONLY')
        if plan['authorizes_execution'] or plan['status']!='PREPARED_NOT_AUTHORIZED':
            raise ValueError('FIXTURE_PLAN_REQUIRED')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=OwnerLock(self.root/'COORDINATOR.lock')
        self.path=self.root/'QUEUE.json'
        if self.path.exists():
            self.lock.close();raise FileExistsError('INSPECT_OR_RECONCILE_NEVER_RESTART')
        self.plan=copy.deepcopy(plan)
        self.value=dict(status='FIXTURE_RUNNING_ZERO_PHYSICS', plan=self.plan,
            reservations={},records=[],actual={key:0 for key in KEYS},
            fixture=True,physics_executed=0,model_calls=0)
        write(self.path,self.value)

    def reserve(self, identifier):
        v=copy.deepcopy(self.value)
        if v['status']!='FIXTURE_RUNNING_ZERO_PHYSICS':raise ValueError('STOPPED_OWNER')
        if identifier not in [r['id'] for r in self.plan['episodes']]:raise ValueError('UNREGISTERED_ID')
        if identifier in v['reservations'] or any(r['id']==identifier for r in v['records']):
            raise ValueError('FIRST_ATTEMPT_ALREADY_RESERVED_OR_FINISHED')
        if len(v['reservations'])>=self.plan['workers']:raise ValueError('ALL_WORKER_SLOTS_RESERVED')
        cost=self.plan['cost']
        if any(v['actual'][k]+sum(r[k] for r in v['reservations'].values())+cost[k]>self.plan['limits'][k] for k in KEYS):
            raise ValueError('GLOBAL_BUDGET_RESERVATION_EXCEEDED')
        output=self.root/'episodes'/identifier
        output.mkdir(parents=True,exist_ok=False)
        v['reservations'][identifier]=copy.deepcopy(cost)
        self.value=v;write(self.path,v)
        row=next(r for r in self.plan['episodes'] if r['id']==identifier)
        return dict(**row,output=str(output),reservation=copy.deepcopy(cost),fixture=True)

    def complete(self, receipt):
        identifier=receipt['id'];v=copy.deepcopy(self.value)
        if identifier not in v['reservations']:raise ValueError('UNKNOWN_OR_DUPLICATE_RECEIPT')
        if receipt['fixture'] is not True or receipt['physics_steps']!=0 or receipt['rgb_pairs']!=0:
            raise ValueError('FIXTURE_MUST_NOT_EXECUTE_PHYSICS')
        expected=self.root/'episodes'/identifier/'WORKER_RECEIPT.json'
        if read(expected)!=receipt:raise ValueError('DURABLE_WORKER_RECEIPT_REQUIRED')
        for key in KEYS:
            x=receipt[key]
            if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('INVALID_ACTUAL_COST')
            if key!='cpu_s' and type(x) is not int:raise ValueError('INTEGER_ACTUAL_COUNT_REQUIRED')
        breach=any(receipt[k]>v['reservations'][identifier][k] for k in KEYS)
        for key in KEYS:v['actual'][key]+=receipt[key]
        del v['reservations'][identifier]
        v['records'].append(copy.deepcopy(receipt))
        if breach or receipt['status']!='FIXTURE_COMPLETE':
            v['status']='FIXTURE_STOP_NO_AUTOMATIC_RETRY'
        self.value=v;write(self.path,v)

    def unknown(self, identifier):
        if identifier not in self.value['reservations']:raise ValueError('UNKNOWN_ID')
        self.value['status']='FIXTURE_STOP_UNRECONCILED_RESERVATION_RETAINED'
        write(self.path,self.value)

    def finish(self):
        if self.value['reservations']:raise ValueError('UNRECONCILED_WORKERS')
        if self.value['status']=='FIXTURE_RUNNING_ZERO_PHYSICS':
            if len(self.value['records'])!=len(self.plan['episodes']):raise ValueError('INCOMPLETE_PLAN')
            self.value['status']='FIXTURE_TERMINAL_ZERO_PHYSICS'
            write(self.path,self.value)
        return copy.deepcopy(self.value)

    def close(self):self.lock.close()
