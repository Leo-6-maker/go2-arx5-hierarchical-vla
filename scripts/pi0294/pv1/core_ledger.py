"""Finite Core policy-stage charges; uncertainty keeps the original reservation."""
from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
import time

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from .decision import AUTHORITY
from .scope import EVAL_ROLE

KINDS=('live_and_recovery_integrations','shadow_integrations')


class Ledger:
    role='core'
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'RESOURCE_LEDGER.json'
        self.m=read(self.root/'RUN_MANIFEST.json');self.a=read(self.root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')
        if self.m['scope']!=EVAL_ROLE or self.m['authority']['protocol_ref']!=AUTHORITY or self.a['authority_release']!=AUTHORITY:
            raise PermissionError('PV1_TYPED_CORE_POLICY_COUNTERS_ONLY')
        self.limits=self.m['limits'];self._guard_at=0.;self._guard_value=None;self._guard_closeout=None
        self.check(read(self.path))

    def check(self,v,*,limits=False,allow_unknown=True):
        if (v['schema']!='PI0294_PV1_CORE_POLICY_LEDGER_V1' or v['run_uuid']!=self.root.name
                or v['authority_release']!=AUTHORITY or type(v['unknown_effects']) is not list
                or v['unknown_effects'] and not allow_unknown
                or any(type(v[k]) is not int or v[k]<0 for k in (*KINDS,'engine_starts','main_starts'))
                or type(v['Core_CPU_s']) not in (int,float) or not math.isfinite(v['Core_CPU_s']) or v['Core_CPU_s']<0):
            raise PermissionError('PV1_CORE_COUNTER_INTEGRITY_OR_UNRESOLVED_NATIVE_EFFECT')
        if type(v['reservations']) is not dict or any(p.get('kind') not in KINDS
                or type(p.get('maximum')) is not int or p['maximum']<1 for p in v['reservations'].values()):
            raise PermissionError('PV1_CORE_ONLY_KNOWN_POSITIVE_PENDING_RESERVATIONS')
        if limits:
            pending=sum(p['maximum'] for p in v['reservations'].values())
            if (sum(v[k] for k in KINDS)+pending>self.limits['physics_ticks']
                    or v['engine_starts']>self.limits['engine_starts'] or v['main_starts']>self.limits['policy_episodes']
                    or v['Core_CPU_s']>=self.limits['CPU_s']):
                raise PermissionError('PV1_RESERVED_CORE_POLICY_PHASE_CAP')
        return v

    @contextmanager
    def transaction(self):
        with self.path.with_suffix('.owner.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            value=self.check(read(self.path))
            yield value
            self.check(value)
            write(self.path,value)
        self._guard_at=0.

    def guard(self,*,closeout=False):
        cutoff=self.a['hard_closeout_epoch'] if closeout else self.a['stop_new_physics_epoch']
        if not self.a['T0_epoch']<=time.time()<min(cutoff,self.limits['operation_deadline_epoch']):
            raise TimeoutError('PV1_ORIGINAL_PHYSICS_OR_CLOSEOUT_CUTOFF')
        if not closeout and any(p.exists() for p in (self.root/'STOP.json',self.root/'SOURCE_PHYSICS_STOP.json',self.root.parent.parent/'STOP.json')):
            raise PermissionError('PV1_CORE_SAFETY_OR_PARENT_STOP')
        now=time.monotonic()
        if self._guard_value is None or self._guard_closeout!=closeout or now-self._guard_at>=.25:
            binding=read(self.root/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
            if lease['status']!='ACTIVE' or lease['run_uuid']!=self.root.name or lease['authority_release']!=AUTHORITY:
                raise PermissionError('PV1_LIVE_CORE_EVALUATION_LEASE')
            self._guard_value=self.check(read(self.path),limits=True,allow_unknown=closeout)
            self._guard_at=now;self._guard_closeout=closeout
        return self._guard_value

    def charge(self,key,count):
        self.guard()
        if key not in ('engine_starts','main_starts') or type(count) is not int or count<0:
            raise ValueError('PV1_CORE_ONLY_NATIVE_CONSTRUCTORS_AND_POLICY_STARTS')
        with self.transaction() as v:
            if v['unknown_effects']:raise PermissionError('PV1_UNKNOWN_NATIVE_EFFECT_NO_NEW_CORE_START')
            v[key]+=count;self.check(v,limits=True,allow_unknown=False)

    @contextmanager
    def steps(self,kind,maximum,identifier):
        self.guard()
        if kind not in KINDS or type(maximum) is not int or maximum<1 or not identifier:
            raise ValueError('PV1_EXACT_LIVE_OR_SHADOW_RESERVATION')
        with self.transaction() as v:
            if identifier in v['reservations'] or v['unknown_effects']:
                raise PermissionError('PV1_PRESERVE_PRIOR_OR_UNKNOWN_PHYSICAL_INTENT')
            v['reservations'][identifier]=dict(kind=kind,maximum=maximum,started_epoch=time.time())
            self.check(v,limits=True,allow_unknown=False)
        meter=dict(actual=0)
        try:
            yield meter
            actual=meter['actual']
            if type(actual) is not int or not 0<=actual<=maximum:
                raise ValueError('PV1_UNKNOWN_ACTUAL_NATIVE_INTEGRATION_COUNT')
        except BaseException as exc:
            with self.transaction() as v:
                v['unknown_effects'].append(dict(identifier=identifier,error=type(exc).__name__,reservation_retained=True))
            raise
        else:
            with self.transaction() as v:
                pending=v['reservations'].pop(identifier)
                if pending['kind']!=kind or pending['maximum']!=maximum:
                    raise PermissionError('PV1_PHYSICAL_RESERVATION_IDENTITY_CHANGED')
                v[kind]+=actual

    def cpu(self,cost):
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost<0:
            raise ValueError('PV1_FINITE_CORE_CPU_COST')
        # Actual closing costs are recorded even if they expose a cap overrun.
        with self.transaction() as v:v['Core_CPU_s']+=cost

    def stop_physics(self,reason,detail):
        p=self.root/'SOURCE_PHYSICS_STOP.json'
        if not p.exists():write(p,dict(status='STOP_NEW_PV1_SOURCE_PHYSICS',reason=reason,detail=detail,observed_epoch=time.time()))
