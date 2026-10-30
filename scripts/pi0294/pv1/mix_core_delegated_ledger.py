"""Explicit finite-custody adapter for the ORIGINAL Core evaluator interfaces.

No SSH or GLOBAL cache. Neural charges use the existing onceIO store; original
RPC telemetry and candidate bookkeeping stay separate, previously prepaid,
confirmed local records. Canonical settlement remains the independent owner's
fresh terminal merge. This module does not replace qualification or safety.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import json
import os
import signal
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.dec1 import source_cpu_once_io as once
from scripts.pi0294.dec1.owner_transport import Journal
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from scripts.pi0294.native_dataset import sha
from . import mix_accounting as a, mix_core_accounting as c, mix_core_identity as identity
from . import mix_core_delegation as contract, mix_core_delegation_store as store
from . import mix_core_delegated_control as control, mix_core_delegated_process as process_reader
from .evaluation_core_scope import validate
from .mix_core_context import LOCAL_BINDING

ACTIVATION='CORE_FINITE_DELEGATION_ACTIVATION.json'
ACTIVATION_SCHEMA='V5_CD_CORE_EXACT_DUAL_ONCE_ACTIVATION_BYTES_V1'
_ACTIVATION_KEYS=('global_bytes','local_bytes','global_intent_bytes','global_receipt_bytes',
                  'local_intent_bytes','local_receipt_bytes')


def need(ok,why):
    if not ok:raise PermissionError('CORE_DELEGATED_LEDGER_'+why)


def selected(manifest,request):
    values=(manifest.get('execution',{}).get('global_accounting_mode'),request.get('global_accounting_mode'))
    if values==(None,None):return False
    need(values==(contract.MODE,contract.MODE) and identity.typed(manifest) and identity.typed(request),
         'EXACT_NEW_COLD_MODE_NO_LEGACY_FALLBACK')
    return True


def activation_bundle(grant_bytes,*,now_epoch,**raws):
    """Serialize already-verified adoption bytes for immutable owner delivery."""
    need(set(raws)==set(_ACTIVATION_KEYS),'SIX_COMPLETE_ORIGINAL_ADOPTION_BYTES')
    contract.activate(grant_bytes,now_epoch=now_epoch,**raws)
    return dict(schema=ACTIVATION_SCHEMA,grant_sha256=a.sha(grant_bytes),activation_epoch=now_epoch,
                raw_base64={k:base64.b64encode(v).decode() for k,v in raws.items()})


def verify_activation(grant_bytes,bundle_bytes,state):
    bundle=json.loads(bundle_bytes)
    need(set(bundle)=={'schema','grant_sha256','activation_epoch','raw_base64'}
         and bundle['schema']==ACTIVATION_SCHEMA and bundle['grant_sha256']==a.sha(grant_bytes)
         and set(bundle['raw_base64'])==set(_ACTIVATION_KEYS),'ACTUAL_ADOPTION_BUNDLE')
    initial=contract.activate(grant_bytes,now_epoch=bundle['activation_epoch'],
        **{k:base64.b64decode(v,validate=True) for k,v in bundle['raw_base64'].items()})
    need(initial['activation']==state['activation'],'EXACT_ORIGINAL_DUAL_ACTIVATION_NOT_A_CALLER_PASS')
    return initial


def _immutable(path):
    raw=once._file(path)
    need(not Path(path).stat().st_mode&0o222,'IMMUTABLE_OWNER_DELIVERED_RECORD:'+Path(path).name)
    return raw


class Ledger:
    role='hopper'
    physical_site='CoreLab'
    successor=True
    accounting=c
    candidate_field=c.CANDIDATES
    delegated=True

    def __init__(self,root,request_path=None,*,resident_observer=False):
        need(type(resident_observer) is bool,'EXPLICIT_READ_ONLY_RESIDENT_MODE')
        self.root=Path(root);self.resident_observer=resident_observer
        self.manifest=(json.loads(_immutable(self.root/'RUN_MANIFEST.json')) if resident_observer else validate(self.root)[0])
        paths=list((self.root/'pbs').glob('alloc-*/REQUEST.json')) if request_path is None else [Path(request_path)]
        need(len(paths)==1,'ONE_EXACT_CURRENT_REQUEST')
        self.request_path=paths[0];raw=_immutable(self.request_path);self.request=json.loads(raw);self.request_sha=a.sha(raw)
        from scripts.pi0294.generalization import minimal_core_accounting as first512
        if first512.typed(self.request):
            self.accounting=first512
            self.candidate_field=first512.CANDIDATES
        need(selected(self.manifest,self.request) and self.request_path.parent.parent.resolve()==(self.root/'pbs').resolve()
             and self.request_path.name=='REQUEST.json'
             and self.request_path.parent.name=='alloc-'+str(self.request['allocation_number']).zfill(2),'EXACT_REGISTERED_REQUEST')
        identity.request_identity(self.manifest,self.request,sha(self.root/'RUN_MANIFEST.json'))
        if resident_observer:
            owner=identity.current_owner(self.root,self.request,request_sha256=self.request_sha)
            need(owner['pid']==os.getpid() and owner['proc_start']==int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                 'ONLY_ACTUAL_REGISTERED_RESIDENT_READ_ONLY_OBSERVER')
        bound=json.loads(_immutable(self.root/'NATIVE_CORE_PROCESS_BOUND.json'))
        self.grant_sha=bound.get('finite_delegation_grant_sha256')
        self.grant_bytes=_immutable(self.root/'evidence'/store.GRANT);self.body=contract._body(self.grant_bytes)
        need(a.sha(self.grant_bytes)==self.grant_sha
             and self.body['request']==self.request and self.body['manifest']==self.manifest
             and bound.get('request_id')==self.request['request_id']
             and bound.get('request_sha256')==self.request_sha
             and bound.get('native_handle')==self.body['issued_row']['native_handle'],'EXACT_BOUND_NATIVE_AND_GRANT')
        confirmed=store.read_confirmed(self.root,expected_grant_sha256=self.grant_sha)
        bundle_raw=_immutable(self.root/'evidence'/ACTIVATION)
        verify_activation(self.grant_bytes,bundle_raw,confirmed['state'])
        self.adopted_global_bytes=base64.b64decode(json.loads(bundle_raw)['raw_base64']['global_bytes'],validate=True)
        self.native_handle=bound['native_handle'];self.actor_pid=os.getpid()
        self.actor_start=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        self.binding=read(self.root/'plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json')
        need(self.manifest['execution']['sealed_inputs'].get('plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json')
             ==sha(self.root/'plans/CORE_GLOBAL_LEDGER_BINDINGS.private.json'),'ORIGINAL_SEALED_PEER_BINDING')
        self.local_binding=read(self.root/LOCAL_BINDING)
        need(self.manifest['execution']['sealed_inputs'].get(LOCAL_BINDING)==sha(self.root/LOCAL_BINDING),
             'ORIGINAL_SEALED_LOCAL_BINDING')
        self.local_path=Path(self.local_binding['canonical_LOCAL_ledger'])
        need(self.local_path.name=='LOCAL_RESOURCE_LEDGER.private.json'
             and self.local_path.resolve().is_relative_to((self.root.parent.parent/'control').resolve()),'CANONICAL_LOCAL_NOT_COPY')
        self.path=self.root/'evidence'/store.STATE
        self.partial_output_guard=None
        self.d25_output_guard=None
        self.fixed512_output_guard=None
        if first512.typed(self.request):
            from scripts.pi0294.generalization.minimal_core_output import Guard
            self.fixed512_output_guard=Guard(self.root,self.manifest,self.request,read(self.local_path))
        if 'D25_local_successor_sha256'in self.request:
            from .mix_core_d25_output_guard import Guard
            self.d25_output_guard=Guard(self.root,self.manifest,self.request,read(self.local_path))
        if self.request.get('original_partial_predecessor_request_id'):
            from .mix_core_partial_resume import OutputGuard
            self.partial_output_guard=OutputGuard(self.root,self.manifest,read(self.local_path))
        self._native_cache=None
        if not resident_observer:
            directory=self.root/'reservations/Core-delegated-rpc'/str(self.actor_pid)/str(self.actor_start)
            directory.mkdir(parents=True,mode=0o700,exist_ok=True)
            self.journal=Journal(directory,dict(run_uuid=self.root.name,job_id=self.native_handle,
                source_sha=self.request['source_sha'],request_sha256=self.request_sha))
        control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)

    def _actor(self,*,neural=False):
        need(os.getpid()==self.actor_pid and int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])==self.actor_start,
             'NO_FORK_OR_REBOUND_WRITER')
        if neural:need(not self.resident_observer and (self.actor_pid,self.actor_start)==(self.body['issued_row']['pid'],self.body['issued_row']['proc_start']),
                       'ONLY_BOUND_NEURAL_MAY_CHARGE_MODEL_OR_QUERY')

    def _confirmed(self):
        return store.read_confirmed(self.root,expected_grant_sha256=self.grant_sha)['state']

    def state(self):
        """Explicit compatibility view, never a canonical ledger snapshot."""
        state=self._confirmed();telemetry=control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)
        return dict(schema='V5_CD_CORE_DELEGATED_RUNTIME_VIEW_V1',mode=contract.MODE,
            grant_sha256=self.grant_sha,request_id=self.request['request_id'],delegated_effects=state,
            canonical_GLOBAL_is_fresh=False,canonical_GLOBAL_is_writable=False,
            canonical_GLOBAL_observed_epoch=self.body['issued_epoch'],CPU_s=telemetry['CPU_s'],
            RPC_calls=telemetry['RPC_calls'],**{self.candidate_field:deepcopy(telemetry['candidates'])})

    def _local_fences(self,*,closeout=False):
        self._actor();now=time.time()
        end=self.body['accounting_not_after_epoch'] if closeout else self.body['execution_not_after_epoch']
        need(type(closeout) is bool and self.body['issued_epoch']<=now<end,'ORIGINAL_FINITE_WINDOW_AND_CLOSEOUT')
        local=json.loads(once._file(self.local_path))
        need(local.get(contract.FIELD,{}).get(self.request['request_id'])==contract._marker(self.grant_bytes)
             and local.get('unknown_effects')==local.get('pending_updates')==local.get('pending_queries')==[],
             'ACTUAL_LOCAL_CUSTODY_HELD_AND_UNKNOWN_FENCES')
        from . import successor_cd as successor
        from scripts.pi0294.generalization import minimal_core_accounting as first512
        if first512.typed(self.request):
            marker=local[first512.FIELD][self.request['request_id']]
            first512.validate_row(marker)
        else:marker=successor.validated(local)
        need(marker['T_CD']==self.request['T_CD'] and marker['deadline_epoch']==self.request['deadline_epoch'],
             'ACTUAL_LOCAL_UNCHANGED_SCOPE_CLOCK')
        physical=Path(self.binding['physical_peer_root'])
        need(physical.name==self.root.name and physical.parent==self.root.parent.parent/'children',
             'ORIGINAL_EXACT_PHYSICAL_PEER')
        for peer in (self.root,physical):
            binding=read(peer/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
            need(lease['status']=='ACTIVE' and lease['run_uuid']==self.root.name
                 and lease['authority_release']==successor.TASK_SOURCE and lease['deadline_epoch']==self.request['deadline_epoch'],
                 'ACTUAL_OWN_ACTIVE_NEURAL_AND_PHYSICAL_LEASES')
        if not closeout:
            need(not any(p.exists() for p in (self.root/'STOP.json',self.root.parent.parent/'STOP.json',physical/'STOP.json',
                 physical/'SOURCE_PHYSICS_STOP.json',physical/'SOURCE_SAFETY_STOP_FROM_CORE.json',
                 self.root/'SOURCE_PHYSICS_STOP.json',self.root/'SOURCE_SAFETY_STOP_FROM_CORE.json')),'ALL_ORIGINAL_STOP_FENCES')
            if not self.resident_observer:read_live_owner(self.root,self.root.name)
            identity.current_owner(self.root,self.request,request_sha256=self.request_sha)
        if self.partial_output_guard is not None:self.partial_output_guard.check()
        if self.d25_output_guard is not None:self.d25_output_guard.check()
        if self.fixed512_output_guard is not None:self.fixed512_output_guard.check()
        return now

    def live_proof(self,*,force_native=True):
        """Original full reservation + fresh actual LOCAL/owner/process/device.

        The source GLOBAL timestamp stays fixed. No canonical resource refresh
        is implied by this proof; only this exact previously held row is usable.
        """
        need(type(force_native) is bool,'EXPLICIT_NATIVE_READ_POLICY')
        now=self._local_fences();state=self._confirmed();control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)
        cached=self._native_cache
        if force_native or cached is None or not 0<=now-cached['observed_epoch']<=5:
            cached=process_reader.observe(self.root,self.body,state['row']);self._native_cache=cached
        contract._live_process(self.body,state['row'],cached)
        now=time.time();need(0<=now-cached['observed_epoch']<=120,'FRESH_ACTUAL_PROCESS_NOT_REFRESHED_GLOBAL')
        row=state['row'];q,loads,cpu=c.progress(row)
        allocation=deepcopy(self.body['admission']['allocation'])
        allocation.update(native_process=deepcopy(cached),native_process_sha256=c.digest(cached))
        return dict(schema='V5_CD_CORE_FINITE_RESERVED_LOCAL_RESOURCE_PROOF_V1',
            status='PASS_EXACT_FINITE_CORE_DELEGATED_RESERVATION',mode=contract.MODE,
            request_id=self.request['request_id'],request_sha256=self.request_sha,grant_sha256=self.grant_sha,
            canonical_GLOBAL_is_fresh=False,canonical_GLOBAL_is_writable=False,
            canonical_GLOBAL_observed_epoch=self.body['issued_epoch'],
            canonical_GLOBAL_sha256=self.body['source_inputs']['global_bytes'],
            local_observed_epoch=now,native_process_sha256=c.digest(cached),allocation=allocation,
            original_admission=deepcopy(self.body['admission']),
            remaining=dict(generation_queries=row['reserved_neural_queries'],model_loads=2-loads,
                           CPU_s=row['CPU_upper_s']-cpu,GPU_s=self.body['limits']['GPU_s']),
            all_remaining_costs_fit=True,shortfalls={},all_remaining_costs_basis='UNCHANGED_EXACT_RESERVED_GRANT',
            original_matrix_and_final_protected=True,model_permission=False,physics_authorized=False)

    def guard(self,*,closeout=False):
        if closeout:self._local_fences(closeout=True);self.state()
        else:self.live_proof(force_native=False)
        return True

    def _mutate(self,operation,invoke):
        self._actor();need(not self.resident_observer,'READ_ONLY_RESIDENT')
        def bounded(intent,timeout):
            # A local metadata stall is still an UNKNOWN write. Preserve any
            # stricter enclosing deadline and never extend the original 40s.
            old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
            began=time.monotonic()
            def expired(*args):raise TimeoutError('FINITE_LOCAL_ONCE_IO_EXCEEDED_ORIGINAL_RPC_BOUND')
            signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,min(timeout,old_timer[0]) if old_timer[0]>0 else timeout)
            try:return invoke()
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
                if old_timer[0]>0:
                    remaining=old_timer[0]-(time.monotonic()-began)
                    if remaining>0:signal.setitimer(signal.ITIMER_REAL,remaining,old_timer[1])
        return self.journal.call('LOCAL_'+operation,'MUTATION',bounded)

    def charge(self,kind,count):
        need(kind in ('generation_queries','model_loads'),'ORIGINAL_MODEL_OR_QUERY_KIND')
        self._actor(neural=True);proof=self.live_proof(force_native=True);state=self._confirmed()
        need(control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)['final_neural_CPU_s'] is None,'NO_EFFECT_AFTER_FINAL_CPU')
        token=f"{self.actor_pid}:{self.actor_start}:{state['sequence']}:DEBIT"
        return self._mutate('DEBIT',lambda:store.consume_once(self.root,expected_grant_sha256=self.grant_sha,effect_id=token,kind=kind,count=count,
            native_facts_bytes=a.raw(proof['allocation']['native_process']),now_epoch=time.time()))

    def cpu(self,cost):
        """Preserve final measured CPU even after STOP; independent terminal pays.

        No fresh living GPU is required during finally. The record remains
        bounded and is not a canonical debit or proof the process has exited.
        """
        self._actor(neural=True);self._local_fences(closeout=True);self._confirmed()
        before=control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)
        return self._mutate('FINAL_CPU',lambda:control.commit(self.root,self.grant_bytes,kind='FINAL_NEURAL_CPU',values={'CPU_s':cost},
            actor_pid=self.actor_pid,actor_start=self.actor_start,now_epoch=time.time(),expected_before_sha256=a.sha(a.raw(before)),adopted_global_bytes=self.adopted_global_bytes))

    @contextmanager
    def transaction(self):
        need(not self.resident_observer,'RESIDENT_OBSERVER_CANNOT_DEBIT_OR_RECORD_METRICS')
        before=self.state();after=deepcopy(before);yield after
        changed={k for k in set(before)|set(after) if before.get(k)!=after.get(k)}
        if not changed:return
        self._local_fences(closeout=True);self._confirmed()
        current=control.read_confirmed(self.root,self.grant_bytes,adopted_global_bytes=self.adopted_global_bytes)
        need(before['CPU_s']==current['CPU_s'] and before['RPC_calls']==current['RPC_calls']
             and before[self.candidate_field]==current['candidates'],'NO_CONCURRENT_STALE_TRANSACTION_RETRY')
        if changed<={'CPU_s','RPC_calls'}:
            kind='METRICS';values=dict(CPU_s=after['CPU_s']-before['CPU_s'],RPC_calls=after['RPC_calls']-before['RPC_calls'])
        elif changed=={self.candidate_field}:
            self._actor(neural=True);self.guard();kind='CANDIDATE';values=after[self.candidate_field]
        else:need(False,'NO_ARBITRARY_GLOBAL_STYLE_TRANSACTION')
        self._mutate(kind,lambda:control.commit(self.root,self.grant_bytes,kind=kind,values=values,actor_pid=self.actor_pid,
            actor_start=self.actor_start,now_epoch=time.time(),expected_before_sha256=a.sha(a.raw(current)),adopted_global_bytes=self.adopted_global_bytes))
