"""Explicit cold-owner integration; never installs itself into an old owner.

The original launcher must supply its validated private host/source bindings,
new run/request/lease, recovery authority and allocation lock before binding.
No registration, qsub, project login or action occurs on import/construction.
"""
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from scripts.pi0292.durable import OwnerLock
from .owner_transport import Journal, ReadOnlyFailure
from .owner_rpc import invoke, program

READ_LABELS = frozenset({
    'source-A40-current-unfiltered-native-hopper',
    'source-A40-current-unfiltered-native-vanda',
    'source-A40-fresh-project-and-native-balance',
    'source-A40-native-before-release-project-quota',
})
HEARTBEAT_LABELS = frozenset({
    'source-A40-same-owner-existing-job-heartbeat',
    'source-A40-fresh-heartbeat-before-once-MODEL_START',
    'source-A40-fresh-heartbeat-before-once-FIRST_UPDATE',
})


class ColdRPC:
    """Transport injected only in the new owner's private process."""
    def __init__(self, journal, *, argv_by_site, remote_root, request_directory,
                 code_root, transport=invoke):
        if set(argv_by_site) != {'hopper','vanda'}:
            raise ValueError('BOTH_ORIGINAL_CANONICAL_HOST_BINDINGS')
        self.journal = journal
        self.argv = {k:tuple(v) for k,v in argv_by_site.items()}
        self.root, self.directory, self.code_root = map(str,(remote_root,request_directory,code_root))
        self.transport = transport
        self.observation = None

    def bind_observation(self, state):
        """Called only after the mature fresh() has validated both platforms."""
        view=state['view'];row=state['rows']['hopper']
        ledger=row.get('ledger_summary')
        if ledger is None:
            ledger=json.loads(base64.b64decode(row['ledger_base64'],validate=True))
        if (view.get('status') != 'PASS_TWO_PLATFORM_CAMPAIGN_VIEW'
                or ledger.get('unknown_effects') != [] or ledger.get('pending_updates') != []):
            raise PermissionError('ORIGINAL_VALIDATED_TWO_PLATFORM_VIEW_AND_EFFECTS')
        self.observation=json.loads(json.dumps(dict(view=view,ledger={
            k:ledger[k] for k in ('unknown_effects','pending_updates')})))
        return state

    def _heartbeat(self, values, intent, timeout, function):
        if (self.observation is None or values.get('VIEW') != self.observation['view']
                or values.get('TARGET') != self.root or values.get('DIRECTORY') != self.directory
                or values.get('JOB') != self.journal.identity['job_id']):
            raise PermissionError('CURRENT_VALIDATED_OBSERVATION_AND_BOUND_REQUEST')
        args=dict(root=self.root,directory=self.directory,job=values['JOB'],
                  view=self.observation['view'],ledger=self.observation['ledger'],intent=intent)
        code=program('scripts.pi0294.dec1.owner_heartbeat_publish',function,args,self.code_root)
        return self.transport(self.argv['vanda'],code,timeout_s=timeout)

    def __call__(self, site, code, values=None, label='UNCLASSIFIED'):
        if site not in self.argv:
            raise ValueError('ORIGINAL_BOUND_SITE_ONLY')
        values=dict(values or {})
        if label in HEARTBEAT_LABELS:
            if site != 'vanda':
                raise PermissionError('HEARTBEAT_ON_BOUND_VANDA_RUN_ONLY')
            return self.journal.call('HEARTBEAT','HEARTBEAT',
                lambda intent,limit:self._heartbeat(values,intent,limit,'publish'),
                readback=lambda intent,limit:self._heartbeat(values,intent,limit,'readback'))
        effect='READ_ONLY' if label in READ_LABELS else 'MUTATION'
        phase=effect+'_'+hashlib.sha256(label.encode()).hexdigest()[:20].upper()
        # A new or misspelled label defaults to effect-bearing, never read retry.
        payload='\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+code
        return self.journal.call(phase,effect,lambda intent,limit:
            self.transport(self.argv[site],payload,timeout_s=limit))


def bind_cold_owner(owner, rpc, *, validate_new_registration):
    """The guard checks new authority/registration and old owner closure.

    This is not a launcher. The guard must reject using a failed/settled request
    or any existing active owner. Returning False is not an admissible proof.
    """
    proof=validate_new_registration()
    if (not isinstance(proof,dict) or proof.get('status') != 'PASS_NEW_REGISTERED_COLD_OWNER'
            or proof.get('identity') != rpc.journal.identity
            or proof.get('prior_owner_closed') is not True
            or proof.get('new_recovery_authority_adopted') is not True):
        raise PermissionError('NEW_REGISTERED_AUTHORIZED_OWNER_REQUIRED')
    if getattr(owner,'cold_rpc_bound',False):
        raise PermissionError('NO_SECOND_COLD_BINDING')
    previous_fresh=owner.fresh
    def fresh(*args,**kwargs):
        return rpc.bind_observation(previous_fresh(*args,**kwargs))
    owner.previous.rpc=rpc
    owner.fresh=fresh
    owner.cold_rpc_bound=True
    return owner


def watch(owner, secret, *, clock=time.time, monotonic=time.monotonic, sleep=time.sleep,scope_binding=None):
    """Original single-owner lifecycle, retrying only explicit read failures.

    Allocation/first-update releases and project authentication keep their
    mature implementation. Their timeouts propagate, preserving once-intents;
    they are never reclassified as read-only. Terminal jobs are returned for
    the original independent audit and once-only settlement/closeout.
    """
    o=owner
    deadline=o.native.T0+156*3600
    if scope_binding is not None:
        from scripts.pi0294.pv1 import successor_cd as cd
        cd.validated({cd.FIELD:scope_binding})
        if getattr(o,'successor_scope',None)!=scope_binding:raise PermissionError('EXPLICIT_REGISTERED_SUCCESSOR_WATCH_BINDING')
        deadline=scope_binding['deadline_epoch']
    if not getattr(o,'cold_rpc_bound',False):
        raise PermissionError('EXPLICIT_NEW_COLD_BINDING_REQUIRED')
    lock=OwnerLock(o.BASE/'SOURCE_ALLOCATION_OWNER.lock')
    try:
        req=o.read(o.BASE/'SOURCE_REQUEST.private.json')['request']
        job=o.read(o.BASE/'SOURCE_JOB_BOUND.private.json')['job_id']
        reg=o.read(o.BASE/'TRAIN_REGISTERED.private.json')
        directory=o.read(o.BASE/'SOURCE_REQUEST.private.json')['directory']
        o.record('SOURCE_OWNER_PROCESS',dict(pid=os.getpid(),started_epoch=clock(),job_id=job,
            existing_owner=True,new_scientific_owner=False,credentials_saved=False))
        last=None;lastauth=0.;errors=0
        while clock()<deadline:
            try:
                state=o.fresh(req['submit_count'],job,light=True)
                jobrow=state['rows']['vanda']['jobs'].get(job,state['rows']['vanda'].get('terminal_job'))
                if (not jobrow or jobrow['Job_Name']!='pi0294_dec1_a40_train'
                        or jobrow['project']!='CFP04-CF-007'):
                    raise PermissionError('EXACT_ORIGINAL_NATIVE_TRAINING_JOB')
                status=jobrow['job_state'];receipts=state['rows']['vanda'].get('receipts',{})
                current=(status,tuple(sorted(receipts)),receipts.get('A40_NATIVE_EFFECT_LEDGER.json',{}).get('optimizer_updates',0))
                if current!=last:
                    latest=dict(job_id=job,observed_epoch=clock(),native_state=status,
                                native_job=jobrow,receipts=receipts,campaign_view=state['view'])
                    o.record('NATIVE-'+str(time.time_ns()),latest)
                    o.update(o.BASE/'LATEST_NATIVE.private.json',latest);last=current
                if status=='F':
                    value=dict(status='NATIVE_TERMINAL_PRESERVED_NEEDS_ACTUAL_EFFECT_SETTLEMENT',
                               job_id=job,native=jobrow,receipts=receipts,observed_epoch=clock(),do_not_resubmit=True)
                    o.record('SOURCE_NATIVE_TERMINAL_UNSETTLED',value)
                    return value
                if status not in ('Q','R'):
                    raise PermissionError('ORIGINAL_Q_OR_R_ONLY')
                o.previous.rpc('vanda',o.HEARTBEAT,dict(TARGET=reg['root'],DIRECTORY=directory,JOB=job,VIEW=state['view']),
                               'source-A40-same-owner-existing-job-heartbeat')
                if status=='R' and 'PBS_READY.json' in receipts and 'START_NEURAL.json' not in receipts:
                    o.release(state,secret=secret);lastauth=monotonic()
                elif status=='R' and 'A40_PRETRAIN_READY.json' in receipts and 'A40_PRETRAIN_RELEASE.json' not in receipts:
                    o.release(state,secret=secret,first_update=True);lastauth=monotonic()
                elif monotonic()-lastauth>300:
                    o.auth(secret);lastauth=monotonic()
                errors=0;sleep(30 if status=='R' else 60)
            except ReadOnlyFailure as error:
                errors+=1;o.record('READ_ONLY_OBSERVATION_INTERRUPTION-'+str(time.time_ns()),
                    dict(type='ReadOnlyFailure',phase_in_durable_RPC_journal=True,
                         unknown_mutations_not_retried=True,job_id=job,
                         diagnostic=getattr(error,'diagnostic',None)))
                if errors>=3:
                    raise
                sleep(30)
        return dict(status='ORIGINAL_TRAIN_STOP_WINDOW_REACHED_NO_NEW_EFFECTS',job_id=job)
    finally:
        lock.close()
