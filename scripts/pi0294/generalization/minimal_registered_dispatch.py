"""Cold metadata-owner handoff after a proved REGISTER CAS, before first qsub.

This is not a retry of registration, qsub, a model or a failed allocation. The
scientific request/capsule remains immutable. A separately registered, sealed
metadata owner may perform its *first* qsub only after exact reconciliation.
Original UNKNOWN records remain unchanged. All release/watch/audit/settlement
methods come from the original scientific capsule.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.dec1 import cold_owner, vanda_campaign as queue
from scripts.pi0294.pv1 import mix_owner as mature, successor_cd
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.generalization import minimal_execution as e
from scripts.pi0294.generalization import minimal_execution_adapter as adapters
from scripts.pi0294.generalization.minimal_native_context import root_context, captured_LOCAL
from scripts.pi0294.generalization.minimal_native_owner import Owner as OriginalOwner, HEARTBEAT
from scripts.pi0294.generalization.minimal_native_launch import effective_queue, collect_effective_queue

SCHEMA = 'ARX5_REGISTERED_PRE_QSUB_METADATA_OWNER_HANDOFF_V1'
OLD_CPU_UPPER = 800
NEW_CPU_UPPER = 2200
ABSENT = ('SOURCE_GLOBAL_RESERVED.private.json', 'SOURCE_QSUB_INTENT.private.json',
          'SOURCE_QSUB_REPLY.private.json', 'SOURCE_JOB_BOUND.private.json')


def raw(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def digest(body):
    return hashlib.sha256(body).hexdigest()


def reconcile(global_bytes, request_bytes, evidence, *, now_epoch):
    """Exact evidence join only; not native dispatch permission or a write."""
    g = json.loads(global_bytes)
    r = json.loads(request_bytes)
    adapter, row, _ = adapters.registered(g, r['request_id'])
    e.need(adapter is adapters.retry and r.get('continuation_generation') == 3
           and row['status'] == e.STATES[0] and row['job_id'] is None
           and row['request_sha256'] == digest(request_bytes)
           and all(row.get(k) == v for k, v in r.items())
           and g['unknown_effects'] == g['pending_updates'] == [],
           'DISPATCH_EXACT_REGISTERED_UNBOUND_REQUEST_NOT_FAILED_OR_SECOND_REQUEST')
    e.need(0 <= now_epoch-evidence['observed_epoch'] <= 120
           and evidence['GLOBAL_sha256'] == digest(global_bytes)
           and evidence['new_remote_writes'] == 0,
           'DISPATCH_FRESH_ACTUAL_FULL_GLOBAL_RECONCILIATION')
    stopped = evidence['old_exit']
    intent = evidence['old_journal']['intent']
    e.need(stopped['returncode'] == 1 and stopped['foreground_wait_completed'] is True
           and stopped['source_sha'] == r['source_sha']
           and evidence['old_journal']['status'] == 'UNKNOWN'
           and intent['phase'] == 'REGISTER_REQUEST' and intent['effect'] == 'MUTATION'
           and intent['identity']['run_uuid'] == r['run_uuid']
           and intent['identity']['source_sha'] == r['source_sha']
           and intent['timeout_s'] == 40 and intent['automatic_write_retry'] is False
           and evidence['old_stop']['error_type'] == 'UnknownMutation'
           and evidence['old_stop']['message'] == 'UNKNOWN_REGISTER_REQUEST_NO_AUTOMATIC_RETRY',
           'DISPATCH_ORIGINAL_KNOWN_TERMINAL_REGISTER_UNKNOWN_ONLY')
    e.need(set(evidence['old_processes']) == {'owner', 'wrapper'}
           and all(v['same_live_procstart'] is False for v in evidence['old_processes'].values()),
           'DISPATCH_OLD_NATIVE_OWNER_AND_WRAPPER_NOT_ALIVE')
    # The old owner inherited four CPUs; its waiting wrapper adds one. This
    # conservative elapsed bound includes all concurrent inherited children.
    elapsed = stopped['finished_epoch']-evidence['old_launch_epoch']
    e.need(0 <= elapsed and math.ceil(5*elapsed)+10 <= OLD_CPU_UPPER
           and OLD_CPU_UPPER+NEW_CPU_UPPER == 3000,
           'DISPATCH_SAME_PREPAID_OWNER_CPU_BANK_NO_SECOND_PAYMENT')
    native = evidence['native_once']
    ib = base64.b64decode(native['intent_base64'], validate=True)
    rb = base64.b64decode(native['receipt_base64'], validate=True)
    iv, rv = json.loads(ib), json.loads(rb)
    e.need(rv['intent_sha256'] == digest(ib)
           and rv['after_sha256'] == iv['proposed_sha256'] == digest(global_bytes)
           and rv['request_id'] == iv['request_id']
           and rv['target'] == iv['target'] == evidence['GLOBAL_path']
           and native['temporary_exists'] is False,
           'DISPATCH_ACTUAL_NATIVE_ONCE_RECEIPT_FULL_AFTER_BYTES_NO_REGISTER_REPLAY')
    e.need(evidence['old_absent'] == {n: True for n in ABSENT}
           and evidence['allocation_files'] == ['EFFECTIVE_QUEUE_LIMITS.json', 'REQUEST.json']
           and evidence['initial_release_exists'] is False
           and evidence['effect_ledger_exists'] is False,
           'DISPATCH_NO_QSUB_SCRIPT_INTENT_REPLY_BIND_RELEASE_OR_EFFECT')
    e.need(now_epoch+r['walltime_s']+r['accounting_pad_s'] < r['stop_new_training_epoch']
           and r['T_CD'] <= now_epoch < r['deadline_epoch'],
           'DISPATCH_UNRESET_FULL_ORIGINAL_TRAINING_WINDOW')
    return dict(schema=SCHEMA, request_id=r['request_id'], request_sha256=digest(request_bytes),
                original_registration_operation_id=intent['operation_id'],
                GLOBAL_sha256=digest(global_bytes), old_UNKNOWN_preserved=True,
                registration_repeated=False, submission_repeated=False,
                old_owner_CPU_upper_s=OLD_CPU_UPPER, new_owner_CPU_upper_s=NEW_CPU_UPPER,
                native_permission=False)


def capture(root, old_launch_epoch, process_identities, *, now=time.time):
    """Small exact files plus live kernel reads, with no model/tensor stream."""
    root = Path(root)
    base = root/'control/resident-owner'
    saved = read(base/'SOURCE_REQUEST.private.json')
    directory = Path(saved['directory'])
    gp = root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
    gb = gp.read_bytes()
    state = read(base/'setup-rpc/RPC_EFFECT_STATE.json')
    matches = []
    for p in gp.parent.glob('V5_CD_*_RECEIPT.json'):
        if p.stat().st_mtime < state['intent']['started_epoch']:
            continue
        rb = p.read_bytes()
        if json.loads(rb).get('after_sha256') == digest(gb):
            ip = p.with_name(p.name.replace('_RECEIPT.json', '_INTENT.json'))
            ib = ip.read_bytes()
            matches.append(dict(intent_base64=base64.b64encode(ib).decode(),
                                receipt_base64=base64.b64encode(rb).decode(),
                                temporary_exists=Path(json.loads(ib)['temporary']).exists()))
    e.need(len(matches) == 1, 'DISPATCH_ONE_EXACT_ORIGINAL_REGISTER_RECEIPT')
    processes = {}
    for name, value in process_identities.items():
        p = Path('/proc', str(value['pid']), 'stat')
        fields = p.read_text().rsplit(')', 1)[1].split() if p.exists() else None
        processes[name] = dict(value, same_live_procstart=bool(fields and fields[0] not in ('Z', 'X')
                               and int(fields[19]) == value['procstart']))
    evidence = dict(observed_epoch=now(), GLOBAL_path=str(gp), GLOBAL_sha256=digest(gb),
                    old_exit=read(root/'control/RESIDENT_OWNER_OS_EXIT.json'), old_journal=state,
                    old_stop=read(base/'PRESERVED_STOP.private.json'), old_processes=processes,
                    old_launch_epoch=old_launch_epoch, native_once=matches[0],
                    old_absent={n: not (base/n).exists() for n in ABSENT},
                    allocation_files=sorted(str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()),
                    initial_release_exists=(directory/'START_NEURAL.json').exists(),
                    effect_ledger_exists=(root/'A40_NATIVE_EFFECT_LEDGER.json').exists(), new_remote_writes=0)
    return gb, (directory/'REQUEST.json').read_bytes(), evidence


def predispatch_queue(global_bytes, request_bytes, evidence, proof, *, now_epoch):
    """Qualify only the proved not-yet-dispatched reservation, keeping raw queue.

    The ordinary classifier correctly fences an unbound ledger row. This
    distinct pre-qsub boundary can explain exactly that one fence after the
    native CAS/no-qsub reconciliation. It never changes generic classification,
    filters another row, or fabricates a current PBS job.
    """
    reconciled = reconcile(global_bytes, request_bytes, evidence, now_epoch=now_epoch)
    rows = queue.queue_rows(proof, now_epoch=now_epoch)
    owned, outside, unknown = queue.classify(json.loads(global_bytes), rows)
    e.need(owned == [] and unknown == [dict(job_id=None,
           reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')],
           'DISPATCH_ONLY_ONE_PROVED_UNSUBMITTED_RESERVATION_AND_NO_UNKNOWN_NATIVE_JOB')
    return dict(reconciliation=reconciled, full_queue=proof, outside=outside,
                owned=owned, original_classification=unknown,
                explained_only_by_exact_no_qsub_reconciliation=True)


def exact_current_cas(path, *, before_sha256, after_sha256):
    """Read-only reconciliation of a returned/late original onceIO write."""
    path = Path(path)
    body = path.read_bytes()
    e.need(digest(body) == after_sha256, 'DISPATCH_EXACT_BIND_CURRENT_AFTER_NOT_OLD_OR_OTHER_BYTES')
    matches = []
    for p in path.parent.glob('V5_CD_*_RECEIPT.json'):
        receipt_bytes = p.read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt.get('before_sha256') != before_sha256 or receipt.get('after_sha256') != after_sha256:
            continue
        ip = p.with_name(p.name.replace('_RECEIPT.json', '_INTENT.json'))
        intent_bytes = ip.read_bytes()
        intent = json.loads(intent_bytes)
        e.need(receipt['intent_sha256'] == digest(intent_bytes)
               and receipt['request_id'] == intent['request_id']
               and receipt['target'] == intent['target'] == str(path)
               and intent['before_sha256'] == before_sha256 and intent['proposed_sha256'] == after_sha256
               and not Path(intent['temporary']).exists(), 'DISPATCH_EXACT_BIND_ONCE_INTENT_RECEIPT_AND_NO_TEMP')
        matches.append(dict(intent_sha256=digest(intent_bytes), receipt_sha256=digest(receipt_bytes),
                            original_once_request_id=receipt['request_id']))
    e.need(len(matches) == 1, 'DISPATCH_ONE_EXACT_BIND_CAS_NOT_A_SECOND_WRITE')
    return dict(before_sha256=before_sha256, after_sha256=after_sha256,
                **matches[0], read_only=True, original_UNKNOWN_preserved=True, write_repeated=False)


class Owner(OriginalOwner):
    """New sealed metadata process; never calls the old submit/register path."""
    def __init__(self, handoff_root):
        self.BASE = Path(handoff_root)
        h = self.handoff = read(self.BASE/'RUN_MANIFEST.json')
        e.need(h['schema'] == SCHEMA and sha(Path(__file__)) == h['owner_code_sha256']
               and Path(__file__).resolve() == (self.BASE/'minimal_registered_dispatch.py').resolve(),
               'DISPATCH_EXECUTING_NEW_IMMUTABLE_METADATA_SOURCE')
        binding = read(self.BASE/'OUTPUT_BINDING.private.json')
        own_lease = read(binding['lease'])
        entries = [v for v in read(binding['registry'])['entries'] if v.get('role') == binding['role']]
        e.need(len(entries) == 1 and entries[0]['path'] == str(self.BASE)
               and binding['run_root'] == str(self.BASE) and binding['manifest_sha256'] == sha(self.BASE/'RUN_MANIFEST.json')
               and own_lease['status'] == 'ACTIVE' and own_lease['manifest_sha256'] == binding['manifest_sha256']
               and own_lease['run_uuid'] == h['run_uuid'] and own_lease['deadline_epoch'] == h['deadline_epoch'],
               'DISPATCH_NEW_CANONICAL_METADATA_ROLE_AND_OWN_ACTIVE_LEASE')
        self.root = Path(h['science_root'])
        original = read(self.root/'control/resident-owner/SOURCE_REQUEST.private.json')
        request_path = Path(original['directory'])/'REQUEST.json'
        self.manifest, self.registered_request = root_context(self.root, request_path, full=True, require_bound=False)
        self.CONFIG = read(self.root/'MIX_OWNER_CONFIG.private.json')
        self.binding = read(self.root/'OUTPUT_BINDING.private.json')
        self.adapter = adapters.manifest(self.manifest)
        self.global_path = self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
        self.successor_scope = successor_cd.validated(read(self.global_path))
        self.code = self.root/self.manifest['code']['capsule_relative']
        self.request_template = self.CONFIG['request_template']
        self.local = captured_LOCAL(self.root)
        e.need(h['science_manifest_sha256'] == sha(self.root/'RUN_MANIFEST.json')
               and h['science_source_sha'] == self.manifest['code']['vla_commit'] == self.CONFIG['source_sha']
               and h['request_sha256'] == sha(request_path)
               and h['limits'] == dict(old_owner_CPU_upper_s=800, new_owner_CPU_upper_s=2200,
                                      original_owner_CPU_bank_s=3000, new_GiB_or_CPU_payment=0)
               and os.getuid() == 22182 and os.uname().nodename.startswith('stdct-login-')
               and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not os.environ.get('PBS_JOBID'),
               'DISPATCH_SAME_SCIENTIFIC_SOURCE_INPUT_REQUEST_CLOCK_AND_PREPAID_BANK')
        self.retry_predecessor = read(self.root/adapters.PREDECESSOR)
        from scripts.pi0294.generalization.minimal_retry_runtime_guard import Guard as Physical
        self.retry_physical = Physical(self.root, self.request_template, self.manifest, lambda **_: None)
        self.last = self.current = self.running = None
        self.cold_rpc_bound = self.preserve_readonly_until_deadline = True
        self.previous = SimpleNamespace(rpc=self.heartbeat_rpc)
        self.HEARTBEAT = HEARTBEAT
        self.setup = mature.new_journal(self.BASE, 'setup-rpc', dict(run_uuid=self.root.name,
            job_id='REGISTERED_NOT_SUBMITTED', source_sha=self.manifest['code']['vla_commit'],
            request_sha256=h['request_sha256']))
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
        resource.setrlimit(resource.RLIMIT_CPU, (2190, 2200))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        self.record('SOURCE_REQUEST', original)

    def cpu_guard(self):
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        used = time.process_time()+children.ru_utime+children.ru_stime
        e.need(used < NEW_CPU_UPPER-40, 'DISPATCH_AGGREGATE_PREPAID_OWNER_AND_CHILD_CPU')

    def program(self, *a, **kw):
        self.cpu_guard()
        return super().program(*a, **kw)

    def fresh(self, *a, **kw):
        self.cpu_guard()
        return super().fresh(*a, **kw)

    def first_dispatch(self, secret=None):
        h = self.handoff
        gb, rb, evidence = capture(self.root, h['old_launch_epoch'], h['old_processes'])
        proof = reconcile(gb, rb, evidence, now_epoch=time.time())
        self.record('ORIGINAL_REGISTER_CAS_RECONCILED', dict(proof=proof, actual=evidence))
        r = json.loads(rb)
        directory = Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory'])
        self.auth(secret)
        quota = self.program('QUOTA')
        full_queue = self.setup.call('CURRENT_FULL_QUEUE', 'READ_ONLY',
                                    lambda intent, limit: queue.collect_queue())
        state = predispatch_queue(gb, rb, evidence, full_queue, now_epoch=time.time())
        from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
        checked_outside(queue.queue_rows(full_queue, now_epoch=time.time()), state['outside'],
                        base64.b64decode(self.CONFIG['outside_confirmation_base64'], validate=True))
        future = e.scope.first512_forecast(self.global_path.read_bytes(), self.local)
        e.need(future['cap_changes_required'] == {} and quota['report_age_s'] <= 3600
               and quota['available_bytes_lower'] >= future['dimensions']['Hopper_storage_bytes']['projected']
                    -read(self.global_path)['new_storage_current_bytes'], 'DISPATCH_FULL_FUTURE_AND_FRESH_SITE_QUOTA')
        current = self.setup.call('CURRENT_EFFECTIVE_QUEUE', 'READ_ONLY',
                                 lambda intent, limit: collect_effective_queue(walltime_s=r['walltime_s']))
        effective_queue(raw(current), created_epoch=time.time(), walltime_s=r['walltime_s'])
        self.record('CURRENT_PRE_DISPATCH_QUEUE', current)
        # Never rewrite the request's old queue proof. The immutable proof was
        # valid at its original created_epoch; this new read qualifies now.
        e.need(sha(directory/'EFFECTIVE_QUEUE_LIMITS.json') == r['effective_queue_proof_sha256'],
               'DISPATCH_ORIGINAL_REQUEST_QUEUE_INPUT_BYTES_UNCHANGED')
        gb, checked, evidence = capture(self.root, h['old_launch_epoch'], h['old_processes'])
        reconcile(gb, checked, evidence, now_epoch=time.time())
        e.need(checked == rb, 'DISPATCH_REQUEST_NOT_REPLACED_DURING_FRESH_READS')
        body = '\n'.join(k+'='+repr(v) for k,v in dict(TARGET=str(self.root), DIRECTORY=str(directory),
            REQUEST=r, RESERVATION=dict(request_sha256=digest(rb), observed_epoch=time.time(),
                                       whole_package_already_reserved=True)).items())+'\n'+self.CONFIG['programs']['SUBMIT']
        def qsub(intent, limit):
            env = {k:v for k,v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES', 'PBS_JOBID')}
            q = subprocess.run([sys.executable, '-'], input=body, capture_output=True, text=True,
                               timeout=min(limit, 40), env=env)
            e.need(q.returncode == 0, 'DISPATCH_QSUB_UNKNOWN_NEVER_REPLAY')
            return json.loads(q.stdout)
        self.record('SOURCE_QSUB_INTENT', dict(request_id=r['request_id'], first_dispatch_only=True))
        reply = self.setup.call('QSUB', 'MUTATION', qsub)
        self.record('SOURCE_QSUB_REPLY', reply)
        e.need(reply['returncode'] == 0 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02', reply['stdout']),
               'DISPATCH_QSUB_REPLY_UNKNOWN_NEVER_REPLAY')
        job = reply['stdout']
        before = self.global_path.read_bytes()
        bind_epoch = time.time()
        proposed = self.adapter.bind(json.loads(before), r['request_id'], job, now_epoch=bind_epoch)
        # Original write_global sets exactly current_revision+1 as well.
        expected = digest(raw(proposed))
        self.record('BIND_EXACT_PROPOSAL', dict(before_sha256=digest(before), after_sha256=expected,
                                              request_id=r['request_id'], job_id=job, bind_epoch=bind_epoch))
        def fixed_bind(g):
            e.need(digest(raw(g)) == digest(before), 'DISPATCH_CURRENT_GLOBAL_UNCHANGED_BEFORE_BIND')
            return self.adapter.bind(g, r['request_id'], job, now_epoch=bind_epoch)
        from scripts.pi0294.dec1.owner_transport import UnknownMutation
        try:
            self.setup.call('BIND', 'MUTATION', lambda intent, limit: self.write_global('BIND', fixed_bind))
        except UnknownMutation:
            # No BIND retry: only the exact native after bytes AND original
            # once intent/receipt allow a distinct post-reconciliation journal.
            state = read(self.setup.state_path)
            e.need(state['status'] == 'UNKNOWN' and state['intent']['phase'] == 'BIND'
                   and state['intent']['effect'] == 'MUTATION', 'DISPATCH_BIND_UNKNOWN_EXACT_PHASE_ONLY')
            proof = exact_current_cas(self.global_path, before_sha256=digest(before), after_sha256=expected)
            self.record('BIND_UNKNOWN_EXACT_READONLY_CONFIRMED', dict(proof=proof,
                original_UNKNOWN_state_sha256=sha(self.setup.state_path), original_journal=state))
            self.setup = mature.new_journal(self.BASE, 'setup-after-exact-bind-cas', self.setup.identity)
        self.record('SOURCE_JOB_BOUND', dict(job_id=job, request_id=r['request_id']))
        self.record('TRAIN_REGISTERED', dict(root=str(self.root), manifest_sha256=sha(self.root/'RUN_MANIFEST.json')))
        self.running = mature.new_journal(self.BASE, 'running-rpc', dict(run_uuid=self.root.name, job_id=job,
            source_sha=r['source_sha'], request_sha256=digest(rb)))
        return job


def main(handoff_root):
    h = read(Path(handoff_root)/'RUN_MANIFEST.json')
    # Same scientific-root flock as the original owner, acquired before any
    # new owner records or dispatch. The old file's content is not rewritten.
    lock = OwnerLock(Path(h['science_root'])/'control/STAGE_SERVER_OWNER.lock')
    owner = None
    secret = sys.stdin.readline(257).rstrip('\n') or None
    try:
        owner = Owner(handoff_root)
        owner.record('SERVER_OWNER_PROCESS', dict(pid=os.getpid(),
            proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            source_sha=owner.manifest['code']['vla_commit'], metadata_source_sha=h['metadata_source_sha'],
            agent_liveness_required=False))
        owner.first_dispatch(secret)
        terminal = cold_owner.watch(owner, secret, scope_binding=owner.successor_scope)
        owner.record('RESIDENT_TERMINAL_RETURNED', terminal)
        result = owner.finish(terminal)
        # The scientific lease was closed by the unmodified original finish.
        # Close this separate metadata-role lease exactly once afterwards.
        from scripts.pi0294.dec1.source_cpu_once_io import commit
        own = read(owner.BASE/'OUTPUT_BINDING.private.json')
        path = Path(own['lease'])
        before = path.read_bytes()
        value = json.loads(before)
        e.need(value['status'] == 'ACTIVE' and value['run_uuid'] == h['run_uuid'],
               'DISPATCH_ONLY_OWN_METADATA_LEASE_CLOSE')
        value.update(status='CLOSED', closed_epoch=time.time(),
                     scientific_close_sha256=sha(owner.BASE/'CLOSED.private.json'))
        key = h['run_uuid']+'-CLOSE'
        owner.running.call('CLOSE_METADATA_LEASE', 'MUTATION', lambda intent, limit: commit(path,
            expected_before_sha256=digest(before), proposed_bytes=raw(value), request_id=key,
            intent_path=path.with_name(key+'.INTENT.json'), receipt_path=path.with_name(key+'.RECEIPT.json'),
            lock_path=path.with_suffix('.owner.lock')))
        owner.record('RESIDENT_EXIT', dict(returncode=0, observed_epoch=time.time(), endpoint_status=result['status']))
        return result
    except BaseException as error:
        if owner is not None:
            owner.record('PRESERVED_STOP', dict(error_type=type(error).__name__, message=str(error)[-3000:],
                observed_epoch=time.time(), old_UNKNOWN_preserved=True, no_automatic_write_retry=True))
        raise
    finally:
        secret = None
        lock.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--handoff-root', required=True)
    main(p.parse_args().handoff_root)
