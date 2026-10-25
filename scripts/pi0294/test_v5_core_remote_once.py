"""Original Journal/onceIO in temp files, captured budget + synthetic C50.

Only the native account assertion is substituted. Never SSH, register or run
models. Transport failure after a real temp CAS must retain UNKNOWN/no retry.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest

from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global
from scripts.pi0294.test_v5_core_stage_accounting import registered, started
from scripts.pi0294.pv1 import evaluation_core_ledger as remote
from scripts.pi0294.pv1 import mix_core_accounting as c, mix_accounting as a, successor_cd as s
from scripts.pi0294.dec1.owner_transport import Journal, UnknownMutation


@pytest.fixture
def cd_remote(captured_global, tmp_path, monkeypatch):
    small, local, row, _ = registered(); process = started(row)
    actor = os.getpid(); start = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    process.update(pid=actor, proc_start=start, native_handle=c.process_handle(process['boot_id'], actor, start))
    small = c.bind(small, local, request_id=row['request_id'], process=process, now_epoch=1205.)['proposed_global']
    g = deepcopy(captured_global)
    for name in (s.FIELD, a.PACKAGE, c.FIELD): g[name] = deepcopy(small[name])
    for name in ('used', 'reserved'):
        for dimension, amount in small[name].items(): g[name][dimension] += amount
    for name in ('CPU_s', 'CPU_reserved_s', 'storage_reserved_bytes', 'new_storage_current_bytes',
                 'optimizer_updates', 'generation_queries', 'model_loads'):
        g[name] = g.get(name, 0) + small[name]
    g['new_storage_peak_bytes'] = max(g['new_storage_peak_bytes'], g['new_storage_current_bytes'])
    g['PBS_submissions'] += deepcopy(small['PBS_submissions'])
    c.checked(g)
    parent = tmp_path/'20261018T182956Z-localpc-pv1'; parent.mkdir()
    path = parent/'HOPPER_RESOURCE_LEDGER.json'; path.write_bytes(a.raw(g))
    code = Path(__file__).resolve().parents[2]
    names = ['scripts/pi0294/pv1/mix_core_accounting.py','scripts/pi0294/pv1/evaluation_core_ledger.py',
             'scripts/pi0294/pv1/neural_ledger.py','scripts/pi0294/dec1/source_cpu_once_io.py']
    seal = tmp_path/'SOURCE_SEAL.json'; seal.write_bytes(a.raw(dict(source_sha=row['source_sha'],
        files={n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in names})))
    root = tmp_path/row['child_run_uuid']; (root/'evidence').mkdir(parents=True)
    journal = root/'temporary-rpc'; journal.mkdir()
    obj = remote.Ledger.__new__(remote.Ledger); obj.root = root
    obj.accounting = c; obj.successor = True; obj.candidate_field = c.CANDIDATES
    obj.actor_pid = actor; obj.actor_start = start; obj.request = row; obj.request_sha = row['request_sha256']
    obj.native_handle = process['native_handle']; obj.sequence = 0; obj.path = root/'evidence/SNAPSHOT.json'
    obj.binding = dict(canonical_parent=str(parent), remote_code=str(code), remote_seal=str(seal),
        remote_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest())
    obj.journal = Journal(journal, dict(run_uuid=root.name, job_id=obj.native_handle,
        source_sha=row['source_sha'], request_sha256=row['request_sha256']))
    actual = "assert os.getuid()==22182 and os.environ.get('USER')=='e1192548' and not os.environ.get('PBS_JOBID')"
    assert actual in remote.REMOTE
    monkeypatch.setattr(remote, 'REMOTE', remote.REMOTE.replace(actual,
        'assert os.getuid()=='+str(os.getuid())+" and not os.environ.get('PBS_JOBID')"))
    obj._argv = lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-C50-once>','exec'))"]
    return obj, path


def test_same_full_captured_budget_program_preserves_old_PBS_history_and_future(cd_remote):
    obj,path=cd_remote;before=json.loads(path.read_bytes())
    assert obj.state()==before
    after=obj._rpc('DEBIT',kind='model_loads',count=1)
    after=obj._rpc('DEBIT',kind='generation_queries',count=1)
    assert after['PBS_submissions']==before['PBS_submissions']
    assert after['training_runs']==before['training_runs'] and after['optimizer_updates']==before['optimizer_updates']
    assert after[a.PACKAGE]==before[a.PACKAGE] and after['model_loads']==before['model_loads']+1
    assert after['reserved']['neural_queries']==before['reserved']['neural_queries']-1
    assert json.loads(path.read_bytes())==after and obj.journal._state()['status']=='CONFIRMED'


def test_actual_control_metrics_recorded_without_second_CPU_charge(cd_remote):
    obj,path=cd_remote;before=json.loads(path.read_bytes())
    after=obj._rpc('METRICS',CPU_s=2.5,RPC_calls=50)
    assert after['CPU_s']==before['CPU_s'] and after['CPU_reserved_s']==before['CPU_reserved_s']
    assert after[a.PACKAGE]==before[a.PACKAGE]
    assert after['V5_CD_prepaid_control_actual_metrics'][obj.request['request_id']]==dict(CPU_s=2.5,RPC_calls=50)


def test_typed_1024_candidate_does_not_overwrite_old_AB_candidate_index(cd_remote):
    obj,path=cd_remote;before=json.loads(path.read_bytes())
    after=obj._rpc('CANDIDATE',candidate='C50')
    assert after.get('development_candidates')==before.get('development_candidates')
    assert after[c.CANDIDATES]['C50']==dict(checkpoint_manifest_sha256=obj.request['checkpoint_manifest_sha256'],global_step=1024)
    with pytest.raises(Exception):obj._rpc('CANDIDATE',candidate='D25')
    assert json.loads(path.read_bytes())==after


def test_applied_write_then_lost_transport_keeps_original_UNKNOWN_and_never_reissues(cd_remote):
    obj,path=cd_remote;before=json.loads(path.read_bytes())
    obj._argv=lambda:[sys.executable,'-c',"import sys;exec(compile(sys.stdin.read(),'<temp-C50-once>','exec'));sys.exit(7)"]
    with pytest.raises(Exception):obj._rpc('DEBIT',kind='model_loads',count=1)
    after=json.loads(path.read_bytes());assert after['model_loads']==before['model_loads']+1
    assert obj.journal._state()['status']=='UNKNOWN'
    with pytest.raises(UnknownMutation):obj._rpc('DEBIT',kind='model_loads',count=1)
    assert json.loads(path.read_bytes())==after and len(list(obj.journal.root.glob('TRANSPORT-*.private.json')))==1
