"""Owner-only terminal admission from fixed raw identities and audit receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

from deadline_guard import atomic_write, install_deadline

SOURCE = '57a02b4923e5001df5fd4b3b5958c372b9cff541'
SEAL_SHA = 'f4919dd5b4ad4f978f182cb0663e6cbdb3b50be34a84d19f56f2d7aa98398f9d'
AUTHORITY = '15cc7957488fb879f5f050e3602327592da7036b'
DOMAIN = 'FIXED_BASE_ARX5_TASK_SIM'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_manifest(folder):
    manifest = read(folder/'FILE_MANIFEST.json')
    names = [row['path'] for row in manifest]
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    assert len(names) == len(set(names)) and set(names) == actual-{'FILE_MANIFEST.json'}
    for row in manifest:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = folder/relative
        assert path.resolve().is_relative_to(folder.resolve())
        cursor = path
        while cursor != folder:
            assert not cursor.is_symlink()
            cursor = cursor.parent
        assert sha(path) == row['sha256']
    return sha(folder/'FILE_MANIFEST.json')


def bind_formal(root, identifier, layout, attempt, seal):
    row = dict(id=identifier, split=layout['split'], group=layout['group_id'],
        seed=layout['seed'], attempted=attempt is not None, admitted=False,
        source_sha=SOURCE, registered=True, windows=0)
    assert isinstance(row['seed'], int) and not isinstance(row['seed'], bool)
    if attempt is None:
        row.update(reason='NOT_ATTEMPTED_ORIGINAL_CUTOFF_OR_TERMINAL_OWNER_STOP')
        return row
    folder = root/'episodes'/identifier
    card, result = read(folder/'CARD.json'), read(folder/'RESULT.json')
    assert all(card[key] == value for key, value in layout.items())
    assert card['source_sha'] == result['source_sha'] == attempt['source_sha'] == SOURCE
    assert card['authority'] == AUTHORITY and card['domain'] == result['domain'] == DOMAIN
    assert card['mode'] == 'normal' and result['id'] == identifier
    assert attempt['status'] == result['status']
    row.update(physics_status=result['status'], physical_success=result.get('score',{}).get('persistent_success'),
        paired_samples=result['paired_samples'], physics_steps=result['physics_steps'],
        result_sha256=sha(folder/'RESULT.json'), card_sha256=sha(folder/'CARD.json'))
    if result['status'] != 'VALID_COMPLETE':
        row.update(reason='PRESERVED_INCOMPLETE_PHYSICS', error=result.get('error'))
        return row
    assert attempt['steps'] == result['physics_steps'] == 16000
    assert attempt['pairs'] == result['paired_samples'] == 641
    manifest_sha = verified_manifest(folder)
    audit_path = root/'evidence'/(identifier+'_RAW_AUDIT_WITH_VIDEO.json')
    if not audit_path.is_file():
        audit_path = root/'evidence'/(identifier+'_RAW_AUDIT.json')
    if not audit_path.is_file():
        row.update(reason='RAW_AUDIT_MISSING', raw_manifest_sha256=manifest_sha)
        return row
    audit = read(audit_path)
    assert audit['audit_script_sha256'] == seal['code_files']['scripts/pi0291_arx5_data/audit_data.py']
    assert audit['integrations'] == audit['model_loads'] == audit['render_calls'] == 0
    assert len(audit['episodes']) == 1 and audit['episodes'][0]['id'] == identifier
    evidence = audit['episodes'][0]
    if audit['status'] != 'PASS_RAW_DATA_AUDIT' or evidence['status'] != 'PASS_FULL_RAW_AUDIT':
        row.update(reason='RAW_AUDIT_FAIL', audit_receipt_sha256=sha(audit_path))
        return row
    assert evidence['source_sha'] == SOURCE and evidence['manifest_sha256'] == manifest_sha
    assert evidence['physics_sha256'] == sha(folder/'physics.npz')
    assert evidence['samples_sha256'] == sha(folder/'samples.json')
    assert evidence['paired_samples'] == 641 and evidence['physics_steps'] == 16000
    assert evidence['actual_committed_H40_windows'] == 601 and evidence['complete']
    assert evidence['privileged_policy_separation'] and evidence['causality_clock_and_reference_prefix_verified']
    assert evidence['independent_score']['persistent_success'] == row['physical_success']
    eligible = bool(row['physical_success'] and evidence['all_action_supervision_valid']
        and evidence['camera_visibility']['initial_and_grasp_carry_release_visible'])
    if row['physical_success']:
        score = evidence['independent_score']
        assert score['safety_valid'] and score['hold_observed'] == score['hold_pass'] == 41
        assert all(score['history'].values())
    row.update(admitted=eligible, windows=601 if eligible else 0,
        reason='OWNER_RAW_SUCCESS_AND_DATA_ADMITTED' if eligible else 'PRESERVED_VALID_QUALITY_FAILURE',
        samples_sha256=evidence['samples_sha256'], physics_sha256=evidence['physics_sha256'],
        raw_manifest_sha256=manifest_sha, audit_receipt_sha256=sha(audit_path),
        independent_score=evidence['independent_score'])
    return row


def publish_plan(root, plans, starts):
    journal_path = root/'evidence/RAW_BANK_FINALIZATION_STATE.json'
    digests = {name:hashlib.sha256(value).hexdigest() for name,value in plans.items()}
    starts_sha = hashlib.sha256(json.dumps(starts,sort_keys=True).encode()).hexdigest()
    journal = dict(status='PREPARED',expected_output_sha256=digests,
        starts_sha256=starts_sha,seal_sha256=SEAL_SHA,helper_sha256=sha(__file__))
    if journal_path.exists():
        old = read(journal_path)
        assert all(old[key] == journal[key] for key in ('expected_output_sha256','starts_sha256','seal_sha256','helper_sha256'))
    else:
        assert not any((root/name).exists() for name in plans), 'Existing unjournaled output: preserve and inspect'
        atomic_write(journal_path,journal)
    for name,value in plans.items():
        path = root/name
        if path.exists():
            assert sha(path) == digests[name], 'Never replace mismatched existing output'
            continue
        temporary = path.with_name(path.name+'.pending-'+str(os.getpid()))
        try:
            with temporary.open('xb') as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)
        assert sha(path) == digests[name]
    journal['status'] = 'COMMITTED_ALL_OUTPUT_HASHES_VERIFIED'
    atomic_write(journal_path,journal)


def finalize(root):
    root = Path(root)
    seal, ledger, state = read(root/'TEACHER_AND_LAYOUT_SEAL.json'), read(root/'RESOURCE_ACCOUNTING.json'), read(root/'evidence/OWNER_N3_QUEUE_STATE.json')
    assert sha(root/'TEACHER_AND_LAYOUT_SEAL.json') == SEAL_SHA
    assert seal['source_sha'] == state['source_sha'] == SOURCE and seal['authority'] == AUTHORITY
    assert not state['physics_running'] and state['current'] is None
    assert ledger['CPU_s']+ledger['unmetered_dependency_setup_CPU_s_upper'] < 48*3600
    assert not any(row['status'] == 'STARTED' for row in ledger['starts'])
    identifiers = [f'T291{i:03d}' for i in range(64)]+[f'D291{i:03d}' for i in range(16)]
    assert seal['formal_order'] == identifiers
    assert len({row['id'] for row in ledger['starts']}) == len(ledger['starts'])
    attempts = {row['id']: row for row in ledger['starts']}
    actual_order = [row['id'] for row in ledger['starts'] if row['id'].startswith(('T','D'))]
    assert actual_order == identifiers[:len(actual_order)]
    assert {name:sha(root/'code'/name) for name in seal['code_files']} == seal['code_files']
    gate = read(root/'PILOT_GATE.json')
    assert gate['status'] == 'PASS' and gate['all_eight_independent_raw_audits_pass']
    assert gate['source_sha'] == SOURCE and gate['teacher_layout_seal_sha256'] == SEAL_SHA
    assert gate['denominator'] == gate['physical_successes'] == 8
    rows = [bind_formal(root, identifier, seal['layouts'][identifier], attempts.get(identifier), seal) for identifier in identifiers]
    counts = {split:dict(planned=n, attempted=sum(r['attempted'] for r in rows if r['split']==split),
        admitted=sum(r['admitted'] for r in rows if r['split']==split),
        unattempted=sum(not r['attempted'] for r in rows if r['split']==split),
        admitted_H40_windows=sum(r['windows'] for r in rows if r['split']==split)) for split,n in [('TRAIN',64),('DEV',16)]}
    admission = dict(run_uuid=root.name, source_sha=SOURCE, authority=AUTHORITY, domain=DOMAIN,
        dataset_status='PARTIAL', native_qualification='NOT_RUN_AT_RAW_ADMISSION',
        complete_attempt_bank=len(actual_order)==80, counts=counts, episodes=rows,
        seed_encoding='Exact Python integers from sealed layouts; no floating-point conversion',
        teacher_layout_seal_sha256=SEAL_SHA, selected='NONE', deployment='NONE')
    if len(actual_order)==80 and counts['TRAIN']['admitted']>=56 and counts['DEV']['admitted']>=14:
        admission['dataset_status'] = 'FULL_RAW_BANK_NATIVE_PENDING'
    lines = []
    for attempt in ledger['starts']:
        if attempt['id'] in identifiers:
            continue
        result = read(root/'episodes'/attempt['id']/'RESULT.json')
        lines.append(dict(**attempt, attempted=True, admitted=False, formal_BC=False,
            cohort='COMMISSIONING' if attempt['id'].startswith('C') else 'PILOT',
            physical_success=result.get('score',{}).get('persistent_success'), result_sha256=sha(root/'episodes'/attempt['id']/'RESULT.json')))
    lines.extend(rows)
    path = root/'evidence/ATTEMPT_LEDGER.jsonl'
    plans = {'DATASET_ADMISSION.json':(json.dumps(admission,indent=2,allow_nan=False)+'\n').encode(),
        'evidence/ATTEMPT_LEDGER.jsonl':''.join(json.dumps(row,allow_nan=False)+'\n' for row in lines).encode(),
        'evidence/PILOT_SCORE.json':(root/'PILOT_GATE.json').read_bytes()}
    publish_plan(root,plans,ledger['starts'])
    assert sha(root/'evidence/PILOT_SCORE.json') == sha(root/'PILOT_GATE.json')
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(status='OWNER_TERMINAL_RAW_BINDING_PASS', CPU_s=usage.ru_utime+usage.ru_stime,
        counts=counts, admission_sha256=sha(root/'DATASET_ADMISSION.json'),
        attempt_ledger_sha256=sha(path), code_files_checked=len(seal['code_files']),
        helper_sha256=sha(__file__), physics_steps=0, render_calls=0, model_loads=0)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    root = Path(args.root)
    receipt_path = root/'evidence'/f'CPU_AUX_OWNER_FINAL_RAW_BINDING_{int(time.time())}_{os.getpid()}.json'
    def save_usage(status):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if not receipt_path.exists():
            atomic_write(receipt_path,dict(status=status,CPU_s=usage.ru_utime+usage.ru_stime,
                helper_sha256=sha(__file__),physics_steps=0,render_calls=0,model_loads=0,
                scope='This unique helper invocation only; failed/timeout invocation remains chargeable'))
    status = 'FAILED_OWNER_BINDING_PRESERVED'
    try:
        install_deadline(root, 'owner_final_raw_binding')
        for signum in (signal.SIGALRM,signal.SIGXCPU):
            previous = signal.getsignal(signum)
            def track_stop(number,frame,handler=previous):
                save_usage('STOP_ORIGINAL_WALL_OR_CPU_BOUND')
                handler(number,frame)
            signal.signal(signum,track_stop)
        finalize(root)
        status = 'OWNER_TERMINAL_RAW_BINDING_PASS'
    finally:
        save_usage(status)
