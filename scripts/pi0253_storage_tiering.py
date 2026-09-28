#!/usr/bin/env python3
"""Explicit manifest unlink only; preserve evidence, active paths and shared assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import socket
import stat
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    p = Path(path); s = p.lstat()
    return dict(path=str(p), realpath=str(p.resolve()), owner=pwd.getpwuid(s.st_uid).pw_name,
                uid=s.st_uid, dev=s.st_dev, inode=s.st_ino, nlink=s.st_nlink,
                logical_bytes=s.st_size, allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                mode=s.st_mode, symlink_target=os.readlink(p) if p.is_symlink() else None)


def allowed(item, protected=False, shared=False, opened=False):
    if protected or item.get('current_candidate') or item['class'] == 'KEEP_ACTIVE':
        return False, 'KEEP_ACTIVE_OR_CURRENT_CANDIDATE'
    if shared:
        return False, 'SHARED_BYTES_RETAINED'
    if opened:
        return False, 'ACTIVE_REFERENCE'
    if item['class'] == 'PURGE_RETIRED' and item.get('terminal_retired') and item.get('no_dependencies'):
        return True, 'WEIGHTS_DISCARDED_INTENTIONALLY'
    if item['class'] == 'KEEP_HF' and not item.get('needed_workset', True):
        receipt = item.get('content_readback_receipt', {})
        if receipt.get('status') in ('PASS_PT_FULL_HF_READBACK','PASS_FULL_FILE_READBACK') and receipt.get('sha256') and item.get('anonymous_reachability') == 'PASS':
            return True, 'FIXED_HF_VERIFIED_LOCAL_RETIRED'
        return False, 'USEFUL_UNARCHIVED_OR_PUBLIC_PENDING'
    return False, 'UNKNOWN_SKIP'


def opened_files():
    opened, unknown, sessions = set(), [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            for fd in (p / 'fd').iterdir():
                try:
                    s = fd.stat(); opened.add((s.st_dev, s.st_ino))
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
        except PermissionError:
            try:
                comm = (p / 'comm').read_text().strip()
                cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
                if comm in ('(sd-pam)', 'sshd-session') or (comm == 'sshd' and cmd.startswith('sshd: ') and '@' in cmd) or (comm == 'systemd' and cmd == '/usr/lib/systemd/systemd --user'):
                    sessions.append(dict(pid=p.name, comm=comm, reason='LOGIN_SESSION_DAEMON_NOT_PROJECT_WORKER'))
                else:
                    unknown.append(p.name)
            except FileNotFoundError:
                pass
    return opened, unknown, sessions


def space(roots):
    result = []
    for root in roots:
        p = Path(root); s = os.statvfs(p)
        result.append(dict(path=str(p), dev=p.stat().st_dev, available_bytes=s.f_bavail*s.f_frsize,
                           free_inodes=s.f_favail, epoch_s=time.time(), quota_headroom='UNKNOWN'))
    return result


def journal(path, event):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())


def cpu_check():
    retired = dict(class_='PURGE_RETIRED', terminal_retired=True, no_dependencies=True)
    retired['class'] = retired.pop('class_')
    assert allowed(retired)[0]  # No backup requirement for expressly retired failure.
    assert not allowed(retired, protected=True)[0]
    assert not allowed({**retired, 'current_candidate': True})[0]
    assert not allowed(retired, shared=True)[0]
    assert not allowed(retired, opened=True)[0]
    assert not allowed({'class':'KEEP_HF', 'needed_workset':False})[0]
    assert not allowed({**retired, 'terminal_retired':False})[0]
    metadata = {'storage_state':'METADATA_ONLY_NOT_RESUMABLE', 'inventory_verified':False}
    assert not (metadata['storage_state'] == 'LOCAL_FULL_VERIFIED' and metadata['inventory_verified'])
    print('PASS_CPU_STORAGE_BOUNDARIES_ZERO_PROJECT_QUERIES', flush=True)


def run(plan_path, preview_path, apply, expected_sha):
    plan = read(plan_path)
    assert plan['authorization'] == 'PI0253_STORAGE_TIERING_AND_RETIRED_FAILURE_PURGE_GO'
    assert plan['authority_commit'] == 'eb2ac1b76338fc5b3486809f6bb28e80529ed4b1'
    assert plan['host'] == socket.gethostname() and plan['active_reference_check'] == 'PASS'
    opened, unknown, sessions = opened_files()
    assert not unknown, 'open-file coverage UNKNOWN for owned PIDs: ' + str(unknown)
    roots = [Path(p).resolve() for p in plan['allow_roots']]
    protected = [Path(p).resolve() for p in plan['protected_paths']]
    protected_inodes = {tuple(x) for x in plan['protected_inodes']}
    if apply:
        publication = read(Path(preview_path).parent / 'publication.json')
        assert expected_sha == sha(preview_path) == publication['preview_sha256'] and publication['url']
        previous = read(preview_path); assert previous['plan_sha256'] == sha(plan_path)
        items = previous['items']
    else:
        items = []
        for item in plan['items']:
            row = dict(item)
            p = Path(row['path']); assert p.is_absolute()
            assert any(p.parent.resolve().is_relative_to(root) for root in roots)
            if not p.exists() and not p.is_symlink():
                row.update(result='ALREADY_ABSENT', action='SKIP'); items.append(row); continue
            current = identity(p); row['pre_delete_identity'] = current
            assert current['uid'] == os.getuid()
            assert stat.S_ISREG(current['mode']) or stat.S_ISLNK(current['mode'])
            if row['class'] == 'KEEP_HF':
                ref = row['content_readback_receipt']; assert sha(ref['path']) == ref['sha256']
            is_protected = any(p.resolve().is_relative_to(q) for q in protected)
            ok, why = allowed(row, protected=is_protected,
                shared=(current['dev'],current['inode']) in protected_inodes,
                opened=(current['dev'],current['inode']) in opened)
            row.update(action='UNLINK' if ok else 'SKIP', result='PREVIEW', decision_reason=why)
            items.append(row)
    before = space(plan['space_roots'])
    if apply:
        for row in items:
            if row['action'] != 'UNLINK':
                continue
            p = Path(row['path'])
            if not p.exists() and not p.is_symlink():
                row['result'] = 'ALREADY_ABSENT_AFTER_PREVIEW'; continue
            current = identity(p)
            if current != row['pre_delete_identity']:
                row['result'] = 'CHANGED_IDENTITY_SKIP'; continue
            if row['class'] == 'KEEP_HF':
                ref = row['content_readback_receipt']; assert sha(ref['path']) == ref['sha256']
            is_protected = any(p.resolve().is_relative_to(q) for q in protected)
            ok, why = allowed(row, protected=is_protected,
                shared=(current['dev'],current['inode']) in protected_inodes,
                opened=(current['dev'],current['inode']) in opened)
            if not ok:
                row['result'] = why + '_SKIP'; continue
            log = Path(preview_path).parent / 'unlink_journal.jsonl'
            journal(log, dict(event='UNLINK_INTENT', path=str(p), identity=current, epoch_s=time.time()))
            p.unlink(); assert not p.exists() and not p.is_symlink()
            row['result'] = 'UNLINKED_VERIFIED_ABSENT'
            journal(log, dict(event='UNLINKED_VERIFIED_ABSENT', path=str(p), epoch_s=time.time()))
            write(p.parent / 'PI0253_LOCAL_WEIGHT_RETIREMENT.json', dict(
                schema='PI0253_LOCAL_PAYLOAD_TOMBSTONE_V1', authorization=plan['authorization'],
                storage_state=row.get('post_retirement_storage_state','METADATA_ONLY_NOT_RESUMABLE'),
                local_resume_admitted=False, archive_history_untouched=True, reason=why, manifest_path=str(preview_path),
                original_card_unchanged=True, intentionally_retired_local_weights=True))
    report = dict(schema='PI0253_STORAGE_PREVIEW_V1' if not apply else 'PI0253_STORAGE_UNLINK_RESULT_V1',
                  host=plan['host'], plan_sha256=sha(plan_path), epoch_s=time.time(), items=items,
                  space_before=before, space_after=space(plan['space_roots']), open_file_check='PASS',
                  inaccessible_login_sessions=sessions,
                  model_queries=0, optimizer_updates=0, selected_checkpoint='NONE')
    deleted = [r for r in items if r['result'] == 'UNLINKED_VERIFIED_ABSENT']
    report['logical_bytes_unlinked'] = sum(r['pre_delete_identity']['logical_bytes'] for r in deleted)
    regular = {}
    for r in deleted:
        s = r['pre_delete_identity']
        if stat.S_ISREG(s['mode']):
            k = (s['dev'],s['inode']); regular.setdefault(k, []).append(s)
    report['unique_inode_allocated_bytes_removed'] = sum(v[0]['allocated_bytes'] for v in regular.values() if len(v) == v[0]['nlink'])
    report['shared_inode_allocated_bytes_retained'] = sum(v[0]['allocated_bytes'] for v in regular.values() if len(v) < v[0]['nlink'])
    target = Path(preview_path).with_name('result.json') if apply else preview_path
    write(target, report)
    print(json.dumps(dict(host=plan['host'],items=len(items),unlink=sum(r['action']=='UNLINK' for r in items),
                         deleted=len(deleted),logical_bytes_unlinked=report['logical_bytes_unlinked'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--plan', type=Path); parser.add_argument('--preview', type=Path)
    parser.add_argument('--apply', action='store_true'); parser.add_argument('--preview-sha')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
    else:
        assert args.plan and args.preview
        run(args.plan, args.preview, args.apply, args.preview_sha)
