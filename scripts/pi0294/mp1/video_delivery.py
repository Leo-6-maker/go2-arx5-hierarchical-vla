"""Publish audited, closed source-recovery clips without reopening their runs."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from scripts.pi0292.durable import read, write, OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.public_checkpoint import reject_private
from scripts.pi0294.checkpoint_delivery import verify_remote
from .scope import validate, charge_cpu

CHILDREN = ['e1-panel-recovery-01', 'e1-panel-recovery-02', 'e1-panel-recovery-03']
REPO = 'leoyuer/pi0294-public-artifacts'


def audited_movies(root):
    root = Path(root)
    rows, files, sources = [], {}, {}
    for identifier in CHILDREN:
        child = root / 'children' / identifier
        binding = read(child / 'OUTPUT_BINDING.private.json')
        if read(binding['lease'])['status'] != 'CLOSED':
            raise PermissionError('CLOSED_CHILD_ONLY_NOT_REOPENED')
        ledger = read(child / 'RESOURCE_LEDGER.json')
        if ledger['reservations'] or ledger['CPU_reserved_s']:
            raise PermissionError('NO_UNSETTLED_CHILD_WORK')
        for decision in sorted((child / 'episodes').glob('*/MP1_ADJUDICATION.json')):
            actual = read(decision)
            zero = actual['main_integrations'] == 0
            audit_folder = child / 'evidence' / ('zero-step-audits' if zero else 'audits') / decision.parent.name
            audit = read(audit_folder / 'AUDIT.private.json')
            required = 'PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE' if zero else 'PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT'
            if audit['status'] != required:
                raise PermissionError('INDEPENDENT_ACTUAL_AUDIT_REQUIRED')
            if audit['attempt_id'] != decision.parent.name or audit['adjudication'] != actual:
                raise ValueError('EXACT_CLOSED_ATTEMPT_ADJUDICATION')
            movie = audit['video']
            expected = audit_folder / ('actual-initial-body-wrist.mp4' if zero else 'actual-body-wrist.mp4')
            if expected.is_symlink() or not expected.resolve().is_relative_to(child.resolve()):
                raise PermissionError('NO_VIDEO_SYMLINK_OR_ESCAPE')
            if (sha(expected) != movie['sha256'] or expected.stat().st_size != movie['bytes']
                    or movie['decoded_frames'] != movie['actual_paired_frames']
                    or movie['padding_or_interpolation']):
                raise ValueError('ACTUAL_VIDEO_IDENTITY_OR_DECODE_GAP')
            name = 'movies/' + audit['attempt_id'] + '.mp4'
            if name in files:
                raise ValueError('UNIQUE_RECOVERY_ATTEMPT_VIDEO')
            sources[name] = expected
            files[name] = dict(bytes=movie['bytes'], sha256=movie['sha256'])
            rows.append(dict(id=audit['attempt_id'], child=identifier,
                status=audit['adjudication']['status'], task=audit['adjudication']['task'],
                scene_group=audit['adjudication']['scene_group'],
                native_sim_s=movie['native_duration_s'], frames=movie['decoded_frames'],
                initial_frame_included=True, task_trajectory=movie['native_duration_s'] > 0,
                autonomous_task_success=audit['adjudication']['status'] == 'SUCCESS',
                artifact_path=name, **files[name]))
    if len(rows) != 10 or sum(v['bytes'] for v in files.values()) > 128 * 1024**2:
        raise PermissionError('EXACT_TEN_RECOVERY_ATTEMPTS_128MIB_UPPER')
    return rows, files, sources


def publish(root, token_file):
    root = Path(root)
    runtime = validate(root, '4080-cpu-data')
    binding = root / 'plans/PUBLIC_RECOVERY_VIDEO_BINDING.json'
    if runtime['execution'].get('public_video_binding_sha256') != sha(binding):
        raise PermissionError('SEALED_PUBLIC_VIDEO_BINDING')
    spec = read(binding)
    if (spec['parent_run'] != root.name or spec['repo_id'] != REPO
            or spec['children'] != CHILDREN or spec['maximum_CPU_s'] != 600
            or spec['public_authority'] != 'DIRECT_USER_PUBLIC_GENERATED_SOURCE_VIDEO_PERMISSION'
            or spec['scope'] != 'AUDITED_GENERATED_SIMULATION_CLIPS_ONLY_NO_RAW_ASSETS_KEYS_OR_WBC'
            or not spec['created_epoch'] <= time.time() < spec['deadline_epoch'] <= spec['created_epoch'] + 1800):
        raise PermissionError('EXPLICIT_BOUNDED_PUBLIC_VIDEO_AUTHORITY')
    signal.alarm(max(1, int(spec['deadline_epoch'] - time.time())))
    owner = OwnerLock(root / 'PUBLIC_RECOVERY_VIDEO_OWNER.lock')
    started = time.process_time()
    output = root / 'artifacts/MP1_RECOVERY_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'
    output.mkdir(mode=0o700)
    try:
        rows, files, sources = audited_movies(root)
        index = dict(schema='MP1_RECOVERY_ACTUAL_VIDEO_INDEX_V1', parent_run=root.name,
            source_sha=runtime['code']['vla_commit'], private=False, rows=rows,
            actual_videos=len(rows), initial_frame_only=sum(not r['task_trajectory'] for r in rows),
            all_videos_are_successes=False, all_videos_are_complete_episodes=False,
            no_interpolation_or_fabricated_camera=True, original_private_bytes_unchanged=True,
            selected='NONE', deployment='NONE')
        reject_private(index)
        write(output / 'VIDEO_INDEX.json', index)
        sources['VIDEO_INDEX.json'] = output / 'VIDEO_INDEX.json'
        files['VIDEO_INDEX.json'] = dict(bytes=sources['VIDEO_INDEX.json'].stat().st_size,
                                        sha256=sha(sources['VIDEO_INDEX.json']))
        os.environ['HF_HUB_OFFLINE'] = '0'
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
        token = Path(token_file).read_text().strip()
        api = HfApi(token=token)
        info = api.repo_info(REPO, repo_type='dataset')
        if info.private:
            raise PermissionError('EXISTING_PUBLIC_REPOSITORY_ONLY_NO_VISIBILITY_CHANGE')
        prefix = 'PI0294/' + root.name + '/MP1_RECOVERY_ACTUAL_AUDITED_VIDEO_PUBLIC_V1'
        if any(n.startswith(prefix + '/') for n in api.list_repo_files(REPO, repo_type='dataset')):
            raise FileExistsError('IMMUTABLE_PUBLIC_PREFIX_NO_BLIND_REUPLOAD')
        state = dict(status='UPLOAD_INTENT_UNKNOWN_NO_BLIND_RETRY', repo_id=REPO,
                     prefix=prefix, files=files, actual_videos=len(rows))
        write(output / 'TRANSFER.private.json', state)
        revision = api.create_commit(REPO, repo_type='dataset', parent_commit=info.sha,
            commit_message='Audited MP1 recovery clips including failures and initial-only attempts',
            operations=[CommitOperationAdd(path_in_repo=prefix + '/' + n,
                        path_or_fileobj=str(sources[n])) for n in files]).oid
        state.update(status='PAYLOAD_COMMITTED_READBACK_PENDING', data_revision=revision)
        write(output / 'TRANSFER.private.json', state)
        def download(*args, **kwargs):
            return hf_hub_download(*args, **kwargs, local_dir=output / 'small-readback')
        verify_remote(api, REPO, prefix, revision, files, download, token)
        ready = dict(status='READY_ACTUAL_AUDITED_RECOVERY_VIDEO_BYTES', parent_run=root.name,
                     actual_videos=len(rows), files=files, data_revision=revision,
                     private=False, READY_uploaded_last=True)
        write(output / 'READY.json', ready)
        final = api.create_commit(REPO, repo_type='dataset', parent_commit=revision,
            commit_message='READY after fixed-revision recovery video hash verification',
            operations=[CommitOperationAdd(path_in_repo=prefix + '/READY.json',
                        path_or_fileobj=str(output / 'READY.json'))]).oid
        final_files = dict(files)
        final_files['READY.json'] = dict(bytes=(output / 'READY.json').stat().st_size,
                                        sha256=sha(output / 'READY.json'))
        verify_remote(api, REPO, prefix, final, final_files, download, token)
        state.update(status='PASS_PUBLIC_RECOVERY_VIDEOS_FIXED_REVISION_READBACK', revision=final,
                     READY_sha256=sha(output / 'READY.json'), READY_uploaded_last=True,
                     payload_bytes=sum(v['bytes'] for v in files.values()))
        write(output / 'TRANSFER.private.json', state)
        return state
    finally:
        charge_cpu(root, time.process_time() - started)
        owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--existing-token-file', required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    def expire(signum, frame):
        raise TimeoutError('PUBLIC_RECOVERY_VIDEO_FIXED_1800S_WALL_BOUND')
    signal.signal(signal.SIGALRM, expire)
    signal.alarm(1800)
    try:
        value = publish(args.root, args.existing_token_file)
    finally:
        signal.alarm(0)
    print(json.dumps({k: value[k] for k in ['status', 'revision', 'prefix', 'actual_videos', 'payload_bytes']}), flush=True)
