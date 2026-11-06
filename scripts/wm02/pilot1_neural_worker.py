"""Finite original-owner worker wrapper; close only this job's actual lease."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write
    spec = json.loads(args.spec.read_text())
    root = Path(spec['registered_root'])
    lease_path = Path(spec['lease'])
    lease = json.loads(lease_path.read_text())
    if lease.get('status') != 'ACTIVE' or lease.get('spec_sha256') != digest(args.spec) or lease.get('run_uuid') != root.name:
        raise PermissionError('EXACT_FRESH_PILOT1_WORKER_REQUIRED')
    if (root / 'SUPERVISOR.private.json').exists():
        raise FileExistsError('DO_NOT_REDISPATCH_EXISTING_OR_UNKNOWN_WORKER')
    write(root / 'SUPERVISOR.private.json', dict(pid=os.getpid(), source_sha=spec['source_sha'], started_epoch=time.time()))
    module = {'ACTOR_DISTILLATION': 'scripts.wm02.train_pilot1_actor',
              'ACTOR_STUDY': 'scripts.wm02.train_pilot2_matrix',
              'ARX_RESIDUAL': 'scripts.wm02.train_pilot1_residual',
              'CONTACT_WM': 'scripts.wm02.train_pilot1_contact_wm',
              'CONTACT_POLICY': 'scripts.wm02.train_pilot1_contact_policy',
              'FROZEN_VLA_SHADOW': 'scripts.wm02.infer_fusion_vla',
              'FROZEN_VLA_LIVE_PAUSED_QUERY': 'scripts.wm02.infer_fusion_paused_vla',
              'FUSION_TRAJECTORY_RESPONSE_OFFLINE': 'scripts.wm02.fit_fusion_trajectory_response',
              'FUSION_RESPONSE_WM': 'scripts.wm02.train_fusion_response',
              'FUSION_STRUCTURED_RESPONSE': 'scripts.wm02.train_fusion_structured',
              'FUSION_TARGET_OBSERVATION_REPLAY': 'scripts.wm02.capture_fusion_observation',
              'FUSION_SOURCE_CAMERA_QUALIFICATION': 'scripts.wm02.qualify_fusion_source_camera',
              'FUSION_CORR_ASSET': 'scripts.wm02.stage_fusion_corr_asset',
              'FUSION_CONTACT_TIMING_BASELINE': 'scripts.wm02.fit_contact_timing_baseline',
              'FUSION_CONTACT_TIMING_FULL_HISTORY_BASELINE': 'scripts.wm02.fit_contact_timing_full_history'}[spec['kind']]
    if spec['kind'] == 'CONTACT_WM' and spec.get('contact_recipe') in (
            'WM02_PILOT2_DIRECT_CONTACT_V1', 'WM02_PILOT2_PREFIX_CONTACT_V1'):
        module = 'scripts.wm02.train_pilot2_contact_wm'
    if spec['kind'] == 'CONTACT_POLICY' and spec.get('policy_recipe') == 'WM02_PILOT2_SUPPORTED_GOAL_POLICY_V1':
        module = 'scripts.wm02.train_pilot2_supported_policy'
    command = [sys.executable, '-B', '-m', module, '--spec', str(args.spec), '--output', str(root / 'fit-01')]
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    failure, exit_code = None, 1
    try:
        with (root / 'TRAIN_STDOUT.private.log').open('wb') as stdout, (root / 'TRAIN_STDERR.private.log').open('wb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=min(spec['wall_limit_s'], spec['deadline_epoch']-time.time()))
        exit_code = result.returncode
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(root / 'WORKER_EXIT.private.json', dict(exit_code=exit_code, argv=command, failure=failure,
              wall_s=time.monotonic()-started, child_CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
        lease = json.loads(lease_path.read_text())
        if lease.get('status') != 'ACTIVE' or lease.get('spec_sha256') != digest(args.spec):
            raise PermissionError('WORKER_LEASE_CHANGED_DURING_EXECUTION')
        lease.update(status='CLOSED', closed_epoch=time.time(), worker_exit_code=exit_code)
        write(lease_path, lease)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
