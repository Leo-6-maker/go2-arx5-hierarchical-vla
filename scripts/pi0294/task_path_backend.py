"""Finite new-stage MPlib backend; private actual-state input, zero physics.

Called by the approved native owner using the already installed planner
environment. First result is preserved; no seed replacement or retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

extra = os.environ.get('S1_RUNTIME_EXTRA_SITE_PACKAGES')
if extra:
    if not Path(extra).is_absolute() or not Path(extra).is_dir() or ':' in extra:
        raise ValueError('EXACT_EXISTING_RUNTIME_SITE_PACKAGES_REQUIRED')
    # Keep the existing planner environment's NumPy/MPlib ahead of the runtime
    # MuJoCo dependency. Never modify either installed environment.
    sys.path.append(extra)

import numpy as np

from .task_path_model import PlanningObjectMotion
from .task_path_planning import load_saved_native, prepare_path
from .task_path_reference import retime_witness


def run(request_path, output_path):
    source, output = Path(request_path), Path(output_path)
    if source.parent.resolve() != output.parent.resolve() or output.exists() or output.with_suffix('.npz').exists():
        raise ValueError('FIRST_SAME_FOLDER_BACKEND_RESULT_PRESERVED')
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    raw = source.read_bytes(); request = json.loads(raw); began = time.process_time()
    if request['schema'] != 'S1_ACTUAL_STAGE_MPLIB_REQUEST_V1': raise ValueError('EXACT_NEW_STAGE_REQUEST')
    result = dict(request_sha256=hashlib.sha256(raw).hexdigest(), physics_integrations=0,
                  stage=request['stage'], selected='NONE', deployment='NONE')
    try:
        native = load_saved_native(request['model_path'], request['model_sha256'], request['actual_sample'])
        motion = PlanningObjectMotion(request['actual_sample'], mode=request['object_motion'],
                                      push_direction=request.get('push_direction'))
        planning, path = prepare_path(native=native, urdf=request['urdf'], srdf=request['srdf'],
            goal_world=request['goal_world'], planning_object_motion=motion)
        result['planning'] = planning; result['object_hypothesis'] = motion.evidence()
        result['status'] = planning['status']
        if path is not None and planning['status'] == 'DISCRETE_NATIVE_CONFIGURATION_CHECK_PASS':
            timing, timed = retime_witness(native, path['joint_witness'], path['time_s'],
                planning_object_motion=motion, linear_speed_m_s=request['linear_speed_m_s'])
            result.update(timing=timing, status=timing['status'])
            if timed is not None:
                artifact = output.with_suffix('.npz')
                with artifact.open('xb') as f:
                    np.savez_compressed(f, time_s=timed['time_s'], poses_world=timed['poses_world'],
                                        joint_witness=timed['joint_witness'])
                result['artifact_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except Exception as exc:
        result.update(status='FINITE_BACKEND_EXCEPTION_PRESERVED', error=type(exc).__name__+': '+str(exc))
    result['CPU_s'] = time.process_time()-began
    with output.open('x') as f:
        os.fchmod(f.fileno(), 0o600); json.dump(result, f, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.request, args.output), allow_nan=False))
