"""Compile an uninstalled monitor and compare four fixed saved-state windows.

This is numerical ABI checking, not native stop-latency or safety admission.
No LD_PRELOAD, native binary, controller, or physics integration is invoked.
"""
import bisect
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Outcome(ctypes.Structure):
    _fields_=[('clearance',ctypes.c_double*2),('error',ctypes.c_double),('upper',ctypes.c_double),
              ('age',ctypes.c_double),('overlap',ctypes.c_int*2),('alarm',ctypes.c_int)]


def run(directory,goal):
    import mujoco
    import numpy as np
    from scripts.wm02.close_goal_contact_highrate_v1 import window_packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    directory,goal=Path(directory),Path(goal);started=time.time()
    if mujoco.__version__!='3.2.7'or mujoco.get_mjcb_control()is not None:
        raise ValueError('FROZEN_MUJOCO_AND_NO_CALLBACKS_REQUIRED')
    source=directory/'close_contact_fence_candidate_v1.cpp';info=read(directory/'SOURCE.private.json')
    if info['source_files']['src/go2arx5_vla/target_mujoco/close_contact_fence_candidate_v1.cpp']!=sha(source):
        raise ValueError('PINNED_CANDIDATE_SOURCE_CHANGED')
    library=directory/'libclose_contact_fence_candidate_v1.so'
    if library.exists():raise FileExistsError('UNIQUE_COMPILE_IDENTITY_REQUIRED_NO_REPLAY')
    package=Path(mujoco.__file__).parent
    command=['g++','-std=c++17','-O2','-fPIC','-shared',str(source),'-I'+str(package/'include'),
             '-L'+str(package),'-Wl,-rpath,'+str(package),'-l:libmujoco.so.3.2.7','-ldl','-o',str(library)]
    built=subprocess.run(command,capture_output=True,text=True,timeout=120)
    if built.returncode:raise RuntimeError('CANDIDATE_COMPILE_FAILED:'+built.stderr)
    module=ctypes.CDLL(str(library),mode=ctypes.RTLD_LOCAL)
    evaluate=module.close_contact_fence_evaluate_v1
    evaluate.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.POINTER(Outcome)]
    evaluate.restype=ctypes.c_int
    windows=read(goal/'safety-highrate-v1/RESULT.private.json');assert windows['exit_code']==0
    originals={e['id']:e for e in read(goal/'safety-margin-v1/FIXED_AUDIT_PLAN.private.json')['episodes']}
    cases=[];calls=0;max_error=0.;model_pins={}
    with forbid_integration(mujoco)as forbidden:
        for entry in windows['windows']:
            artifact=entry['artifact'];assert sha(artifact['path'])==artifact['sha256']
            saved=read(artifact['path']);expected={r['step']:r for r in saved['states']}
            original=originals[entry['window']['id']];native=Path(original['case'])/'native-01'
            spec=read(Path(original['case'])/'EXECUTION_SPEC.json');model_path=spec['model_path']
            assert sha(model_path)==spec['frozen_files'][model_path]
            model_pins[entry['window']['id']]=sha(model_path)
            model=mujoco.MjModel.from_binary_path(model_path);data=mujoco.MjData(model)
            commands=[json.loads(l)for l in(native/'COMMANDS.private.jsonl').read_text().splitlines()]
            boundary={c['native_step']:c for c in commands if c.get('phase')=='mobile-accepted-boundary-050'};ticks=sorted(boundary)
            alarms=[]
            for packet in window_packets(native/'state.bin.journal',entry['window']['lower'],entry['window']['upper']):
                step=packet['step'];data.qpos[:]=packet['qpos'];data.qvel[:]=packet['qvel'];data.ctrl[:]=packet['ctrl'];data.time=packet['time_s']
                witness=boundary[ticks[bisect.bisect_right(ticks,step)-1]]
                before=hashlib.sha256(b''.join(np.asarray(a).tobytes()for a in(data.qpos,data.qvel,data.ctrl))).hexdigest()
                value=Outcome();code=evaluate(model._address,data._address,witness['native_time_s'],witness['accepted_pose7'][2],ctypes.byref(value));calls+=1
                if code not in(0,1):raise ValueError('CPP_REJECTED_VALID_FIXED_HISTORICAL_STATE')
                wanted=expected[step]
                errors=[abs(value.error-wanted['actual_minus_held_receiver_z_m']),abs(value.upper-wanted['conditional_upper_error_m']),abs(value.age-wanted['receiver_knot_age_s'])]
                for k,p in enumerate(wanted['pads']):
                    errors.append(abs(value.clearance[k]-p['clearance_m']))
                    if bool(value.overlap[k])!=p['overlaps_table_xy']:raise ValueError('PAD_TABLE_XY_GEOMETRY_DIFFERENCE')
                max_error=max(max_error,*errors)
                if max(errors)>1e-10 or bool(value.alarm)!=wanted['conditional_rate_bound_alarm']:
                    raise ValueError('CPP_AND_FIXED_RECORDED_STATE_PREDICATE_DIFFER')
                after=hashlib.sha256(b''.join(np.asarray(a).tobytes()for a in(data.qpos,data.qvel,data.ctrl))).hexdigest()
                if before!=after:raise ValueError('READONLY_ABI_CHANGED_INPUT_STATE')
                if code==1:alarms.append(step)
            cases.append(dict(id=entry['window']['id'],states=len(expected),alarms=len(alarms),first_alarm=None if not alarms else alarms[0],original_outcome_unchanged=True))
        fault_codes=[]
        for timestamp,z in((data.time+.002,.64),(data.time-.052,.64),(data.time,float('nan'))):
            value=Outcome();fault_codes.append(evaluate(model._address,data._address,timestamp,z,ctypes.byref(value)))
        if fault_codes!=[2,2,2]:raise ValueError('INVALID_OR_STALE_WITNESS_MUST_FAIL_CLOSED')
        if forbidden:raise RuntimeError('UNEXPECTED_PYTHON_INTEGRATION')
    if calls!=1204:raise ValueError('FIXED_FOUR_WINDOW_DENOMINATOR')
    return dict(schema='SLA_CLOSE_UNINSTALLED_CPP_PREDICATE_NUMERICAL_CHECK_V1',exit_code=0,
        status='NUMERICAL_ABI_MATCH_NOT_SAFETY_QUALIFIED',candidate_origin_source_sha=info['source_sha'],
        candidate_cpp_sha256=sha(source),library_sha256=sha(library),mujoco='3.2.7',cases=cases,
        saved_state_evaluations=calls,maximum_geometry_reference_difference_m_or_s=max_error,
        original_input_qpos_qvel_ctrl_unchanged=True,invalid_future_stale_nonfinite_witness_codes=fault_codes,
        original_four_window_result_sha256=sha(goal/'safety-highrate-v1/RESULT.private.json'),scene_sha256=model_pins,
        historical_full_cohort_zero_false_block_screen='FAILED_1_OF_38_ORIGINAL_SUCCESSES',
        installed=False,native_stop_latency_proven=False,independent_admission=False,
        wrapper_stop_hook_not_native_qualified=True,new_integrations=0,new_physical_requests=0,
        new_model_generations=0,new_optimizer_updates=0,wall_s=time.time()-started,CPU_s=time.process_time())


if __name__=='__main__':
    import sys
    print(json.dumps(run(*sys.argv[1:3])))
