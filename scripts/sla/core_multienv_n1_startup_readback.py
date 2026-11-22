"""Localize divergence in the frozen startup traces for the first six N1 short cards."""
import argparse
import json
import os
from pathlib import Path
import struct
import time

import mujoco
import numpy as np

from go2arx5_vla.world_model.init_trace_reader import RECORD
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest

START = 5703
END = START + 175
DIAGNOSTIC_ATOL = 1e-9
OLD_TEARDOWN_SHA = 'd783ca87c33a7691473f86063319cef8cbc43df228be0575fe219b874541a50b'


def read(path):
    return json.loads(path.read_bytes())


def snapshots(path):
    """Extract measured states, previous controller versions, and actual leg handoff ticks."""
    qpos = np.empty((END, 41))
    qvel = np.empty((END, 38))
    ctrl = np.empty((END, 17))
    prior_ctrl = np.empty_like(ctrl)
    handoffs = {}
    before = after = 0
    with path.open('rb') as stream:
        header = stream.read(16)
        size, capacity = struct.unpack('<II',header[8:])
        if header[:8] != b'WM02IT01' or size != RECORD.size or capacity not in (65536,131072,262144,524288):
            raise ValueError('Unexpected original trace record ABI')
        for seq, raw in enumerate(iter(lambda: stream.read(size),b'')):
            ticket, _, timestamp, kind, count, _, _, position, velocity, step, *v = RECORD.unpack(raw)
            if ticket != seq:
                raise ValueError('Native trace sequence discontinuity')
            if kind == 8 and position == velocity and position:
                handoffs.setdefault(position,after)
            if kind not in (11,12):
                continue
            if count != 96 or not np.isfinite(v).all():
                raise ValueError('Unexpected 41/38/20 trace payload')
            if kind == 11:
                if step != before+1 or abs(timestamp-.002*(step-1)) > 1e-8:
                    raise ValueError('Original pre-step clock differs')
                prior_ctrl[before] = v[79:96]
                before += 1
            else:
                if step != after+1 or step != before or abs(timestamp-.002*step) > 1e-8:
                    raise ValueError('Original post-step clock differs')
                qpos[after], qvel[after], ctrl[after] = v[:41],v[41:79],v[79:96]
                after += 1
                if after == END:
                    break
    if before != END or after != END:
        raise ValueError('Missing original paired 2 ms initialization steps')
    return dict(qpos=qpos,qvel=qvel,ctrl=ctrl,prior_ctrl=prior_ctrl,
                leg_policy_first_applied=[dict(token=int(k),after_step=v) for k,v in list(handoffs.items())[:2]])


def trace_pin(case, terminal=None):
    folder = case/'native-01'
    trace = folder/'init.trace'
    if terminal is not None:
        end = read_pin(terminal)
        expected = end['trace_audit']['input_files_sha256']['init.trace']
        evidence = terminal
    else:
        resp = case.parents[1]/'E01_NATIVE_FAILED_TEARDOWN_RECONCILIATION.private.json'
        if digest(resp) != OLD_TEARDOWN_SHA:
            raise PermissionError('Original retained teardown receipt changed')
        old = read(resp)
        if (old['case_spec_sha256'] != digest(case/'EXECUTION_SPEC.json')
                or old['native_requests'] != 1 or old['native_ticks'] != 10703
                or old['native_trace_status'] != 'RECORDED_TRACE_CHECKS_PASS'):
            raise PermissionError('Expected the original teardown-reconciled short card')
        expected = old['trace_sha256']
        evidence = dict(path=str(resp),sha256=OLD_TEARDOWN_SHA)
    if digest(trace) != expected:
        raise PermissionError('Native trace differs from the sealed receipt')
    return dict(path=str(trace),sha256=expected,evidence=evidence)


def columns(model):
    ctl = [model.actuator(i).name or f'control[{i}]' for i in range(model.nu)]
    base = next(i for i in range(model.njnt)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[i] == 0)
    if (model.nq,model.nv,model.nu) != (41,38,20) or base != 0:
        raise ValueError('Expected the frozen native state and control dimensions')
    arm = {model.joint(name).id for name in (f'x5_joint{i}' for i in range(1,7))}
    arm_ctrl = [i for i in range(17) if int(model.actuator_trnid[i,0]) in arm]
    if not arm_ctrl:
        raise ValueError('The inline controller trace does not identify any arm actuator')
    return dict(arm_qpos=sorted(int(model.jnt_qposadr[i]) for i in arm), arm_ctrl=arm_ctrl,
                observed_ctrl_names=ctl[:17], unavailable_ctrl_names=ctl[17:])


def first_difference(a, b, indices, name):
    delta = np.abs(a[name][:,indices]-b[name][:,indices])
    hit = np.flatnonzero(np.max(delta,axis=1) > DIAGNOSTIC_ATOL)
    if not len(hit):
        return None
    i = int(hit[0])
    j = int(np.argmax(delta[i]))
    return dict(step=i+1, index=indices[j], absolute_difference=float(delta[i,j]),
                before_first_task_query=i+1<START)


def audit(campaign, summary):
    authority = read_pin(read(campaign/'EXECUTION_SPEC.json')['sim_only_project_authorization'])
    fixtures = read_pin(authority['fixtures'])
    records = {v['id']:v for v in read(campaign/'RECORDS.private.json')}
    expected = [f'MEV1-ENG-N1-SHORT-{n:02d}' for n in range(6)]
    if ([f['slot']['id'] for f in fixtures[:6]] != expected or any(cid not in records for cid in expected)
            or mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the original six engineering pairs and isolated reader')
    outputs = []
    now = time.monotonic()
    with forbid_integration(mujoco) as attempts:
        for f in fixtures[:6]:
            cid = f['slot']['id']
            hist = Path(f['recording']['path'])
            if digest(hist) != f['recording']['sha256']:
                raise PermissionError('Original 50 ms anchor recording changed')
            previous, current = hist.parents[1], campaign/'cases'/cid
            legacy = previous/'RESULT.private.json'
            old_spec, new_spec = (read(p/'EXECUTION_SPEC.json') for p in (previous,current))
            model = Path(new_spec['model_path'])
            amount = digest(model)
            if (amount != new_spec['frozen_files'][str(model)]
                    or amount != old_spec['frozen_files'][old_spec['model_path']]
                    or old_spec['mobile_condition']['reset'] != new_spec['mobile_condition']['reset']
                    or old_spec['mobile_condition']['seed'] != new_spec['mobile_condition']['seed']):
                raise PermissionError('Cannot compare mismatched original scenes or resets')
            pins = [trace_pin(previous,dict(path=str(legacy),sha256=digest(legacy)) if legacy.is_file() else None),
                    trace_pin(current,records[cid]['terminal'])]
            a,b = (snapshots(Path(p['path'])) for p in pins)
            axes = columns(mujoco.MjModel.from_binary_path(str(model)))
            control = first_difference(a,b,axes['arm_ctrl'],'prior_ctrl')
            qual = records[cid]['measurement']['status']
            outputs.append(dict(condition=cid, historical=f['old_id'], qualification=qual,
                trace_files=pins, model_sha256=amount, arm_controls=axes,
                first_base_position_difference=first_difference(a,b,list(range(3)),'qpos'),
                first_arm_position_difference=first_difference(a,b,axes['arm_qpos'],'qpos'),
                first_observed_arm_input_difference=control,
                first_base_velocity_difference=first_difference(a,b,list(range(6)),'qvel'),
                previous_arm_input_at_task_start=a['ctrl'][START-1,axes['arm_ctrl']].tolist(),
                current_arm_input_at_task_start=b['ctrl'][START-1,axes['arm_ctrl']].tolist(),
                base_gap_at_task_start_mm=float(1000*np.linalg.norm(a['qpos'][START-1,:3]-b['qpos'][START-1,:3])),
                recorded_leg_policy_first_applied=dict(previous=a['leg_policy_first_applied'],current=b['leg_policy_first_applied']),
                controller_selected_policy='NOT_EXPORTED_BY_FROZEN_CONTROLLER'))
    if attempts:
        raise RuntimeError('Readback attempted a new native integration')
    result = dict(status='ORIGINAL_N1_STARTUP_TIMELINE_RECONCILED',pairs=outputs,
                  masks=dict(absolute_difference_for_diagnostic_only=DIAGNOSTIC_ATOL,original_tolerance_unchanged=True),
                  native_states_per_trace=END, first_query_tick=START, seconds=time.monotonic()-now,
                  new_native_requests=0,new_native_integrations=0,new_model_generations=0,new_optimizer_updates=0)
    with summary.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
    return result


if __name__ == '__main__':
    opts = argparse.ArgumentParser(description=__doc__)
    opts.add_argument('campaign',type=Path)
    opts.add_argument('summary',type=Path)
    args = opts.parse_args()
    value = audit(args.campaign,args.summary)
    print(json.dumps(dict(status=value['status'],receipt=str(args.summary),sha256=digest(args.summary))))
