"""Account for real matched MuJoCo starts, scores, grip rejection and contact evidence."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('MAIN-phase00000','MAIN-phase02000','MAIN-phase03800')


def load(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def contact(path):
    with path.open('rb') as stream:
        geoms = {row['id']:row for row in json.loads(stream.readline())['geoms']}
        stream.seek(max(0,path.stat().st_size-1024**2))
        stream.readline()
        records = [json.loads(line) for line in stream]
    bad = next((r for r in records if r.get('contact_stop')),None)
    if bad is None:
        return None
    pairs = {}
    for c in bad['contacts']:
        a,b = geoms[c['geom1']],geoms[c['geom2']]
        pair = tuple(g['name'] or g['body'] for g in (a,b))
        if any(name.startswith('x5_') or 'pad' in name or 'gripper' in name for name in pair):
            pairs[pair] = min(c['dist'],pairs.get(pair,0))
    return dict(time_s=bad['contact_time_s'],arm_pairs=[dict(pair=pair,depth_mm=-1000*d) for pair,d in pairs.items()])


def gripper(root, reject):
    past = root/f"{reject['query']-1:04d}"/'QUERY_ACTUAL.private.json'
    here = root/f"{reject['query']:04d}"
    record = load(here/'QUERY_ACTUAL.private.json')
    action = here/'model-response/ACTION.npz'
    if sha(action) != record['action']['sha256']:
        raise ValueError('Action archive differs from the actual query response')
    with np.load(here/'INPUT.npz',allow_pickle=False) as inp, np.load(action,allow_pickle=False) as proposal:
        previous = float(inp['previous_width_m'])
        age = float(inp['query_native_s']-inp['previous_source_native_s'])
        width = .002+.040*proposal['physical_H40'][:,9]
        elapsed = float(inp['query_native_s'])
        predicted_speed = abs(width[0]-previous)/age
    actual = load(past)['confirmed_intervals'][-1] if past.is_file() else None
    measured = None if actual is None else actual.get('measured_gripper_width_m')
    if actual and previous!=actual['emitted_gripper_width_m']:
        raise ValueError('Previous command is different from the last physically emitted atom')
    if abs(predicted_speed-reject['original_gripper_speed']['first_seam_speed_m_s'])>1e-8:
        raise ValueError('Native seam score and physical command delta differ')
    jumps = np.abs(np.diff(np.r_[previous,width]))/.05*1000
    first = np.flatnonzero(jumps>40+1e-7)
    return dict(query=reject['query']+1,install_ACKs=len(record['install_ACKs']),
                confirmed_atoms=record['confirmed_atoms'],tick=record['native_start_step'],
                age_native_ms=1000*age,time_native_s=elapsed,previous_command_mm=1000*previous,
                measured_width_mm=None if measured is None else 1000*measured,proposal_first_mm=1000*width[0],
                first_seam_mm_s=1000*predicted_speed,max_internal_mm_s=float(jumps[1:].max()),
                first_exceedance=int(first[0]) if first.size else None,
                previous_P10_confirmed_atoms=len(load(past)['confirmed_intervals']) if past.is_file() else 0,
                action_sha256=sha(action))


def case(folder):
    native = folder/'native-01'
    integration = native/'BEFORE_FIRST_INTEGRATION.json'
    ready = native/'g7-queries/0000/INPUT.npz'
    terminal = folder/'RESULT.private.json'
    result = dict(confirmed_start=integration.is_file(),terminal=terminal.is_file())
    if integration.is_file():
        before = load(integration)
        result['initial_actuator'] = {k:before[k] for k in ('q_target','q','dq','kp','kd','tau','ctrl') if k in before}
        result['initial_zero_clock'] = before['sim_time_s']==0
    if ready.is_file():
        with np.load(ready,allow_pickle=False) as data:
            result['first_input'] = data['state_eef10'].tolist()
            result['first_RGB'] = {k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in ('body_rgb','wrist_rgb')}
        meta = load(ready.with_name('INPUT_META.private.json'))
        result['first_query_tick'] = meta['native_step']
        result['initial_leg_motiontime'] = meta['pause_before']['motiontime']
    if not terminal.is_file():
        return result
    row = load(terminal)
    detail = row['detail'] or {}
    score_path = native/'MULTITASK_SCORE_PROGRESS.private.json'
    score = detail.get('score') or (load(score_path) if score_path.is_file() else {})
    stop = row['physics_stops'][0] if row['physics_stops'] else row['mobile_safety_stop']
    events = score.get('events',{})
    result.update(requests=row['physical_requests'],ticks=row['actual_native_integrations'],
                  generations=row['model_generations'],updates=row['optimizer_updates'],visits=row['training_visits'],
                  first_failure=stop['reason'] if stop else detail.get('terminal_reason'),
                  success=bool(row['metrics']['task_and_required_motion_success']),
                  grasp=score.get('ever_target_grasp'),lift20=score.get('ever_target_lift20mm'),
                  lift_mm=1000*score.get('maximum_lift_m',0.) if score else None,
                  hold_s=score.get('longest_grasp_lift_hold_s'),events=events,technical=row['technical_failure'],
                  stop=stop,stop_after_integrations=(row['actual_native_integrations']-stop['steps']) if stop and 'steps' in stop else None,
                  contacts=contact(native/'physics.jsonl') if stop and stop['reason']=='ILLEGAL_CONTACT' else None,
                  result_sha256=sha(terminal))
    reject = detail.get('valid_policy_rejection')
    if reject and reject['kind']=='G7_ACTUAL_PREVIOUS_COMMAND_SPEED_REJECTED':
        result['gripper_rejection'] = gripper(native/'g7-queries',reject)
    return result


def readback(baseline, candidate):
    by_arm = {}
    starts = {}
    for arm,root in zip(ARMS,(baseline,baseline,candidate),strict=True):
        specs = root/'blocks'/f'{arm}-DEV-00'
        panel = specs/'PANEL.private.json'
        rows = {}
        if panel.is_file():
            for item in load(panel):
                rows[item['instruction']] = case(specs/'cases'/item['id'])
        by_arm[arm] = rows
    for task,row in by_arm[ARMS[-1]].items():
        if 'first_input' not in row:
            continue
        starts[task] = {}
        for arm in ARMS[:2]:
            prior = by_arm[arm][task]
            a,b = (np.asarray(r['first_input']) for r in (prior,row))
            starts[task][arm] = dict(actuator_match=prior['initial_actuator']==row['initial_actuator'],
                TCP_distance_mm=float(1000*np.linalg.norm(a[:3]-b[:3])),rotation6_max_delta=float(np.max(abs(a[3:9]-b[3:9]))),
                finger_fraction_delta=float(abs(a[9]-b[9])),same_RGB=prior['first_RGB']==row['first_RGB'])
    return dict(arms=by_arm,initial_comparisons=starts)


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('baseline',type=Path)
    cli.add_argument('candidate',type=Path)
    args = cli.parse_args()
    print(json.dumps(readback(args.baseline,args.candidate),sort_keys=True))
