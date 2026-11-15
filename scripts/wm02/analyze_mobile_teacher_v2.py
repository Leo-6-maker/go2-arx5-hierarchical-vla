"""Read sealed mobile records; no simulator, controller, learner or certifier call.

Output includes private coordinates and paths. Only aggregate reports may be
published. P9 is the last consumed command row; P10 is its 0.5s wire endpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def xyz(p):
    return np.asarray(p, float)[:3, 3]


def norm(v):
    return float(np.linalg.norm(v))


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def within_span(segment,goal,key):
    first=None;previous=None;maximum=0.
    for row in segment:
        position=xyz(row[key])if key=='actual_TCP'else np.asarray(row[key])[:3]
        if norm(position-goal)<=.004:
            if first is None or previous!=row['native_start_step']:first=row['native_end_step']
            maximum=max(maximum,(row['native_end_step']-first)*.002)
            previous=row['native_end_step']
        else:first=None;previous=None
    return maximum


def analyze_case(path):
    spec = read(path)
    root = Path(path).parent
    if not (root / 'RESULT.private.json').exists():
        return None
    result = read(root / 'RESULT.private.json')
    assert read(root / 'control/LEASE.json')['status'] == 'CLOSED'
    assert result['optimizer_updates'] == result['training_visits'] == 0
    out = root / 'native-01'
    commands = rows(out / 'COMMANDS.private.jsonl')
    samples = rows(out / 'MOBILE_SAMPLES.private.jsonl')
    before = {r['native_end_step']: r for r in samples}
    relevant = [r for r in samples if r['native_start_step'] >= 5703]
    proposals = {r['wire_sha256']: r for r in commands if r['phase'] == 'proposed'}
    initial = before[5703]
    cube = xyz(initial['object_pose'])
    rotation = np.asarray(initial['actual_TCP'])[:3, :3]
    pregrasp = cube + [0, -.0017, .080]
    descend = cube + [0, -.0017, .012] + rotation @ [-.012, 0, 0]
    qevents = [r for r in commands if r['phase'] in
               ('mobile-expert-query', 'F4-real-expert-prefix', 'mobile-task-expert','mobile-continuous-expert-query',
                'mobile-student-query', 'mobile-task-student')]
    model_responses = sorted((out / 'queries').glob('*/RESPONSE.private.json'))
    by_query = {e['query']: e for e in qevents}
    # Rejected student proposals have a response but no accepted-query event.
    query_ids = sorted(set(by_query) | {int(p.parent.name) for p in model_responses})
    queries = []
    for q in query_ids:
        e = by_query.get(q)
        response = out / 'queries' / f'{q:02d}' / 'RESPONSE.private.json'
        if response.exists():
            meta = read(response.parent / 'INPUT_META.private.json')
            start = meta['native_step']
        else:
            start = 5703 + q * 250
        prior = before.get(start)
        if prior is None:
            raise ValueError('MISSING_QUERY_BOUNDARY:' + root.name + ':' + str(q))
        actual0 = xyz(prior['actual_TCP'])
        reference0_logged = np.asarray(prior['installed_EEF10'])[:3]
        boundary_reference = np.asarray(prior['target_endpoint_EEF10'])[:3]
        if e and 'submitted_policy_H40' in e:
            raw = np.asarray(e['submitted_policy_H40'])
        elif e and 'mapped_target_H40' in e:
            raw = np.asarray(e['mapped_target_H40'])
        else:
            r = read(response)
            assert sha(r['action']) == r['action_sha256']
            from scripts.pi0292.eval_contract import matrix10
            frame = np.asarray(spec['policy_from_world'])
            with np.load(r['action'], allow_pickle=False) as packet:
                transforms = np.linalg.inv(frame) @ matrix10(packet['physical_H40'])
            raw = np.zeros((40, 10))
            raw[:, :3] = transforms[:, :3, 3]
        goal = np.asarray(e['privileged_goal']) if e and 'privileged_goal' in e else None
        segment = [r for r in relevant if start <= r['native_start_step'] < start + 250]
        errors = [norm(xyz(r['actual_TCP']) - np.asarray(r['target_endpoint_EEF10'])[:3])
                  for r in segment]
        gripper = [r['qpos'] for r in segment]
        target = [np.asarray(r['target_endpoint_EEF10'])[:3] for r in segment]
        last = segment[-1] if segment else prior
        wires = [proposals[r['installed_reference_sha256']] for r in segment]
        installed_query = np.asarray(wires[0]['action_EEF10']) if wires else None
        base_errors, preview_spans = [], []
        for r, wire in zip(segment, wires):
            doc = json.loads(wire['wire'])
            base = doc.get('base')
            if base:
                points = base['waypoints']
                cursor = (r['native_end_step'] - wire['native_step']) // 25
                base_errors.append(norm(np.asarray(points[cursor]['position']) - xyz(r['measured_MPC_mount'])))
                preview_spans.append(max(norm(np.asarray(p['position']) - points[0]['position']) for p in points))
        record = dict(query=q, native_start_step=start,
                      phase=e['phase'] if e else 'STUDENT_PROPOSAL_REJECTED',
                      stage=e.get('stage') if e else None,
                      goal_basis='ORIGINAL_TEACHER_REQUEST' if goal is not None else 'NO_PRIVILEGED_STUDENT_GOAL',
                      actual_TCP_start=actual0.tolist(), actual_TCP_end=xyz(last['actual_TCP']).tolist(),
                      reference_start_logged=reference0_logged.tolist(),
                      wire_boundary_reference=boundary_reference.tolist(),
                      reference_boundary_minus_logged_m=norm(boundary_reference - reference0_logged),
                      requested_H40_end_displacement_m=norm(raw[39, :3] - raw[0, :3]),
                      requested_P9_displacement_m=norm(raw[9, :3] - raw[0, :3]),
                      requested_P10_displacement_m=norm(raw[10, :3] - raw[0, :3]),
                      actual_query_displacement_m=norm(xyz(last['actual_TCP']) - actual0),
                      confirmed_points=len(segment),
                      tracking_endpoint_RMS_m=rms(errors),
                      tracking_endpoint_maximum_m=max(errors, default=None),
                      sampled_native_grasp_points=sum(bool(r['native_grasp']) for r in segment),
                      installed_gripper_min=min((r['installed_EEF10'][9] for r in segment), default=None),
                      installed_gripper_max=max((r['installed_EEF10'][9] for r in segment), default=None),
                      actual_BASE_displacement_m=norm(xyz(last['actual_mount']) - xyz(prior['actual_mount'])),
                      BASE_reference_present=bool(base_errors), BASE_tracking_RMS_m=rms(base_errors),
                      maximum_BASE_preview_span_m=max(preview_spans, default=0),
                      closest_pregrasp_actual_m=min([norm(xyz(r['actual_TCP'])-pregrasp) for r in segment] or [norm(actual0-pregrasp)]),
                      closest_descend_actual_m=min([norm(xyz(r['actual_TCP'])-descend) for r in segment] or [norm(actual0-descend)]),
                      actual_pregrasp_within4mm_points=sum(norm(xyz(r['actual_TCP'])-pregrasp)<=.004 for r in segment),
                      actual_descend_within4mm_points=sum(norm(xyz(r['actual_TCP'])-descend)<=.004 for r in segment))
        if goal is not None:
            record.update(privileged_goal=goal.tolist(), query_goal_distance_m=norm(goal-actual0),
                          query_end_goal_distance_m=norm(goal-xyz(last['actual_TCP'])),
                          H40_endpoint_goal_distance_m=norm(goal-raw[39,:3]),
                          consumed_reference_end_goal_distance_m=norm(goal-target[-1]) if target else None,
                          closest_consumed_reference_to_goal_m=min((norm(goal-p) for p in target), default=None),
                          actual_goal_progress_m=norm(goal-actual0)-norm(goal-xyz(last['actual_TCP'])),
                          installed_reference_goal_progress_m=norm(goal-boundary_reference)-norm(goal-target[-1]) if target else None,
                          closest_actual_to_goal_m=min([norm(goal-xyz(r['actual_TCP']))for r in segment]or[norm(goal-actual0)]),
                          actual_goal_within4mm_points=sum(norm(goal-xyz(r['actual_TCP']))<=.004 for r in segment),
                          reference_goal_within4mm_points=sum(norm(goal-p)<=.004 for p in target),
                          longest_sampled_actual_goal_within4mm_s=within_span(segment,goal,'actual_TCP'),
                          longest_sampled_reference_goal_within4mm_s=within_span(segment,goal,'target_endpoint_EEF10'),
                          installed_H40_endpoint_goal_distance_m=norm(goal-installed_query[-1,:3]) if installed_query is not None else None,
                          installed_query_boundary_jump_m=norm(installed_query[0,:3]-boundary_reference) if installed_query is not None else None,
                          teacher_15mm_cap=norm(raw[39,:3]-raw[0,:3])>=.015-1e-9)
        queries.append(record)
    # Native contact bytes are original, not contacts recomputed by forward/IK.
    import mujoco
    model = mujoco.MjModel.from_binary_path(spec['model_path'])
    assert sha(spec['model_path']) == spec['frozen_files'][spec['model_path']]
    pads = [int(model.geom('task_'+s+'_pad').id) for s in ('left','right')]
    finger_q = [int(model.jnt_qposadr[model.joint('task_'+s+'_slide').id]) for s in ('left','right')]
    cube_geom = int(model.geom('task_cube_geom').id)
    for q in queries:
        q.update(original_left_pad_contact_ticks=0, original_right_pad_contact_ticks=0,
                 original_bilateral_pad_contact_ticks=0, actual_gripper_min=None, actual_gripper_max=None)
    q_index = {q['native_start_step']+k: q for q in queries for k in range(1,251)}
    journal = out/'state.bin.journal'
    journal_hash = hashlib.sha256()
    packet_count = 0
    with journal.open('rb') as f:
        while size := f.read(4):
            n, = struct.unpack('<I', size)
            blob = f.read(n)
            assert len(blob)==n
            journal_hash.update(size);journal_hash.update(blob)
            magic,step,t,nq,nv,nu=struct.unpack_from('<8sQdIII',blob)
            packet_count+=1
            assert magic==b'GRTSTATE' and step==packet_count and abs(t-step*.002)<1e-7
            q=q_index.get(step)
            if q is None:continue
            values=np.frombuffer(blob,dtype='<f8',count=nq,offset=36)
            opening=float(values[finger_q].mean()/.044)
            q['actual_gripper_min']=opening if q['actual_gripper_min']is None else min(opening,q['actual_gripper_min'])
            q['actual_gripper_max']=opening if q['actual_gripper_max']is None else max(opening,q['actual_gripper_max'])
            offset=36+8*(nq+nv+nu);nc,=struct.unpack_from('<I',blob,offset);offset+=4;contact=[False,False]
            for _ in range(nc):
                a,b,dist,force=struct.unpack_from('<IIdd',blob,offset)
                for i,pad in enumerate(pads):contact[i]|={a,b}=={pad,cube_geom}and force>=.05
                offset+=160
            q['original_left_pad_contact_ticks']+=contact[0]
            q['original_right_pad_contact_ticks']+=contact[1]
            q['original_bilateral_pad_contact_ticks']+=all(contact)
    assert packet_count==result['actual_native_integrations']
    assert result['model_generations']==len(model_responses)
    errors=[norm(xyz(r['actual_TCP'])-np.asarray(r['target_endpoint_EEF10'])[:3])for r in relevant]
    return dict(case=root.name,condition=spec['mobile_condition'],source_sha=spec['source_sha'],
                anchors={str(p.relative_to(root)):sha(p)for p in [Path(path),root/'RESULT.private.json',out/'MOBILE_SAMPLES.private.jsonl',out/'COMMANDS.private.jsonl']},
                original_state_journal_sha256=journal_hash.hexdigest(),
                result=result,queries=queries,
                summary=dict(queries=len(queries),stage_queries=dict(Counter(q['stage']for q in queries if q['stage'])),
                             native_integrations=packet_count,tracking_RMS_m=rms(errors),
                             original_left_pad_contact_ticks=sum(q['original_left_pad_contact_ticks']for q in queries),
                             original_right_pad_contact_ticks=sum(q['original_right_pad_contact_ticks']for q in queries),
                             original_bilateral_pad_contact_ticks=sum(q['original_bilateral_pad_contact_ticks']for q in queries),
                             confirmed_task_or_prefix_points=len(relevant),native_grasp_points=sum(bool(r['native_grasp'])for r in relevant)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+',required=True);args=parser.parse_args()
    start=time.time();cpu=time.process_time();cases=[]
    for group in args.groups:
        for path in read(Path(group)/'EXECUTION_SPEC.json')['cases']:
            value=analyze_case(path)
            if value is not None:cases.append(value)
    print(json.dumps(dict(schema='MOBILE_TEACHER_P10_TIMING_READONLY_V2',cases=cases,
                          read_only=True,new_physical_requests=0,new_integrations=0,new_model_queries=0,
                          new_training_updates=0,new_training_visits=0,new_certifier_calls=0,
                          new_forward_calls=0,wall_s=time.time()-start,CPU_s=time.process_time()-cpu,
                          snapshot_epoch=time.time()),separators=(',',':')))


if __name__=='__main__':main()
