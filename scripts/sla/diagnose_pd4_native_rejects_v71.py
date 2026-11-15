"""Inspect the first two saved PD4 FINAL joint rejects, without integration.

Exactly one original-certifier clone per closed request; no new inference,
alternative IK setting, candidate replacement or native execution permission.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time

import numpy as np

PINS = {
    'sla-pd4-student-train4-v71-20261115-01-train01-pd4_final_plan-r16':
        '13f8a0af2363fcbc457a567ce021da7594432db3267808f94d1afd8bd6f1199e',
    'sla-pd4-student-train4-v71-20261115-01-train04-pd4_final_plan-r16':
        '4c4230294f19d2d2eff9998bfe81e716ea59547b6e00aa8378d64d404e096504',
}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acceleration_partitions(rates):
    """Prioritize any history-containing triple over a prefix center label."""
    acceleration=np.asarray(rates['acceleration_rad_s2'])
    first=np.arange(len(acceleration))-rates['history_points']
    center=first+1
    return {label:float(np.max(np.abs(acceleration[mask]))) for label,mask in (
        ('reference_join_triples_with_history',first<0),
        ('H40_prefix_center1_9_without_history',(first>=0)&(center<10)),
        ('H40_suffix_center10_38',center>=10)) if np.any(mask)}


def run(group, output):
    import mujoco
    from scripts.pi0294 import task_path_model as original
    from scripts.pi0294.task_state import ActualState, parse_state
    from scripts.sla.audit_query_feasibility_v7 import rejected_joint_diagnostic
    from scripts.sla.diagnose_ik_precision_v71 import ORIGINAL_SHA
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write

    group, output = Path(group), Path(output)
    assert mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None
    assert sha(inspect.getsourcefile(original.certify_chunk)) == ORIGINAL_SHA
    assert sha(group/'EXECUTION_SPEC.json') == 'a07b1071288af69b328360240ccdd96f6814011a237f1f6d1afe8be72984643f'
    outcomes = {r['native_run']:r for r in read(group/'OUTCOMES.private.json')}
    output.mkdir(exist_ok=False)
    started, cpu = time.time(), time.process_time()
    attempted = completed = native_models_loaded = 0
    rows = []
    try:
        for name, pin in PINS.items():
            root = group/'runs'/name; native = root/'native-01'
            row = outcomes[name]
            assert row['result_sha256'] == sha(native/'RESULT.json') == pin
            assert row['technical_valid'] and not row['task_success'] and row['actual_SLA_queries'] == 1
            assert row['first_failure'] == 'ValueError: ORIGINAL_JOINT_ACCELERATION_REJECT'
            for actor in (root,root.with_name(root.name+'-gpu')):
                assert read(actor/'control/LEASE.json')['status'] == 'CLOSED'
            spec = read(root/'EXECUTION_SPEC.json'); result = read(native/'RESULT.json')
            assert result['shutdown_clean'] and all(p['exit_code']==0 for p in result['owned_process_exits'])
            assert sha(spec['model_path']) == result['scene_sha256']
            model = mujoco.MjModel.from_binary_path(spec['model_path'])
            native_models_loaded += 1
            capture = read(native/'TASK_LEARNING_CAPTURES.private.json')['records'][0]
            state = Path(capture['path']).parent/'STATE.private.bin'
            assert sha(state) == capture['state_sha256']
            with forbid_integration(mujoco) as calls:
                sample = ActualState(model).update(parse_state(state.read_bytes()))
            assert not calls and sample['native_step'] == capture['native_step'] == 5703
            decision_path = native/'TASK_LEARNING_DECISION_000.private.json'
            decision = read(decision_path)
            assert decision['index'] == 0 and not decision['SLA_raw_H40_consumed']
            assert not decision['native_execution_authorized'] and not decision['teacher_fallback']
            assert decision['command_clips'] == 0
            assert np.allclose(sample['tcp'],decision['original_actual_TCP'],rtol=0,atol=1e-9)
            query = root.with_name(root.name+'-gpu')/'fit-01/queries/00'
            assert sha(query/'ACTION.npz') == read(query/'RESULT.json')['action_sha256'] == decision['action_sha256']
            command_path = native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
            assert not command_path.exists() or not read(command_path)['records']
            events = [json.loads(x) for x in (native/'COMMANDS.private.jsonl').read_text().splitlines()]
            events = [e for e in events if e.get('native_step')==5703
                and e.get('phase')=='reference-certificate-rejected' and 'action_EEF10' in e]
            assert len(events) == 1
            assert events[0]['reason'] == 'ORIGINAL_JOINT_ACCELERATION_REJECT'
            installed = np.asarray(events[0]['action_EEF10']); assert installed.shape == (40,10)
            preparation = native/'STABLE_TASK_CYCLES/-1/RESULT.private.json'
            past = read(preparation)['windows'][-2:]
            assert [w['native_end_step'] for w in past] == [5678,5703]
            from scripts.pi0292.eval_contract import matrix10
            history = [dict(end_native_step=w['native_end_step'],
                pose=matrix10(w['actual_installed_consumed_EEF10_BASE7'][:10])) for w in past]
            intent = dict(native_run=name,query=0,state_sha256=sha(state),native_result_sha256=pin,
                decision_sha256=sha(decision_path),command_journal_sha256=sha(native/'COMMANDS.private.jsonl'),
                preparation_sha256=sha(preparation),additional_certifier_calls=1,automatic_retry=False)
            attempted += 1
            with (output/'CLONE_CALLS.private.jsonl').open('a') as f:
                f.write(json.dumps(dict(intent,event='INTENT'))+'\n')
            before = [sample[k].copy() for k in ('qpos','qvel','ctrl')]
            diagnostic = rejected_joint_diagnostic(model,sample,installed,history,events[0]['stage'])
            completed += 1
            assert all(np.array_equal(v,sample[k]) for v,k in zip(before,('qpos','qvel','ctrl')))
            assert diagnostic['status'] == 'REJECT_OFFLINE_CLONE_ONLY'
            assert diagnostic['error'] == 'ORIGINAL_JOINT_ACCELERATION_REJECT'
            rates = diagnostic['joint_rates']; partitions=acceleration_partitions(rates)
            joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'x5_joint{i}') for i in range(1,7)]
            assert all(i>=0 for i in joints)
            saved = dict(intent,diagnostic=diagnostic,maximum_absolute_acceleration_rad_s2=partitions,
                acceleration_partition_index='HISTORY_TRIPLES_FIRST; then H40 centers1..9 prefix and10..38 suffix',
                actual_arm_qpos=sample['qpos'][model.jnt_qposadr[joints]].tolist(),
                full_WBC_state_restored=False,original_native_result_unchanged=True)
            rows.append(saved)
            with (output/'ROWS.private.jsonl').open('a') as f:f.write(json.dumps(saved,allow_nan=False)+'\n')
            with (output/'CLONE_CALLS.private.jsonl').open('a') as f:
                f.write(json.dumps(dict(intent,event='COMPLETED',result=diagnostic['error']))+'\n')
        result = dict(schema='SLA_PD4_FIRST_TWO_FINAL_JOINT_REJECT_DIAGNOSTICS_V71',rows=rows,
            clone_certifier_attempts=attempted,clone_certifier_calls_completed=completed,
            original_certifier_sha256=ORIGINAL_SHA,new_model_generations=0,SLA_model_loads=0,
            native_kinematic_model_loads=native_models_loaded,
            optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0,
            formal_results_or_thresholds_changed=False,full_WBC_state_restored=False,
            same_state_expert_final_plan_pairs=0,wall_s=time.time()-started,CPU_s=time.process_time()-cpu)
        write(output/'RESULT.private.json',result)
        return result
    finally:
        write(output/'COUNTERS.private.json',dict(clone_certifier_attempts=attempted,
            clone_certifier_calls_completed=completed,completed_rows=len(rows),new_model_generations=0,
            native_kinematic_model_loads=native_models_loaded,SLA_model_loads=0,
            optimizer_updates=0,physical_requests=0,physics_integrations=0))


if __name__ == '__main__':
    def interrupted(*args):raise InterruptedError('FINITE_OWNED_CPU_DIAGNOSTIC_STOP')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('group');p.add_argument('output');a=p.parse_args()
    result=run(a.group,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
