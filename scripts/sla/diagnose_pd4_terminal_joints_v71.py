"""Complete the original8 terminal-joint attribution with five new clones.

Two prior FINAL results are reused verbatim. No formal condition, model
generation, numerical convergence experiment or physics is run again.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time

import numpy as np

NEW={
    'train04-pd4_executed_ref':(1,'ded3704d29530f80a406f58baea019a78596f6059166d33375601965c651e6a8'),
    'train07-pd4_executed_ref':(0,'c568fe53bdcd1fea3331ee0d9308ad88677f665211e39a77e4acc1ebee138e2d'),
    'train07-pd4_final_plan':(1,'f59726fc86910e59097a9d6eb13f6b1ff098f83c4f5ec3a1f38366cb0feb05fd'),
    'train10-pd4_final_plan':(5,'56a49cb9a7c0cd758de651864e42297487cd9089c0087c957b43bc49c23ec9d3'),
    'train10-pd4_executed_ref':(0,'3174fc942d78f6c61466dec177f3ed4921db92290a2bcae59edaf9d4d869ac44'),
}
PREFIX='sla-pd4-student-train4-v71-20261115-01-'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(spec,output):
    import mujoco
    from scripts.pi0294 import task_path_model as original
    from scripts.pi0294.task_state import ActualState,parse_state
    from scripts.pi0292.eval_contract import matrix10
    from scripts.sla.audit_query_feasibility_v7 import rejected_joint_diagnostic
    from scripts.sla.diagnose_ik_precision_v71 import ORIGINAL_SHA
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write
    assert spec['maximum_new_calls']==5 and spec['physics']is False and spec['optimizer_updates']==0
    assert mujoco.__version__=='3.2.7'and mujoco.get_mjcb_control()is None
    assert sha(inspect.getsourcefile(original.certify_chunk))==ORIGINAL_SHA
    G=Path(spec['group']);prior=Path(spec['prior2'])
    assert sha(G/'RESULT.private.json')=='d9185b7dff07e364368efccd04a1e2d26c382900c6b02000c7a8904d6e6a2401'
    terminal=read(G/'RESULT.private.json');assert terminal['exit_code']==0 and terminal['summary']['terminal']==8
    assert read(G/'control/LEASE.json')['status']=='CLOSED'
    assert sha(prior)=='09071b2dbac16edb34e6be274c4ce8d4d711dbca9dc561847adb57386067fb3d'
    old=read(prior);assert old['clone_certifier_calls_completed']==2 and len(old['rows'])==2
    outcomes={r['native_run']:r for r in terminal['outcomes']};rows=[]
    for r in old['rows']:
        assert outcomes[r['native_run']]['result_sha256']==r['native_result_sha256']
        assert r['diagnostic']['error']=='ORIGINAL_JOINT_ACCELERATION_REJECT'
        rows.append(dict(r,reused_prior_result=True,new_clone_calls=0))
    output=Path(output);output.mkdir(exist_ok=False);start,cpu=time.time(),time.process_time()
    attempted=completed=loads=0;models={}
    try:
        for suffix,(query,pin)in NEW.items():
            name=PREFIX+suffix+'-r16';row=outcomes[name];R=G/'runs'/name;N=R/'native-01';A=R.with_name(R.name+'-gpu')
            assert row['result_sha256']==sha(N/'RESULT.json')==pin
            assert row['technical_valid']and row['actual_SLA_queries']==query+1
            assert row['first_failure']=='ValueError: ORIGINAL_JOINT_ACCELERATION_REJECT'
            for actor in(R,A):assert read(actor/'control/LEASE.json')['status']=='CLOSED'
            ns=read(R/'EXECUTION_SPEC.json');native=read(N/'RESULT.json');assert native['shutdown_clean']
            assert sha(ns['model_path'])==native['scene_sha256'];key=(ns['model_path'],native['scene_sha256'])
            if key not in models:models[key]=mujoco.MjModel.from_binary_path(ns['model_path']);loads+=1
            model=models[key];capture=read(N/'TASK_LEARNING_CAPTURES.private.json')['records'][query]
            state=Path(capture['path']).parent/'STATE.private.bin';assert sha(state)==capture['state_sha256']
            with forbid_integration(mujoco)as forbidden:sample=ActualState(model).update(parse_state(state.read_bytes()))
            step=5703+250*query;assert not forbidden and sample['native_step']==capture['native_step']==step
            decision_path=N/f'TASK_LEARNING_DECISION_{query:03d}.private.json';decision=read(decision_path)
            assert decision['index']==query and np.allclose(sample['tcp'],decision['original_actual_TCP'],rtol=0,atol=1e-9)
            action=A/'fit-01/queries'/f'{query:02d}'/'ACTION.npz';assert sha(action)==decision['action_sha256']
            assert not decision['teacher_fallback']and not decision['endpoint_basis_applied']and decision['command_clips']==0
            cp=N/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json';commands=read(cp)['records']if cp.exists()else[]
            assert len(commands)==10*query and not any(c['student_decision_sha256']==sha(decision_path)for c in commands)
            history=list(read(N/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
            history.extend(dict(c,actual_installed_consumed_EEF10_BASE7=c['installed_consumed_EEF10'])for c in commands)
            history=[c for c in history if c['native_end_step']<=step][-2:]
            assert [c['native_end_step']for c in history]==[step-25,step]
            history=[dict(end_native_step=c['native_end_step'],pose=matrix10(c['actual_installed_consumed_EEF10_BASE7'][:10]))for c in history]
            journal=N/'COMMANDS.private.jsonl';events=[json.loads(x)for x in journal.read_text().splitlines()]
            events=[e for e in events if e.get('native_step')==step and e.get('phase')=='reference-certificate-rejected'and'action_EEF10'in e]
            assert len(events)==1 and events[0]['reason']=='ORIGINAL_JOINT_ACCELERATION_REJECT'
            candidate=np.asarray(events[0]['action_EEF10']);assert candidate.shape==(40,10)
            identity=dict(native_run=name,query=query,native_step=step,native_result_sha256=pin,
                state_sha256=sha(state),decision_sha256=sha(decision_path),command_journal_sha256=sha(journal))
            attempted+=1;assert attempted<=5
            with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='INTENT'))+'\n')
            before=[sample[k].copy()for k in('qpos','qvel','ctrl')]
            diagnostic=rejected_joint_diagnostic(model,sample,candidate,history,events[0]['stage']);completed+=1
            assert all(np.array_equal(a,sample[k])for a,k in zip(before,('qpos','qvel','ctrl')))
            assert diagnostic['status']=='REJECT_OFFLINE_CLONE_ONLY'and diagnostic['error']=='ORIGINAL_JOINT_ACCELERATION_REJECT'
            saved=dict(identity,diagnostic=diagnostic,reused_prior_result=False,new_clone_calls=1,
                full_WBC_state_restored=False,original_native_result_unchanged=True)
            rows.append(saved)
            with(output/'ROWS.private.jsonl').open('a')as f:f.write(json.dumps(saved,allow_nan=False)+'\n')
            with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='COMPLETED'))+'\n')
        assert attempted==completed==5 and len(rows)==7
        result=dict(schema='SLA_PD4_ORIGINAL8_TERMINAL_JOINT_ATTRIBUTION_V71',rows=rows,
            prior2_analysis_sha256=sha(prior),prior_clone_calls_reused=2,new_clone_certifier_attempts=attempted,
            new_clone_certifier_calls_completed=completed,total_joint_rejects_diagnosed=7,native_kinematic_model_loads=loads,
            new_SLA_generations=0,SLA_model_loads=0,optimizer_updates=0,training_visits=0,physical_requests=0,
            physics_integrations=0,full_WBC_state_restored=False,same_state_expert_final_plan_pairs=0,
            formal_results_or_thresholds_changed=False,original_certifier_sha256=ORIGINAL_SHA,
            CPU_s=time.process_time()-cpu,wall_s=time.time()-start)
        write(output/'RESULT.private.json',result);return result
    finally:
        write(output/'COUNTERS.private.json',dict(new_clone_certifier_attempts=attempted,
            new_clone_certifier_calls_completed=completed,prior_clone_calls_reused=2,completed_new_rows=len(rows)-2,
            new_SLA_generations=0,optimizer_updates=0,physical_requests=0,physics_integrations=0))


if __name__=='__main__':
    def interrupted(*args):raise InterruptedError('FINITE_OWNED_TERMINAL_DIAGNOSTIC_STOP')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec');p.add_argument('output');a=p.parse_args()
    result=run(read(a.spec),a.output);print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
