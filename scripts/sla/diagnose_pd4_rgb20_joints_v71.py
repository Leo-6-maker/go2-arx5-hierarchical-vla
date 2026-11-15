"""Certify only the new64 RGB20 outputs at the same saved expert contexts.

The completed81-call RGB10/label baseline is reused, not executed again.
This CPU diagnostic does not load SLA, integrate a world or authorize actions.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time

import numpy as np

from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS

OPERATION='SLA_PD4_RGB20_SAVED_CONTEXT_MAX64_CLONES_V71'
PROBES=[0,61,124,185,22,82,148,208,32,93,2,81,147,213,23,57]
BASELINE_SHA='571cb1b9f9dfa0fbd7538442c82015a4621db95678dba8e9236525cca26411de'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def admission(s):
    if (s.get('operation')!=OPERATION or s.get('indices')!=PROBES or s.get('noise_ids')!=[0,1]
            or s.get('planned_variants')!=64 or s.get('maximum_clone_calls')!=64
            or s.get('sampling_steps')!=20 or s.get('baseline64_rerun')is not False
            or s.get('label_checks_rerun')is not False or s.get('new_model_generations')!=0
            or s.get('optimizer_updates')!=0 or s.get('physics')is not False
            or s.get('native_execution_admitted')is not False):
        raise PermissionError('ONLY_NEW64_RGB20_CLONES_REUSE_EXISTING81_NO_REGENERATION')


def passed(r):
    return bool(r['diagnostic']and r['diagnostic']['status']=='PASS_OFFLINE_CLONE_ONLY')


def pair(before,after):
    return 'BOTH_PASS'if before and after else'ONLY10_PASS'if before else'ONLY20_PASS'if after else'BOTH_FAIL'


def run(s,output):
    admission(s)
    import mujoco
    from scripts.pi0294.task_state import ActualState,parse_state
    from scripts.pi0292.eval_contract import matrix10
    from scripts.sla.audit_query_feasibility_v7 import rejected_joint_diagnostic
    from scripts.sla.diagnose_ik_precision_v71 import ORIGINAL_SHA,original
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
    from scripts.pi0292.durable import write
    assert mujoco.__version__=='3.2.7'and mujoco.get_mjcb_control()is None
    assert sha(inspect.getsourcefile(original.certify_chunk))==ORIGINAL_SHA
    baseline=Path(s['baseline81']);assert sha(baseline)==BASELINE_SHA
    old=read(baseline);assert old['clone_certifier_calls_completed']==81 and len(old['rows'])==96
    old_rows={(r['view'],r['index'],r['noise']):r for r in old['rows']if r['kind']=='ORIGINAL_RGB10_PREDICTION'}
    assert len(old_rows)==64
    prediction=Path(s['sampling_result']['path']);assert sha(prediction)==s['sampling_result']['sha256']
    new=read(prediction)
    assert(new['actual_generations'],new['model_executions_completed'],new['generation_attempts'])==(64,64,64)
    assert(new['formal_sampler_steps'],new['diagnostic_sampler_steps'])==(10,20)
    assert new['fixed_indices']==PROBES and new['noise_ids']==[0,1]and new['optimizer_updates']==0
    expected=[(v,i,n)for v in MANIFESTS for i in PROBES for n in(0,1)]
    assert [(r['view'],r['index'],r['noise'])for r in new['rows']]==expected
    new_rows={(r['view'],r['index'],r['noise']):r for r in new['rows']}
    owner=Path(s['PD4_owner']);assert sha(owner/'RESULT.private.json')=='c0b2037ffe1fbef373724b12a3e3c4fe50e3448d3163df85c2175fb5a8736220'
    arms={r['view']:r for r in read(owner/'RESULT.private.json')['completed_arms']}
    data_root=Path(read(Path(arms['PD4_FINAL_PLAN']['root'])/'EXECUTION_SPEC.json')['dataset'])
    assert sha(data_root/'MANIFEST.private.json')==MANIFESTS['PD4_FINAL_PLAN'];manifest=read(data_root/'MANIFEST.private.json')
    output=Path(output);output.mkdir(exist_ok=False);started,cpu=time.time(),time.process_time()
    attempted=completed=loads=0;rows=[];contexts={}
    try:
        for index in PROBES:
            ep=manifest['episodes'][index//61];anchor=index%61;source=Path(ep['source_root']);native=source/'native-01'
            assert sha(source/'EXECUTION_SPEC.json')==ep['source_spec_sha256']
            assert sha(native/'RESULT.json')==ep['source_native_result_sha256']
            assert read(source/'control/LEASE.json')['status']=='CLOSED'
            spec=read(source/'EXECUTION_SPEC.json');native_result=read(native/'RESULT.json')
            if ep['id']not in contexts:
                assert sha(spec['model_path'])==native_result['scene_sha256']
                model=mujoco.MjModel.from_binary_path(spec['model_path']);loads+=1
                p=native/'FINAL_PLAN_CAPTURE_ACTUAL.private.json';assert sha(p)==ep['final_plan_summary_sha256']
                labels=data_root/ep['labels']['file'];assert sha(labels)==ep['labels']['sha256']
                with np.load(labels,allow_pickle=False)as z:values={k:z[k].copy()for k in('state_eef10','phase')}
                past=list(read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
                past.extend(dict(r,actual_installed_consumed_EEF10_BASE7=r['installed_consumed_EEF10'])for r in read(native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json')['records'])
                contexts[ep['id']]=dict(model=model,values=values,plans=read(p)['plans'],past=past,
                    captures=read(native/'TASK_LEARNING_CAPTURES.private.json')['records'])
            c=contexts[ep['id']];capture=c['captures'][anchor];state_file=Path(capture['path']).parent/'STATE.private.bin'
            state_sha=sha(state_file);assert state_sha==capture['state_sha256']
            with forbid_integration(mujoco)as calls:sample=ActualState(c['model']).update(parse_state(state_file.read_bytes()))
            assert not calls and sample['native_step']==capture['native_step']==5703+250*anchor
            state=c['values']['state_eef10'][anchor];frame=np.asarray(spec['task_phase_source_from_target'])
            assert np.allclose(matrix10(state),frame@sample['tcp'],rtol=0,atol=1e-9)
            past=[r for r in c['past']if r['native_end_step']<=sample['native_step']][-2:]
            assert [r['native_end_step']for r in past]==[sample['native_step']-25,sample['native_step']]
            history=[dict(end_native_step=r['native_end_step'],pose=matrix10(r['actual_installed_consumed_EEF10_BASE7'][:10]))for r in past]
            plan=c['plans'][anchor];assert plan['native_step']==sample['native_step'];base=np.asarray(plan['base_pose_hold'])
            phase=str(c['values']['phase'][anchor]);stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(phase,'approach')
            for view in MANIFESTS:
                for noise in(0,1):
                    before=old_rows[view,index,noise];record=new_rows[view,index,noise]
                    assert before['state_sha256']==state_sha and before['source_spec_sha256']==ep['source_spec_sha256']
                    assert before['source_native_result_sha256']==ep['source_native_result_sha256']and before['stage']==stage
                    assert record['file']==f'{view}/{index:04d}-{noise}.npz'
                    action=prediction.parent/record['file'];action_sha=sha(action)
                    with np.load(action,allow_pickle=False)as z:
                        assert hashlib.sha256(z['physical_H40'].tobytes()).hexdigest()==record['physical_sha256']
                        assert hashlib.sha256(z['normalized'].tobytes()).hexdigest()==record['normalized_sha256']
                    program=diagnostic=None;failure=None
                    try:program,_=compile_proposal(action,action_sha,source_from_target=frame,query_state10=state,
                        actual_TCP=sample['tcp'],consumed_reference_TCP=history[-1]['pose'],base_hold=base)
                    except ValueError as error:failure=str(error)
                    assert (program is not None)==record['EEF_full_H40_pass']
                    identity=dict(view=view,index=index,noise=noise,state_sha256=state_sha)
                    if program is not None:
                        attempted+=1;assert attempted<=64
                        with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='INTENT',attempt=attempted))+'\n')
                        snapshot=[sample[k].copy()for k in('qpos','qvel','ctrl')]
                        diagnostic=rejected_joint_diagnostic(c['model'],sample,program.reference_path,history,stage);completed+=1
                        assert all(np.array_equal(a,sample[k])for a,k in zip(snapshot,('qpos','qvel','ctrl')))
                        with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='COMPLETED',call=completed,status=diagnostic['status']))+'\n')
                    row=dict(identity,phase=phase,stage=stage,action_file_sha256=action_sha,physical_sha256=record['physical_sha256'],
                        raw_compiler_pass=program is not None,raw_compiler_failure=failure,diagnostic=diagnostic,
                        baseline10_EEF_pass=before['raw_compiler_pass'],baseline10_joint_pass=passed(before),
                        source_spec_sha256=ep['source_spec_sha256'],full_WBC_state_restored=False)
                    rows.append(row)
                    with(output/'ROWS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        assert len(rows)==64 and attempted==completed<=64
        summaries={}
        for view in MANIFESTS:
            group=[r for r in rows if r['view']==view]
            summaries[view]=dict(variants=32,baseline10_EEF_pass=sum(r['baseline10_EEF_pass']for r in group),
                baseline10_joint_pass=sum(r['baseline10_joint_pass']for r in group),
                diagnostic20_EEF_pass=sum(r['raw_compiler_pass']for r in group),diagnostic20_joint_pass=sum(passed(r)for r in group),
                paired_joint_classification=dict(Counter(pair(r['baseline10_joint_pass'],passed(r))for r in group)))
        result=dict(schema=OPERATION,rows=rows,summaries=summaries,recorded_windows=16,variants=64,
            clone_certifier_attempts=attempted,clone_certifier_calls_completed=completed,native_kinematic_model_loads=loads,
            original81_calls_reused_not_rerun=True,baseline81_sha256=BASELINE_SHA,sampling_result_sha256=sha(prediction),
            new_model_generations=0,SLA_model_loads=0,optimizer_updates=0,training_visits=0,physical_requests=0,
            physics_integrations=0,formal_results_or_thresholds_changed=False,full_WBC_state_restored=False,
            historical_student_state_expert_final_plan_pairs=0,original_certifier_sha256=ORIGINAL_SHA,
            wall_s=time.time()-started,CPU_s=time.process_time()-cpu)
        write(output/'RESULT.private.json',result);return result
    finally:
        write(output/'COUNTERS.private.json',dict(clone_certifier_attempts=attempted,clone_certifier_calls_completed=completed,
            completed_rows=len(rows),native_kinematic_model_loads=loads,new_model_generations=0,
            optimizer_updates=0,physical_requests=0,physics_integrations=0))


if __name__=='__main__':
    def interrupted(*args):raise InterruptedError('FINITE_RGB20_CLONE_DIAGNOSTIC_STOP')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec');p.add_argument('output');a=p.parse_args()
    result=run(read(a.spec),a.output);print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
