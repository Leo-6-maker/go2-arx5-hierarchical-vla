"""Fixed16 recorded PD4 contexts, two label views and original64 RGB outputs.

At most81 original-certifier calls:32 labels plus the49 originally EEF-valid
predictions. No new generation, alternate IK setting or full WBC restoration.
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

from go2arx5_vla.sla_adapter.pd4_config_v71 import MANIFESTS, STATISTICS

OPERATION='SLA_PD4_FIXED16_RECORDED_STATE_ORIGINAL64_CLONES_V71'
PROBES=[0,61,124,185,22,82,148,208,32,93,2,81,147,213,23,57]
METRICS={'PD4_EXECUTED_REF':'95965f8b446fc2b3c13c07264a0a37a8c025546394288e884432fad38984a74b',
    'PD4_FINAL_PLAN':'411345bb7dda0acb81d15bb42a60395b206529e52f0cb1bca57b3ac84c26f5c4'}


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def admission(spec):
    if (spec.get('operation')!=OPERATION or spec.get('indices')!=PROBES
            or spec.get('planned_variants')!=96 or spec.get('maximum_clone_calls')!=81
            or spec.get('original_sampling_steps')!=10 or spec.get('new_model_generations')!=0
            or spec.get('optimizer_updates')!=0 or spec.get('physics')is not False
            or spec.get('native_execution_admitted')is not False):
        raise PermissionError('FIXED16_TWO_LABELS_ORIGINAL64_ONLY_MAX81_CLONES')


def run(spec, output):
    admission(spec)
    import mujoco
    from scripts.pi0294.task_state import ActualState,parse_state
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.sla.audit_query_feasibility_v7 import rejected_joint_diagnostic
    from scripts.sla.diagnose_ik_precision_v71 import ORIGINAL_SHA,original
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
    from scripts.pi0292.durable import write

    assert mujoco.__version__=='3.2.7' and mujoco.get_mjcb_control()is None
    assert sha(inspect.getsourcefile(original.certify_chunk))==ORIGINAL_SHA
    owner=Path(spec['PD4_owner']);assert sha(owner/'RESULT.private.json')=='c0b2037ffe1fbef373724b12a3e3c4fe50e3448d3163df85c2175fb5a8736220'
    terminal=read(owner/'RESULT.private.json');assert terminal['actual_optimizer_updates']==1000
    arms={a['view']:a for a in terminal['completed_arms']}
    output=Path(output);output.mkdir(exist_ok=False)
    started,cpu=time.time(),time.process_time();attempts=calls=models_loaded=0;rows=[];datasets={};contexts={}
    try:
        for view in MANIFESTS:
            arm=Path(arms[view]['root']);assert sha(arm/'RESULT.private.json')==arms[view]['result_sha256']
            s=read(arm/'EXECUTION_SPEC.json');D=Path(s['dataset'])
            assert sha(D/'MANIFEST.private.json')==MANIFESTS[view];m=read(D/'MANIFEST.private.json')
            assert sha(D/m['statistics']['file'])==STATISTICS[view]
            p=arm/'candidate500/PREDICTIONS32.private.json';assert sha(p)==METRICS[view]
            metrics=read(p);assert [(r['index'],r['noise'])for r in metrics]==[(i,n)for i in PROBES for n in(0,1)]
            labels={}
            for e in m['episodes']:
                assert sha(D/e['labels']['file'])==e['labels']['sha256']
                with np.load(D/e['labels']['file'],allow_pickle=False)as z:labels[e['id']]={k:z[k].copy()for k in ('state_eef10','nominal_policy_H40','phase','native_step')}
            datasets[view]=dict(root=D,manifest=m,labels=labels,metrics={(r['index'],r['noise']):r for r in metrics},arm=arm)
        for index in PROBES:
            first=datasets['PD4_FINAL_PLAN'];ep=first['manifest']['episodes'][index//61];anchor=index%61
            source=Path(ep['source_root']);native=source/'native-01'
            assert read(source/'control/LEASE.json')['status']=='CLOSED'
            assert sha(source/'EXECUTION_SPEC.json')==ep['source_spec_sha256']
            s=read(source/'EXECUTION_SPEC.json');assert sha(native/'RESULT.json')==ep['source_native_result_sha256']
            native_result=read(native/'RESULT.json');assert sha(s['model_path'])==native_result['scene_sha256']
            key=str(source)
            if key not in contexts:
                model=mujoco.MjModel.from_binary_path(s['model_path']);models_loaded+=1
                plans_path=native/'FINAL_PLAN_CAPTURE_ACTUAL.private.json';assert sha(plans_path)==ep['final_plan_summary_sha256']
                commands=read(native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json')['records']
                past=list(read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
                past.extend(dict(r,actual_installed_consumed_EEF10_BASE7=r['installed_consumed_EEF10'])for r in commands)
                contexts[key]=dict(model=model,plans=read(plans_path)['plans'],past=past,
                    captures=read(native/'TASK_LEARNING_CAPTURES.private.json')['records'])
            c=contexts[key];capture=c['captures'][anchor];state_path=Path(capture['path']).parent/'STATE.private.bin'
            assert sha(state_path)==capture['state_sha256']
            with forbid_integration(mujoco)as forbidden:sample=ActualState(c['model']).update(parse_state(state_path.read_bytes()))
            assert not forbidden and sample['native_step']==capture['native_step']==5703+250*anchor
            state=first['labels'][ep['id']]['state_eef10'][anchor];frame=np.asarray(s['task_phase_source_from_target'])
            assert np.allclose(matrix10(state),frame@sample['tcp'],rtol=0,atol=1e-9)
            past=[r for r in c['past']if r['native_end_step']<=sample['native_step']][-2:]
            assert [r['native_end_step']for r in past]==[sample['native_step']-25,sample['native_step']]
            history=[dict(end_native_step=r['native_end_step'],pose=matrix10(r['actual_installed_consumed_EEF10_BASE7'][:10]))for r in past]
            plan=c['plans'][anchor];assert plan['native_step']==sample['native_step']
            base=np.asarray(plan['base_pose_hold']);assert base.shape==(7,)
            phase=str(first['labels'][ep['id']]['phase'][anchor]);stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(phase,'approach')
            for view,d in datasets.items():
                other=d['manifest']['episodes'][index//61];data=d['labels'][other['id']]
                assert other['source_root']==ep['source_root']and np.array_equal(data['state_eef10'][anchor],state)
                for noise in (None,0,1):
                    kind='LABEL'if noise is None else'ORIGINAL_RGB10_PREDICTION'
                    action_pin=None;program=None;failure=None
                    if noise is not None:
                        prior=d['metrics'][index,noise];action=d['arm']/'candidate500/work/step500-RGB'/prior['file']
                        action_pin=sha(action)
                        with np.load(action,allow_pickle=False)as z:
                            assert hashlib.sha256(z['physical_H40'].tobytes()).hexdigest()==prior['physical_sha256']
                    try:
                        if noise is None:
                            raw=data['nominal_policy_H40'][anchor]
                            mapped=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9])
                            program=FiftyMsProgram(mapped,sample['tcp'],history[-1]['pose'],base)
                        else:
                            program,_=compile_proposal(action,action_pin,source_from_target=frame,query_state10=state,
                                actual_TCP=sample['tcp'],consumed_reference_TCP=history[-1]['pose'],base_hold=base)
                    except ValueError as error:failure=str(error)
                    if noise is not None:assert (program is not None)==prior['EEF_full_H40_pass']
                    identity=dict(view=view,index=index,noise=noise,kind=kind,state_sha256=sha(state_path))
                    diagnostic=None
                    if program is not None:
                        attempts+=1;assert attempts<=81
                        with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='INTENT',attempt=attempts))+'\n')
                        before=[sample[k].copy()for k in ('qpos','qvel','ctrl')]
                        diagnostic=rejected_joint_diagnostic(c['model'],sample,program.reference_path,history,stage);calls+=1
                        assert all(np.array_equal(a,sample[k])for a,k in zip(before,('qpos','qvel','ctrl')))
                        with(output/'CLONE_CALLS.private.jsonl').open('a')as f:f.write(json.dumps(dict(identity,event='COMPLETED',call=calls,status=diagnostic['status']))+'\n')
                    row=dict(identity,episode=ep['id'],anchor=anchor,stage=stage,source_spec_sha256=ep['source_spec_sha256'],
                        source_native_result_sha256=ep['source_native_result_sha256'],scene_sha256=native_result['scene_sha256'],
                        action_file_sha256=action_pin,raw_compiler_pass=program is not None,raw_compiler_failure=failure,
                        diagnostic=diagnostic,full_WBC_state_restored=False,historical_student_state_expert_final_plan_pair=False)
                    rows.append(row)
                    with(output/'ROWS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        assert len(rows)==96 and attempts==calls<=81
        summary={}
        for view in MANIFESTS:
            summary[view]={}
            for kind in ('LABEL','ORIGINAL_RGB10_PREDICTION'):
                group=[r for r in rows if(r['view'],r['kind'])==(view,kind)]
                summary[view][kind]=dict(variants=len(group),EEF_pass=sum(r['raw_compiler_pass']for r in group),
                    joint_pass=sum(bool(r['diagnostic']and r['diagnostic']['status']=='PASS_OFFLINE_CLONE_ONLY')for r in group),
                    outcomes=dict(Counter(r['diagnostic']['error']if r['diagnostic']and'error'in r['diagnostic']else'PASS_OFFLINE_CLONE_ONLY'if r['diagnostic']else'RAW_EEF_REJECT'for r in group)))
        result=dict(schema=OPERATION,rows=rows,summaries=summary,recorded_windows=16,variants=96,
            clone_certifier_attempts=attempts,clone_certifier_calls_completed=calls,native_kinematic_model_loads=models_loaded,
            new_model_generations=0,SLA_model_loads=0,optimizer_updates=0,training_visits=0,physical_requests=0,
            physics_integrations=0,full_WBC_state_restored=False,same_state_historical_student_expert_final_plan_pairs=0,
            original_certifier_sha256=ORIGINAL_SHA,formal_results_or_thresholds_changed=False,
            source_metrics_sha256=METRICS,CPU_s=time.process_time()-cpu,wall_s=time.time()-started)
        write(output/'RESULT.private.json',result);return result
    finally:
        write(output/'COUNTERS.private.json',dict(clone_certifier_attempts=attempts,clone_certifier_calls_completed=calls,
            completed_rows=len(rows),native_kinematic_model_loads=models_loaded,new_model_generations=0,SLA_model_loads=0,
            optimizer_updates=0,physical_requests=0,physics_integrations=0))


if __name__=='__main__':
    def interrupted(*args):raise InterruptedError('FINITE_RECORDED_STATE_DIAGNOSTIC_STOP')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('spec');p.add_argument('output');a=p.parse_args()
    result=run(read(a.spec),a.output);print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
