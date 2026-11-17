"""V4 finite SHIFT7-only R1; first80 then conditional TOTAL180; zero training."""
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.close_goal_g1_v1 import wait_resource_owner,width_metrics
from scripts.sla.hybrid_saved96_v1 import summarize
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.hybrid_eef_grip_v1 import coordination
from go2arx5_vla.sla_adapter.shift7_cn38_r1_v4 import CONTRACT,panel,evaluate
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal

def read(p):return json.loads(Path(p).read_bytes())


def run(spec):
    if (spec['physics'] or spec['new_physical_requests_budget']!=0
            or spec['new_optimizer_updates_budget']!=0 or spec['new_training_visits_budget']!=0
            or spec['new_complete_generations_budget']!=180 or spec['goal_plan_sha']!=CONTRACT['plan_sha']):
        raise PermissionError('FIXED_V4_R1_ZERO_TRAIN_NATIVE_MAX180_REQUIRED')
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    authorization=read_pin(spec['resource_authorization'])
    if (authorization['stage']!='R1' or authorization['max_generations']!=180
            or authorization['native_requests']!=0 or authorization['new_updates']!=0
            or authorization['plan_sha']!=CONTRACT['plan_sha'] or not authorization['user_granted']):
        raise PermissionError('INDEPENDENT_R1_RESOURCE_GRANT_REQUIRED')
    root=Path(spec['registered_root']);start=time.monotonic();cpu=time.process_time()
    bank=read_pin(spec['CN38_bank']);cases=panel(bank);entries={r['index']:r for r in bank['records']}
    old=read_pin(spec['HYBRID_records']);previous={(r['index'],r['noise']):r for r in old}
    if len(old)!=180 or len(previous)!=180:raise ValueError('SEALED_MATCHED_HYBRID180_REQUIRED')
    # Persist the complete ordered panel and criteria BEFORE importing/loading GPU.
    write(root/'PANEL.private.json',cases);write(root/'GATE_CONTRACT.private.json',CONTRACT)
    import jax
    from go2arx5_vla.sla_adapter.site_v2 import validate
    site=validate(spec)
    from go2arx5_vla.sla_adapter.compilation_cache_binding_v1 import bind_declared_cache
    cache_binding=bind_declared_cache(spec,jax.config)
    from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import Provider
    package=spec['SHIFT7_package']
    provider=Provider(package['path'],manifest_sha256=package['sha256'],parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
    write(root/'MODEL_LOAD_ACTUAL.private.json',dict(parent_loads=1,adapter_packages=1,optimizer_loaded=False,package=package,
        load_wall_s=time.monotonic()-start,GPU_memory=jax.devices()[0].memory_stats(),site=site,compilation_cache=cache_binding))
    attempted=completed=0;rows=[];first=None;journal=Journal(root/'control/journal',root.name)
    for case in cases:
        if attempted==80:
            first=evaluate(rows,expected=80);write(root/'FIRST80_GATE.private.json',first)
            if not first['GO']:break
        if (root/'STOP.json').exists() or time.time()>=spec['deadline_epoch']-60:
            raise TimeoutError('FINITE_R1_STOP_OR_DEADLINE_NO_AUTO_REPLAY')
        entry=entries[case['entry']];noise=case['noise'];base=previous[(entry['index'],noise)]
        if (base['episode']!=entry['episode'] or base['phase']!=entry['phase']
                or base['input_sha256']!=entry['input']['sha256'] or base['previous_reference']!=entry['previous_reference']):
            raise ValueError('IDENTICAL_OBSERVATION_NOISE_REFERENCE_REQUIRED')
        for pin in (entry['input'],entry['truth'],base['output']):
            if sha(pin['path'])!=pin['sha256']:raise ValueError('SEALED_R1_INPUT_OR_BASELINE_CHANGED')
        with np.load(entry['input']['path'],allow_pickle=False)as z:
            state=z['state_eef10'].copy();images=[z['body_rgb'].copy(),z['wrist_rgb'].copy()];frame=z['T_policy_from_mount'].copy()
        with np.load(entry['truth']['path'],allow_pickle=False)as z:
            expert=z['physical_H40'].copy();reference=z['accepted_reference_TCP'].copy()
        with np.load(base['output']['path'],allow_pickle=False)as z:hybrid=z['physical_H40'].copy()
        message='R1-%03d-%d'%(entry['index'],noise)
        # Existing Journal records INTENT before any neural effect. A killed
        # generation cannot be resent under this or a replacement identity.
        def generate():
            nonlocal attempted,completed
            attempted+=1
            write(root/'GENERATION_COUNTS.private.json',dict(attempted=attempted,completed=completed,cap=180,new_updates=0,new_visits=0,new_native_requests=0))
            t=time.monotonic();value=provider.predict(state,images,entry['language'],frame,noise_id=noise,previous_reference=entry['previous_reference']);latency=time.monotonic()-t
            output=root/'data'/(message+'.npz')
            with output.open('xb')as f:np.savez(f,normalized=value['normalized'],physical_H40=value['physical_H40'])
            completed+=1
            write(root/'GENERATION_COUNTS.private.json',dict(attempted=attempted,completed=completed,cap=180,new_updates=0,new_visits=0,new_native_requests=0))
            return dict(path=str(output),sha256=sha(output),model_wall_s=latency)
        output=journal.once(message,dict(input=entry['input'],previous_reference=entry['previous_reference'],noise=noise,package=package),generate)
        with np.load(output['path'],allow_pickle=False)as z:action=z['physical_H40'].copy()
        env=analyze(action,state,np.asarray(entry['policy_from_world']),reference)
        from scripts.pi0292.eval_contract import matrix10
        from go2arx5_vla.world_model.task_learning_student_v1 import StudentProposalRejected
        policy=np.asarray(entry['policy_from_world']);compiler_pass=True;compiler_failure=None
        try:
            compile_proposal(output['path'],output['sha256'],source_from_target=policy,query_state10=state,
                actual_TCP=np.linalg.inv(policy)@matrix10(state),consumed_reference_TCP=reference,base_hold=np.r_[np.zeros(6),1.])
        except StudentProposalRejected as error:compiler_pass=False;compiler_failure=str(error)
        if compiler_pass!=env['original_full_H40_pass']:raise ValueError('ORIGINAL_COMPILER_AND_ALL_CHANNEL_GATE_DISAGREE')
        w=width_metrics(action,expert)
        row=dict(case,input_sha256=entry['input']['sha256'],previous_reference=entry['previous_reference'],output=output,
            envelope=env,original_compiler_pass=compiler_pass,original_compiler_failure=compiler_failure,width=w,
            coordination=coordination(action,expert,state,entry['previous_reference'],policy),
            EEF_position_errors_m=np.linalg.norm(action[:,:3]-expert[:,:3],axis=1).tolist(),
            HYBRID_EEF_position_errors_m=np.linalg.norm(hybrid[:,:3]-expert[:,:3],axis=1).tolist(),
            terminal_intent_matches=bool((.002+.04*action[-1,9]>.022)==(.002+.04*expert[-1,9]>.022)),
            contact_clearance_shadow='UNKNOWN_NO_CONTROLLER_SELECTED_REFERENCE_OR_DYNAMIC_NATIVE_STATE_PROOF',
            branch_generations=1,new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0)
        rows.append(row);write(root/'R1_RECORDS.private.json',rows)
        write(root/'goal_state.json',dict(phase='R1_80'if len(rows)<=80 else'R1_180_CONDITIONAL',completed=len(rows),attempted_generations=attempted,
            completed_generations=completed,source_sha=spec['source_sha'],plan_sha=CONTRACT['plan_sha'],epoch=time.time(),new_updates=0,new_visits=0,new_native_requests=0,goal_met=False))
    if len(rows)==180:
        final=evaluate(rows,expected=180)
    elif len(rows)==80 and first is not None and not first['GO']:
        final=first
    else:raise ValueError('EXACT80_NEGATIVE_OR180_TERMINAL_REQUIRED')
    passed=len(rows)==180 and final['GO']
    result=dict(schema='SHIFT7_CN38_R1_V4_ACTUAL',exit_code=0,status='R1_FULL_PASS'if passed else'R1_NEGATIVE_PRESERVED',
        source_sha=spec['source_sha'],plan_sha=CONTRACT['plan_sha'],first80=first,final=final,
        selected_architecture='SHIFT7_SINGLE'if passed else'FM_EEF_TIME_GRIP',
        architecture_selection='GLOBAL_NO_PER_QUERY_SWITCH',by_episode={ep:summarize([r for r in rows if r['episode']==ep])for ep in sorted({r['episode']for r in rows})},
        by_phase={p:summarize([r for r in rows if r['phase']==p])for p in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')},
        actual_predictions=len(rows),attempted_generations=attempted,completed_generations=completed,unstarted_of180=180-len(rows),
        new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,new_task_episodes=0,
        GPU_memory=jax.devices()[0].memory_stats(),site=site,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        latency_p50_p95_p99_max_s=np.percentile([r['output']['model_wall_s']for r in rows],[50,95,99,100]).tolist(),
        evidence=dict(records=dict(path=str(root/'R1_RECORDS.private.json'),sha256=sha(root/'R1_RECORDS.private.json')),
            panel_sha256=sha(root/'PANEL.private.json'),criteria_sha256=sha(root/'GATE_CONTRACT.private.json')),
        expert_state_offline_only=True,student_closed_loop_success_proven=False,native_admission=False,goal_met=False)
    write(root/'RESULT.private.json',result);return result


if __name__=='__main__':
    spec=read(sys.argv[2]);root=Path(spec['registered_root']);lock=None
    try:
        lock=wait_resource_owner(spec);result=run(spec)
    except BaseException:
        result=dict(exit_code=1,status='TECHNICAL_OUTCOME_NO_AUTO_REPLAY',failure=traceback.format_exc(),
            actual_counts=read(root/'GENERATION_COUNTS.private.json')if(root/'GENERATION_COUNTS.private.json').exists()else{},
            new_updates=0,new_visits=0,new_native_requests=0,new_integrations=0,goal_met=False)
        write(root/'RESULT.private.json',result)
    finally:
        if lock:lock.close()
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec,result['exit_code']);raise SystemExit(result['exit_code'])
