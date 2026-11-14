"""Compare saved labels/predictions at verified recorded measured clone states.

No approximate pose reset, full WBC recovery, expert-final-plan pairing or live
qualification is implied. All paths keep the original raw compiler and gates.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
from scripts.pi0294.task_state import ActualState,parse_state
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0291_arx5_data.native_dataset import pose10
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
from go2arx5_vla.sla_adapter.s16_offline_v7 import DATA_SHA,STATS_SHA,fixed_indices
from scripts.sla.diagnose_ik_precision_v71 import CONFIGS,ORIGINAL_SHA,evaluate,original


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(spec,output):
    import mujoco
    assert mujoco.__version__=='3.2.7' and mujoco.get_mjcb_control() is None
    assert sha(inspect.getsourcefile(original.certify_chunk))==ORIGINAL_SHA
    output=Path(output);output.mkdir(exist_ok=False);start=time.time();cpu=time.process_time()
    D=Path(spec['dataset']);assert sha(D/'MANIFEST.private.json')==DATA_SHA
    m=json.loads((D/'MANIFEST.private.json').read_bytes());assert sha(D/m['statistics']['file'])==STATS_SHA
    selection_path=Path(spec['selection']);assert sha(selection_path)==spec['selection_sha256']
    selection=json.loads(selection_path.read_bytes());indices=fixed_indices(m,selection)
    matrix=[(e,k)for e in m['episodes']for k in range(61)];cache={};models={};rows=[];calls=0
    R=Path(spec['fixed160_root']);result=json.loads((R/'RESULT.private.json').read_bytes())
    assert sha(R/'RESULT.private.json')==spec['fixed160_result_sha256'] and result['actual_generations']==160
    assert json.loads((R/'control/LEASE.json').read_bytes())['status']=='CLOSED'
    for index in indices:
        ep,anchor=matrix[index]
        if ep['id'] not in cache:
            assert sha(D/ep['labels']['file'])==ep['labels']['sha256']
            with np.load(D/ep['labels']['file'],allow_pickle=False)as z:cache[ep['id']]={k:z[k].copy()for k in ('state_eef10','nominal_policy_H40')}
        source=ep['mount']['source_spec'];assert sha(source['path'])==source['sha256'];s=json.loads(Path(source['path']).read_bytes())
        native=Path(s['registered_root'])/'native-01';captures=json.loads((native/'TASK_LEARNING_CAPTURES.private.json').read_bytes())['records'];capture=captures[anchor]
        state_file=Path(capture['path']).parent/'STATE.private.bin';assert state_file.exists(),'EXACT_SAVED_STATE_REQUIRED_NO_APPROXIMATION'
        assert sha(state_file)==capture['state_sha256']
        scene=Path(s['model_path']);assert sha(scene)==ep['mount']['scene_sha256']
        if str(scene)not in models:models[str(scene)]=mujoco.MjModel.from_binary_path(str(scene))
        model=models[str(scene)]
        with forbid_integration(mujoco):sample=ActualState(model).update(parse_state(state_file.read_bytes()))
        state=cache[ep['id']]['state_eef10'][anchor];frame=np.asarray(s['task_phase_source_from_target'])
        error=float(np.max(np.abs(matrix10(state)-frame@sample['tcp'])))
        assert error<=1e-9 and sample['native_step']==capture['native_step']==5703+250*anchor,'SAME_RECORDED_CLOCK_TCP_REQUIRED'
        past=list(json.loads((native/'STABLE_TASK_CYCLES/-1/RESULT.private.json').read_bytes())['windows'])
        commands=json.loads((native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json').read_bytes())['records']
        past.extend(dict(v,actual_installed_consumed_EEF10_BASE7=v['installed_consumed_EEF10'])for v in commands)
        past=[v for v in past if v['native_end_step']<=sample['native_step']][-2:]
        assert [v['native_end_step']for v in past]==[sample['native_step']-25,sample['native_step']]
        history=[dict(end_native_step=v['native_end_step'],pose=matrix10(v['actual_installed_consumed_EEF10_BASE7'][:10]))for v in past]
        base=np.asarray(past[-1]['actual_installed_consumed_EEF10_BASE7'][10:]);assert base.shape==(7,)
        stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(ep['expert_stage_audit_only'][anchor],'approach')
        variants=[]
        for noise in (None,0,1):
            kind='RECORDED_EXECUTED_REFERENCE_LABEL' if noise is None else 'S16_PREDICTION'
            program=None;failure=None;pin=None
            try:
                if noise is None:
                    raw=cache[ep['id']]['nominal_policy_H40'][anchor]
                    mapped=pose10(np.linalg.inv(frame)@matrix10(raw),raw[:,9])
                    program=FiftyMsProgram(mapped,sample['tcp'],history[-1]['pose'],base)
                else:
                    action=R/'inference/step0500-sampling10'/f'{index:04d}-{noise}.npz';pin=sha(action)
                    program,_=compile_proposal(action,pin,source_from_target=frame,query_state10=state,actual_TCP=sample['tcp'],consumed_reference_TCP=history[-1]['pose'],base_hold=base)
            except ValueError as e:failure=str(e)
            configs=[]
            if program is not None:
                for precision,settings in enumerate(CONFIGS):
                    calls+=1
                    configs.append(evaluate(model,sample,program.reference_path,history,stage,settings,output/f'{index:04d}-{noise}-precision{precision}.npz'))
            variants.append(dict(kind=kind,noise=noise,action_file_sha256=pin,raw_compiler_pass=program is not None,raw_compiler_failure=failure,settings=configs))
        row=dict(index=index,episode=ep['id'],anchor=anchor,native_step=sample['native_step'],stage=stage,state_sha256=sha(state_file),scene_sha256=sha(scene),source_spec_sha256=source['sha256'],TCP_max_component_error=error,reference_history=[dict(end_native_step=v['end_native_step'],pose=v['pose'].tolist())for v in history],variants=variants,full_WBC_state_restored=False,historical_student_state_expert_final_plan_pair=False)
        rows.append(row)
        with(output/'ROWS.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    out=dict(schema='SLA_S16_RECORDED_STATE_LABEL_PREDICTION_CLONE_V7_1',rows=rows,verified_windows=len(rows),predictions=32,label_plans=16,clone_certifier_calls=calls,new_model_generations=0,optimizer_updates=0,training_visits=0,physical_requests=0,physics_integrations=0,full_WBC_state_restored=False,same_state_historical_expert_final_plan_pairs=0,formal_results_changed=False,wall_s=time.time()-start,CPU_s=time.process_time()-cpu)
    (output/'RESULT.private.json').write_text(json.dumps(out,indent=2,allow_nan=False));return out
