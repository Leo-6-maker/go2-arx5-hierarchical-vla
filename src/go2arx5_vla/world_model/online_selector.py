"""Finite real source decisions: fsynced model choice before safety admission and first400ms."""
from pathlib import Path
import hashlib,json,os,time,traceback
import numpy as np
from .decision_native import bank,proposals,local_guard
from .decision_diagnostics import frozen_model,rule_scores,stable_choice
from .native_adapter import SourceAdapter
from .action_compiler import compile_reference,SOURCE
from .contracts import FRAME,TCP
from .replay import tensor_state
from .storage import write_json,dump_private
from .contact_identification import compare_repeats


def choose(model,history,current,context,raw,blind):
    import torch
    from .contact_training import predict
    # A blind model has identical inputs for every proposal. Evaluate that input
    # once: batched matrix rounding must never become a surrogate action signal.
    n=1 if blind else len(raw)
    def repeat(x):return torch.tensor(x,dtype=torch.float32)[None].expand(n,*np.shape(x))
    with torch.no_grad():
        _,event,_=predict(model,repeat(np.stack([tensor_state(r['state_before'])for r in history])),
            repeat(tensor_state(current)),repeat(np.stack([r['reference_compiled']for r in history])),repeat(context),
            repeat(np.array([[r['t0']/32,r['native_progress']['point']*.05]for r in history])),
            repeat(np.array([current['time_s']/32,0.])),torch.tensor(raw[:n],dtype=torch.float32),blind=blind)
    scores=event[:,-1,2].numpy()
    return np.repeat(scores,len(raw))if blind else scores


def rollout(adapter,anchor,models,spec,folder,policy,scene):
    adapter.restore(anchor['saved']);history=list(anchor['history']);rows=[];selections=[];live=0;guard=local_guard(spec)
    ref=adapter.save_complete(folder/'initial-restore.pkl.gz')
    for decision in range(2):
        guard();state=adapter.snapshot(force=True);names,raw=proposals('preclose',tensor_state(state),anchor['context'],float(adapter.last_command[-1]),scene,decision)
        if policy in models:scores=choose(models[policy],history[-8:],state,anchor['context'],raw,policy=='no_future_action');index=stable_choice(scores,raw)
        elif policy=='same_observation_rule':scores=rule_scores(tensor_state(state),raw);index=stable_choice(scores,raw)
        elif policy=='fixed_random':
            index=int.from_bytes(hashlib.sha256(f'WM02_ONLINE_RANDOM_V3/{scene}/{decision}'.encode()).digest()[:4],'big')%len(names);scores=None
        else:raise ValueError('Unknown fixed selector')
        selection=dict(scene=scene,policy=policy,decision=decision,time_s=state['time_s'],candidates=names,
            scores=scores.tolist()if scores is not None else None,selected=names[index],action_raw=raw[index].tolist(),
            original_history_atoms=8,uses_future_truth=False,Teacher_phase_input=False,
            selector_version='V4',score_tie_tolerance=1e-6,tie_break='SHA256 of raw float64 LE bytes',
            blind_single_prediction_broadcast=policy=='no_future_action')
        # Exclusive/fsynced file exists BEFORE native gate, shadow forecast, or live step.
        path=folder/f'decision-{decision:02d}-BEFORE_EFFECT.json';write_json(path,selection)
        selection_pin=hashlib.sha256(path.read_bytes()).hexdigest()
        plan=compile_reference(raw[index],state['tcp'],state['time_s'],backend=SOURCE,frame=FRAME,tcp=TCP)
        gate=adapter.gate(plan);preflight=adapter.preflight(plan,guard)if gate['decision']!='REJECT'else None
        if gate['decision']=='REJECT'or preflight['decision']!='PASS':
            write_json(folder/f'decision-{decision:02d}-REJECT.json',dict(selection_sha256=selection_pin,gate=gate,preflight=preflight,physics_ticks=0))
            selections.append(dict(decision=decision,selected=names[index],selection_sha256=selection_pin,status='REJECTED_ZERO_PHYSICS'));break
        block=[]
        for k in range(8):
            row=adapter.advance_recorded(plan,k,run_id=folder.parent.name,actor_version='FROZEN_WM_SELECTOR_V4',restore_ref=ref)
            block.append(row);live+=row['physics_ticks']
            if row['terminal']or row['truncated']:break
        dump_private(folder/f'decision-{decision:02d}-ACTUAL.pkl.gz',dict(rows=block,selection_sha256=selection_pin))
        rows.extend(block);history.extend(block);observed=adapter.snapshot(force=True)
        selections.append(dict(decision=decision,selected=names[index],selection_sha256=selection_pin,status='EXECUTED',
            actual_atoms=len(block),observed_time_s=observed['time_s'],observed_grasp=bool(observed['native_grasp'])))
        if len(block)!=8 or block[-1]['terminal']or block[-1]['truncated']:break
    state=adapter.snapshot(force=True);saved=adapter.capture();dump_private(folder/'ACTUAL_ROWS.pkl.gz',dict(rows=rows,selections=selections,final_restore=saved))
    return dict(rows=rows,restore=saved,summary=dict(policy=policy,decisions=selections,actual_transitions=len(rows),live_physics_ticks=live,
        endpoint_grasp=bool(state['native_grasp']),terminal=bool(rows and rows[-1]['terminal']),full_task_success=False))


def scene(job):
    import torch
    from .neural_training import parameter_hash
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,{job['cpu']})
    spec=job['spec'];entry=job['entry'];folder=Path(job['out']);folder.mkdir();cpu=time.process_time();start=time.monotonic()
    card=json.loads(Path(entry['card']).read_text());adapter=None;result={};failure=None;live=0;repeat=None
    try:
        models={k:frozen_model(v)for k,v in spec['models'].items()};pins={k:parameter_hash(v)for k,v in models.items()}
        adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed']);anchors=bank(adapter,entry)
        anchor=anchors['preclose'];runs={}
        for policy in ('action_conditioned','no_future_action','same_observation_rule','fixed_random'):
            p=folder/policy;p.mkdir();r=rollout(adapter,anchor,models,spec,p,policy,entry['scene']);runs[policy]=r;result[policy]=r['summary'];live+=r['summary']['live_physics_ticks']
        p=folder/'conditioned-repeat';p.mkdir();r=rollout(adapter,anchor,models,spec,p,'action_conditioned',entry['scene']);live+=r['summary']['live_physics_ticks']
        repeat=compare_repeats(runs['action_conditioned']['rows'],r['rows'],runs['action_conditioned']['restore'],r['restore'])
        if any(parameter_hash(v)!=pins[k]for k,v in models.items()):raise RuntimeError('Frozen selector changed weights')
    except BaseException:failure=traceback.format_exc()
    finally:
        shadow=adapter.shadow_ticks if adapter else 0
        if adapter:adapter.close()
    report=dict(scene=entry['scene'],policies=result,conditioned_repeat=repeat,failure=failure,
        live_physics_ticks=live,shadow_ticks=shadow,actual_transitions=live//25,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(folder/'RESULT.json',report);return report


def run(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if sorted(x['scene']for x in spec['prefix_inputs'])!=list(range(16,20)):raise ValueError('Exact four DEV parents required')
    start=time.monotonic();jobs=[dict(spec=spec,entry=e,cpu=spec['cpus'][i%len(spec['cpus'])],out=str(Path(output)/f"scene-{e['scene']:02d}"))for i,e in enumerate(spec['prefix_inputs'])]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:scenes=list(pool.map(scene,jobs))
    result=dict(status='FINITE_FROZEN_SOURCE_SELECTOR_EXECUTED',source_sha=spec['source_sha'],models={k:dict(sha256=v['sha256'],parent_checkpoint_sha256=v['parent_checkpoint_sha256'])for k,v in spec['models'].items()},
        scenes=scenes,declared_policy_episodes=16,declared_main_decisions=32,repeat_episodes=4,
        actual_transitions=sum(x['actual_transitions']for x in scenes),live_physics_ticks=sum(x['live_physics_ticks']for x in scenes),
        shadow_ticks=sum(x['shadow_ticks']for x in scenes),CPU_s=sum(x['CPU_s']for x in scenes),wall_s=time.monotonic()-start,
        WM_updates=0,Actor_updates=0,target_integrations=0,complete_tasks=0,formal_reward=False,DEV_only=True)
    write_json(Path(output)/'RESULT.json',result);return result
