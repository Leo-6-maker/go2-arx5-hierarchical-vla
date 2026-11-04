"""Prospective B2: frozen recommendations precede all new S counterfactual effects."""
from pathlib import Path
import hashlib,json,os,time,traceback
import numpy as np
from .storage import digest,write_json,dump_private
from .replay import tensor_state,SLICES
from .decision_native import KINDS,bank,proposals,local_guard
from .decision_collection import construct
from .contact_identification import execute,compare_repeats,supported
from .decision_diagnostics import stable_choice,rule_scores

METHODS=('pure_WM','rule','OLS_transport_rule_contact','hybrid')


def ols_fit(path,expected):
    """Same fixed20 FIT0..3 transport samples; never new validation labels."""
    if digest(path)!=expected:raise ValueError('Original FIT replay identity differs')
    with np.load(path,allow_pickle=False)as f:
        keep=(f['partition']=='FIT')&(f['scene']<4)&(f['anchor']=='transport')
        if keep.sum()!=20:raise ValueError('Original20 FIT OLS samples required')
        state=f['state'][keep];raw=f['raw'][keep]
    x=np.c_[np.ones(20),raw[...,:3].sum(1)/.004]
    y=np.concatenate([state[:,9:25,SLICES[n]]-state[:,8,None,SLICES[n]]for n in('tcp_position','object_position')],-1)
    return np.linalg.lstsq(x,y.reshape(20,-1),rcond=None)[0]


def forecast_inputs(history,current,context,raw):
    import torch
    def repeat(value):return torch.as_tensor(np.asarray(value),dtype=torch.float32)[None].expand(len(raw),*np.shape(value))
    states=np.stack([tensor_state(r['state_before'])for r in history]+[tensor_state(current)])
    return dict(state=repeat(states),reference=repeat(np.stack([r['reference_compiled']for r in history])),context=repeat(context),clock=repeat(np.array([[r['t0']/32,r['native_progress']['point']*.05]for r in history]+[[current['time_s']/32,0.]])),raw=torch.as_tensor(raw,dtype=torch.float32))


def selectors(kind,current,context,raw,pred,event,ols):
    s=tensor_state(current);p=pred[:,-1];grasp=event[:,-1,2]
    if kind=='preclose':wm=grasp;rule=rule_scores(s,raw)
    elif kind=='transport':
        wm=grasp-10*np.linalg.norm(p[:,SLICES['object_position']][:,:2]-context[:2],axis=1)
        rule=-np.linalg.norm(s[SLICES['object_position']][:2]+raw[:,:,:2].sum(1)-context[:2],axis=1)
    else:
        # No native support head exists. This limited proxy must face native support in Q.
        wm=(1-grasp)*np.clip(p[:,SLICES['gripper_fraction']].ravel(),0,1)
        rule=raw[:,:,6].mean(1)
    if kind=='transport':
        delta=(np.c_[np.ones(len(raw)),raw[...,:3].sum(1)/.004]@ols).reshape(len(raw),16,6)
        score_ols=-np.linalg.norm(s[SLICES['object_position']][:2]+delta[:,-1,3:5]-context[:2],axis=1)
    else:score_ols=rule
    fallback=kind!='preclose'or float(np.ptp(grasp))<.1
    hybrid=score_ols if fallback else wm
    scores=dict(zip(METHODS,(wm,rule,score_ols,hybrid)))
    return {k:dict(scores=v.tolist(),index=stable_choice(v,raw),fallback=(kind!='transport'if k=='OLS_transport_rule_contact'else fallback if k=='hybrid'else False))for k,v in scores.items()}


def actual_quality(kind,state,context,complete):
    if not complete:return -1e6
    if kind=='preclose':return float(state['native_grasp'])
    if kind=='transport':return float(state['native_grasp'])-10*float(np.linalg.norm(state['object_pose'][:2,3]-context[:2]))
    return float(state['supported']) * (1-float(state['native_grasp']))*float(state['gripper_fraction'])


def scene(job):
    import torch
    from .native_adapter import SourceAdapter
    from .temporal_diagnostics import load_model
    from .response_eval_audit import canonical_forecast
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);os.sched_setaffinity(0,{job['cpu']})
    spec=job['spec'];folder=Path(job['out']);folder.mkdir();start=time.monotonic();cpu=time.process_time();adapter=None;failure=None;phases={};live=shadow=0;prefix=None
    try:
        prefix=construct(dict(job,out=str(folder/'construction')))
        live+=prefix['live_physics_ticks'];shadow+=prefix['shadow_ticks']
        if prefix['failure']:raise RuntimeError('Original construction failed: '+prefix['failure'])
        model=load_model(spec['model']);adapter=SourceAdapter(job['card'],spec['asset_root'],executor_id=spec['executor_id'],seed=job['card']['seed']);anchors=bank(adapter,prefix['entry']);guard=local_guard(spec)
        for kind in KINDS:
            if kind not in anchors:phases[kind]=dict(status='ANCHOR_NOT_FOUND',intended_candidates=5,intended_methods=4);continue
            anchor=anchors[kind];adapter.restore(anchor['saved']);state=adapter.snapshot(force=True);names,raw=proposals(kind,tensor_state(state),anchor['context'],float(adapter.last_command[-1]),job['card']['ident_index'])
            part=folder/kind;part.mkdir();ref=adapter.save_complete(part/'common-restore.pkl.gz')
            b=forecast_inputs(anchor['history'],state,anchor['context'],raw);infer=time.process_time()
            with torch.no_grad():pred,event,_=canonical_forecast(model,b)
            pred=pred.numpy();event=event.numpy();choices=selectors(kind,state,anchor['context'],raw,pred,event,np.asarray(spec['OLS_coefficients']))
            # Native/reference physics and future outcomes have not run yet.
            selection=dict(kind=kind,scene=job['card']['ident_index'],names=names,physical_raw=raw,raw_sha256=[hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()for a in raw],methods=choices,model_sha256=spec['model']['sha256'],parent_checkpoint_sha256=spec['model']['parent_checkpoint_sha256'],model_inference_CPU_s=time.process_time()-infer,uses_future_truth=False,prospective_validation=True)
            write_json(part/'BEFORE_ANY_BRANCH.json',selection);pin=digest(part/'BEFORE_ANY_BRANCH.json');np.savez_compressed(part/'PREDICTIONS_BEFORE_EFFECT.npz',state=pred,event=event,raw=raw)
            truth={};runs={}
            def effect(name,action,*,tail=False):
                nonlocal live
                adapter.restore(anchor['saved']);before=adapter.engine.steps
                try:r=execute(adapter,action,part/(name+'.pkl.gz'),ref,folder.name,guard,tail=tail)
                finally:live+=max(0,adapter.engine.steps-before)
                observed=adapter.snapshot(force=True);observed['supported']=supported(adapter,observed)
                if any(x['terminal']or x['truncated']for x in r['rows']):raise RuntimeError('First native terminal: preserve partial stream, no further same-parent physics')
                complete=len(r['rows'])==16 and not any(x['terminal']or x['truncated']for x in r['rows'])and not r['events']
                return r,dict(complete=complete,transitions=len(r['rows']),rejected_blocks=len(r['events']),Q=actual_quality(kind,observed,anchor['context'],complete),native_grasp=bool(observed['native_grasp']),native_support=bool(observed['supported']),terminal=any(x['terminal']for x in r['rows']))
            # Actually execute every selected action, even ties; do not alias outcomes.
            online={}
            for method in METHODS:
                r,z=effect('online-'+method,raw[choices[method]['index']]);z.update(selected=names[choices[method]['index']],selection_sha256=pin,fallback=choices[method]['fallback']);online[method]=z
            for name,action in zip(names,raw):
                r,z=effect('candidate-'+name,action);truth[name]=z;runs[name]=r
            control='nominal'if kind=='preclose'else'hold'if kind=='transport'else'release';checks={}
            for label,tail in(('repeat',False),('tail-only',True)):
                r,z=effect(label,raw[names.index(control)],tail=tail);checks[label]=compare_repeats(runs[control]['rows'],r['rows'],runs[control]['final_restore'],r['final_restore'])
            oracle=max(z['Q']for z in truth.values());rule=online['rule']['Q']
            for method,z in online.items():z.update(oracle_Q=oracle,regret_Q=max(0.,oracle-z['Q']),improvement_vs_rule_Q=z['Q']-rule)
            errors=[]
            for h in(1,8,16):
                valid=[(i,n)for i,n in enumerate(names)if len(runs[n]['rows'])>=h]
                row=dict(horizon=h,observed_candidates=len(valid))
                if valid:
                    states=np.stack([tensor_state(runs[n]['rows'][h-1]['state_after'])for i,n in valid]);pi=pred[[i for i,n in valid],h-1]
                    for field in('tcp_position','object_position'):row[field+'_RMSE_m']=float(np.sqrt(np.mean(np.sum((pi[:,SLICES[field]]-states[:,SLICES[field]])**2,axis=-1))))
                    row['grasp_Brier']=float(np.mean((event[[i for i,n in valid],h-1,2]-np.array([runs[n]['rows'][h-1]['state_after']['native_grasp']for i,n in valid]))**2))
                errors.append(row)
            phases[kind]=dict(status='ACTUAL_FINITE_S_EXECUTED',selection_sha256=pin,online=online,candidates=truth,controls=checks,oracle_Q=oracle,oracle_improvement_over_rule_Q=max(0.,oracle-rule),forecast_errors=errors,scope='Partial0.8s source diagnostic Q, not formal task reward/success',no_support_prediction_head=True)
            write_json(part/'RESULT.json',phases[kind])
            if any(not x['qualified']for x in checks.values()):raise RuntimeError('New native restore/tail qualification failed; preserve first failure')
            if any(x['terminal']for x in truth.values())or any(x['terminal']for x in online.values()):raise RuntimeError('Native source terminal; no further same-parent physics')
    except BaseException:failure=traceback.format_exc()
    finally:
        if adapter:shadow+=adapter.shadow_ticks;adapter.close()
    result=dict(scene=job['card']['ident_index'],partition='PROSPECTIVE_B2_NEVER_FIT',phases=phases,failure=failure,construction=prefix,prefix_transitions=prefix['prefix_transitions']if prefix else 0,live_physics_ticks=live,shadow_ticks=shadow,actual_transitions=live//25,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(folder/'RESULT.json',result);return result


def run(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    output=Path(output);start=time.monotonic();p=Path(spec['preregistration'])
    if digest(p)!=spec['preregistration_sha256']:raise ValueError('Frozen B2 preregistration differs')
    plan=json.loads(p.read_text());cards=plan['cards']
    if len(cards)!=8 or [c['ident_index']for c in cards]!=list(range(40,48))or any(c['split']!='DEV'or c['ident_partition']!='PROSPECTIVE_B2_NEVER_FIT'for c in cards):raise ValueError('Independent frozen8 parents required')
    if digest(spec['B1_gate'])!=spec['B1_gate_sha256']or json.loads(Path(spec['B1_gate']).read_text())['all_development_gates_pass']is not True:raise ValueError('Actual prespecified M4-B joint gate required')
    for name,pin in spec['asset_files'].items():
        if digest(Path(spec['asset_root'])/name)!=pin:raise ValueError('Frozen S asset differs:'+name)
    coefficients=ols_fit(spec['replay'],spec['replay_sha256']);spec=dict(spec,OLS_coefficients=coefficients.tolist());write_json(output/'OLS_FIT_ONLY.json',dict(coefficients=coefficients,source_data_sha256=spec['replay_sha256'],FIT_parents=[0,1,2,3],samples=20,closed_form_solves=1));write_json(output/'CARDS.json',cards)
    jobs=[dict(spec=spec,card=c,cpu=spec['cpus'][i%4],out=str(output/f"scene-{c['ident_index']:02d}"))for i,c in enumerate(cards)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:
        scenes=[]
        for x in pool.map(scene,jobs):scenes.append(x);print(json.dumps({k:x[k]for k in('scene','actual_transitions','failure')}),flush=True)
    result=dict(status='PROSPECTIVE_B2_TERMINAL',source_sha=spec['source_sha'],model=spec['model'],preregistration_sha256=spec['preregistration_sha256'],FIT_data_sha256=spec['replay_sha256'],intended_parents=8,intended_anchors=24,intended_method_episodes=96,intended_candidate_branches=120,scenes=scenes,actual_transitions=sum(x['actual_transitions']for x in scenes),live_physics_ticks=sum(x['live_physics_ticks']for x in scenes),shadow_ticks=sum(x['shadow_ticks']for x in scenes),CPU_s=sum(x['CPU_s']for x in scenes),wall_s=time.monotonic()-start,WM_updates=0,Actor_updates=0,target_integrations=0,full_tasks=0,formal_reward=False,old_holdout_reused=False)
    write_json(output/'RESULT.json',result);return result
