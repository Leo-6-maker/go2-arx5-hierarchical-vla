"""Frozen NEXT1 replay reproduction, TRAIN-parent CV and explicit response fits."""
def score(pred,data,meta,ids,*,compact_output=False):
    import numpy as np
    from go2arx5_vla.world_model.next2_response import compact,action
    from go2arx5_vla.world_model.target_arx import CausalARX
    sl=CausalARX(meta).layout['tcp_position_m'];truth=data['state'][ids];guess=pred[...,:3]if compact_output else pred[...,sl];actual=truth[:,:,sl];a=action(data,ids);hold=np.all(np.abs(a)<1e-6,axis=1);rows={}
    if hold.sum()<1:raise ValueError('MATCHED_REAL_HOLD_REQUIRED')
    for h in [1,8,16]:
        pd=guess[:,h-1]-actual[:,8];td=actual[:,8+h]-actual[:,8];pe=pd-pd[hold].mean(0);te=td-td[hold].mean(0);chosen=~hold;axes=np.argmax(np.abs(a[chosen]),axis=1)
        rows[str(h)]=dict(absolute_TCP_RMSE_m=float(np.sqrt(np.mean((guess[:,h-1]-actual[:,8+h])**2))),action_delta_RMSE_m=float(np.sqrt(np.mean((pe[chosen]-te[chosen])**2))),direction_fraction=float(np.mean([p[k]*t[k]>0 for p,t,k in zip(pe[chosen],te[chosen],axes)])),predicted_delta_norm_m=float(np.linalg.norm(pe[chosen],axis=1).mean()),actual_delta_norm_m=float(np.linalg.norm(te[chosen],axis=1).mean()))
    return rows


def run(spec,output):
    import time,json
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.target_arx import CausalARX
    from go2arx5_vla.world_model.next2_response import HoldResponse,future_choice_mask,action
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.target_replay_v2 import select_samples
    from go2arx5_vla.world_model.storage import digest,write_json
    start=time.monotonic();cpu=time.process_time();out=Path(output);out.mkdir(exist_ok=False)
    if digest(spec['replay'])!=spec['replay_sha256']or digest(spec['ARX'])!=spec['ARX_sha256']:raise ValueError('PINNED_NEXT1_DATA_OR_ARX_CHANGED')
    data,meta=load(spec['replay']);train=select_samples(data,meta,'TRAIN');dev=select_samples(data,meta,'DEV');parents=np.asarray([meta['episodes'][int(i)]['parent']for i in data['episode_index']]);names=sorted(set(parents[train]));masked=future_choice_mask(data);arx=CausalARX(meta).fit(data,train)
    with np.load(spec['ARX'],allow_pickle=False)as z:
        old=CausalARX.from_arrays(meta,z);error=max(float(np.max(np.abs(arx.arrays()[k]-z[k])))for k in arx.arrays())
    with(out/'ARX_REPRODUCED.npz').open('xb')as f:np.savez_compressed(f,**arx.arrays())
    if error>1e-8:raise ValueError('ORIGINAL_ARX_NOT_NUMERICALLY_REPRODUCED:'+str(error))
    variants=[(baseline,ridge)for baseline in ['constant','velocity']for ridge in [.01,.1,1.,10.]];cv=[]
    for parent in names:
        fit=np.asarray([i for i in train if parents[i]!=parent]);held=np.asarray([i for i in train if parents[i]==parent]);fold=dict(held_parent=parent,fit_parents=sorted(set(parents[fit])),fit_windows=len(fit),held_windows=len(held),normalization_fit_only=True,models={})
        for label,ds in [('ARX',data),('NO_FUTURE_CHOICE_ARX',masked)]:
            m=CausalARX(meta).fit(ds,fit);fold['models'][label]=score(m.predict(ds,held),data,meta,held)
        for baseline,ridge in variants:
            m=HoldResponse(meta,baseline=baseline,ridge=ridge).fit(data,fit);fold['models'][baseline+'_'+str(ridge)]=score(m.predict(data,held),data,meta,held,compact_output=True)
        cv.append(fold)
    ranking=[]
    for baseline,ridge in variants:
        key=baseline+'_'+str(ridge);loss=float(np.mean([fold['models'][key][h]['action_delta_RMSE_m']+.25*fold['models'][key][h]['absolute_TCP_RMSE_m']for fold in cv for h in ['8','16']]));ranking.append(dict(baseline=baseline,ridge=ridge,TRAIN_LOPO_score_m=loss))
    ranking.sort(key=lambda r:(r['TRAIN_LOPO_score_m'],r['baseline'],r['ridge']));chosen=ranking[0];response=HoldResponse(meta,baseline=chosen['baseline'],ridge=chosen['ridge']).fit(data,train);no_choice=CausalARX(meta).fit(masked,train)
    for name,m in [('HOLD_RESPONSE',response),('NO_FUTURE_CHOICE_ARX',no_choice)]:
        with(out/(name+'.npz')).open('xb')as f:np.savez_compressed(f,**m.arrays())
    frozen=dict(schema='WM02_NEXT2_FROZEN_LOCAL_PREDICTORS_V1',source_sha=spec['source_sha'],FIT_replay_sha256=spec['replay_sha256'],metadata=meta,ARX=dict(path=str(out/'ARX_REPRODUCED.npz'),sha256=digest(out/'ARX_REPRODUCED.npz')),NO_FUTURE_CHOICE_ARX=dict(path=str(out/'NO_FUTURE_CHOICE_ARX.npz'),sha256=digest(out/'NO_FUTURE_CHOICE_ARX.npz')),HOLD_RESPONSE=dict(path=str(out/'HOLD_RESPONSE.npz'),sha256=digest(out/'HOLD_RESPONSE.npz'),baseline=chosen['baseline'],ridge=chosen['ridge']),identified_actions='HOLD/+X/-X/+Y1mm only; no Z/rotation/gripper or100mm prediction qualification',candidate_selection='All choices by4 TRAIN-parent LOPO; historical DEV already opened; no prospective fit',normalizer_TRAIN_only=True,prediction_not_policy_or_contact_planner=True)
    # Freeze the actual fitted identities before opening even historical DEV.
    write_json(out/'FROZEN_PREDICTORS.private.json',frozen);evaluation={}
    for split,ids in [('TRAIN',train),('DEV_ALREADY_OPENED',dev)]:
        evaluation[split]={}
        for parent in sorted(set(parents[ids])):
            ix=np.asarray([i for i in ids if parents[i]==parent]);evaluation[split][parent]=dict(ARX=score(arx.predict(data,ix),data,meta,ix),NO_FUTURE_CHOICE_ARX=score(no_choice.predict(masked,ix),data,meta,ix),HOLD_RESPONSE=score(response.predict(data,ix),data,meta,ix,compact_output=True))
    tolerances={}
    tcp=arx.layout['tcp_position_m']
    for h in [8,16]:
        differences=[]
        for parent in names:
            ix=np.asarray([i for i in train if parents[i]==parent]);a=action(data,ix);holds=ix[np.all(np.abs(a)<1e-6,axis=1)]
            if len(holds)!=2:raise ValueError('EXACT_TWO_TRAIN_HOLD_REPEATS_REQUIRED')
            delta=data['state'][holds,8+h,tcp]-data['state'][holds,8,tcp];differences.append((delta[1]-delta[0])/np.sqrt(2))
        tolerances[str(h)]=max(1e-6,float(np.sqrt(np.mean(np.asarray(differences)**2))))
    tol=dict(schema='WM02_NEXT2_TRAIN_ONLY_TOLERANCE_V1',absolute_TCP_additive_tolerance_m=tolerances,estimator='RMS independent HOLD-pair difference divided sqrt2 across4 TRAIN parents/coordinates; exploratory, not confidence bound',prospective_direction='Every parent3/3 atH8/H16',future_choice_ablation_minimum_delta_RMSE_improvement=.2)
    write_json(out/'FROZEN_THRESHOLDS.private.json',tol)
    # Per-axis R(0)=0 follows from an un-biased response head, checked exactly.
    result=dict(schema='WM02_NEXT2_MODEL_FAIRNESS_AND_CAUSALITY_V1',source_sha=spec['source_sha'],status='CPU_RESPONSE_CANDIDATES_FROZEN',replay_sha256=spec['replay_sha256'],ARX_reproduction_max_array_error=error,ARX_reproduced_sha256=frozen['ARX']['sha256'],ARX_original_sha256=spec['ARX_sha256'],TRAIN_parent_CV=cv,TRAIN_candidate_ranking=ranking,chosen_response=chosen,evaluation=evaluation,frozen_predictors_sha256=digest(out/'FROZEN_PREDICTORS.private.json'),frozen_thresholds=tol,hold_response_output='TCP_world3+BASE_world3+TCP_in_BASE3; compact local dynamics, not full safety world model',HOLD_response_at_zero_action='EXACT_ZERO_NO_RESPONSE_INTERCEPT',historical_NO_ACTION='All17 reference coordinates masked in both history and future; old results preserved',new_ablation='Only chosen future EEF position reset to actual past HOLD reference; other reference/clock/history retained',ARX_vs_GRU='State+state delta and fully TRAIN-standardized action/time features; one-step ridge vs16-step free106D GRU objective, not architecture-only contrast',new_closed_form_linear_solves=4+len(cv)*(2+2*len(variants)),new_integrations=0,new_neural_optimizer_updates=0,Actor_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    # Count whole model fit calls: original ARX +10 per fold +response +masked ARX.
    result['new_closed_form_model_fits']=3+len(cv)*(2+len(variants));write_json(out/'RESULT.private.json',result);return result


def main(argv=None):
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()),a.output);print(json.dumps({k:v[k]for k in ['status','ARX_reproduction_max_array_error','ARX_reproduced_sha256','chosen_response','CPU_s']}));return 0
if __name__=='__main__':raise SystemExit(main())
