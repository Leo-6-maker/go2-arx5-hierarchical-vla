"""Score previously sealed predictions on prospective actual physical responses."""
def run(spec):
    import json,time
    from pathlib import Path
    import numpy as np
    from go2arx5_vla.world_model.target_replay import load
    from go2arx5_vla.world_model.storage import digest
    from scripts.wm02.audit_next2_models import score
    data,meta=load(spec['replay']);f=json.loads(Path(spec['frozen_predictors']).read_text());tol=json.loads(Path(spec['frozen_thresholds']).read_text())
    if digest(spec['replay'])!=spec['replay_sha256']or digest(spec['frozen_predictors'])!=spec['frozen_predictors_sha256']or not meta.get('prospective_prediction_first'):raise ValueError('FROZEN_PROSPECTIVE_IDENTITIES_REQUIRED')
    predictions={k:[]for k in ['ARX','NO_FUTURE_CHOICE_ARX','HOLD_RESPONSE']};pins=[];same_prefix=[]
    for i,episode in enumerate(meta['episodes']):
        root=Path(spec['campaign_root'])/'runs'/episode['episode'];p=root/'native-01/PREDICTIONS_BEFORE_ACTION.private.json';pv=json.loads(p.read_text());e=json.loads((root/'EPISODE_MANIFEST.private.json').read_text());chosen=next(c for c in pv['candidates']if np.allclose(c['offset_xyz_m'],e['offset_xyz_m'],atol=1e-10,rtol=0))
        if not np.array_equal(np.asarray(pv['observed_history']),data['state'][i,:9])or pv['models_sha256']!={k:f[k]['sha256']for k in predictions}or pv['thresholds_sha256']!=digest(spec['frozen_thresholds']):raise ValueError('SAVED_REAL_PREFIX_OR_PRE_SAMPLING_MODEL_IDENTITY_DIFFERS')
        for k in predictions:predictions[k].append(chosen['predictions'][k])
        differences={}
        for k in predictions:
            sl=slice(0,3)if k=='HOLD_RESPONSE'else slice(86,89);hold=np.asarray(pv['candidates'][0]['predictions'][k]);differences[k]={str(h):{candidate['action']:(np.asarray(candidate['predictions'][k])[h-1,sl]-hold[h-1,sl]).tolist()for candidate in pv['candidates'][1:]}for h in [1,8,16]}
        same_prefix.append(dict(episode=episode['episode'],TCP_world_action_differences_m=differences));pins.append(dict(episode=episode['episode'],prediction_sha256=digest(p),native_result_sha256=episode['result_sha256'],state_journal_sha256=episode['state_journal_sha256']))
    pred={k:np.asarray(v)for k,v in predictions.items()};parents=np.asarray([e['parent']for e in meta['episodes']]);rows={}
    for parent in sorted(set(parents)):
        ids=np.flatnonzero(parents==parent)
        if len(ids)!=5:raise ValueError('FIVE_PREREGISTERED_ATTEMPTS_PER_PARENT_REQUIRED')
        scores={k:score(v[ids],data,meta,ids,compact_output=k=='HOLD_RESPONSE')for k,v in pred.items()};gates={}
        for k in ['ARX','HOLD_RESPONSE']:
            gates[k]={}
            for h in ['8','16']:
                q=scores[k][h];ab=scores['NO_FUTURE_CHOICE_ARX'][h];ar=scores['ARX'][h];improvement=1-q['action_delta_RMSE_m']/max(ab['action_delta_RMSE_m'],1e-30);gates[k][h]=dict(delta_improvement=improvement,all3_directions_correct=q['direction_fraction']==1.,absolute_within_TRAIN_tolerance=q['absolute_TCP_RMSE_m']<=ar['absolute_TCP_RMSE_m']+tol['absolute_TCP_additive_tolerance_m'][h],passed=improvement>=.2 and q['direction_fraction']==1. and q['absolute_TCP_RMSE_m']<=ar['absolute_TCP_RMSE_m']+tol['absolute_TCP_additive_tolerance_m'][h])
        rows[parent]=dict(scene_id=meta['episodes'][ids[0]]['scene_id'],original_split=meta['episodes'][ids[0]]['split'],scores=scores,gates=gates)
    verified={k:all(r['gates'][k][h]['passed']for r in rows.values()for h in ['8','16'])for k in ['ARX','HOLD_RESPONSE']};aggregate={k:{h:{metric:float(np.sqrt(np.mean([r['scores'][k][h][metric]**2 for r in rows.values()])))for metric in ['absolute_TCP_RMSE_m','action_delta_RMSE_m']}for h in ['1','8','16']}for k in pred}
    return dict(schema='WM02_NEXT2_ZERO_UPDATE_PROSPECTIVE_RESPONSE_V1',status='PROSPECTIVE_RESPONSE_VERIFIED'if any(verified.values())else'PROSPECTIVE_RESPONSE_NOT_VERIFIED_FAILURES_PRESERVED',predictor_verified=verified,parents=rows,aggregate_equal_parent_RMSE=aggregate,production_attempts=40,accepted_windows=len(meta['episodes']),rejected_production_attempts=40-len(meta['episodes']),recorded_model_atoms=len(meta['episodes'])*24,replay_sha256=spec['replay_sha256'],model_sha256={k:f[k]['sha256']for k in pred},prediction_and_native_pins=pins,same_prefix_model_action_differences=same_prefix,new_model_updates=0,Actor_updates=0,full_task_successes=0,counterfactual='MATCHED_INDEPENDENT_RESTARTS_NOT_EXACT_NATIVE_FORKS',POLICY_READY=False)

def main(argv=None):
    import argparse,json
    from pathlib import Path
    from go2arx5_vla.world_model.storage import write_json
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);v=run(json.loads(a.spec.read_text()));write_json(a.output,v);print(json.dumps({k:v[k]for k in ['status','predictor_verified','aggregate_equal_parent_RMSE']}));return 0
if __name__=='__main__':raise SystemExit(main())
