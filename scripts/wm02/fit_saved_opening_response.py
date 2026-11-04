"""One FIT-only scalar response fit and post-hoc causal saved-release evaluation."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.storage import digest,write_json
from go2arx5_vla.world_model.replay import load_rows,SLICES
from go2arx5_vla.world_model.actuator_observer import SourceFingerTarget
from go2arx5_vla.world_model.opening_response import reference_curve,predict_opening,fit_lag
from go2arx5_vla.world_model.decision_diagnostics import stable_choice

DATA='c12293bc877758254a14d19ec112f147fe19c93ffacf4c8399ae8ebfcb42c808'
MANIFEST='d6593c1b37432164ca38fd88da1c60036b20b2b7126b7e9f767acce856c38c01'

def run(a):
    start=time.monotonic();cpu=time.process_time();os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});a.output.mkdir(exist_ok=False)
    if digest(a.replay)!=DATA or digest(a.manifest)!=MANIFEST:raise ValueError('Original input data identity differs')
    meta=json.loads(a.audit.read_text())
    if meta['data_sha256']!=DATA or meta['source_native_sha']!='7e41cd69c3c215e91cb2f85688e5426d4a7f4d2c':raise ValueError('Original FIT stream audit required')
    old_pins={(r['scene'],r['anchor'],r['candidate']):r['sha256']for r in meta['stream_pins']};new_pins=json.loads(a.manifest.read_text())['files'];used={}
    def rows(p,pin):
        if digest(p)!=pin:raise ValueError('Saved stream changed:'+p.name)
        used[str(p)]=pin;r,events=load_rows(dict(path=str(p),sha256=pin))
        if not r or events or any(x['terminal']or x['truncated']for x in r):raise ValueError('Complete original executed stream required')
        return r
    def prefix_states(p,pin):
        r=rows(p,pin)
        if abs(r[0]['t0'])>1e-12 or r[0]['controller_memory']['steps']!=0:raise ValueError('Complete known-reset prefix required')
        memory=SourceFingerTarget(known_reset=True);states={};last=0.
        for row in r:
            if abs(row['t0']-last)>1e-9:raise ValueError('Missing causal prefix time')
            for c in row['reference_consumed']:memory.consume(float(c['eef10'][9]))
            states[round(row['t1'],8)]=memory.interval()[0];last=row['t1']
        return states
    with np.load(a.replay,allow_pickle=False)as f:
        keep=np.flatnonzero((f['partition']=='FIT')&(f['anchor']=='supported_release'));raw=f['raw'][keep];initial=f['state'][keep,8,SLICES['gripper_fraction']].ravel();truth=f['state'][keep,9:25,SLICES['gripper_fraction']].squeeze(-1);scenes=f['scene'][keep];names=f['candidate'][keep]
    if len(keep)!=120 or set(scenes)!=set(range(24)):raise ValueError('Only original24 FIT release parents allowed')
    prefixes={};examples=[]
    for i,(scene,name)in enumerate(zip(scenes,names)):
        folder=a.fit_root/f'scene-{scene:02d}';p=folder/'construction/PREFIX.pkl.gz'
        if int(scene)not in prefixes:
            result=json.loads((folder/'construction/RESULT.json').read_text());pin=result['entry']['files'][str(p)];prefixes[int(scene)]=prefix_states(p,pin)
        future=rows(folder/f'branches/supported_release/{name}.pkl.gz',old_pins[(int(scene),'supported_release',str(name))])
        z=prefixes[int(scene)][round(future[0]['t0'],8)];reference=reference_curve(z,raw[i])
        if len(future)!=16 or not np.allclose([r['state_after']['gripper_fraction']for r in future],truth[i],rtol=0,atol=1e-12):raise ValueError('FIT numeric/real stream labels differ')
        examples.append((float(initial[i]),reference,truth[i]))
    lag,n=fit_lag(examples);model=dict(schema='WM02_STATIONARY_OPENING_SCALAR_V1',source_sha=a.source_sha,data_sha256=DATA,FIT_parents=list(range(24)),lag_fraction=lag,parameter_count=1,FIT_examples=120,FIT_moving_horizons=n,limitations='Stationary opening floor only; requires known-reset causal command prefix; no closing/contact generalization or M4 integration')
    write_json(a.output/'MODEL.json',model);loaded=json.loads((a.output/'MODEL.json').read_text());model_sha=digest(a.output/'MODEL.json')
    eval_scenes=[];truth_all=[];before_all=[];after_all=[];regret_before=regret_after=0;effect=[]
    for scene in range(40,48):
        prefix=f'scene-{scene:02d}/';p=a.new_root/prefix/'construction/PREFIX.pkl.gz';states=prefix_states(p,new_pins[prefix+'construction/PREFIX.pkl.gz']);base=prefix+'supported_release/'
        selection=json.loads((a.new_root/base/'BEFORE_ANY_BRANCH.json').read_text())
        if selection['uses_future_truth']or not selection['prospective_validation']:raise ValueError('Original before-effect forecasts required')
        for name in('BEFORE_ANY_BRANCH.json','PREDICTIONS_BEFORE_EFFECT.npz','RESULT.json'):
            if digest(a.new_root/base/name)!=new_pins[base+name]:raise ValueError('Frozen prospective forecast/result changed')
        actual=json.loads((a.new_root/base/'RESULT.json').read_text());candidate_names=selection['names']
        with np.load(a.new_root/base/'PREDICTIONS_BEFORE_EFFECT.npz',allow_pickle=False)as f:planned=f['raw'];before=f['state'][:,:,SLICES['gripper_fraction']].squeeze(-1);prob=f['event'][:,-1,2]
        after=[];true=[];quality=[]
        for i,name in enumerate(candidate_names):
            future=rows(a.new_root/base/f'candidate-{name}.pkl.gz',new_pins[base+f'candidate-{name}.pkl.gz'])
            z=states[round(future[0]['t0'],8)];reference=reference_curve(z,planned[i]);g0=float(future[0]['state_before']['gripper_fraction']);prediction=predict_opening(g0,reference,loaded['lag_fraction'])
            if not np.array_equal(prediction,predict_opening(g0,reference,lag)):raise ValueError('Scalar reload changed predictions')
            after.append(prediction);true.append([r['state_after']['gripper_fraction']for r in future]);quality.append(actual['candidates'][name]['Q'])
        after=np.asarray(after);true=np.asarray(true);quality=np.asarray(quality);old_scores=(1-prob)*np.clip(before[:,-1],0,1);new_scores=(1-prob)*after[:,-1]
        if not np.allclose(old_scores,selection['methods']['pure_WM']['scores'],rtol=0,atol=1e-7):raise ValueError('Original proxy formula differs')
        old=stable_choice(old_scores,planned);new=stable_choice(new_scores,planned);oracle=float(max(quality));rb=max(0.,oracle-quality[old]);ra=max(0.,oracle-quality[new]);regret_before+=int(rb>1e-12);regret_after+=int(ra>1e-12)
        r=candidate_names.index('release');d=candidate_names.index('delay2');effect.append(dict(scene=scene,old_effect=float(before[r,-1]-before[d,-1]),response_effect=float(after[r,-1]-after[d,-1]),actual_effect=float(true[r,-1]-true[d,-1])))
        eval_scenes.append(dict(scene=scene,old_selected=candidate_names[old],response_selected=candidate_names[new],old_regret_Q=float(rb),response_regret_Q=float(ra)));truth_all.append(true);before_all.append(before);after_all.append(after)
    truth_all=np.asarray(truth_all);before_all=np.asarray(before_all);after_all=np.asarray(after_all)
    metrics={f'H{h}':dict(frozen_M4B_RMSE_fraction=float(np.sqrt(np.mean((before_all[:,:,h-1]-truth_all[:,:,h-1])**2))),response_RMSE_fraction=float(np.sqrt(np.mean((after_all[:,:,h-1]-truth_all[:,:,h-1])**2))))for h in(1,8,16)}
    result=dict(schema='WM02_SAVED_STATIONARY_OPENING_RESPONSE_ACTUAL_V1',source_sha=a.source_sha,scope='Post-hoc already-opened release diagnostic; no blind test, deployed method, new score or control improvement claim',model_sha256=model_sha,model=model,input_data_sha256=DATA,stream_manifest_sha256=MANIFEST,FIT_parents=24,FIT_candidates=120,eval_parents=8,eval_candidates=40,opening_metrics=metrics,positive_regret_parents=dict(frozen_M4B=regret_before,opening_response_with_same_frozen_grasp=regret_after),effect_diagnostics=effect,scenes=eval_scenes,private_input_pins=used,scalar_reload_max_abs=0.,new_physics=0,new_neural_inference=0,new_WM_optimizer_updates=0,new_Actor_updates=0,closed_form_fits=1,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(a.output/'RESULT.private.json',result);print(json.dumps({k:v for k,v in result.items()if k in('source_sha','model_sha256','opening_metrics','positive_regret_parents','closed_form_fits','scalar_reload_max_abs','CPU_s','wall_s')}));return 0

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in('fit-root','replay','audit','new-root','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-sha',required=True);return run(p.parse_args(argv))
if __name__=='__main__':raise SystemExit(main())
