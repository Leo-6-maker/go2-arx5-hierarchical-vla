"""Read-only NumPy inspection of frozen contact models and observed grid timing."""
import argparse,gzip,json,time
from pathlib import Path


def run(manifest,output):
    import numpy as np
    from go2arx5_vla.world_model.storage import digest,write_json
    from go2arx5_vla.world_model.pilot2_contact_model import replay,metrics
    from go2arx5_vla.world_model.pilot2_contact_timing import transition_timing
    from go2arx5_vla.world_model.pilot2_contact_numpy import predict
    if manifest['schema']!='WM02_PILOT2_FROZEN_CONTACT_INSPECTION_V1':raise ValueError('FROZEN_INSPECTION_MANIFEST_REQUIRED')
    start=time.monotonic();cpu=time.process_time();episodes=[]
    for item in manifest['bundles']:
        if digest(item['path'])!=item['sha256']:raise ValueError('ACTUAL_CONTACT_BUNDLE_CHANGED')
        value=json.loads(gzip.decompress(Path(item['path']).read_bytes()))
        if value['schema']!='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1':raise ValueError('ACTUAL_SUPPORTED_CONTACT_BUNDLE_REQUIRED')
        episodes.extend(value['episodes'])
    data=replay(episodes);ids={s:np.flatnonzero(data['split']==s)for s in ('TRAIN','DEV')}
    if not all(len(v)for v in ids.values()):raise ValueError('ACTUAL_WHOLE_PARENT_TRAIN_DEV_REQUIRED')
    result=dict(schema='WM02_PILOT2_FROZEN_CONTACT_INSPECTION_RESULT_V1',source_sha=manifest['source_sha'],
        bundles_sha256=[b['sha256']for b in manifest['bundles']],models={},baselines={},
        model_updates=0,native_integrations=0,full_state_rollout=False,TEST_used=False)
    def measure(delta,p):
        return {s:dict(forecast_metrics=metrics(delta,p,data,ix),grid_transition_timing=transition_timing(p,data,ix))for s,ix in ids.items()}
    for item in manifest['models']:
        if item['arm']not in ('ACTION_CONDITIONED','NO_FUTURE_PROGRAM')or digest(item['path'])!=item['sha256']:
            raise ValueError('FROZEN_DECLARED_CONTACT_MODEL_REQUIRED')
        with np.load(item['path'],allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
        future=data['plan'if item['arm']=='ACTION_CONDITIONED'else'blind_plan'];raw=predict(arrays,np.c_[data['history'],future])
        probability=1/(1+np.exp(-np.clip(raw[...,6:],-50,50)))
        result['models'][item['name']]=dict(portable_model_sha256=item['sha256'],arm=item['arm'],metrics=measure(raw[...,:6],probability))
    persistence=np.repeat(data['current_events'][:,None],len(data['horizons']),1);frequency=[]
    for j in range(len(data['horizons'])):
        ix=ids['TRAIN'][data['valid'][ids['TRAIN'],j]]
        if not len(ix):raise ValueError('ACTUAL_TRAIN_FORECASTS_REQUIRED_FOR_CONSTANT_BASELINE')
        frequency.append(data['events'][ix,j].mean(0))
    for name,p in [('PERSISTENCE',persistence),('TRAIN_EVENT_FREQUENCY',np.broadcast_to(frequency,data['events'].shape))]:
        result['baselines'][name]=measure(np.zeros_like(data['delta_mm']),p)
    result.update(wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write_json(output,result);return dict(status='FROZEN_CONTACT_PREDICTIONS_INSPECTED',output_sha256=digest(output),
        models=len(manifest['models']),optimizer_updates=0,native_integrations=0,wall_s=result['wall_s'],CPU_s=result['CPU_s'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(run(json.loads(a.manifest.read_text()),a.output)));return 0


if __name__=='__main__':raise SystemExit(main())
