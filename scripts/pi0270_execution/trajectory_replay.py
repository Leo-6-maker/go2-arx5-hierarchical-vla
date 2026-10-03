"""Saved-reference/executor/same-tick/V3/reliability CPU replay, never a new episode."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import argparse
import ast
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from admission import BASE, ROOT, read, sha, write

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main(lane):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30,16*2**30)); os.nice(19)
    if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    sys.path[:0] = [str(BASE/'PI0267/code'), str(BASE/'PI0266/code'), str(BASE/'PI0268/R1/code'),
                    str(BASE/'PI0250/runtime/reference-317a/src')]
    from pi0267_eval_report import score, independent_entry
    from reliability import reliable
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    scope = dict(np=np, Rotation=Rotation, Slerp=Slerp)
    sources = [BASE/'PI0267/code/pi0267_cpu.py', BASE/'PI0269/code/virtual_episode.py']
    expected_sources = ['474868a33dd92cf8b4a5a62fc5b375b4b40f486a7e3480f8748263479287f0aa',
                        '639e9ebadc9b92c52fc1581ab38098e8d61cf1495aed4e4df9093fddba15bd34']
    assert all(sha(p)==h for p,h in zip(sources,expected_sources))
    import pi0267_metrics_v3, pi0267_r1_cpu, reliability
    scoring_sources = [Path(m.__file__) for m in (pi0267_metrics_v3,pi0267_r1_cpu,reliability)]
    assert sha(scoring_sources[-1])=='cdbe67ab2a0f159c286fe1be835aba1e537e258e1ee0fde2a344877364aa6c28'
    for path, name in zip(sources, ('interpolate','advance')):
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    advance = scope['advance']
    scene = read(BASE/'PI0267/design/SCENE_CARD.json'); anchor = np.asarray(scene['policy_anchor_world_m'])
    if lane == 'collection':
        point = 'PREFIX4096'; card_path = ROOT/'design/COLLECTION_CARD.json'
        rows = read(card_path)['cases']
        closed = {r['family'] for r in rows if (ROOT/'evidence'/('SUPPORT_CPU_'+r['family']+'.json')).exists()}
        cases = [r['case'] for r in rows if r['family'] in closed]
    else:
        point = 'AGG4096'; card_path = ROOT/'design/EVALUATION_CARD.json'
        assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
        cases = sorted(p.parent.name for p in (RAW/lane/point).glob('*/result.json'))
        assert len(cases) == (64 if lane=='virtual' else 32)
    assert cases
    checks = []
    for case in cases:
        folder = RAW/lane/point/case; result_path = folder/'result.json'; record = read(result_path)
        assert record['point']==point
        field = 'collection_card_sha256' if lane=='collection' else 'evaluation_card_sha256'
        assert record[field] == sha(card_path)
        destination = ROOT/'evidence/replay'/lane/(case+'.trajectory.json')
        if destination.exists():
            prior = read(destination); assert prior['source_result_sha256']==sha(result_path)
            assert prior['status'] in ('PASS_SAVED_EXECUTOR_SAME_TICK_V3_RELIABILITY','PRESERVED_COMPONENT_PREQUERY_EXCLUSION')
            checks.append(dict(path=str(destination),sha256=sha(destination))); continue
        if record['validity']=='PREQUERY_EXCLUDED':
            assert lane=='component' and not record['queries']
            write(destination,dict(status='PRESERVED_COMPONENT_PREQUERY_EXCLUSION',source_result_sha256=sha(result_path),
                  reasons=record['reasons'],qualification_authority=False,model_queries=0,renderer_calls=0,physics_calls=0))
            checks.append(dict(path=str(destination),sha256=sha(destination))); continue
        assert record['validity']=='VALID' and sha(folder/'trajectory.npz')==record['arrays_sha256']
        with np.load(folder/'trajectory.npz') as z:
            t=z['time'].copy();p=z['actual_p'].copy();r=z['actual_r'].copy();ip=z['installed_p'].copy();ir=z['installed_r'].copy()
            spec=json.loads(str(z['spec_json']))
        assert np.array_equal(t,np.arange(len(t))*.02)
        expected_p=[p[0].copy()];expected_r=[r[0].copy()];desired_p=[ip[0].copy()];desired_r=[ir[0].copy()]
        gates={g['replan']:g['result'] for g in record['gates']}
        for index,query in enumerate(record['queries']):
            request=query['query'];assert query['replan']==index
            if query.get('observation_sha256'): assert sha(request['input'])==query['observation_sha256']
            with np.load(request['input']) as obs:
                assert abs(float(obs['timestamp'])-.4*index)<1e-10
                assert np.max(abs(obs['position_world']-expected_p[-1]))<=1e-12
                assert np.max(abs(obs['rotation_world']-expected_r[-1]))<=1e-12
            if index not in gates or gates[index]['decision']=='REJECT':
                assert index==len(record['queries'])-1 and record['stop_reason'] in ('NONFINITE_POLICY_OUTPUT','INVALID_ROT6D_POLICY_OUTPUT','GATE_REJECT')
                break
            assert sha(request['output'])==query['prediction_sha256']
            with np.load(request['output']) as z: decoded=z['decoded'][0].copy()
            lp=decoded[:,:3].astype(float)+anchor;lr=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
            ap,ar,dp,dr=advance(expected_p[-1],expected_r[-1],lp,lr,record['backend'])
            n=min(20,len(t)-len(expected_p));assert n>0
            if n<20:
                assert index==len(record['queries'])-1 and record['stop_reason']=='VIRTUAL_TOOL_COLLISION'
                assert record['collision_events'][-1]['tick']==len(t)-1
            expected_p.extend(ap[:n]);expected_r.extend(ar[:n]);desired_p.extend(dp[:n]);desired_r.extend(dr[:n])
        expected_p,expected_r,desired_p,desired_r=map(np.asarray,(expected_p,expected_r,desired_p,desired_r))
        assert expected_p.shape==p.shape and expected_r.shape==r.shape
        differences={name:float(np.max(abs(a-b))) for name,a,b in
            [('actual_p',expected_p,p),('actual_r',expected_r,r),('installed_p',desired_p,ip),('installed_r',desired_r,ir)]}
        assert all(d<=1e-12 for d in differences.values()), (case,differences)
        verdict=score(t,p,r,ip,ir,spec,record['scorer']['safety_failure']);assert verdict==record['scorer']
        if spec['task']=='M3':
            broad,entry,_=independent_entry(t,p,r,spec)
            assert entry==verdict['entry_index']
            assert (None if broad is None else float(t[broad]))==verdict['earliest_broad_readiness_s']
        retention=reliable(t,p,r,spec['goal'],spec['goal_rotation'],verdict,
            gate_refusals=sum(g['result']['decision']=='REJECT' for g in record['gates']),
            collisions=len(record['collision_events']),execution_failure=record['stop_reason'] is not None)
        write(destination,dict(status='PASS_SAVED_EXECUTOR_SAME_TICK_V3_RELIABILITY',point=point,case=case,lane=lane,
              source_result_sha256=sha(result_path),source_arrays_sha256=record['arrays_sha256'],reconstruction_max_abs=differences,
              reference_final_xyz_m=float(np.linalg.norm(ip[-1]-spec['goal'])),actual_final_xyz_m=float(np.linalg.norm(p[-1]-spec['goal'])),
              same_tick_tracking_xyz_rms_m=float(np.sqrt(np.mean(np.sum((p-ip)**2,axis=1)))),
              same_tick_tracking_so3_rms_rad=float(np.sqrt(np.mean(Rotation.from_matrix(r@ir.transpose(0,2,1)).magnitude()**2))),reliability=retention,
              shared_original_V3_and_V2_exact=True,independent_M3_entry_checked=spec['task']=='M3',
              frozen_executor_arithmetic_shared_not_an_independent_physics_backend=True,
              source_functions=[dict(path=str(s),sha256=sha(s)) for s in sources],source_sha256=sha(__file__),
              scoring_sources=[dict(path=str(s),sha256=sha(s)) for s in scoring_sources],
              qualification_authority=lane=='virtual',model_queries=0,teacher_calls=0,renderer_calls=0,physics_calls=0))
        checks.append(dict(path=str(destination),sha256=sha(destination)))
        print('SAVED_TRAJECTORY_CPU_REPLAYED',lane,case,flush=True)
    destination=ROOT/'evidence'/('TRAJECTORY_CPU_'+lane+'_'+str(len(checks))+'.json')
    write(destination,dict(status='PASS_SAVED_EXECUTOR_SAME_TICK_V3_RELIABILITY_BATCH',lane=lane,cases=len(checks),checks=checks,
          collection_all192_complete=lane=='collection' and len(checks)==192,source_sha256=sha(__file__),
          model_queries=0,teacher_calls=0,renderer_calls=0,physics_calls=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('SAVED_TRAJECTORY_CPU_BATCH_COMPLETE',lane,len(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lane',choices=('collection','virtual','component'),default='collection')
    main(parser.parse_args().lane)
