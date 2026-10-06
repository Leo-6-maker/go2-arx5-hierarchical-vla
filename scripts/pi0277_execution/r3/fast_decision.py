"""Native task/input/noise decision replay, without another neural or physics call."""
import os
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code')
import numpy as np
from evaluation_card import ROOT,RAW,COMMON,read,sha,freeze
from panel import scores
R3=ROOT.with_name('R3');RELEASE='a5a7e569684d52170f02b0819ae006a7587800dd'

def decision(point,lane):
    full=ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json'
    p=full if full.exists() else R3/'evidence'/f'FAST_{point}_{lane}.json'
    v=read(p);assert v['point']==point and v['lane']==lane
    if p==full:assert v['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
    else:
        assert v['status']=='PASS_NATIVE_TASK_RECONSTRUCTION_FOR_DECISION' and all(v['checks'].values())
        assert sha(ROOT/'model_bindings'/f'{point}.json')==v['binding_sha256']
        assert sha(ROOT/'design/EVALUATION_CARD.json')==v['card_sha256']
        for row in v['result_files']:assert sha(row['path'])==row['sha256']
    expected=read(ROOT/'design/EVALUATION_CARD.json')[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]
    assert v['task']['denominator']==len(expected) and [r['seed'] for r in v['task']['rows']]==expected
    return p,v

def replay(point,lane):
    seal=R3/'EXECUTION_FILESET.json'
    for r in read(seal)['files']:assert sha(r['path'])==r['sha256']
    done_path=ROOT/'runtime'/f'{point}-{lane}.done.json'
    if not done_path.exists():done_path=ROOT/'runtime'/f'R3-{point}-{lane}.done.json'
    done=read(done_path);assert done['returncode']==0 and done['arguments']==[lane,'--point',point]
    binding=read(ROOT/'model_bindings'/f'{point}.json');admission=read(ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json')
    assert admission['status']=='PASS_PI0277_FIXED_POLICY_IDENTITY' and admission['model_binding_sha256']==sha(ROOT/'model_bindings'/f'{point}.json')
    card=read(ROOT/'design/EVALUATION_CARD.json');seeds=card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]
    noise={(r['lane'],r['condition'],r['index']):r for r in read(ROOT/'design/EVALUATION_NOISE.json')['rows']}
    attempts={r['ordinal']:r for r in [json.loads(s) for s in (COMMON/'evidence/model_attempts.jsonl').read_text().splitlines()]}
    heads=collections.Counter(json.loads(s)['attempt'] for s in (COMMON/'evidence/model_heads.jsonl').read_text().splitlines())
    proofs=[];envelope=[];results=[]
    for seed in seeds:
        paths=[p for p in (RAW/'episodes'/f'{lane}-{seed}-{point}').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')];assert len(paths)==1
        folder=paths[0].parent;rec=read(paths[0]);results.append(dict(path=str(paths[0]),sha256=sha(paths[0])))
        for name,h in rec['files'].items():assert sha(folder/name)==h
        initial=read(folder/('INITIAL_SEEDED_NATIVE_RESET.json' if lane=='CONFIRM' else 'INITIAL_FIXED_PHYSICS_MATCH.json'))
        if lane!='CONFIRM':assert max(initial['differences'].values())<=1e-6
        queries=read(folder/'QUERIES.json')
        with np.load(folder/'observations.npz',allow_pickle=False) as obs:
            for q in queries:
                output=Path(q['output']);side=read(output.with_suffix('.receipt.json'));a=attempts[side['attempt']]
                assert a['point']==side['point']==point and heads[side['attempt']]==4
                assert side['binding']==binding and side['no_teacher_or_privileged_inputs'] and side['output_sha256']==sha(output)
                assert output.resolve().is_relative_to((RAW/'quality'/point).resolve())
                req=a['request'];expected=noise[lane,seed,q['index']]
                assert req['output']==str(output) and req['input']==str(output.with_name('input.npz'))
                assert req['noise_file']==expected['file'] and req['noise_name']==expected['name'] and req['noise_sha256']==expected['sha256']==side['noise_sha256']
                index=8*q['index'];assert q['time_s']==index*.05
                with np.load(req['input'],allow_pickle=False) as inp:
                    assert set(inp.files)=={'body_rgb','wrist_rgb','state_v1'}
                    assert np.array_equal(inp['state_v1'],obs['state_v1'][index])
                    assert np.array_equal(inp['body_rgb'],obs['base_camera_96'][index]) and np.array_equal(inp['wrist_rgb'],obs['hand_camera_96'][index])
                    state=inp['state_v1'].copy()
                with np.load(output,allow_pickle=False) as z:
                    receipt=json.loads(str(z['receipt_json']));assert receipt==q['receipt'] and np.array_equal(z['state'],state)
                    actual=z['noise'];g=z['decoded'][0,:,9].copy()
                with np.load(req['noise_file'],allow_pickle=False) as z:assert np.array_equal(actual,z[req['noise_name']])
                import hashlib
                assert hashlib.sha256(actual.tobytes()).hexdigest()==side['noise_sha256']
                bad=np.flatnonzero((g<-.1)|(g>1.1));early=bool((bad<8).any());late=bool((bad>=8).any())
                finite=bool(np.isfinite(g).all());category=('both' if early and late else ('prefix_only' if early else ('suffix_only' if late else 'neither'))) if finite else 'nonfinite'
                envelope.append(dict(seed=seed,query=q['index'],time_s=q['time_s'],category=category,min=float(g.min()) if finite else None,max=float(g.max()) if finite else None,indices=bad.tolist(),terminal=rec['status'] if q==queries[-1] else None))
                proofs.append(dict(output=str(output),sha256=side['output_sha256'],attempt=side['attempt'],heads=4,input_sha256=side['input_sha256'],noise_sha256=side['noise_sha256']))
    task=scores(point,lane,seeds)
    suffix_cases=[r['seed'] for r in task['rows'] if r['status']=='VALID_MODEL_GRIPPER_ENVELOPE_FAIL' and next(x for x in reversed(envelope) if x['seed']==r['seed'])['category']=='suffix_only']
    v=dict(status='PASS_NATIVE_TASK_RECONSTRUCTION_FOR_DECISION',point=point,lane=lane,governing_release=RELEASE,runtime_release=card['release'],
        execution_fileset_sha256=sha(seal),binding_sha256=sha(ROOT/'model_bindings'/f'{point}.json'),card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),
        checks=dict(result_hashes=True,policy_binding=True,seed_initial_identity=True,input_whitelist=True,fresh_same_tick_RGB_state=True,frozen_actual_noise=True,actual_four_heads=True,native_reconstruction=True,full_execution_required_for_success=True),
        result_files=results,queries=proofs,task=task,envelope=envelope,suffix_only_terminal_failures=suffix_cases,
        full_S4_replay='PASS_ALREADY_COMPLETED' if (ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json').exists() else 'PENDING',new_model_queries=0,new_physics_steps=0,selected_checkpoint='NONE',deployment='NONE')
    freeze(R3/'evidence'/f'FAST_{point}_{lane}.json',v);decision(point,lane)
    print(json.dumps(dict(status=v['status'],point=point,persistent_stacks=task['persistent_stacks'],suffix_only_terminal_failures=suffix_cases)),flush=True)

if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=('TRAIN_PHYSICS','TEST','CONFIRM'));a=p.parse_args();replay(a.point,a.lane)
