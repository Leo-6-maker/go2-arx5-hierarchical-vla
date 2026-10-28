"""Exact registered outputs and literal attempts/heads/frame/start accounting."""
import json
from collections import Counter
from pathlib import Path
from quality_worker import BASE,ROOT,read,sha,write,admitted

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')
POINTS=('BROAD5120','TARGET5120')

def journal(name,cap):
    path=ROOT/'evidence'/f'{name}.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
    assert len(rows)<=cap and [r['ordinal'] for r in rows]==list(range(1,len(rows)+1))
    return rows

def main():
    admitted();assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    outputs={};records=[];registered_cases=set();observations=set()
    def prediction(path,point):
        path=Path(path);assert path.is_relative_to(RAW) and str(path) not in outputs
        receipt=read(path.with_suffix('.receipt.json'))
        assert receipt['status']=='VALID' and receipt['point']==point and receipt['output_sha256']==sha(path)
        assert receipt['binding']==read(ROOT/'model_bindings'/f'{point}.json')
        assert receipt['heads']==4 and receipt['execution_phase']=='quality' and receipt['observed_query_dtypes']==['torch.bfloat16']*4
        assert receipt['native_S4'] and receipt['source_noise_dtype']=='numpy.float32'
        request=receipt['frozen_request'];assert request['output']==str(path)
        if request.get('expected_condition_sha256'):assert receipt['input_sha256']==request['expected_condition_sha256']
        outputs[str(path)]=dict(path=str(path),point=point,sha256=receipt['output_sha256'],receipt_sha256=sha(path.with_suffix('.receipt.json')))
    offline=read(ROOT/'design/RUNTIME_OFFLINE_CARD.json')
    early=read(ROOT/'design/NEW_FIT192.json')['rows'];late=read(ROOT/'design/SUPPORT_CARD.json')['late']
    families=[f['family_id'] for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
    component=read(ROOT/'design/RUNTIME_COMPONENT_CARD.json')['rows']
    for point in POINTS:
        for domain in ('fit','factor','legacy'):
            for row in offline[domain+'_rows']:
                path=RAW/'offline'/point/domain/row['uid']/'record.json';rec=read(path)
                assert rec['point']==point and rec['row']==row and rec['card_sha256']==sha(ROOT/'design/RUNTIME_OFFLINE_CARD.json')
                if rec['status']=='PREQUERY_EXCLUDED_KEEP_DENOMINATOR':assert domain=='factor' and not row['admitted'] and not rec.get('tensor')
                else:
                    assert rec['status']=='VALID' and rec['tensor_sha256']==sha(rec['tensor']);prediction(rec['tensor'],point)
                records.append(dict(path=str(path),sha256=sha(path)))
        for lane,rows,card in (('newfit',early,'NEW_FIT192.json'),('late',late,'SUPPORT_CARD.json')):
            for row in rows:
                for j in ((None,) if lane=='newfit' else (0,1)):
                    folder=RAW/lane/point/row['input_id']
                    if j is not None:folder=folder/f'j{j}'
                    path=folder/'record.json';rec=read(path)
                    assert rec['status']=='VALID' and rec['point']==point and rec['row']==row and rec['card_sha256']==sha(ROOT/'design'/card)
                    if j is not None:assert rec['j']==j
                    assert rec['tensor_sha256']==sha(rec['tensor']);prediction(rec['tensor'],point)
                    records.append(dict(path=str(path),sha256=sha(path)))
        for lane,cases in (('virtual',[f'{f}-{t}-c{c}-{b}' for f in families for t in ('M1','M3') for c in (0,1) for b in ('E0','E1')]),
                           ('component',[r['case'] for r in component])):
            assert len(cases)==(64 if lane=='virtual' else 32)
            for case in cases:
                registered_cases.add((point,case));path=RAW/lane/point/case/'result.json';rec=read(path)
                assert rec['point']==point and rec['evaluation_card_sha256']==sha(ROOT/'design/RUNTIME_EVALUATION_CARD.json')
                if lane=='component':assert rec['component_start']==next(r for r in component if r['case']==case)
                if rec['validity']=='PREQUERY_EXCLUDED':assert lane=='component' and not rec['queries']
                else:
                    assert rec['validity']=='VALID' and rec['arrays_sha256']==sha(path.parent/'trajectory.npz')
                    assert len(rec['queries'])<=(100 if lane=='virtual' else 20)
                    for q in rec['queries']:
                        assert q['prediction_sha256']==sha(q['query']['output']);prediction(q['query']['output'],point)
                observations.update(str(p) for p in (path.parent/'queries').glob('*/observation.npz'))
                records.append(dict(path=str(path),sha256=sha(path)))
    assert len(records)==2*(968+192+592+64+32)
    actual={str(p) for lane in ('offline','newfit','late','virtual','component') for p in (RAW/lane).rglob('prediction.npz')}
    assert actual==set(outputs)
    attempts=journal('model_attempts',18432);heads=journal('model_heads',73728)
    assert {a['request']['output'] for a in attempts}==set(outputs)
    counts=Counter(r['attempt'] for r in heads);assert set(counts)<={r['ordinal'] for r in attempts}
    technical=[]
    for a in attempts:
        path=Path(a['request']['output']);receipt=read(path.with_suffix('.receipt.json'))
        assert a['point']==receipt['point'] and a['request']==receipt['frozen_request'] and a['input_sha256']==receipt['input_sha256']
        assert a['noise_sha256']==receipt['noise_sha256']
        assert 0<=counts[a['ordinal']]<=4
        if receipt['attempt']==a['ordinal']:assert counts[a['ordinal']]==4
        else:
            assert receipt['attempt']>a['ordinal'];technical.append(dict(attempt=a['ordinal'],heads=counts[a['ordinal']],first_valid=receipt['attempt']))
    starts=journal('virtual_starts',208);frames=journal('virtual_frame_starts',15360)
    assert {(r['point'],r['case']) for r in starts}==registered_cases and len(registered_cases)==192
    assert all((r['point'],r['case']) in registered_cases for r in frames) and len(observations)<=len(frames)
    assert len(outputs)<=17584 and len(attempts)-len(outputs)==len(technical)
    write(ROOT/'evidence/QUALITY_OUTPUT_INVENTORY.json',dict(status='PASS_EXACT_ALL_REGISTERED_RESULTS_AND_LITERAL_BUDGETS',
        outputs=list(outputs.values()),records=records,attempts=len(attempts),heads=len(heads),technical_attempts=technical,
        starts=len(starts),frame_attempts=len(frames),saved_frame_pairs=len(observations),
        journals={n:sha(ROOT/'evidence'/f'{n}.jsonl') for n in ('model_attempts','model_heads','virtual_starts','virtual_frame_starts')},
        model_queries_by_audit=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_EXACT_ALL_REGISTERED_RESULTS_AND_LITERAL_BUDGETS',len(outputs),len(attempts),len(heads),flush=True)

if __name__=='__main__':main()
