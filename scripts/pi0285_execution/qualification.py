"""Owner route from real fixed panels and independent CPU receipts, never averages."""
import argparse,json
from contract import ROOT,sha,write
from policy import tasks


def decide(point,train,dev=None,confirm=None):
    selected=tasks(point,'TRAIN');stage_b=len(selected)==2
    fit=all(train[t]>=7 for t in selected)
    if dev is None:
        if fit or point=='SHARED4096-4096':return 'RUN_FIXED_DEV'
        return 'CONTINUE_SAME_PHASE_4096' if point.endswith('-2048') and not point.startswith('REPAIR') else ('COLLECT_FIXED_CORRECTIONS' if not point.startswith('REPAIR') else 'STOP_VALID_QUALITY_FAIL')
    qualified=fit and all(dev[t]>=13 for t in selected) and (not stage_b or dev['paired_both']>=12)
    if not qualified:return 'CONTINUE_SAME_PHASE_4096' if point.endswith('-2048') and not point.startswith('REPAIR') else ('COLLECT_FIXED_CORRECTIONS' if not point.startswith('REPAIR') else 'STOP_VALID_QUALITY_FAIL')
    if confirm is None:return 'FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_B' if stage_b else 'FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_A'
    passed=all(confirm[t]>=26 for t in selected) and (not stage_b or confirm['paired_both']>=24)
    return ('PROTECT_SHARED_TWO_TASK_POLICY' if stage_b else 'PROTECT_PLACE_V1_BEFORE_SHARED') if passed else 'STOP_CONFIRMATION_VALID_QUALITY_FAIL_NO_SECOND_CANDIDATE'


def panel(point,name):
    source=ROOT/'source/evidence'/f'{point}_{name}.json';cpu=ROOT/'audit'/f'{point}_{name}_CPU.json'
    if not source.exists():return None,None
    value=json.loads(source.read_text());check=json.loads(cpu.read_text())
    assert check['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS' and check['source_sha256']==sha(source)
    expected=tasks(point,name);size=dict(TRAIN=8,DEV=16,CONFIRM_A=32,CONFIRM_B=32)[name]
    assert len(value['rows'])==len(check['rows'])==size*len(expected)
    counts={t:sum(r['persistent_success'] for r in value['rows'] if r['task']==t) for t in expected}
    assert all(sum(r['task']==t for r in value['rows'])==size and value['tasks'][t]==dict(requested=size,success=counts[t]) for t in expected)
    assert all(r['status'].startswith('VALID') and r['point']==point and r['split']==name for r in value['rows'])
    assert [(r['task'],r['layout'],r['persistent_success']) for r in check['rows']]==[(r['task'],r['layout'],r['persistent_success']) for r in value['rows']]
    if len(expected)==2:
        counts['paired_both']=sum(all(next(r['persistent_success'] for r in value['rows'] if r['task']==t and r['layout']==seed) for t in expected) for seed in {r['layout'] for r in value['rows']})
        assert counts['paired_both']==value['paired_both_success']
    return counts,dict(source_path=str(source),source_sha256=sha(source),CPU_path=str(cpu),CPU_sha256=sha(cpu))


def route(point):
    train,tproof=panel(point,'TRAIN');assert train is not None
    dev,dproof=panel(point,'DEV');confirmation='CONFIRM_B' if len(tasks(point,'TRAIN'))==2 else 'CONFIRM_A'
    confirmed,cproof=panel(point,confirmation)
    next_action=decide(point,train,dev,confirmed);candidate=ROOT/'qualification'/(confirmation+'_CANDIDATE.json')
    if next_action.startswith('FREEZE_FIRST'):
        binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point]
        value=dict(point=point,model_binding=binding,confirm_allowed=True,development_panels={f'{point}_TRAIN.json':tproof['source_sha256'],f'{point}_DEV.json':dproof['source_sha256']},development_CPU=dict(TRAIN=tproof,DEV=dproof),release='7e840324c71aef47b401bd861b22710774334c97',selected='NONE',deployment='NONE')
        if candidate.exists():assert json.loads(candidate.read_text())==value,'The first qualified candidate is already frozen; no second candidate'
        else:write(candidate,value)
    value=dict(status='REAL_FIXED_PANEL_OWNER_ROUTE_NOT_EXTRA_SCIENTIFIC_AUTHORITY',point=point,TRAIN=train,DEV=dev,confirmation=confirmed,next_action=next_action,proofs=dict(TRAIN=tproof,DEV=dproof,CONFIRM=cproof),selected='NONE',deployment='NONE')
    write(ROOT/'qualification'/(point+'_ROUTE.json'),value);print(json.dumps(value));return value


def self_check():
    assert decide('PLACE4096-2048',{'T1':6})=='CONTINUE_SAME_PHASE_4096'
    assert decide('PLACE4096-2048',{'T1':7})=='RUN_FIXED_DEV'
    assert decide('PLACE4096-2048',{'T1':7},{'T1':13})=='FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_A'
    assert decide('PLACE4096-2048',{'T1':7},{'T1':13},{'T1':25}).startswith('STOP_CONFIRMATION')
    assert decide('PLACE4096-2048',{'T1':7},{'T1':13},{'T1':26})=='PROTECT_PLACE_V1_BEFORE_SHARED'
    assert decide('SHARED4096-4096',{'T1':0,'T2':0})=='RUN_FIXED_DEV'
    t={'T1':7,'T2':7};d={'T1':13,'T2':13,'paired_both':12}
    assert decide('SHARED4096-2048',t,d,{'T1':26,'T2':26,'paired_both':23}).startswith('STOP_CONFIRMATION')
    assert decide('SHARED4096-2048',t,d,{'T1':26,'T2':26,'paired_both':24})=='PROTECT_SHARED_TWO_TASK_POLICY'
    assert decide('REPAIR_A2048-2048',{'T1':6})=='STOP_VALID_QUALITY_FAIL'
    print('PASS_FIRST_CANDIDATE_FIXED_DENOMINATORS_TASK_AND_PAIRED_GATES_NO_CONFIRM_RETRY; model/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point');p.add_argument('--self-check',action='store_true');a=p.parse_args();self_check() if a.self_check else route(a.point)
