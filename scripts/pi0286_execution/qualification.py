"""One predeclared endpoint choice, then one genuinely fresh confirmation."""
import argparse,json,math,time
from pathlib import Path
from contract import ROOT,INSTRUCTIONS,sha,write,guard
from policy import ENDPOINTS,PARENT,panel_seeds

CPU_STATUS='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'
from r2 import CONFIRM_START_STOP,MODEL_STOP as SCIENCE_STOP


def immutable(path,value):
    path=Path(path)
    if path.exists():assert json.loads(path.read_text())==value,'Never replace a frozen candidate/evidence'
    else:write(path,value)


def panel(point,name):
    source=ROOT/'source/evidence'/f'{point}_{name}.json';cpu=ROOT/'audit'/f'{point}_{name}_CPU.json'
    if not source.exists():return None,None
    v=json.loads(source.read_text());c=json.loads(cpu.read_text());seeds=panel_seeds(point,name)
    assert c['status']==CPU_STATUS and c['source_sha256']==sha(source)
    assert len(v['rows'])==len(c['rows'])==len(seeds)
    assert sorted(r['layout'] for r in v['rows'])==seeds
    assert all(r['task']=='T1' and r['point']==point and r['split']==name and r['status'].startswith('VALID') and r['instruction']==INSTRUCTIONS['T1'] for r in v['rows'])
    assert [(r['task'],r['layout'],r['persistent_success']) for r in c['rows']]==[(r['task'],r['layout'],r['persistent_success']) for r in v['rows']]
    count=sum(bool(r['persistent_success']) for r in v['rows']);assert v['tasks']['T1']==dict(success=count,requested=len(seeds))
    return {'T1':count},dict(source_path=str(source),source_sha256=sha(source),CPU_path=str(cpu),CPU_sha256=sha(cpu))


def decide(train,dev=None,confirm=None):
    if dev is None:return 'RUN_FIXED_DEV' if train['T1']>=7 else 'ENDPOINT_CLOSED_VALID_TRAIN_FAIL'
    if train['T1']<7 or dev['T1']<13:return 'ENDPOINT_CLOSED_VALID_DEVELOPMENT_FAIL'
    if confirm is None:return 'ELIGIBLE_WAIT_PREDECLARED_SELECTION'
    return 'PROTECT_PLACE_V2_SOURCE_QUALIFIED' if confirm['T1']>=26 else 'STOP_VALID_NEW_CONFIRM_FAIL_NO_OTHER_ARM_CONFIRMATION'


def route(point):
    guard();assert point in ENDPOINTS
    train,tp=panel(point,'TRAIN');assert train is not None
    dev,dp=panel(point,'DEV');confirmed,cp=panel(point,'CONFIRM_C')
    value=dict(status='PI0286_ACTUAL_INDEPENDENT_CPU_ENDPOINT_ROUTE',point=point,TRAIN=train,DEV=dev,confirmation=confirmed,next_action=decide(train,dev,confirmed),proofs=dict(TRAIN=tp,DEV=dp,CONFIRM_C=cp),selected='NONE',deployment='NONE')
    write(ROOT/'qualification'/(point+'_ROUTE.json'),value);return value


def rank(rows):
    assert rows and len({r['point'] for r in rows})==len(rows)
    return sorted(rows,key=lambda r:(-r['DEV'],-r['TRAIN'],r['DIAG_rejects'],ENDPOINTS.index(r['point'])))[0]


def reserve(point,now=None):
    now=time.time() if now is None else now
    durations=[]
    for p in ROOT.glob('source/evidence/*_TRAIN.json'):
        if any(p.name.startswith(x+'_') for x in ENDPOINTS):
            durations.extend(r['elapsed_wall_s'] for r in json.loads(p.read_text())['rows'] if r['status']=='VALID_SOURCE_COMPLETE')
    for p in ROOT.glob('source/evidence/*_DEV.json'):
        if any(p.name.startswith(x+'_') for x in ENDPOINTS):
            durations.extend(r['elapsed_wall_s'] for r in json.loads(p.read_text())['rows'] if r['status']=='VALID_SOURCE_COMPLETE')
    assert durations and all(math.isfinite(x) and x>0 for x in durations),'Measured complete model episodes required'
    p90=sorted(durations)[math.ceil(.9*len(durations))-1]
    # Owner receipt supplies measured transfer/load/CPU audit reserve, rather than inventing speed.
    path=ROOT/'design/CONFIRM_C_TIME_RESERVE.json';extra=json.loads(path.read_text())
    assert extra['status']=='OWNER_MEASURED_CONFIRM_LOAD_CPU_AUDIT_AND_HANDOFF_RESERVE'
    assert extra['point']==point and extra['measured_extra_seconds']>=0 and now<=extra['valid_until_epoch']
    required=max(2700.,1.25*p90*32+extra['measured_extra_seconds'])
    assert now<=CONFIRM_START_STOP and now+required<=SCIENCE_STOP,('INSUFFICIENT_FRESH_CONFIRM_RESERVE',required,SCIENCE_STOP-now)
    return dict(p90_seconds=p90,completed_episode_measurements=len(durations),required_seconds=required,extra_receipt_sha256=sha(path),checked_epoch=now)


def non_exposure():
    path=ROOT/'design/CONFIRM_C_NON_EXPOSURE.json';v=json.loads(path.read_text())
    assert v['status']=='PASS_FRESH_CONFIRM_C_UNEXPOSED_ALL_RETAINED_HISTORY'
    assert v['layouts']==list(range(285500,285532)) and v['collisions']==[] and v['queried_or_rendered_before_freeze']==0
    return dict(path=str(path),sha256=sha(path))


def select():
    guard();frozen=ROOT/'qualification/CONFIRM_C_CANDIDATE.json'
    if frozen.exists():return json.loads(frozen.read_text())
    close=ROOT/'design/COMPARISON_CLOSE.json';closing=json.loads(close.read_text())
    assert closing['status']=='OWNER_CLOSED_PREDECLARED_PI0286_COMPARISON'
    assert closing['reason'] in ('BOTH_ENDPOINTS_CLOSED','C1_CORRECTION_NOT_ADMITTED','MEASURED_RESERVE_INCOMPLETE_COMPARISON')
    eligible=[];closed=[]
    for point in ENDPOINTS:
        train,tp=panel(point,'TRAIN')
        if train is None:continue
        dev,dp=panel(point,'DEV')
        assert train['T1']<7 or dev is not None,'Close each qualified TRAIN gate through fixed DEV before choice'
        closed.append(point)
        if train['T1']<7 or dev['T1']<13:continue
        diag=ROOT/'audit'/(point+'_DIAG_CPU.json');d=json.loads(diag.read_text())
        assert d['status']=='PASS_CPU_FIXED_DIAG_NATIVE_DECODE_NOISE_PREFIX_FULL_H40_AND_GATE' and d['point']==point and len(d['rows'])==96
        assert d['source_sha256']==sha(ROOT/'diagnostics'/point/'RESULT.json')
        eligible.append(dict(point=point,TRAIN=train['T1'],DEV=dev['T1'],DIAG_rejects=sum(r['gate']['decision']=='REJECT' for r in d['rows']),proofs=dict(TRAIN=tp,DEV=dp),DIAG_CPU_sha256=sha(diag)))
    assert sorted(closed)==sorted(closing['closed_endpoints'])
    if closing['reason']=='BOTH_ENDPOINTS_CLOSED':assert closed==list(ENDPOINTS)
    if not eligible:return dict(status='NO_DEVELOPMENT_QUALIFIED_CANDIDATE',closed_endpoints=closed)
    chosen=rank(eligible);point=chosen['point'];timing=reserve(point);fresh=non_exposure()
    value=dict(status='FROZEN_ONE_PI0286_CANDIDATE_BEFORE_ANY_CONFIRM_C',point=point,model_binding=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point],
               development_panels={f'{point}_{split}.json':chosen['proofs'][split]['source_sha256'] for split in ('TRAIN','DEV')},development_CPU=chosen['proofs'],
               comparison=eligible,comparison_close_sha256=sha(close),comparison_incomplete=len(closed)!=2,selection_order='DEV_DESC_TRAIN_DESC_DIAG_REJECT_ASC_C0',
               new_confirmation=fresh,time_reserve_at_freeze=timing,confirm_allowed=True,noise_rule='PI0286_CONDITION_QUERY_INDEX_NATIVE_BF16_K4_FIXED_ACROSS_ARMS',K=4,selected='NONE',deployment='NONE')
    immutable(frozen,value);return value


def confirmation_gate(point):
    guard();assert point in ENDPOINTS
    candidate=ROOT/'qualification/CONFIRM_C_CANDIDATE.json';c=json.loads(candidate.read_text())
    assert c['point']==point and c['confirm_allowed'] and c['model_binding']==json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point]
    assert non_exposure()==c['new_confirmation']
    for name,h in c['development_panels'].items():assert sha(ROOT/'source/evidence'/name)==h
    train,_=panel(point,'TRAIN');dev,_=panel(point,'DEV');assert train['T1']>=7 and dev['T1']>=13
    assert not any((ROOT/'source/evidence'/f'{other}_CONFIRM_C.json').exists() for other in ENDPOINTS if other!=point)
    reserve(point)
    return c


def self_check():
    assert decide({'T1':6})=='ENDPOINT_CLOSED_VALID_TRAIN_FAIL'
    assert decide({'T1':7})=='RUN_FIXED_DEV'
    assert decide({'T1':7},{'T1':13},{'T1':25}).startswith('STOP_VALID_NEW_CONFIRM_FAIL')
    assert decide({'T1':7},{'T1':13},{'T1':26})=='PROTECT_PLACE_V2_SOURCE_QUALIFIED'
    a=dict(point=ENDPOINTS[0],TRAIN=7,DEV=13,DIAG_rejects=0);b=dict(a,point=ENDPOINTS[1])
    assert rank([b,a])==a and rank([a,dict(b,DEV=14)])['point']==ENDPOINTS[1]
    assert rank([dict(a,DIAG_rejects=1),b])['point']==ENDPOINTS[1]
    print('PASS_PI0286_FIXED_GATES_SINGLE_CANDIDATE_SELECTION_TIEBREAK_AND_CONFIRM25_REJECTION; model/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point');p.add_argument('--select',action='store_true');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else print(json.dumps(select() if a.select else route(a.point)))
