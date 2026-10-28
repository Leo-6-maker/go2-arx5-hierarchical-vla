"""Reuse the original native episode/score, with PI0278's fixed screens and gates."""
import argparse,fcntl,importlib.util,json,sys
from pathlib import Path
from eval_config import bind,COMMON,ROOT,RAW,OLD,OLD_RAW,RELEASE
c=bind();sys.path.pop(0)
spec=importlib.util.spec_from_file_location('qualified_panel','/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code/panel.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,write,freeze,packet=c.read,c.sha,c.write,c.freeze,c.packet

def snapshot_identity(point,ready,receipt,run,contract,ready_sha,contract_sha,run_sha):
    assert ready['point']==point and not ready['resumable']
    if ready['release']==RELEASE:return
    assert ready['release']=='1a20200ff066ef5a003650110cd20106f9a4bf64'
    assert receipt['status']=='PASS_LEGACY_CREATOR_METADATA_RECONCILIATION'
    assert receipt['governing_release']==RELEASE and receipt['point']==point
    assert receipt['original_ready_sha256']==ready_sha and receipt['readout_sha256']==contract_sha and receipt['training_run_sha256']==run_sha
    assert receipt['creator_sha256']=='eb790dbea1e8fc69bb328369aeadc32917e41bae5500d1e4f44f27aab93c55cf'
    assert run['release']==RELEASE and run['arm']=='COVER93' and run['phase']=='main' and run['limit']==4096 and run['route']=='FAST_CACHE2'
    assert ready['arm']=='COVER93' and ready['step']==int(point.removeprefix('COVER93'))
    assert ready['source_full_trainer']==run['root']+'/COVER93/main/training/checkpoint-'+str(ready['step'])
    assert contract['governing_release']==RELEASE and contract['arm']=='COVER93'
    assert contract['initial_weights_revision']=='LOCAL_FOCUS4096@93e500e8ecb6a719d4a53be09d23b35bdeeb83fcc61e75416247a2c0380f4604'

def admit(point):
    assert point in read(ROOT/'design/EVALUATION_CARD.json')['points']
    if point=='FOCUS4096':
        binding=read(COMMON.with_name('PI0277')/'R2/model_bindings/FOCUS4096.json');view=Path(binding['view'])
    else:
        view=RAW/'models'/point;ready=read(view/'READY.json')
        if ready['release']==RELEASE:snapshot_identity(point,ready,None,None,None,None,None,None)
        else:
            source=view/'TRAINING_RUN_SOURCE.json'
            snapshot_identity(point,ready,read(view/'READY_AUTHORITY_RECONCILIATION.json'),read(source),read(view/'PI0277_READOUT_CONTRACT.json'),sha(view/'READY.json'),sha(view/'PI0277_READOUT_CONTRACT.json'),sha(source))
        for r in ready['files']:assert (view/r['path']).stat().st_size==r['bytes'] and sha(view/r['path'])==r['sha256']
        binding=dict(schema='PI0278_LOCAL_MODEL_ONLY',point=point,view=str(view),view_type='model-only',source_revision='LOCAL@'+sha(view/'READY.json'),effective_contract='GRIP',
            readout_contract_sha256=sha(view/'PI0277_READOUT_CONTRACT.json'),statistics_sha256=sha(ROOT/'contracts/TASK/source/dataset/statistics.json'),
            model_files_sha256={r['path']:r['sha256'] for r in ready['files'] if r['path'] in ['config.json','model.safetensors.index.json'] or r['path'].endswith('.safetensors')})
    for n,h in binding['model_files_sha256'].items():assert sha(view/n)==h
    freeze(ROOT/'model_bindings'/f'{point}.json',binding)
    for r in read(ROOT/'design/EVALUATION_FILESET.json')['files']:assert sha(r['path'])==r['sha256']
    freeze(ROOT/'evidence'/f'POLICY_ADMISSION_{point}.json',dict(status='PASS_PI0277_FIXED_POLICY_IDENTITY',legacy_status_name_only=True,governing_release=RELEASE,canonical=6027913340,point=point,model_binding_sha256=sha(ROOT/'model_bindings'/f'{point}.json'),evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),HF='DEFERRED_USER_PRIORITY'))

def runtime(point,lane,seeds):
    # Original source runtime is loaded in its own module. Limit replacements to four,
    # in addition to the original first-valid protection and the new hard ledgers.
    factory=old.runtime(point,lane,seeds)
    base=factory.__globals__['Episode']
    class Episode(base):
        def __init__(self,*a,**kw):
            starts=list((RAW/'episodes').glob('*/attempt-*/START.json'));assert len(starts)-len({p.parent.parent.name for p in starts})<=4
            slot=a[0] if a else kw['slot']
            if (RAW/'episodes'/slot).exists():assert len(starts)-len({p.parent.parent.name for p in starts})<4
            super().__init__(*a,**kw)
    factory.__globals__['Episode']=Episode
    return factory

def panel(point,lane):
    sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code');from client import Server
    card=read(ROOT/'design/EVALUATION_CARD.json');assert point in card['points']
    if lane=='TEST':
        assert int(point.removeprefix('COVER93'))>=1024
        train=read(COMMON/'R3/evidence'/f'FAST_{point}_TRAIN_PHYSICS.json');assert all(train['checks'].values()) and train['task']['persistent_stacks']>=3
        entries=sorted((ROOT/'evidence').glob('DEV_ADMISSION_*.json'));permit=ROOT/'evidence'/f'DEV_ADMISSION_{point}.json'
        if not permit.exists():
            assert len(entries)<2
            if entries:assert point=='COVER934096' and all(read(p)['point']!=point for p in entries)
            freeze(permit,dict(point=point,governing_release=RELEASE,TRAIN_decision_sha256=sha(COMMON/'R3/evidence'/f'FAST_{point}_TRAIN_PHYSICS.json'),DEV_ordinal=len(entries)+1))
    if lane=='CONFIRM':assert read(ROOT/'evidence/FROZEN_CANDIDATE.json')['point']==point
    seeds=[] if lane=='FIT_ONLY' else card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]
    noises={(r['lane'],r['condition'],r['index']):r for r in read(ROOT/'design/EVALUATION_NOISE.json')['rows']};server=Server(point)
    try:
        if lane in ('FIT_ONLY','TRAIN_PHYSICS'):
            results=[]
            for row in card['fit']:
                folder=RAW/'quality'/point/'FIT'/row['uid']/'n0';server.query(packet(row),noises['FIT',row['uid'],0],folder)
                results.append(dict(uid=row['uid'],output=str(folder/'output.npz'),sha256=sha(folder/'output.npz')))
            freeze(ROOT/'evidence'/f'FIT16_{point}.json',dict(results=results,point=point,actual_rows=48,legacy_filename_only=True))
        run=runtime(point,lane,seeds) if seeds else None
        for seed in seeds:run(seed,lane,server,noises);print('PHYSICAL_CASE_RETAINED',point,lane,seed,flush=True)
    finally:server.close()
    if seeds:print(json.dumps(old.scores(point,lane,seeds)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['admit','FIT_ONLY','TRAIN_PHYSICS','TEST','CONFIRM']);p.add_argument('--point',required=True);a=p.parse_args()
    (COMMON/'runtime').mkdir(exist_ok=True)
    with (COMMON/'runtime/quality_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='admit':admit(a.point)
        else:panel(a.point,a.stage)
